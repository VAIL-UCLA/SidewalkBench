"""
Pedestrian behavior state machine.

Replaces the ping-pong trajectory logic with goal-directed movement driven
by activity programs from the LLM generator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import torch

from sidewalkbench.pg.behavior.orca import compute_orca_velocities
from sidewalkbench.pg.behavior.scene_graph import ACTIONS, POI
from sidewalkbench.pg.utils.scene_random import derive_seed


@dataclass
class ActionStep:
    """One atomic action in a pedestrian's program."""

    action: str  # key in ACTIONS dict
    target_poi_name: str | None  # name of the POI to go to (None for in-place actions)
    duration: float  # seconds to spend on this action
    motion_prompt: str = ""  # text prompt for Motion-LCM


class PedState(Enum):
    WALKING = auto()
    PERFORMING = auto()


class PedestrianAgent:
    """State machine that generates actions on-the-fly with ORCA collision avoidance."""

    INTERACTION_OFFSET = 0.5  # meters
    RADIUS = 0.3  # collision radius (meters)
    MAX_SPEED = 2.0  # m/s
    ARRIVAL_THRESHOLD = 0.3  # meters — close enough to goal

    def __init__(
        self,
        agent_id: int,
        pois: list[POI],
        start_position: np.ndarray,
        terrain_origin: np.ndarray,
        walkable_sampler=None,
        dt: float = 0.2,
        rng_seed: int = 0,
        spline_data: tuple | None = None,  # (arc_to_xy_callable, total_length)
        centerline_projector: Callable | None = None,
    ):
        self.agent_id = agent_id
        self.pois = pois
        self.position = start_position.copy()
        self.velocity = np.zeros(2)  # current 2D velocity
        self.yaw = 0.0
        self.terrain_origin = terrain_origin
        self._walkable_sampler = walkable_sampler
        self.dt = dt
        self._rng = np.random.RandomState(rng_seed)
        self.walk_speed = self._rng.uniform(1.1, 1.65)  # typical walking speed in m/s

        self._state = PedState.WALKING
        self._action_timer = 0.0
        self._goal: np.ndarray | None = None  # (x, y, z) target world position
        self._s_target: float | None = None  # arc-length target on centerline
        self._poi_goal_world: np.ndarray | None = None  # final POI world position
        self._current_motion_prompt = ACTIONS["walk_to"]["motion_prompt"]
        self._pending_perform: ActionStep | None = None
        self._spline_data = spline_data
        self._centerline_projector = centerline_projector

        self._pick_next_action()

    @property
    def state(self) -> PedState:
        return self._state

    @property
    def motion_prompt(self) -> str:
        return self._current_motion_prompt

    def step(self) -> bool:
        """Advance the state machine by one control step.

        Returns:
            ``True`` if the agent is walking and needs a new velocity this step,
            ``False`` if it is performing an in-place action (or finished).
        """
        if self._state == PedState.WALKING:
            return True
        if self._state == PedState.PERFORMING:
            self._tick_performing()
        return False

    # ------------------------------------------------------------------
    # On-the-fly action selection
    # ------------------------------------------------------------------

    def _pick_next_action(self):
        """Randomly decide what to do next."""
        objects = [p for p in self.pois if p.category != "waypoint"]
        waypoints = [p for p in self.pois if p.category == "waypoint"]

        roll = self._rng.random()
        if roll < 0.5 and objects:
            poi = objects[self._rng.randint(len(objects))]
            self._pending_perform = ActionStep(
                action="look_around",
                target_poi_name=None,
                duration=5.0,
                motion_prompt=ACTIONS["look_around"]["motion_prompt"],
            )
        elif waypoints:
            poi = waypoints[self._rng.randint(len(waypoints))]
            self._pending_perform = None
        else:
            poi = (objects or waypoints)[0]
            self._pending_perform = None

        goal = self._interaction_point(poi)
        self._poi_goal_world = goal.copy()
        # For sidewalk waypoints, randomize x across the walkable width
        if poi.category == "waypoint" and self._walkable_sampler is not None:
            xx, _ = self._walkable_sampler(walkable=True, num_samples=1, rng=self._rng)
            goal[0] = float(xx[0]) + self.terrain_origin[0]
            self._poi_goal_world = goal.copy()
        self._goal = goal
        self._s_target = None
        # Project POI onto centerline for arc-length navigation (local coords)
        if self._spline_data is not None and self._centerline_projector is not None:
            local_x = goal[0] - self.terrain_origin[0]
            local_y = goal[1] - self.terrain_origin[1]
            _s, _ = self._centerline_projector(local_x, local_y)
            self._s_target = _s
        self._state = PedState.WALKING
        self._current_motion_prompt = ACTIONS["walk_to"]["motion_prompt"]

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def preferred_velocity(self) -> np.ndarray | None:
        """Preferred (goal-seeking) velocity for the current walk goal.

        Advances any arrival/transition logic.  Returns ``None`` when the agent
        transitioned this step (arrived, picked a new action, or is mid-approach)
        and therefore should not be integrated by the caller.
        """
        if self._goal is None:
            self._pick_next_action()
            return None

        # Centerline-based navigation
        if (
            self._s_target is not None
            and self._spline_data is not None
            and self._centerline_projector is not None
            and self._poi_goal_world is not None
        ):
            dist_to_poi = np.linalg.norm(self._poi_goal_world[:2] - self.position[:2])
            # Already at the object — handle arrival
            if dist_to_poi < self.ARRIVAL_THRESHOLD:
                self.velocity = np.zeros(2)
                if self._pending_perform is not None:
                    step = self._pending_perform
                    self._pending_perform = None
                    self._state = PedState.PERFORMING
                    self._action_timer = step.duration
                    self._current_motion_prompt = step.motion_prompt
                else:
                    self._pick_next_action()
                return None
            # Close enough — walk directly toward the object
            if dist_to_poi < 2.0:
                direction = self._poi_goal_world[:2] - self.position[:2]
            else:
                # Follow centerline (projector uses local coords relative to terrain_origin)
                arc_to_xy, total_len = self._spline_data
                local_x = self.position[0] - self.terrain_origin[0]
                local_y = self.position[1] - self.terrain_origin[1]
                s_current, _ = self._centerline_projector(local_x, local_y)
                s_diff = self._s_target - s_current
                # Reached nearest centerline point — done
                if abs(s_diff) < 0.5:
                    self._pick_next_action()
                    return None
                # Look-ahead along centerline
                sign = 1.0 if s_diff > 0 else -1.0
                s_ahead = s_current + sign * min(3.0, abs(s_diff))
                s_ahead = max(0.0, min(s_ahead, total_len))
                xy = arc_to_xy(np.array([s_ahead]))
                goal_xy = np.array([float(xy[0, 0]) + self.terrain_origin[0], float(xy[0, 1]) + self.terrain_origin[1]])
                direction = goal_xy - self.position[:2]
        else:
            # Check arrival — straight-line
            dist_to_goal = np.linalg.norm(self._goal[:2] - self.position[:2])
            if dist_to_goal < self.ARRIVAL_THRESHOLD:
                self.velocity = np.zeros(2)
                if self._pending_perform is not None:
                    step = self._pending_perform
                    self._pending_perform = None
                    self._state = PedState.PERFORMING
                    self._action_timer = step.duration
                    self._current_motion_prompt = step.motion_prompt
                else:
                    self._pick_next_action()
                return None
            # Compute preferred velocity toward goal
            direction = self._goal[:2] - self.position[:2]

        return _normalized_safe(direction) * self.walk_speed

    def apply_velocity(self, new_vel: np.ndarray):
        """Integrate a (possibly collision-avoiding) velocity into the pose."""
        self.velocity = new_vel
        self.position[:2] = self.position[:2] + new_vel * self.dt
        # Update yaw to face movement direction
        if _norm_sq_safe(new_vel) > 0.1:
            self.yaw = float(np.arctan2(new_vel[1], new_vel[0]))

    def _tick_performing(self):
        self._action_timer -= self.dt
        if self._action_timer <= 0:
            self._pick_next_action()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _interaction_point(self, poi: POI) -> np.ndarray:
        direction = self.position[:2] - poi.position[:2]
        dist = np.linalg.norm(direction)
        if dist > 0.01:
            direction = direction / dist
        else:
            direction = np.array([1.0, 0.0])
        offset_2d = poi.position[:2] + direction * self.INTERACTION_OFFSET
        z = poi.position[2] if len(poi.position) > 2 else 0.0
        return np.array([offset_2d[0], offset_2d[1], z])


def _normalized_safe(v: np.ndarray) -> np.ndarray:
    l = np.dot(v, v)
    if l < 1e-12:
        return np.array([1.0, 0.0])
    return v / np.sqrt(l)


def _norm_sq_safe(v: np.ndarray) -> float:
    return float(np.dot(v, v))


# ---------------------------------------------------------------------------
# Manager: orchestrates all pedestrian agents for a terrain cell
# ---------------------------------------------------------------------------


class PedestrianBehaviorManager:
    """
    Drop-in replacement for the ping-pong trajectory logic in
    SidewalkScenePG.write_data_to_sim().

    Usage:
        manager = PedestrianBehaviorManager(...)
        # In write_data_to_sim():
        manager.tick()
        states = manager.get_states(terrain_idx)  # returns list of (x,y,z,yaw,traj_idx,dir)
    """

    def __init__(self, robot_radius: float = 0.4):
        self.agents: dict[int, list[PedestrianAgent]] = {}
        self._prev_prompts: dict[int, list[str]] = {}
        self.animation_requests: list[tuple[int, int, str]] = []
        self._static_obstacles: dict[int, list[tuple[np.ndarray, float]]] = {}  # terrain_idx -> [(pos2d, radius)]
        self._robot_states: dict[
            int, tuple[np.ndarray, np.ndarray, float]
        ] = {}  # terrain_idx -> (pos2d, vel2d, radius)
        self._active_masks: dict[int, np.ndarray] = {}  # terrain_idx -> bool [num_agents]
        self.robot_radius = robot_radius
        self.dt = 0.2
        self.max_speed = PedestrianAgent.MAX_SPEED

    def set_active_masks(self, masks: dict[int, np.ndarray] | None):
        """Restrict full ORCA to active agents; others use goal steering.

        Args:
            masks: ``terrain_idx -> bool array`` over that cell's agents, or
                ``None`` to treat every agent as active (default behavior).
        """
        self._active_masks = masks or {}

    def set_static_obstacles(self, terrain_idx: int, obstacles: list[tuple[np.ndarray, float]]):
        """Set static obstacle positions for a terrain cell. obstacles = [(pos2d, radius), ...]"""
        self._static_obstacles[terrain_idx] = obstacles

    def set_robot_state(self, terrain_idx: int, pos2d: np.ndarray, vel2d: np.ndarray):
        """Update robot state for collision avoidance."""
        self._robot_states[terrain_idx] = (pos2d, vel2d, self.robot_radius)

    def register_agents(
        self,
        terrain_idx: int,
        pois: list[POI],
        initial_positions: list[np.ndarray],
        terrain_origin: np.ndarray,
        walkable_sampler=None,
        dt: float = 0.2,
        seed: int = 0,
        spline_data: tuple | None = None,  # (arc_to_xy_callable, total_length)
        centerline_projector: Callable | None = None,
    ):
        agents = []
        for i, pos in enumerate(initial_positions):
            agents.append(
                PedestrianAgent(
                    agent_id=i,
                    pois=pois,
                    start_position=pos,
                    terrain_origin=terrain_origin,
                    walkable_sampler=walkable_sampler,
                    dt=dt,
                    rng_seed=derive_seed(seed, "agent", i),
                    spline_data=spline_data,
                    centerline_projector=centerline_projector,
                )
            )
        self.agents[terrain_idx] = agents
        self._prev_prompts[terrain_idx] = [a.motion_prompt for a in agents]
        if agents:
            self.dt = agents[0].dt
            self.max_speed = agents[0].MAX_SPEED

    def unregister_agents(self, terrain_idx: int):
        """Drop all agents and cached state for a terrain cell."""
        self.agents.pop(terrain_idx, None)
        self._prev_prompts.pop(terrain_idx, None)
        self._static_obstacles.pop(terrain_idx, None)
        self._robot_states.pop(terrain_idx, None)
        self._active_masks.pop(terrain_idx, None)

    def tick(self):
        """Advance all agents. Populates ``animation_requests``.

        Active agents (in the robot front-camera field of view, per
        :meth:`set_active_masks`) use the vectorized ORCA solver across the whole
        terrain cell in one call.  All other agents fall back to cheap goal
        steering without collision avoidance, which keeps off-camera
        pedestrians moving without the O(P^2) ORCA cost.
        """
        self.animation_requests.clear()
        for tidx, agents in self.agents.items():
            prev = self._prev_prompts.get(tidx, [])
            static_obs = self._static_obstacles.get(tidx, [])
            robot = self._robot_states.get(tidx)
            mask = self._active_masks.get(tidx)
            num_agents = len(agents)
            if num_agents == 0:
                continue

            # 1. Advance state machines and gather preferred velocities.
            preferred: list[np.ndarray | None] = [None] * num_agents
            walking = [False] * num_agents
            for i, agent in enumerate(agents):
                if agent.step():
                    walking[i] = True
                    preferred[i] = agent.preferred_velocity()

            is_active = [
                walking[i] and preferred[i] is not None and (mask is None or bool(mask[i]))
                for i in range(num_agents)
            ]
            active_idx = [i for i in range(num_agents) if is_active[i]]

            # 2. Vectorized ORCA for the whole terrain cell (single call). Every
            #    agent participates as a neighbor so active pedestrians still
            #    avoid dormant ones; ORCA velocities are only applied to active
            #    pedestrians below.
            new_velocities = compute_orca_velocities(
                np.stack([a.position[:2] for a in agents]),
                np.stack([a.velocity for a in agents]),
                np.stack([preferred[i] if preferred[i] is not None else np.zeros(2) for i in range(num_agents)]),
                np.array([a.RADIUS for a in agents], dtype=float),
                self.max_speed,
                static_obstacles=static_obs,
                robot=robot,
                tau=3.0,
                dt=self.dt,
            )

            # 3. Active agents integrate the ORCA velocity; dormant walking
            #    agents fall back to cheap goal steering without collision
            #    avoidance.
            for i in range(num_agents):
                if not (walking[i] and preferred[i] is not None):
                    continue
                agents[i].apply_velocity(new_velocities[i] if is_active[i] else preferred[i])

            # 4. Prompt tracking. Dormant agents keep their previous prompt so a
            #    change is regenerated once they become active again.
            active_set = set(active_idx)
            new_prev = list(prev) + [a.motion_prompt for a in agents[len(prev) :]]
            for i, agent in enumerate(agents):
                old = prev[i] if i < len(prev) else agent.motion_prompt
                if i in active_set:
                    new_prompt = agent.motion_prompt
                    if old != new_prompt:
                        self.animation_requests.append((tidx, i, new_prompt))
                    new_prev[i] = new_prompt
                else:
                    new_prev[i] = old
            self._prev_prompts[tidx] = new_prev

    def get_states(self, terrain_idx: int) -> torch.Tensor | None:
        """
        Returns a tensor of shape (num_agents, 6) matching the format of
        clone_dynamic_states: (x, y, z, yaw, traj_idx, direction).

        traj_idx and direction are set to 0/1 for compatibility with the
        existing rendering pipeline.
        """
        if terrain_idx not in self.agents:
            return None
        agents = self.agents[terrain_idx]
        if not agents:
            return None

        states = []
        for agent in agents:
            pos = agent.position
            yaw = agent.yaw
            # traj_idx=0, direction=1 — dummy values for render compat
            states.append(
                torch.tensor(
                    [pos[0], pos[1], pos[2], yaw, 0.0, 1.0],
                    dtype=torch.float32,
                )
            )
        return torch.stack(states, dim=0)
