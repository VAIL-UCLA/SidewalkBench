"""Per-step route checks (coco-gs @73c1d32 mesh-route-eval semantics).

One :class:`RouteEvaluator` per active route. Each sim step it is fed the robot
state and returns the terminating :class:`Outcome` (or ``None`` while ongoing),
applying the termination priority (as PG: its EvalTracker records success whenever the arrive term fires,
whatever else fires on the same step, and its long-horizon loop checks arrive first, then one failure in the
order dynamic (pedestrian) > static collision > out-of-region > freeze):

    arrive(success)  >  timeout  >  pedestrian contact  >  collision (cone, contact force)  >  off-sidewalk  >  freeze

Rules and defaults (reference CLI defaults unless marked "research spec"):
  * arrive       : research spec -- project start, goal and robot onto the annotated
                   centerline; arrived when robot progress >= goal progress -
                   ``success_tolerance_m`` (0.1 m), one-sided as PG ``arrive``.
  * off-sidewalk : research spec -- robot XY more than ``off_sidewalk_margin_m`` (1.0 m) from the
                   nearest cell centre of the route's walkable region on the semantic grid
                   (flood-filled from the route centerline; ``sampler.SemanticBounds``),
                   or more than ``sampler.OFF_ROUTE_M`` (2.5 m) from the route centerline.
                   (The reference's own default is > 2 m from the centerline.)
  * collision    : summed contact force on the robot's NON-WHEEL bodies >=
                   ``collision_force_threshold_n`` (1.0 N: touching the scene mesh), or any contact between the robot (wheels included) and a cone,
                   or the robot touching a pedestrian (``human_contact``).
  * freeze       : XY displacement over the last ``freeze_window_s`` (5.0 s, window
                   must be full) is < ``freeze_min_displacement_m`` (0.1 m).
  * timeout      : sim time on the route >= ``route_timeout_s`` (PG ``episode_length_s``:
                   60 s short horizon, 3600 s long horizon; set by isaac_env.py).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .sampler import RouteRuntime


# Outcome / reason strings (match coco-gs status vocabulary).
SUCCESS = "success"
COLLISION = "collision"
HUMAN_CONTACT = "human_contact"  # the reason of a collision with a pedestrian
OFF_SIDEWALK = "off_sidewalk"
FREEZE = "freeze"
TIMEOUT = "timeout"

_REASON = {
    SUCCESS: "goal_reached",
    COLLISION: "collision_force",
    OFF_SIDEWALK: "off_sidewalk",
    FREEZE: "freeze",
    TIMEOUT: "timeout",
}

@dataclass
class CheckConfig:
    success_tolerance_m: float = 0.1
    off_sidewalk_margin_m: float = 1.0
    collision_force_threshold_n: float = 1.0
    freeze_window_s: float = 5.0
    freeze_min_displacement_m: float = 0.1
    route_timeout_s: float = 60.0


@dataclass
class StepInput:
    x: float
    y: float
    sim_time_s: float
    collision_force_n: float = 0.0
    cone_contact: bool = False
    human_contact: bool = False


@dataclass
class Outcome:
    status: str
    reason: str


class RouteEvaluator:
    """Evaluates one route step-by-step and reports the terminating outcome."""

    def __init__(self, runtime: RouteRuntime, config: CheckConfig):
        self.runtime = runtime
        self.config = config
        # override the runtime's success tolerance with the check config
        self.runtime.success_tolerance_m = config.success_tolerance_m
        # (sim_time, x, y) samples inside the freeze window
        self._hist: deque[tuple[float, float, float]] = deque()

    def _frozen(self, sim_time_s: float, x: float, y: float) -> bool:
        cfg = self.config
        if cfg.freeze_window_s <= 0.0:
            return False
        h = self._hist
        h.append((sim_time_s, x, y))
        while h and (sim_time_s - h[0][0]) > cfg.freeze_window_s:
            h.popleft()
        age = sim_time_s - h[0][0]
        # Need a full window before judging (reference: age >= window - 1e-6).
        if age < cfg.freeze_window_s - 1.0e-6:
            return False
        _, x0, y0 = h[0]
        return math.hypot(x - x0, y - y0) < cfg.freeze_min_displacement_m

    def reset_freeze(self) -> None:
        """Restart the freeze window (long-horizon recovery teleports the robot)."""
        self._hist.clear()

    def step(self, si: StepInput) -> Outcome | None:
        cfg = self.config
        rt = self.runtime

        # 1. arrive (success): wins over anything else on the same step, as in PG
        if rt.reached_goal(si.x, si.y):
            return Outcome(SUCCESS, _REASON[SUCCESS])

        # 2. timeout (route sim time)
        if si.sim_time_s >= cfg.route_timeout_s:
            return Outcome(TIMEOUT, _REASON[TIMEOUT])

        # 3. collision: a pedestrian first (PG's dynamic collision), then static (cone, contact force)
        if si.human_contact:
            return Outcome(COLLISION, HUMAN_CONTACT)
        if si.cone_contact:
            return Outcome(COLLISION, "cone_contact")
        if cfg.collision_force_threshold_n > 0.0 and si.collision_force_n >= cfg.collision_force_threshold_n:
            return Outcome(COLLISION, _REASON[COLLISION])

        # 4. off-sidewalk (distance to the route's semantic walkable region)
        if rt.is_off_sidewalk(si.x, si.y):
            return Outcome(OFF_SIDEWALK, _REASON[OFF_SIDEWALK])

        # 5. freeze
        if self._frozen(si.sim_time_s, si.x, si.y):
            return Outcome(FREEZE, _REASON[FREEZE])

        return None
