# Copyright (c) 2022-2025, The SidewalkBench-PG Project Developers.
# Author: Honglin He
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import copy
import json
import os
from glob import glob

import numpy as np
import nvdiffrast.torch as dr
import omni.kit.commands
import omni.usd
import torch
import trimesh
from isaacsim.core.cloner import Cloner
from PIL import Image
from pxr import PhysxSchema, UsdPhysics

from sidewalkbench.pg.utils.compat import patch_physx_manager_for_cloner

patch_physx_manager_for_cloner()

import isaaclab.sim as sim_utils
from isaaclab.scene.interactive_scene import InteractiveScene
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import set_prim_visibility
from sidewalkbench.pg.utils.rotations import axis_angle_to_matrix, matrix_to_axis_angle
from sidewalkbench.pg.utils.math import _quat_to_yaw, _yaw_to_quat, look_at, lookat_from_pose, perspective, spawn_mesh_cuboid
from sidewalkbench.pg.primitives.navigation.random_env_cfg import SceneCfg
from sidewalkbench.pg.utils.scene_random import SceneRandom, derive_seed


class SidewalkScenePG(InteractiveScene):
    """
    A scene that contains entities added to the simulation.
    """

    def __init__(self, cfg: SceneCfg):
        """Initializes the scene.

        Args:
            cfg: The configuration class for the scene.
        """
        self.cfg = cfg
        self.scenario_generation_method = self.cfg.pg_config["scenario_generation_method"]
        self.cfg.replicate_physics = True
        self.pg_config = self.cfg.pg_config
        self.cfg.pg_config = None
        self.randomization = SceneRandom(
            self.pg_config.get("seed", 42), long_horizon=self.pg_config.get("long_horizon", False)
        )
        self.retrying_reset = False
        self.count = 0
        super().__init__(self.cfg)
        self.simplified = self.pg_config["simplified"]
        self.num_walkable_objects_per_terrain = self.pg_config["num_walkable_objects_per_terrain"]
        self.num_non_walkable_objects_per_terrain = self.pg_config["num_non_walkable_objects_per_terrain"]
        self.num_peds_per_terrain = self.pg_config["num_peds_per_terrain"]
        self.scenario_type = self.pg_config.get("scenario_type", "default")
        self.terrain_segment = self.pg_config.get("terrain_segment", "straight")
        self.ped_lateral = self.pg_config.get("ped_lateral", "center")
        self.ped_crossing_direction = self.pg_config.get("ped_crossing_direction", "left_to_right")
        self.robot_bbox = self.pg_config.get("robot_bbox", [0.75953799486, 0.60900770638, 0.70764710009])
        self.robot_height_offset = self.pg_config.get("robot_height_offset", 0.27668)
        self.obstacle_id = self.pg_config.get("obstacle_id", -1)

        stage = omni.usd.get_context().get_stage()
        self.obs_cloner = Cloner(stage=stage)
        self.obs_cloner.define_base_env("/World/Dataset")
        self.clone_static_paths = {}
        self._clone_static_counter = 0
        self.clone_static_states = None
        self.clone_dynamic_states = None
        self.clone_dynamic_trajectories = None
        self.clone_dynamic_velocities = None
        self._ped_speeds = None
        # Synchronized-episode pedestrian control: ``frozen_env_mask`` (bool per env,
        # set by the environment) marks environments that finished the current round;
        # their pedestrians are not ticked. ``pause_pedestrians`` freezes the pedestrian
        # clock entirely (used across reset + the round-start warm-up step).
        self.frozen_env_mask = None
        self.pause_pedestrians = False
        self.env_goals = torch.zeros(self.num_envs, 2, device=self.device)
        # Forward heading [rad] at the goal, when the terrain sampler provides it
        # (NaN otherwise); used to orient the optional goal camera.
        self.env_goal_yaws = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.route_lengths = torch.zeros(self.num_envs, device=self.device)
        self.last_arc_length = torch.zeros(self.num_envs, device=self.device)
        self.arc_length_goal = torch.zeros(self.num_envs, device=self.device)
        self.arc_length_start = torch.zeros(self.num_envs, device=self.device)
        self.decimation = None
        self.dt = None
        self.dtype = torch.float32
        self.render_device = self.device
        self.pre_render_device = "cpu"

        # setup behavior system
        self.use_behavior = (
            self.pg_config.get("use_behavior", False) and self.pg_config.get("terrain_type") == "block_sidewalk"
        )
        self._behavior_manager = None
        self._scene_graph = None
        if self.use_behavior:
            from sidewalkbench.pg.behavior.pedestrian_agent import PedestrianBehaviorManager
            from sidewalkbench.pg.behavior.scene_graph import SceneGraph

            self._scene_graph = SceneGraph()
            self._behavior_manager = PedestrianBehaviorManager(robot_radius=0.5)

        # Behavior efficiency controls: keep pedestrians inside the robot
        # front-camera view (or close to the robot) on full ORCA + animation
        # updates, and let the rest fall back to cheap goal steering.
        self.behavior_fov_cull = bool(self.pg_config.get("behavior_fov_cull", True))
        self.behavior_fov_margin = float(self.pg_config.get("behavior_fov_margin", 0.15))
        self.behavior_radius = float(self.pg_config.get("behavior_radius", 25.0))
        # Pedestrians within this distance of the robot stay fully active
        # (ORCA + animation + rendering) regardless of camera orientation.
        self.behavior_proximity_radius = float(self.pg_config.get("behavior_proximity_radius", 5.0))
        # Pedestrian behavior/ORCA update rate [Hz], decoupled from the policy
        # step rate (20 Hz for the streaming policies).
        self.behavior_update_hz = float(self.pg_config.get("behavior_update_hz", 5.0))
        self._behavior_period: int | None = None
        self._animation_cache: dict[str, torch.Tensor] = {}

        # setup human renderer
        from smplx import SMPL

        self.smpl = SMPL(model_path="assets/smplx/SMPL_NEUTRAL.pkl").to(self.pre_render_device)
        self.smpl_faces = torch.tensor(self.smpl.faces.astype("int32"), device=self.pre_render_device)
        from sidewalkbench.pg.integrations.motionlcm import load_motion_lcm

        self.motion_lcm_model = load_motion_lcm()
        smpl_uv_mesh = trimesh.load("assets/smplx/smpl_uv.obj")
        smpl_uv_mesh.merge_vertices(merge_tex=True)
        self.smpl_uv_coords = torch.tensor(
            smpl_uv_mesh.visual.uv, device=self.pre_render_device, dtype=self.dtype
        )  # [V, 2] in [0,1]
        self.glctx = dr.RasterizeCudaContext(device=self.render_device)

        hide_robots = self.num_envs > 1
        if hide_robots:
            for i in range(self.num_envs):
                prim = self.stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")
                if prim.IsValid():
                    set_prim_visibility(prim, False)

        # Resample textures when env 0 resets unless fixed textures are requested.
        self._terrain_material_randomizer = None
        if not self.pg_config.get("use_fixed_texture", False):
            from sidewalkbench.pg.utils.terrain_materials import TerrainMaterialRandomizer

            terrain_cfg = self.terrain.cfg
            self._terrain_material_randomizer = TerrainMaterialRandomizer(
                ground_prim_path=terrain_cfg.prim_path,
                walkable_group_mdl_paths=[
                    getattr(material, "mdl_path", None) for material in (terrain_cfg.walkable_visual_materials or [])
                ],
                non_walkable_group_mdl_paths=[
                    getattr(material, "mdl_path", None)
                    for material in (terrain_cfg.non_walkable_visual_materials or [])
                ],
                seed=int(self.pg_config.get("terrain_texture_seed", 0)),
                sky_light_prim_path=getattr(self.cfg.sky_light, "prim_path", None),
            )

    def begin_reset(self, env_ids):
        """Select per-cell streams once, before start/goal and object sampling."""
        rows = self.terrain.terrain_levels[env_ids].tolist()
        cols = self.terrain.terrain_types[env_ids].tolist()
        self.randomization.begin_reset(zip(rows, cols), retry=self.retrying_reset)

    def _ped_period_steps(self) -> int:
        """Number of physics substeps between pedestrian/behavior updates."""
        if self._behavior_period is None:
            dt = self.dt if self.dt else 0.005
            self._behavior_period = max(1, int(round(1.0 / (max(self.behavior_update_hz, 1e-3) * dt))))
        return self._behavior_period

    @property
    def ped_dt(self) -> float:
        """Elapsed simulation time between pedestrian/behavior updates [s]."""
        return (self.dt if self.dt else 0.0) * self._ped_period_steps()

    def write_data_to_sim(self):
        super().write_data_to_sim()
        # While paused, pedestrians neither advance nor (trigger). The clock is held
        # too, so the first tick lands on the first action step of the new round.
        if self.pause_pedestrians:
            return
        if self.clone_dynamic_trajectories and self.count % self._ped_period_steps() == 0:
            if self.use_behavior and self._behavior_manager and self._behavior_manager.agents:
                self._tick_behavior()
            else:
                self._tick_scripted()
        self.count += 1

    def _frozen_terrains(self) -> set[int]:
        """Terrain cells whose environments have all finished the current round.

        Pedestrians belong to a terrain cell; a cell is frozen only when every
        environment mapped to it is frozen (guards against envs sharing a cell).
        """
        mask = self.frozen_env_mask
        if mask is None or not bool(mask.any()):
            return set()
        num_cols = self.terrain.terrain_origins.shape[1]
        env_terrain = (self.terrain.terrain_levels * num_cols + self.terrain.terrain_types).tolist()
        per_terrain: dict[int, list[bool]] = {}
        for env_id, tidx in enumerate(env_terrain):
            per_terrain.setdefault(tidx, []).append(bool(mask[env_id]))
        return {tidx for tidx, flags in per_terrain.items() if all(flags)}

    def clear_terrains(self, terrain_idxs) -> None:
        """Remove pedestrians, spawned clones, and behavior agents for terrain cells.

        Used when the long-horizon route advances to a new terrain row so that
        finished rows do not keep pedestrians or spawned objects alive.
        """
        stage = self.stage
        for raw_tidx in terrain_idxs:
            tidx = int(raw_tidx)
            for path in getattr(self, "clone_static_paths", {}).pop(tidx, []):
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    stage.RemovePrim(path)
            for name in (
                "clone_static_states",
                "clone_dynamic_states",
                "clone_dynamic_trajectories",
                "clone_dynamic_velocities",
                "clone_dynamic_animations",
                "clone_dynamic_textures",
                "_ped_speeds",
            ):
                store = getattr(self, name, None)
                if isinstance(store, dict):
                    store.pop(tidx, None)
            if self._behavior_manager is not None:
                self._behavior_manager.unregister_agents(tidx)

    def _tick_behavior(self):
        """Feed robot state to ORCA, advance all agents, and refresh animations."""
        num_rows, num_cols = self.terrain.terrain_origins.shape[:2]
        for env_id in range(self.num_envs):
            tidx = (self.terrain.terrain_levels[env_id] * num_cols + self.terrain.terrain_types[env_id]).item()
            robot_pos = self["robot"].data.root_pos_w[env_id, :2].cpu().numpy()
            robot_vel = self["robot"].data.root_lin_vel_w[env_id, :2].cpu().numpy()
            self._behavior_manager.set_robot_state(tidx, robot_pos, robot_vel)

        # Restrict full ORCA to pedestrians inside the front-camera view
        if self.behavior_fov_cull:
            self._behavior_manager.set_active_masks(self._compute_behavior_active_masks())
        else:
            self._behavior_manager.set_active_masks(None)

        self._behavior_manager.tick()
        for n in self.clone_dynamic_states.keys():
            new_states = self._behavior_manager.get_states(n)
            if new_states is not None:
                self.clone_dynamic_states[n] = new_states.to(self.device)
        self._apply_animation_requests()

    def _tick_scripted(self):
        """Advance scripted (non-behavior) pedestrians along their scripted trajectories."""
        frozen = self._frozen_terrains()
        for n in self.clone_dynamic_trajectories.keys():
            if self.clone_dynamic_trajectories[n] is None or n in frozen:
                continue
            robot_xy = self["robot"].data.root_pos_w[n, :2].cpu().numpy()
            for i, (ped_traj, ped_state) in enumerate(
                zip(self.clone_dynamic_trajectories[n], self.clone_dynamic_states[n])
            ):
                self._update_scripted_ped(n, i, ped_traj, ped_state, robot_xy)

    def _update_scripted_ped(self, n, i, ped_traj, ped_state, robot_xy):
        """Per-scenario update for one scripted pedestrian.

        Mutates ``self.clone_dynamic_states[n][i]`` in place and updates the
        cached velocity. ``n`` is the terrain cell index and ``i`` the pedestrian.
        """
        old_pos = ped_state[:2].clone()
        traj_idx, direction = ped_state[-2], ped_state[-1]
        ped_xy = ped_state[:2].cpu().numpy()
        state = self.clone_dynamic_states[n][i]

        if self.scenario_type in ("obstructing", "conversing", "queueing", "human_lying"):
            # Static: stay at initial position, no movement
            pass
        elif self.scenario_type == "gesturing":
            # Static position, but face the robot dynamically
            vec = robot_xy - ped_xy
            state[3] = float(np.arctan2(vec[1], vec[0]))
        elif self.scenario_type == "active_collision":
            dist = float(np.linalg.norm(robot_xy - ped_xy))
            if dist < 1.0 or state[-1] == 2:
                state[-1] = 2
            else:
                # Chase the robot
                state[-1] = 1
                vec = robot_xy - ped_xy
                vec_norm = float(np.linalg.norm(vec)) or 1.0
                speed = getattr(self, "_ped_speeds", {}).get(n, 1.0)
                new_xy = ped_xy + speed * self.ped_dt * (vec / vec_norm)
                state[:2] = torch.tensor(new_xy, device=self.device, dtype=self.dtype)
                state[3] = float(np.arctan2(vec[1], vec[0]))
        elif self.scenario_type == "lateral_approaching":
            self._update_scripted_lateral(n, i, ped_traj, ped_state, robot_xy, ped_xy)
        elif self.scenario_type == "overtaking":
            self._update_scripted_overtaking(n, i, ped_traj, ped_state, robot_xy, ped_xy)
        elif self.scenario_type != "default":
            # One-way: advance until end, then stop
            if int(traj_idx) < ped_traj.shape[0] - 1:
                traj_idx = traj_idx + 1
                state[-2] = traj_idx
                state[:3] = ped_traj[int(traj_idx), :3]
                state[3] = ped_traj[int(traj_idx), 3]
        else:
            # Default: back-and-forth
            if int(traj_idx) == ped_traj.shape[0] - 1:
                direction = 0
            if int(traj_idx) == 0:
                direction = 1
            traj_idx = traj_idx + (1 if direction == 1 else -1)
            state[-1] = direction
            state[-2] = traj_idx
            state[:3] = ped_traj[int(traj_idx), :3]
            yaw = ped_traj[int(traj_idx), 3]
            if direction == 0:
                yaw = yaw + np.pi
            state[3] = yaw

        # Compute velocity for this pedestrian
        if self.clone_dynamic_velocities[n] is not None:
            new_pos = self.clone_dynamic_states[n][i][:2]
            self.clone_dynamic_velocities[n][i] = (new_pos - old_pos) / self.ped_dt

    def _update_scripted_lateral(self, n, i, ped_traj, ped_state, robot_xy, ped_xy):
        from sidewalkbench.pg.terrains.intersection import _cut_in_params

        num_cols = self.terrain.cfg.terrain_generator.num_cols
        row, col = n // num_cols, n % num_cols
        sampler = self.terrain.terrain_trajectory_samplers[row][col]
        cut = _cut_in_params.get(id(sampler))
        state = self.clone_dynamic_states[n][i]
        if cut is None:
            if self.clone_dynamic_velocities[n] is not None:
                self.clone_dynamic_velocities[n][i][:] = 0.0
            return
        traj_idx = ped_state[-2]
        if traj_idx == 0:
            # Waiting: trigger when robot is near path midpoint
            projector = self.terrain.terrain_centerline_projectors[row][col]
            terrain_origin = self.terrain.terrain_origins[row, col]
            robot_local = self["robot"].data.root_pos_w[n, :2] - terrain_origin[:2]
            s, _ = projector(robot_local[0].item(), robot_local[1].item())
            s_goal = self.arc_length_goal[n].item()
            if s >= s_goal / 2.0 - 3.0:
                # Switch to cutting: swap to walking animation, capture robot speed
                self.clone_dynamic_animations[n][i] = self._generate_animation("a person is walking.")
                v_xy = self["robot"].data.root_vel_w[n, :2].cpu().numpy()
                delta_xy = robot_xy - ped_xy
                angle = np.arccos(delta_xy.dot(v_xy) / ((np.linalg.norm(delta_xy) * np.linalg.norm(v_xy)) + 1e-6))
                v_robot = float(np.linalg.norm(v_xy))
                v_lat = np.abs(v_robot * np.sin(angle) / np.cos(angle)) if v_robot > 0 else 0.0
                state[-2] = float(v_lat)  # store lat speed as flag
        elif traj_idx > 0:
            # Cutting: walk laterally at constant stored speed
            v_lat = ped_state[-2].item()
            step = v_lat * self.ped_dt
            accum = ped_state[-1].item() + step
            if accum < cut["sw"] - 0.6:
                state[:2] = torch.tensor(ped_xy - step * np.array([cut["px"], cut["py"]]), device=self.device, dtype=self.dtype)
                state[-1] = accum

    def _update_scripted_overtaking(self, n, i, ped_traj, ped_state, robot_xy, ped_xy):
        from sidewalkbench.pg.terrains.intersection import _overtake_params

        num_cols = self.terrain.cfg.terrain_generator.num_cols
        row, col = n // num_cols, n % num_cols
        sampler = self.terrain.terrain_trajectory_samplers[row][col]
        ot = _overtake_params.get(id(sampler))
        state = self.clone_dynamic_states[n][i]
        if ot is None:
            if self.clone_dynamic_velocities[n] is not None:
                self.clone_dynamic_velocities[n][i][:] = 0.0
            return
        traj_idx = ped_state[-2]
        dt = self.ped_dt
        if int(traj_idx) == 0:
            # Phase 1: overtaking — dynamic speed = v_robot + 1.0
            v_xy = self["robot"].data.root_vel_w[n, :2].cpu().numpy()
            speed = float(np.linalg.norm(v_xy)) + 1.0
            cos_a, sin_a = np.cos(np.deg2rad(10)), np.sin(np.deg2rad(10))
            fwd, lat = speed * cos_a * dt, speed * sin_a * dt
            new_xy = ped_xy + fwd * np.array(ot["forward"]) + lat * np.array(ot["toward"])
            vel = fwd * np.array(ot["forward"]) + lat * np.array(ot["toward"])
            accum = ped_state[-1].item() + lat
            if accum >= 1.0:
                # Phase 2: snap to closest point on ped_traj
                dists = np.linalg.norm(ped_traj[:, :2].cpu().numpy() - new_xy, axis=1)
                closest = int(np.argmin(dists))
                state[-2] = closest
                state[:2] = ped_traj[closest, :2]
                state[3] = ped_traj[closest, 3]
            else:
                state[-1] = accum
                state[:2] = torch.tensor(new_xy, device=self.device, dtype=self.dtype)
                state[3] = float(np.arctan2(vel[1], vel[0]))
        else:
            # Phase 2: advance along ped_traj (one-way)
            if int(traj_idx) < ped_traj.shape[0] - 1:
                traj_idx = traj_idx + 1
                state[-2] = traj_idx
                state[:3] = ped_traj[int(traj_idx), :3]
                state[3] = ped_traj[int(traj_idx), 3]

    def _postprocess_smpl_verts(self, verts: torch.Tensor) -> torch.Tensor:
        """Ground-align and convert SMPL vertices from (z-forward, y-up) to (x-forward, z-up).

        Args:
            verts: SMPL vertices in source frame, shape ``(B, L, V, 3)``.

        Returns:
            Vertices in target frame, same shape.
        """
        verts = verts.clone()
        verts[:, :, :, 1] -= verts[:, :, :, 1].amin(dim=-1, keepdim=True)  # align feet to the ground
        # (x, y, z) -> (z, x, y)
        return torch.stack([verts[..., 2], verts[..., 0], verts[..., 1]], dim=-1)

    def _smpl_animation_from_keypoints(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Run MotionLCM joints through SMPL for a batch of pedestrians.

        Args:
            keypoints: MotionLCM joints, shape ``(B, L, J, 3)``.

        Returns:
            SMPL vertices in the render frame, shape ``(B, L, V, 3)``.
        """
        from smplx.lbs import blend_shapes, vertices2joints

        from sidewalkbench.pg.integrations.smplx_ext import posed_joints_to_axis_angle

        num_peds, length = keypoints.shape[0], keypoints.shape[1]
        flat = keypoints.reshape(-1, keypoints.shape[2], keypoints.shape[3])
        betas = torch.zeros((flat.shape[0], 10), device=self.pre_render_device, dtype=self.dtype)
        transl = torch.zeros_like(flat[:, 0])
        v_shaped = self.smpl.v_template + blend_shapes(betas, self.smpl.shapedirs)
        rest_joints = vertices2joints(self.smpl.J_regressor, v_shaped)[:, : flat.shape[1]]
        pose = posed_joints_to_axis_angle(flat, rest_joints, self.smpl.parents[: flat.shape[1]])
        pose = torch.cat([pose, torch.zeros((flat.shape[0], 2, 3), device=self.pre_render_device, dtype=self.dtype)], dim=1)
        with torch.no_grad():
            out = self.smpl(
                global_orient=pose[:, 0], body_pose=pose[:, 1:].reshape(flat.shape[0], -1), betas=betas, transl=transl
            )
        verts = out.vertices.contiguous().reshape(num_peds, length, -1, 3)
        return self._postprocess_smpl_verts(verts)

    def _generate_animation(self, prompt: str) -> torch.Tensor:
        """Generate SMPL vertex animation for a single motion prompt. Returns (update_length, V, 3).

        Results are cached per prompt so identical prompts (very common across
        pedestrians and terrain cells) are generated at most once.
        """
        cached = self._animation_cache.get(prompt)
        if cached is not None:
            return cached
        animation_length = 100
        animation_fps = 20
        update_fps = 1.0 / self.ped_dt

        with torch.no_grad():
            generator = torch.Generator(device=next(self.motion_lcm_model.parameters()).device)
            generator.manual_seed(derive_seed(self.randomization.seed, "behavior_animation", prompt))
            kp = self.motion_lcm_model(
                {"length": [animation_length], "text": [prompt]}, generator=generator
            )[0]
            kp = torch.stack(kp, dim=0)  # (1, frames, joints, 3)
        kp = kp[:, :: int(animation_fps / update_fps), :, :].to(self.pre_render_device)
        v = self._smpl_animation_from_keypoints(kp).squeeze(0)  # (update_length, V, 3)
        self._animation_cache[prompt] = v
        return v

    def _apply_animation_requests(self):
        """Regenerate animations for changed prompts, reusing one tensor per prompt."""
        requests = self._behavior_manager.animation_requests
        if not requests:
            return
        resolved: dict[str, torch.Tensor] = {}
        for _, _, prompt in requests:
            if prompt not in resolved:
                resolved[prompt] = self._generate_animation(prompt)
        for tidx, agent_idx, prompt in requests:
            animations = self.clone_dynamic_animations.get(tidx)
            if animations is not None and agent_idx < len(animations):
                animations[agent_idx] = resolved[prompt]

    def _sync_static_obstacles(self):
        """Push static obstacle geometry to the behavior manager (done once per resample)."""
        if not (self.use_behavior and self._behavior_manager is not None and self.clone_static_states):
            return
        for tidx, states in self.clone_static_states.items():
            if states is None or not isinstance(states, torch.Tensor) or states.numel() == 0:
                continue
            obstacles = []
            for k in range(states.shape[0]):
                pos2d = states[k, :2].cpu().numpy()
                bbox = states[k, 3:6].cpu().numpy()
                radius = float(max(bbox[0], bbox[1]) * 0.5)
                obstacles.append((pos2d, radius))
            self._behavior_manager.set_static_obstacles(tidx, obstacles)

    @staticmethod
    def _camera_mvp(camera_pos, camera_quat, camera_intrinsics, device, dtype):
        """Build per-camera ``P @ V`` matrices from a camera pose and intrinsics.

        Performed entirely on the compute device (no GPU->CPU sync) so it does not
        stall the physics loop when called from the behavior tick.
        """

        def _to_tensor(value):
            return (value.torch if hasattr(value, "torch") else torch.as_tensor(value)).to(
                device=device, dtype=dtype
            )

        pos = _to_tensor(camera_pos)
        quat = _to_tensor(camera_quat)
        intrinsics = _to_tensor(camera_intrinsics)
        x, y, z, w = quat.unbind(-1)
        rot = torch.stack(
            [
                torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
            ],
            dim=-2,
        )
        forward = rot @ torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        up = rot @ torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        projection = perspective(intrinsics, device=device, dtype=dtype)
        view = look_at(pos, pos + forward, up, device=device, dtype=dtype)
        return (projection @ view).contiguous()

    def _project_persons(self, points, mvp, margin):
        """Return per-camera visibility for world points.

        Args:
            points: World positions [m], shape ``(N, 3)``.
            mvp: Projection-view matrices, shape ``(B, 4, 4)``.
            margin: Extra NDC margin (fraction of half-image) treated as visible.

        Returns:
            Bool tensor ``(B, N)``: in front of each camera and inside the frustum
            plus ``margin``.
        """
        points_h = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)  # (N,4)
        clip = torch.einsum("nj,bij->bni", points_h, mvp.to(points.dtype))  # (B,N,4)
        w = clip[..., 3]
        in_front = w > 1e-6
        ndc = clip[..., :2] / w.clamp(min=1e-6).unsqueeze(-1)
        return in_front & (ndc[..., 0].abs() <= 1.0 + margin) & (ndc[..., 1].abs() <= 1.0 + margin)

    def _compute_behavior_active_masks(self) -> dict[int, np.ndarray]:
        """Per-terrain active-agent masks from the robot front camera frustum.

        A pedestrian is active when it lies inside the front-camera frustum (with
        :attr:`behavior_fov_margin`) within :attr:`behavior_radius` of the robot
        root, or within :attr:`behavior_proximity_radius` of the robot root in any
        direction.
        """
        if not (self.use_behavior and self._behavior_manager is not None):
            return {}
        try:
            camera = self["camera"]
            camera_pos = camera.data.pos_w
            camera_quat = camera.data.quat_w_world
            camera_intrinsics = camera.data.intrinsic_matrices
            if camera_pos is None or camera_intrinsics is None:
                return {}
            mvp = self._camera_mvp(camera_pos, camera_quat, camera_intrinsics, self.render_device, self.dtype)
            robot_root_w = self["robot"].data.root_pos_w
        except Exception:
            return {}

        num_rows, num_cols = self.terrain.terrain_origins.shape[:2]
        robot_terrain_idxs = self.terrain.terrain_levels * num_cols + self.terrain.terrain_types
        robot_xy_all = robot_root_w.to(device=self.render_device, dtype=self.dtype)

        # Compute every terrain's mask on the device, then transfer once to host to
        # avoid one GPU->CPU sync per terrain cell.
        terrain_order: list[int] = []
        mask_tensors: list[torch.Tensor] = []
        for tidx, states in self.clone_dynamic_states.items():
            if states is None or not isinstance(states, torch.Tensor) or states.numel() == 0:
                continue
            env_ids = (robot_terrain_idxs == tidx).nonzero(as_tuple=False).flatten()
            if env_ids.numel() == 0:
                continue
            points = states[:, :3].to(device=self.render_device, dtype=self.dtype)
            visible = self._project_persons(points, mvp[env_ids], self.behavior_fov_margin).any(dim=0)  # (N,)

            robot_xy = robot_xy_all[env_ids, :2]  # (E,2)
            distance = torch.linalg.norm(points[:, None, :2] - robot_xy[None, :, :], dim=-1).amin(dim=1)  # (N,)

            # In-frustum within the radius, OR close to the robot in any direction.
            active = (visible & (distance <= self.behavior_radius)) | (distance <= self.behavior_proximity_radius)
            terrain_order.append(tidx)
            mask_tensors.append(active)

        if not mask_tensors:
            return {}
        flat = torch.cat(mask_tensors).cpu().numpy()
        masks: dict[int, np.ndarray] = {}
        offset = 0
        for tidx, tensor in zip(terrain_order, mask_tensors):
            n = tensor.shape[0]
            masks[tidx] = flat[offset : offset + n]
            offset += n
        return masks

    def generate_scene(self, remove_current_scene: bool = False):
        """
        Args:
            remove_current_scene (bool, optional): _description_. Defaults to False.
        """
        if self.scenario_generation_method == "limited async procedural generation":
            self.generate_limited_async_procedural_scene()
        else:
            raise NotImplementedError(
                f"Scenario generation method {self.scenario_generation_method} not implemented yet."
            )

    def generate_limited_async_procedural_scene(self):
        static_usd_paths = "assets/usds/"
        all_assets = os.listdir(static_usd_paths)
        all_assets = [asset for asset in all_assets if asset.endswith(".usd") and "non_metric" not in asset]

        static_asset_cfg = []
        self._static_bboxes = []
        # Build UUID -> config path lookup from adj_parameter_folder
        adj_folder = "./assets/adj_parameter_folder/"
        uuid_to_config = {}
        for cfg_file in os.listdir(adj_folder):
            if not cfg_file.endswith(".json"):
                continue
            uuid = cfg_file.rsplit("-", 1)[1][:-5]  # extract UUID from "prefix-UUID.json"
            uuid_to_config[uuid] = cfg_file
        asset_buildings = [
            a for a in all_assets if ("building" in a.lower()) or ("tree" in a.lower()) or ("wall" in a.lower())
        ]
        asset_not_buildings = [a for a in all_assets if a not in asset_buildings]

        # hardcoded asset list
        with open("assets/filtered_assets.txt") as f:
            asset_ids = [line.strip() for line in f if line.strip()]

        asset_not_buildings = [a for a in asset_not_buildings if a[:-4] in asset_ids]

        current_obstacle_id = 0

        for asset_path in sorted(asset_not_buildings):
            usd_path = copy.deepcopy(asset_path)
            asset_valid = False
            # Extract UUID from USD filename: "category_UUID.usd"
            uuid = usd_path[:-4].rsplit("_", 1)[-1]
            cfg_file = uuid_to_config.get(uuid)
            if cfg_file:
                obj_info = json.load(open(os.path.join(adj_folder, cfg_file)))
                if obj_info["general"]["length"] > 1.0 or obj_info["general"]["width"] > 1.0:
                    asset_valid = False
                else:
                    if current_obstacle_id == self.obstacle_id or self.obstacle_id == -1:
                        asset_valid = True
                    else:
                        asset_valid = False
                    current_obstacle_id += 1
            if not asset_valid:
                continue

            if self.simplified:
                usd_cfg = sim_utils.MeshCuboidCfg(
                    size=(obj_info["general"]["length"], obj_info["general"]["width"], obj_info["general"]["height"]),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visible=False,
                )
            else:
                usd_cfg = sim_utils.UsdFileCfg(
                    usd_path=os.path.join(static_usd_paths, usd_path),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visible=False,
                )
            static_asset_cfg.append([usd_cfg, obj_info])
            self._static_bboxes.append(
                np.array([obj_info["general"]["length"], obj_info["general"]["width"], obj_info["general"]["height"]])
            )

        num_static_assets = len(static_asset_cfg)
        if self.num_walkable_objects_per_terrain == 0:
            num_static_assets = 1
        self._static_obstacle_prim_paths: list[str] = []

        for i in range(num_static_assets):
            prim_path = f"/World/Dataset/Static_{i}_0"
            obj_cfg, obj_info = static_asset_cfg[i]
            self._static_obstacle_prim_paths.append(prim_path)
            if self.simplified:
                prim = spawn_mesh_cuboid(prim_path, obj_cfg)
            else:
                prim = spawn_from_usd(prim_path, obj_cfg)

            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)

        self._static_bboxes = self._static_bboxes[
            :num_static_assets
        ]  # only keep the bboxes for the static assets that are actually used in the scene
        self._static_bboxes = np.stack(self._static_bboxes, axis=0)

    def resample(self, env_ids):
        stage = self.stage
        # Global terrain retexturing is tied to env 0's episode so that later
        # resets of other environments do not flicker the shared terrain.
        if self._terrain_material_randomizer is not None and bool((torch.as_tensor(env_ids) == 0).any().item()):
            self._terrain_material_randomizer.apply()
        # terrain_origins: [10, 10, 3] - lower right corner of each terrain
        # terrain_walkable_samplers: list[callable] - functions to sample positions in walkable regions
        terrain_origins = self.terrain.terrain_origins  # [10, 10, 3]
        terrain_walkable_samplers = self.terrain.terrain_walkable_samplers  # list[callable]
        terrain_trajectory_samplers = self.terrain.terrain_trajectory_samplers
        subterrain_size = self.terrain.cfg.terrain_generator.size

        num_rows, num_cols = terrain_origins.shape[0], terrain_origins.shape[1]

        # Convert env_ids to terrain indices
        robot_terrain_idxs = self.terrain.terrain_levels[env_ids] * num_cols + self.terrain.terrain_types[env_ids]
        terrain_idxs_to_resample = torch.unique(robot_terrain_idxs).tolist()

        # Only remove static clones for the terrains being resampled
        for tidx in terrain_idxs_to_resample:
            if tidx in self.clone_static_paths:
                for path in self.clone_static_paths[tidx]:
                    stage.RemovePrim(path)
            self.clone_static_paths[tidx] = []

        # Flatten terrain data
        num_walkable_objects_per_terrain = self.num_walkable_objects_per_terrain
        num_non_walkable_objects_per_terrain = self.num_non_walkable_objects_per_terrain
        num_peds_per_terrain = self.num_peds_per_terrain

        def sample_in_walkable_region(num_samples_per_terrain: int):
            """Sample positions between left and right walkable edges for each terrain."""
            positions_list = []
            env_idx = []
            for tidx in terrain_idxs_to_resample:
                i = tidx // num_cols
                j = tidx % num_cols
                walkable_terrain_sampler = terrain_walkable_samplers[i][j]
                terrain_base = terrain_origins[i, j]  # [3]
                if self.scenario_type == "obstacle_obstruction":
                    # Place obstacle at the obstruction point (same as pedestrian_obstruction)
                    trajectory_sampler = terrain_trajectory_samplers[i][j]
                    traj = trajectory_sampler(
                        1, scenario_type=self.scenario_type, ped_lateral=self.ped_lateral,
                        rng=self.randomization.numpy(i, j, "object_position"),
                    )
                    xx = torch.tensor(traj[:num_samples_per_terrain, 0], device=self.device)
                    yy = torch.tensor(traj[:num_samples_per_terrain, 1], device=self.device)
                else:
                    xx, yy = walkable_terrain_sampler(
                        walkable=True, num_samples=num_samples_per_terrain,
                        rng=self.randomization.numpy(i, j, "object_position"),
                    )  # sample in the walkable region
                    xx = torch.tensor(xx, device=self.device)
                    yy = torch.tensor(yy, device=self.device)
                pos = (
                    torch.stack(
                        [
                            xx,
                            yy,
                            torch.ones(num_samples_per_terrain, device=self.device)
                            * self.terrain.cfg.terrain_generator.sidewalk_height,
                        ],
                        dim=-1,
                    )
                    + terrain_base
                )
                yaw = torch.rand(
                    num_samples_per_terrain, device=self.device,
                    generator=self.randomization.torch(i, j, "object_orientation", device=self.device),
                ) * 2 * np.pi
                # convert to quaternion (x, y, z, w)
                rot = _yaw_to_quat(yaw)
                positions_list.append(torch.cat([pos, rot], dim=-1))
                for _ in range(num_samples_per_terrain):
                    env_idx.append(tidx)
            return torch.cat(positions_list, dim=0), env_idx  # [total_samples, 7], [total_samples]

        def sample_pedestrian_trajectories(num_samples_per_terrain: int):
            trajectory_list = []
            env_idx = []
            update_fps = 1.0 / self.ped_dt
            for tidx in terrain_idxs_to_resample:
                i = tidx // num_cols
                j = tidx % num_cols
                trajectory_sampler = terrain_trajectory_samplers[i][j]
                terrain_base = terrain_origins[i, j]  # [3]
                if self.scenario_type == "overtaking" or self.scenario_type == "intersection":
                    pedestrian_speed = 0.5
                else:
                    pedestrian_speed = self.randomization.numpy(i, j, "pedestrian_speed").uniform(1.1, 1.65)
                self._ped_speeds[tidx] = pedestrian_speed
                delta_distance = pedestrian_speed * self.ped_dt
                num_points = int(subterrain_size[1] / pedestrian_speed * update_fps)
                trajs = trajectory_sampler(
                    num_points,
                    num_samples=num_samples_per_terrain,
                    delta_distance=delta_distance,
                    scenario_type=self.scenario_type,
                    ped_lateral=self.ped_lateral,
                    ped_crossing_direction=self.ped_crossing_direction,
                    rng=self.randomization.numpy(i, j, "pedestrian_trajectory"),
                )
                # trajs: (num_samples, n_pts, 3) -- n_pts may differ from num_points
                n_pts = trajs.shape[1]
                for k in range(num_samples_per_terrain):
                    traj = trajs[k]
                    traj = torch.tensor(traj, device=self.device, dtype=self.dtype)
                    use_ground_height = self.terrain_segment == "crosswalk"
                    if use_ground_height:
                        height = self.terrain.cfg.terrain_generator.ground_height
                    else:
                        height = self.terrain.cfg.terrain_generator.sidewalk_height
                    traj_pos = torch.cat([traj[:, :2], torch.ones(n_pts, 1, device=self.device) * height], dim=-1)
                    traj_pos = traj_pos + terrain_base
                    traj_yaw = traj[:, [2]]
                    trajectory_list.append(torch.cat([traj_pos, traj_yaw], dim=-1))
                    env_idx.append(tidx)
            return trajectory_list, env_idx  # [total_samples, num_points, 7]

        # Pedestrians (dynamic objects) - sample on WALKABLE region
        if num_peds_per_terrain > 0:
            self.count = 0
            if self.clone_dynamic_states is None:
                self.clone_dynamic_states = {}
            if self.clone_dynamic_trajectories is None:
                self.clone_dynamic_trajectories = {}
            if self.clone_dynamic_velocities is None:
                self.clone_dynamic_velocities = {}
            if self._ped_speeds is None:
                self._ped_speeds = {}
            for tidx in terrain_idxs_to_resample:
                self.clone_dynamic_states[tidx] = []
                self.clone_dynamic_trajectories[tidx] = []
                self.clone_dynamic_velocities[tidx] = []
            update_fps = 1.0 / (self.ped_dt)
            ped_trajs, env_idx = sample_pedestrian_trajectories(num_peds_per_terrain)
            ped_keys = [
                (*divmod(int(tidx), num_cols), index % num_peds_per_terrain)
                for index, tidx in enumerate(env_idx)
            ]
            env_idx = torch.tensor(env_idx, device=self.device)
            ped_states = []

            # randomize pedestrian initial states (current_traj_idx, forward/reverse)
            for (row, col, ped_id), ped_traj in zip(ped_keys, ped_trajs):
                if self.scenario_type != "default":
                    init_ts, init_dir = 0, 0  # start at beginning, move forward (one-way)
                    init_pos = ped_traj[init_ts, :3]
                    init_yaw = ped_traj[init_ts, 3]
                else:
                    rng = self.randomization.numpy(row, col, "pedestrian_initial", ped_id)
                    init_ts, init_dir = rng.randint(0, ped_traj.shape[0]), rng.randint(0, 2)
                    init_pos = ped_traj[init_ts, :3]
                    init_yaw = ped_traj[init_ts, 3]
                    if init_dir == 0:
                        init_yaw = init_yaw + np.pi

                init_state = torch.cat(
                    [init_pos, torch.tensor([init_yaw, init_ts, init_dir], device=self.device, dtype=self.dtype)],
                    dim=-1,
                )
                ped_states.append(init_state)

            for env_id, ped_state, ped_traj in zip(env_idx, ped_states, ped_trajs):
                self.clone_dynamic_states[int(env_id)].append(ped_state)
                self.clone_dynamic_trajectories[int(env_id)].append(ped_traj)

            for tidx in terrain_idxs_to_resample:
                val = self.clone_dynamic_states[tidx]
                self.clone_dynamic_states[tidx] = torch.stack(val, dim=0) if len(val) > 0 else None
                num_peds = len(val)
                vel_val = self.clone_dynamic_velocities[tidx]
                self.clone_dynamic_velocities[tidx] = (
                    torch.stack(vel_val, dim=0)
                    if len(vel_val) > 0
                    else (torch.zeros(num_peds, 2, device=self.device, dtype=self.dtype) if num_peds > 0 else None)
                )

            # sample textures
            if not hasattr(self, "clone_dynamic_textures") or self.clone_dynamic_textures is None:
                self.clone_dynamic_textures = {}
            for tidx in terrain_idxs_to_resample:
                self.clone_dynamic_textures[tidx] = []
            all_texture_list = sorted(glob("assets/smplx/SMPLitex-textures/*.png"))
            for env_id, (row, col, ped_id) in zip(env_idx, ped_keys):
                texture_path = self.randomization.numpy(row, col, "pedestrian_texture", ped_id).choice(all_texture_list)
                uv_tex = Image.open(texture_path).convert("RGB")
                uv_tex = torch.tensor(np.array(uv_tex), device=self.pre_render_device, dtype=self.dtype) / 255.0
                uv_tex = uv_tex[:, :, :3]  # [H, W, 3] in [0,1]
                # Sample texture per vertex using UV coords
                uv_x = (self.smpl_uv_coords[:, 0] * (uv_tex.shape[1] - 1)).long().clamp(0, uv_tex.shape[1] - 1)
                uv_y = ((1 - self.smpl_uv_coords[:, 1]) * (uv_tex.shape[0] - 1)).long().clamp(0, uv_tex.shape[0] - 1)
                vertex_tex_color = uv_tex[uv_y, uv_x]  # [V, 3]
                self.clone_dynamic_textures[int(env_id)].append(vertex_tex_color)

            # sample animation
            if not hasattr(self, "clone_dynamic_animations") or self.clone_dynamic_animations is None:
                self.clone_dynamic_animations = {}
            for tidx in terrain_idxs_to_resample:
                self.clone_dynamic_animations[tidx] = []
            total_pedestrians = len(ped_states)
            update_fps = 1.0 / (self.ped_dt)

            if self.scenario_type == "gesturing":
                # Load pre-recorded stop.npy animation
                motion_raw = np.load("assets/motion/stop.npy", allow_pickle=True).item()
                motion_1 = {
                    "root_orient": matrix_to_axis_angle(
                        motion_raw["pred_rotmat"][:, 0].float().to(self.pre_render_device)
                    ),
                    "pose_body": matrix_to_axis_angle(
                        motion_raw["pred_rotmat"][:, 1:].float().to(self.pre_render_device)
                    ).reshape(-1, 69),
                    "trans": motion_raw["pred_trans"].float().squeeze(1).to(self.pre_render_device),
                    "betas": motion_raw["pred_shape"].float().to(self.pre_render_device),
                }
                # hardcoded
                motion_1["root_orient"][:, :] = 0.0
                motion_1["pose_body"][:, :] = 0.0
                motion_1["pose_body"][:, 3 * 16 - 3 : 3 * 16] = torch.tensor([[0.0, -np.pi / 2, 0.0]]).to(
                    self.pre_render_device
                )
                motion_1["pose_body"][:, 3 * 17 - 3 : 3 * 17] = torch.tensor([[0.0, 0.0, 1.4]]).to(
                    self.pre_render_device
                )
                motion_1["pose_body"][:, 3 * 20 - 3 : 3 * 20] = torch.tensor([[0.0, 0.0, np.pi / 2]]).to(
                    self.pre_render_device
                )

                animation_fps = 30
                update_length = int(np.ceil(motion_1["trans"].shape[0] * update_fps / animation_fps))
                # Normalize to root frame
                norm_transform = torch.eye(4, device=self.pre_render_device)
                norm_transform[:3, :3] = axis_angle_to_matrix(motion_1["root_orient"][0])
                norm_transform[:3, 3] = motion_1["trans"][0]
                norm_transform = torch.inverse(norm_transform)
                smpl_transform = (
                    torch.eye(4, device=self.pre_render_device).unsqueeze(0).repeat(motion_1["trans"].shape[0], 1, 1)
                )
                smpl_transform[:, :3, :3] = axis_angle_to_matrix(motion_1["root_orient"])
                smpl_transform[:, :3, 3] = motion_1["trans"]
                smpl_transform = norm_transform.unsqueeze(0) @ smpl_transform
                motion_1["trans"] = torch.zeros_like(smpl_transform[:, :3, 3])
                motion_1["root_orient"] = matrix_to_axis_angle(smpl_transform[:, :3, :3])
                # Subsample
                step = int(animation_fps / update_fps)
                r = motion_1["root_orient"][::step]
                p = motion_1["pose_body"][::step]
                t = motion_1["trans"][::step]
                b = motion_1["betas"][::step]
                n_frames = r.shape[0]
                global_orient = r.repeat(total_pedestrians, 1)
                body_pose = p.repeat(total_pedestrians, 1)
                betas = b.repeat(total_pedestrians, 1)
                transl = torch.zeros_like(global_orient)
                with torch.no_grad():
                    out = self.smpl(global_orient=global_orient, body_pose=body_pose, betas=betas, transl=transl)
                all_smpl_verts = self._postprocess_smpl_verts(
                    out.vertices.contiguous().reshape(total_pedestrians, n_frames, -1, 3)
                )
            else:
                animation_length = 100
                animation_fps = 20
                if self.scenario_type in ["obstructing", "queueing", "lateral_approaching"]:
                    animation_prompt = "a person is standing straight."
                elif self.scenario_type == "human_lying":
                    animation_prompt = "a man crawls forward on his stomach."
                elif self.scenario_type == "conversing":
                    animation_prompt = "a person is standing straight while having a conversation."
                else:
                    animation_prompt = "a person is walking."
                batch = {
                    "length": [animation_length] * total_pedestrians,
                    "text": [animation_prompt] * total_pedestrians,
                }
                with torch.no_grad():
                    motion_device = next(self.motion_lcm_model.parameters()).device
                    generators = [
                        self.randomization.torch(row, col, "pedestrian_animation", ped_id, device=motion_device)
                        for row, col, ped_id in ped_keys
                    ]
                    keypoints = self.motion_lcm_model(batch, generator=generators)[0]
                keypoints = torch.stack(keypoints, dim=0)
                keypoints = keypoints[:, :: int(animation_fps / update_fps), :, :].to(self.pre_render_device)
                all_smpl_verts = self._smpl_animation_from_keypoints(keypoints)

            for env_id, smpl_verts in zip(env_idx, all_smpl_verts):
                self.clone_dynamic_animations[int(env_id)].append(smpl_verts)

        # Static objects on WALKABLE region
        if num_walkable_objects_per_terrain > 0 and len(self._static_obstacle_prim_paths) > 0:
            if self.clone_static_states is None:
                self.clone_static_states = {}
            for tidx in terrain_idxs_to_resample:
                self.clone_static_states[tidx] = []  # (x,y,z,l,w,h,yaw)
            walkable_obj_positions, env_idx = sample_in_walkable_region(num_walkable_objects_per_terrain)
            obj_mask = np.logical_and(
                self._static_bboxes[:, 0] <= 1, self._static_bboxes[:, 1] <= 1
            )  # only use small objects for walkable region
            walkable_static_obstacle_prim_paths = np.array(self._static_obstacle_prim_paths)[obj_mask]

            obj_idx = torch.cat([
                torch.randint(
                    0, len(walkable_static_obstacle_prim_paths), (num_walkable_objects_per_terrain,),
                    device=self.device,
                    generator=self.randomization.torch(
                        *divmod(tidx, num_cols), "object_asset", device=self.device
                    ),
                )
                for tidx in terrain_idxs_to_resample
            ])
            env_idx = torch.tensor(env_idx, device=self.device)

            for n, static_path in enumerate(walkable_static_obstacle_prim_paths):
                num_clones = (obj_idx == n).sum().item()
                env_idx_n = env_idx[obj_idx == n].tolist()
                if num_clones > 0:
                    root_path = "/World/Dataset"
                    clone_paths = []
                    for i in range(int(num_clones)):
                        clone_paths.append(root_path + f"/Clone_Static_{env_idx_n[i]}_{self._clone_static_counter}")
                        self._clone_static_counter += 1
                        self.clone_static_paths[env_idx_n[i]].append(clone_paths[-1])
                        pos = walkable_obj_positions[obj_idx == n][i][:3]
                        bbox = torch.tensor(self._static_bboxes[obj_mask][n], device=self.device)
                        yaw = torch.tensor(
                            [_quat_to_yaw(walkable_obj_positions[obj_idx == n][i][3:])], device=self.device
                        )
                        self.clone_static_states[env_idx_n[i]].append(torch.cat([pos, bbox, yaw]))
                    self.obs_cloner.clone(
                        source_prim_path=static_path,
                        prim_paths=clone_paths,
                        replicate_physics=False,
                        clone_in_fabric=False,
                        root_path=root_path,
                        positions=walkable_obj_positions[obj_idx == n][:, :3],
                        orientations=walkable_obj_positions[obj_idx == n][:, [6, 3, 4, 5]],
                    )
                    for clone_path in clone_paths:
                        prim = stage.GetPrimAtPath(clone_path)
                        set_prim_visibility(prim, True)
            for tidx in terrain_idxs_to_resample:
                if (
                    tidx in self.clone_static_states
                    and isinstance(self.clone_static_states[tidx], list)
                    and len(self.clone_static_states[tidx]) > 0
                ):
                    self.clone_static_states[tidx] = torch.stack(self.clone_static_states[tidx], dim=0)

        # Static obstacles are constant within an episode: feed them to ORCA once.
        self._sync_static_obstacles()

        # Static objects on NON-WALKABLE region
        assert num_non_walkable_objects_per_terrain == 0

        # --- Behavior agent registration (after static objects are spawned) ---
        if num_peds_per_terrain > 0 and self.use_behavior and self._behavior_manager is not None:
            pois_per_terrain = self._scene_graph.build_from_scene(
                clone_static_states=self.clone_static_states if self.clone_static_states else {},
                clone_static_categories={},
                terrain_origins=terrain_origins.cpu().numpy(),
                terrain_walkable_samplers=terrain_walkable_samplers,
                terrain_size=subterrain_size,
                num_cols=num_cols,
                terrain_centerline_splines=self.terrain.terrain_centerline_splines,
                terrain_indices=terrain_idxs_to_resample,
                rngs={
                    tidx: self.randomization.numpy(*divmod(tidx, num_cols), "behavior_pois")
                    for tidx in terrain_idxs_to_resample
                },
            )
            behavior_dt = self.ped_dt
            for tidx in terrain_idxs_to_resample:
                pois = pois_per_terrain.get(tidx, [])
                states = self.clone_dynamic_states.get(tidx)
                if states is None:
                    continue
                n_peds = states.shape[0]
                initial_positions = [states[k, :3].cpu().numpy() for k in range(n_peds)]
                i_row, j_col = tidx // num_cols, tidx % num_cols
                spline_data = None
                projector = None
                if self.terrain.terrain_centerline_splines is not None:
                    spline_data = self.terrain.terrain_centerline_splines[i_row][j_col]
                if self.terrain.terrain_centerline_projectors is not None:
                    projector = self.terrain.terrain_centerline_projectors[i_row][j_col]
                self._behavior_manager.register_agents(
                    terrain_idx=tidx,
                    pois=pois,
                    initial_positions=initial_positions,
                    terrain_origin=terrain_origins[i_row, j_col].cpu().numpy(),
                    walkable_sampler=terrain_walkable_samplers[i_row][j_col],
                    dt=behavior_dt,
                    seed=self.randomization.seed_for(i_row, j_col, "behavior"),
                    spline_data=spline_data,
                    centerline_projector=projector,
                )
        # height scanner logic
        if hasattr(self, "height_scanner"):
            self["height_scanner"]._raycast_targets_cfg.append(
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/Dataset/Clone_.*", track_mesh_transforms=False)
            )
            self["height_scanner"]._initialize_impl()
            self["height_scanner"].reset()

    def _empty_render(self, camera_shape, batch_size: int, return_depth: bool):
        """Return an all-background render when no pedestrian is visible."""
        height, width = int(camera_shape[0]), int(camera_shape[1])
        img = torch.zeros((batch_size, height, width, 3), device=self.render_device, dtype=self.dtype)
        mask = torch.zeros((batch_size, height, width, 1), device=self.render_device, dtype=self.dtype)
        if return_depth:
            return img, mask, torch.zeros_like(mask)
        return img, mask

    def render_human(
        self,
        camera_pos=None,
        camera_quat=None,
        camera_intrinsics=None,
        camera_shape=None,
        return_depth=False,
        first_env_only=False,
        all_envs_single_image=False,
    ):
        if camera_pos is None or camera_quat is None or camera_intrinsics is None:
            camera_pos = self["camera"].data.pos_w
            camera_quat = self["camera"].data.quat_w_world
            camera_intrinsics = self["camera"].data.intrinsic_matrices
            camera_shape = self["camera"].data.image_shape
        # In single-image mode, render all humans in world space from the first camera only.
        if all_envs_single_image and camera_pos.shape[0] > 1:
            camera_pos = camera_pos[:1]
            camera_quat = camera_quat[:1]
            camera_intrinsics = camera_intrinsics[:1]
        eye, at, up = lookat_from_pose(camera_pos, camera_quat)
        P = perspective(camera_intrinsics, device=self.render_device, dtype=self.dtype)
        V = look_at(eye, at, up, device=self.render_device, dtype=self.dtype)
        MVP = (P @ V).contiguous()  # [B,4,4]

        num_rows, num_cols = self.terrain.terrain_origins.shape[:2]
        robot_terrain_idxs = self.terrain.terrain_levels * num_cols + self.terrain.terrain_types

        # Only rasterize pedestrians inside the rendering camera's frustum and
        # within ``behavior_radius`` of the robot (or within
        # ``behavior_proximity_radius`` in any direction).  Gated behind
        # ``use_behavior``.
        cull = self.use_behavior and self.behavior_fov_cull
        _visibility_cache: dict[tuple[int, int, int], torch.Tensor] = {}

        def _visible_persons(key: int, env_id: int, cam_index: int) -> torch.Tensor:
            cache_key = (key, env_id, cam_index)
            if cache_key in _visibility_cache:
                return _visibility_cache[cache_key]
            states = self.clone_dynamic_states.get(key)
            if states is None or not isinstance(states, torch.Tensor) or states.numel() == 0:
                _visibility_cache[cache_key] = torch.zeros(0, dtype=torch.bool, device=self.render_device)
                return _visibility_cache[cache_key]
            points = states[:, :3].to(device=self.render_device, dtype=self.dtype)
            in_frustum = self._project_persons(points, MVP[cam_index : cam_index + 1], self.behavior_fov_margin)[0]
            # Cap by distance from the robot root, and always render pedestrians
            # close to the robot even when outside the camera view.
            robot_xy = self["robot"].data.root_pos_w[env_id, :2].to(device=self.render_device, dtype=self.dtype)
            distance = torch.linalg.norm(points[:, :2] - robot_xy, dim=-1)
            visible = (in_frustum & (distance <= self.behavior_radius)) | (
                distance <= self.behavior_proximity_radius
            )
            _visibility_cache[cache_key] = visible
            return visible

        def _collect_single_env_vertices_and_colors(env_id: int, cam_index: int):
            """Batched vertices/colors for one env's visible pedestrians, shape (k, V, 3)."""
            key = robot_terrain_idxs[env_id].item()
            animations = self.clone_dynamic_animations.get(key)
            states = self.clone_dynamic_states.get(key)
            if not animations or states is None:
                return None, None
            frame = (self.count // self._ped_period_steps()) % animations[0].shape[0]
            if cull:
                selected = _visible_persons(key, env_id, cam_index).nonzero(as_tuple=False).flatten().tolist()
            else:
                selected = list(range(len(animations)))
            if not selected:
                return None, None
            verts = torch.stack([animations[i][frame] for i in selected], dim=0)  # (k, V, 3) on CPU
            states_sel = states[selected].cpu()  # (k, 6)
            yaw = states_sel[:, 3]
            cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
            x, y, z = verts[..., 0], verts[..., 1], verts[..., 2]
            verts = torch.stack(
                [cos_yaw[:, None] * x - sin_yaw[:, None] * y, sin_yaw[:, None] * x + cos_yaw[:, None] * y, z],
                dim=-1,
            )
            verts = verts + states_sel[:, None, :3]
            colors = torch.stack([self.clone_dynamic_textures[key][i] for i in selected], dim=0)
            return verts, colors

        if all_envs_single_image:
            merged_verts = []
            merged_colors = []
            for env_id in range(self.num_envs):
                env_verts, env_colors = _collect_single_env_vertices_and_colors(env_id, 0)
                if env_verts is not None:
                    merged_verts.append(env_verts)
                    merged_colors.append(env_colors)
            if not merged_verts:
                return self._empty_render(camera_shape, 1, return_depth)
            verts = torch.cat(merged_verts, dim=0).to(self.render_device).reshape(1, -1, 3)
            colors = torch.cat(merged_colors, dim=0).to(self.render_device).reshape(1, -1, 3)
            all_render_smpl_verts = verts
            all_render_smpl_colors = colors
            num_people_per_render = verts.shape[1] // self.smpl_uv_coords.shape[0]
        else:
            render_env_count = 1 if first_env_only else self.num_envs
            env_pairs = []
            for env_id in range(render_env_count):
                cam_index = 0 if first_env_only else env_id
                env_pairs.append(_collect_single_env_vertices_and_colors(env_id, cam_index))
            max_people = max((verts.shape[0] for verts, _ in env_pairs if verts is not None), default=0)
            if max_people == 0:
                return self._empty_render(camera_shape, render_env_count, return_depth)
            num_verts_per_person = self.smpl_uv_coords.shape[0]
            all_render_smpl_verts = torch.full(
                (render_env_count, max_people * num_verts_per_person, 3),
                1e6,
                device=self.render_device,
                dtype=self.dtype,
            )
            all_render_smpl_colors = torch.zeros_like(all_render_smpl_verts)
            for b, (env_verts, env_colors) in enumerate(env_pairs):
                if env_verts is None:
                    continue
                verts = env_verts.to(self.render_device).reshape(-1, 3)
                colors = env_colors.to(self.render_device).reshape(-1, 3)
                all_render_smpl_verts[b, : verts.shape[0]] = verts
                all_render_smpl_colors[b, : colors.shape[0]] = colors
            num_people_per_render = max_people

        V_num = all_render_smpl_verts.shape[1] // max(num_people_per_render, 1)
        faces_list = [self.smpl_faces + p * V_num for p in range(num_people_per_render)]
        faces_merged = torch.cat(faces_list, dim=0).to(self.render_device)
        # Transform verts -> clip space
        ones = torch.ones(
            (all_render_smpl_verts.shape[0], all_render_smpl_verts.shape[1], 1),
            device=self.render_device,
            dtype=self.dtype,
        )
        verts_h = torch.cat([all_render_smpl_verts, ones], dim=-1)  # [B,V,4]
        verts_clip = verts_h @ MVP.transpose(1, 2)  # [B,V,4]
        rast, _ = dr.rasterize(self.glctx, verts_clip, faces_merged, resolution=camera_shape)
        img, _ = dr.interpolate(all_render_smpl_colors, rast, faces_merged)  # [B,H,W,3]
        # Mask background (rast[..., 3] is triangle id / coverage; <=0 means background)
        mask = (rast[..., 3:4] > 0).to(torch.float32)

        if return_depth:
            verts_view = verts_h @ V.transpose(1, 2)  # [B, NUM_PEOPLE*V, 4]
            verts_view_depth = (-verts_view[..., 2:3]).contiguous()  # positive metric depth
            depth, _ = dr.interpolate(verts_view_depth, rast, faces_merged)  # [B,H,W,1]
            depth = depth.flip(dims=[1])

        img = img.flip(
            dims=[1]
        )  # IMPORTANT: The memory order of image data in OpenGL, and consequently in nvdiffrast, is bottom-up
        mask = mask.flip(dims=[1])
        if return_depth:
            return img, mask, depth
        else:
            return img, mask
