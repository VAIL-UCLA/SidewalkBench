# Copyright (c) 2022-2025, The SidewalkBench-PG Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluation metric tracking for navigation experiments."""

import numpy as np
import torch

from sidewalkbench.pg.utils.scenario_rules import GESTURING_MIN_PROGRESS, classify_outcome


def prepare_model_input(obs, model_input_length, model_input_shape, model_type):
    """Slice and resize an observation batch for model inference.

    Args:
        obs: Observation tensor ``(B, context, C, H, W)``.
        model_input_length: Number of context frames to keep.
        model_input_shape: Target ``(H, W)`` fed to the model.
        model_type: Model name. CityWalker, ViNT and NoMaD prepare the frames in their wrappers and get the
            camera frames unresized: ViNT/NoMaD remap the 512x288 camera image to their training camera
            (``use_train_intrinsics``, defined for the full frame), which a pre-resized frame breaks.

    Returns:
        Model-ready tensor ``(B, context, C, H, W)``.
    """
    obs_model = obs[:, -model_input_length:]
    if model_type in ("CityWalker", "ViNT", "NoMaD"):
        return obs_model
    return torch.nn.functional.interpolate(
        obs_model.flatten(0, 1), size=model_input_shape, mode="bilinear", align_corners=False,
    ).unflatten(0, obs_model.shape[:2])


def disable_frame_rate_limit():
    """Disable the Kit main-loop frame-rate limiter so the eval loop runs unbounded."""
    import carb

    carb.settings.get_settings().set("/app/runLoops/main/rateLimitEnabled", False)


def _point_to_segment_distance(point, seg_start, seg_end):
    """Minimum distance from a 2D point to a 2D line segment.

    Args:
        point: torch.Tensor of shape (2,) or (N, 2).
        seg_start: torch.Tensor of shape (2,).
        seg_end: torch.Tensor of shape (2,).

    Returns:
        torch.Tensor scalar: minimum Euclidean distance.
    """
    ab = seg_end - seg_start
    ap = point - seg_start
    ab_norm2 = torch.dot(ab, ab)
    if ab_norm2 < 1e-12:
        return torch.linalg.norm(ap)
    t = torch.dot(ap, ab) / ab_norm2
    t = torch.clamp(t, 0.0, 1.0)
    closest = seg_start + t * ab
    return torch.linalg.norm(point - closest)


class EvalTracker:
    """Track per-episode metrics across multiple environments and resets.

    Args:
        num_envs: Number of parallel environments.
        device: Torch device for internal tensors.
        dt_step: Wall-clock seconds per env step.
        max_resets: Episodes to collect per environment before stopping.
        scenario_type: Scenario-specific success rule; defaults to the scene's type.
    """

    def __init__(self, num_envs, device, dt_step, max_resets=10, scenario_type=None):
        self.num_envs = num_envs
        self.device = device
        self.dt_step = dt_step
        self.max_resets = max_resets
        self.scenario_type = scenario_type

        # Per-step accumulators
        self.path_length = torch.zeros(num_envs, device=device)
        self.last_pos = None
        self.last_vel = None
        self.episode_steps = torch.zeros(num_envs, device=device)
        self.reset_count = torch.zeros(num_envs, dtype=torch.long, device=device)

        # New per-step accumulators
        self._energy_accum = torch.zeros(num_envs, device=device)
        self._in_ps_steps = torch.zeros(num_envs, device=device)
        self._in_proj_ps_steps = torch.zeros(num_envs, device=device)
        self._min_human_dist = torch.full((num_envs,), float("inf"), device=device)

        # Per-env centerline tracking (cached to survive resets)
        self._cached_arc_length = torch.zeros(num_envs, device=device)
        self._cached_arc_start = torch.zeros(num_envs, device=device)
        self._cached_arc_goal = torch.zeros(num_envs, device=device)
        self._cached_cum_lateral = torch.zeros(num_envs, device=device)
        self._cached_lat_steps = torch.zeros(num_envs, device=device)
        self._off_sidewalk_steps = torch.zeros(num_envs, device=device)

        # Per-env episode result lists (existing, kept for minimal disruption)
        self.success = [[] for _ in range(num_envs)]
        self.collision = [[] for _ in range(num_envs)]
        self.oob = [[] for _ in range(num_envs)]
        self.timeout = [[] for _ in range(num_envs)]
        self.freeze = [[] for _ in range(num_envs)]
        self.goal_reached = [[] for _ in range(num_envs)]
        self.group_intrusion = [[] for _ in range(num_envs)]
        self.ignore_gesture = [[] for _ in range(num_envs)]
        self.episode_results = []
        self.episode_time = [[] for _ in range(num_envs)]
        self.episode_path_length = [[] for _ in range(num_envs)]
        self.episode_route_completion = [[] for _ in range(num_envs)]

        # New per-episode result lists
        self.episode_collision_static = [[] for _ in range(num_envs)]
        self.episode_collision_dynamic = [[] for _ in range(num_envs)]
        self.episode_time_efficiency = [[] for _ in range(num_envs)]
        self.episode_energy_efficiency = [[] for _ in range(num_envs)]
        self.episode_in_ps_fraction = [[] for _ in range(num_envs)]
        self.episode_in_proj_ps_fraction = [[] for _ in range(num_envs)]
        self.episode_min_human_distance = [[] for _ in range(num_envs)]
        self.episode_optimal_path = [[] for _ in range(num_envs)]

    @property
    def all_done(self):
        return (self.reset_count >= self.max_resets).all().item()

    def init_pos(self, robot_pos_xy, env=None, env_ids=None):
        """Initialize positions after a reset, optionally for selected envs."""
        if env_ids is None or self.last_pos is None:
            self.last_pos = robot_pos_xy.clone()
            self.last_vel = torch.zeros(self.num_envs, 2, device=self.device)
        else:
            self.last_pos[env_ids] = robot_pos_xy[env_ids]
            self.last_vel[env_ids] = 0.0
        if env is not None:
            scene = env.unwrapped.scene
            ids = range(self.num_envs) if env_ids is None else torch.as_tensor(env_ids).tolist()
            for i in ids:
                row, col = scene.terrain.terrain_levels[i], scene.terrain.terrain_types[i]
                projector = scene.terrain.terrain_centerline_projectors[row][col]
                if projector is None:
                    continue
                local_pos = robot_pos_xy[i] - scene.env_origins[i, :2]
                s_start, _ = projector(local_pos[0].item(), local_pos[1].item())
                goal_local = scene.env_goals[i, :2] - scene.env_origins[i, :2]
                s_goal, _ = projector(goal_local[0].item(), goal_local[1].item())
                self._cached_arc_start[i] = s_start
                self._cached_arc_goal[i] = s_goal
                self._cached_arc_length[i] = s_start

    def reset_attempt(self, env_ids, env):
        """Clear rejected attempt metrics while retaining accepted episode counts."""
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        for name in ("path_length", "episode_steps", "_energy_accum", "_in_ps_steps", "_in_proj_ps_steps",
                     "_cached_cum_lateral", "_cached_lat_steps", "_off_sidewalk_steps"):
            getattr(self, name)[ids] = 0.0
        self._min_human_dist[ids] = float("inf")
        self.init_pos(env.unwrapped.scene["robot"].data.root_pos_w[:, :2], env, env_ids=ids)

    def _get_terrain_idx(self, scene, env_i):
        num_cols = scene.terrain.terrain_origins.shape[1]
        row = scene.terrain.terrain_levels[env_i]
        col = scene.terrain.terrain_types[env_i]
        return int(row * num_cols + col)

    def _check_personal_spaces(self, scene, env_i, robot_pos):
        """Check if robot is inside personal space (1.2m) or projected personal space.

        Returns:
            (in_ps, in_proj_ps, min_dist): in_ps/in_proj_ps are bools, min_dist is
            closest distance to any human (inf if none).
        """
        terrain_idx = self._get_terrain_idx(scene, env_i)
        dynamic_states = scene.clone_dynamic_states
        dynamic_vels = scene.clone_dynamic_velocities

        if dynamic_states is None or terrain_idx not in dynamic_states:
            return False, False, float("inf")

        hum_states = dynamic_states[terrain_idx]
        if hum_states is None or len(hum_states) == 0:
            return False, False, float("inf")

        robot_vel = self.last_vel[env_i] if self.last_vel is not None else torch.zeros(2, device=self.device)

        in_ps = False
        in_proj_ps = False
        min_dist = float("inf")

        for j in range(len(hum_states)):
            hum_pos = hum_states[j, :2]
            dist = torch.linalg.norm(robot_pos - hum_pos).item()
            if dist < min_dist:
                min_dist = dist
            if dist < 1.2:
                in_ps = True

            # Projected personal space
            if dynamic_vels is not None and terrain_idx in dynamic_vels and dynamic_vels[terrain_idx] is not None:
                hum_vel = dynamic_vels[terrain_idx][j]
            else:
                hum_vel = torch.zeros(2, device=self.device)
            v_rel = robot_vel - hum_vel
            ray_end = robot_pos + 3.0 * v_rel
            proj_dist = _point_to_segment_distance(hum_pos, robot_pos, ray_end).item()
            if proj_dist < 1.2:
                in_proj_ps = True

        return in_ps, in_proj_ps, min_dist

    def step(self, env, record_ids=None):
        """Call after env.step(). Records metrics for terminated envs.

        Args:
            env: The gymnasium environment (unwrapped access needed).
            record_ids: If provided, only these environment indices are recorded
                as episodes. Required for synchronized episodes, where terminated
                environments stay terminated (deferred reset) and would otherwise
                be recorded again on every step. ``None`` records every terminated
                environment (default behavior).

        Returns:
            done_info: list of dicts for each env that terminated this step.
        """
        manager = env.unwrapped.termination_manager
        term_dones = manager._term_dones
        # Read by name so adding a termination cannot silently change metrics.
        # The fallback also supports the original six-term test fixtures.
        term_names = getattr(manager, "active_terms", (
            "time_out", "out_of_region", "collision_static", "collision_dynamic",
            "arrive", "freeze", "pedestrian_group_intrusion",
        ))
        term_indices = {name: i for i, name in enumerate(term_names) if i < term_dones.shape[1]}
        any_done = term_dones.any(dim=-1)
        if record_ids is None:
            record_done = any_done
        else:
            record_done = torch.zeros_like(any_done)
            ids = record_ids if isinstance(record_ids, torch.Tensor) else torch.as_tensor(record_ids, device=any_done.device)
            if ids.numel() > 0:
                record_done[ids.to(device=any_done.device, dtype=torch.long)] = True
            record_done &= any_done
        current_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, 0:2].clone()
        current_vel = env.unwrapped.scene["robot"].data.root_vel_w[:, :2].clone()

        # Accumulate path length for alive envs
        alive = ~any_done
        if alive.any() and self.last_pos is not None:
            self.path_length[alive] += torch.linalg.norm(current_pos[alive] - self.last_pos[alive], dim=-1)

        # Per-step accumulation for alive envs
        scene = env.unwrapped.scene
        if alive.any():
            alive_indices = alive.nonzero(as_tuple=False).squeeze(-1).tolist()

            # Energy consumption
            robot = scene["robot"]
            try:
                torque = robot.data.applied_torque
                jvel = robot.data.joint_vel
                power = (torch.abs(torque * jvel)).sum(dim=-1)
                self._energy_accum[alive] += power[alive] * self.dt_step
            except (AttributeError, NotImplementedError):
                pass

            # Personal space checks per-alive-env
            for i in alive_indices:
                in_ps, in_proj_ps, min_dist = self._check_personal_spaces(scene, i, current_pos[i])
                if in_ps:
                    self._in_ps_steps[i] += 1
                if in_proj_ps:
                    self._in_proj_ps_steps[i] += 1
                if min_dist < self._min_human_dist[i]:
                    self._min_human_dist[i] = min_dist

        # Record metrics for newly terminated envs
        done_info = []
        done_envs = record_done.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_envs.tolist():
            if self.reset_count[idx] >= self.max_resets:
                continue

            def flag(name):
                return bool(term_dones[idx, term_indices[name]].item()) if name in term_indices else False

            is_timeout = flag("time_out")
            is_oob = flag("out_of_region")
            is_collision_static = flag("collision_static")
            is_collision_dynamic = flag("collision_dynamic")
            goal_reached = flag("arrive")
            is_freeze = flag("freeze")
            group_intrusion = flag("pedestrian_group_intrusion")
            is_any_collision = is_collision_static or is_collision_dynamic

            # Per-episode computed values
            cpl = self.path_length[idx].item()
            ep_steps = self.episode_steps[idx].item()
            if ep_steps == 0:
                continue
            ep_time = ep_steps * self.dt_step
            s_start = self._cached_arc_start[idx].item()
            s_current = self._cached_arc_length[idx].item()
            s_goal = self._cached_arc_goal[idx].item()
            if getattr(env.unwrapped, "defer_resets", False):
                # A deferred reset preserves the terminal pose. In particular,
                # the gesturing threshold must use this pose, not a stale cache.
                row, col = scene.terrain.terrain_levels[idx], scene.terrain.terrain_types[idx]
                projector = scene.terrain.terrain_centerline_projectors[row][col]
                if projector is not None:
                    local_pos = current_pos[idx] - scene.env_origins[idx, :2]
                    s_current, _ = projector(local_pos[0].item(), local_pos[1].item())
            total_route = s_goal - s_start
            route_completion = min((s_current - s_start) / total_route, 1.0) if total_route > 0 else 0.0
            route_completion = max(route_completion, 0.0)
            if goal_reached:
                route_completion = 1.0
            scenario = self.scenario_type or getattr(scene, "scenario_type", None)
            is_success, failure_reason = classify_outcome(
                scenario, goal_reached=goal_reached, freeze=is_freeze, progress=route_completion,
                collision_static=is_collision_static, collision_dynamic=is_collision_dynamic,
                oob=is_oob, group_intrusion=group_intrusion,
            )
            optimal_path = s_current - s_start
            lat_steps = self._cached_lat_steps[idx].item()
            avg_lateral = self._cached_cum_lateral[idx].item() / lat_steps if lat_steps > 0 else 0.0
            off_sidewalk_time = self._off_sidewalk_steps[idx].item() * self.dt_step

            # New per-episode metrics
            time_efficiency = (optimal_path / ep_time) if ep_time > 0 else 0.0
            total_energy = self._energy_accum[idx].item()
            energy_efficiency = (optimal_path / total_energy) * 1000 if total_energy > 0 else 0.0
            in_ps_frac = (self._in_ps_steps[idx].item() / ep_steps) if ep_steps > 0 else 0.0
            in_proj_ps_frac = (self._in_proj_ps_steps[idx].item() / ep_steps) if ep_steps > 0 else 0.0
            min_human_distance = self._min_human_dist[idx].item()
            if is_collision_dynamic:
                min_human_distance = 0.0

            # Append to per-env lists
            self.success[idx].append(is_success)
            self.collision[idx].append(is_any_collision)
            self.oob[idx].append(is_oob)
            self.timeout[idx].append(is_timeout)
            self.freeze[idx].append(is_freeze)
            self.goal_reached[idx].append(goal_reached)
            self.group_intrusion[idx].append(group_intrusion)
            self.ignore_gesture[idx].append(failure_reason == "ignore_gesture")
            self.episode_path_length[idx].append(cpl)
            self.episode_route_completion[idx].append(route_completion)
            if is_success:
                self.episode_time[idx].append(ep_time)

            # New lists
            self.episode_collision_static[idx].append(is_collision_static)
            self.episode_collision_dynamic[idx].append(is_collision_dynamic)
            self.episode_time_efficiency[idx].append(time_efficiency)
            self.episode_energy_efficiency[idx].append(energy_efficiency)
            self.episode_in_ps_fraction[idx].append(in_ps_frac)
            self.episode_in_proj_ps_fraction[idx].append(in_proj_ps_frac)
            self.episode_min_human_distance[idx].append(min_human_distance)
            self.episode_optimal_path[idx].append(optimal_path)

            self.reset_count[idx] += 1

            info = {
                "env_idx": idx,
                "episode_num": self.reset_count[idx].item(),
                "success": is_success,
                "goal_reached": goal_reached,
                "failure_reason": failure_reason,
                "group_intrusion": group_intrusion,
                "ignore_gesture": failure_reason == "ignore_gesture",
                "collision": is_any_collision,
                "collision_static": is_collision_static,
                "collision_dynamic": is_collision_dynamic,
                "oob": is_oob,
                "timeout": is_timeout,
                "freeze": is_freeze,
                "path_length": cpl,
                "route_completion": route_completion,
                "optimal_path": optimal_path,
                "avg_lateral_offset": avg_lateral,
                "off_sidewalk_time": off_sidewalk_time,
                "episode_time": ep_time,
                "time_efficiency": time_efficiency,
                "energy_efficiency": energy_efficiency,
                "in_ps_fraction": in_ps_frac,
                "in_proj_ps_fraction": in_proj_ps_frac,
                "min_human_distance": min_human_distance,
            }
            # Compute path efficiency for this episode
            path_eff_ep = (optimal_path / cpl) if cpl > 1e-6 else 0.0
            path_eff_ep = min(path_eff_ep, 1.0)
            outside_ps_ep = 1.0 - in_ps_frac
            outside_proj_ps_ep = 1.0 - in_proj_ps_frac
            overall_ep = float(is_success) * path_eff_ep

            done_info.append(info)
            self.episode_results.append(info)
            print(
                f"[Eval] Env {idx} episode {info['episode_num']}/{self.max_resets}  "
                f"success={is_success}  col_static={is_collision_static}  col_dyn={is_collision_dynamic}  "
                f"oob={is_oob}  timeout={is_timeout}  freeze={is_freeze}  "
                f"goal_reached={goal_reached}  group_intrusion={group_intrusion}  reason={failure_reason}  "
                f"route_comp={route_completion:.3f}  path_len={cpl:.3f}m  opt_path={optimal_path:.3f}m  "
                f"path_eff={path_eff_ep:.4f}  time={ep_time:.2f}s  "
                f"time_eff={time_efficiency:.3f}  energy_eff={energy_efficiency:.4f}  "
                f"out_ps={outside_ps_ep:.4f}  out_proj_ps={outside_proj_ps_ep:.4f}  "
                f"min_hum_dist={min_human_distance:.3f}  overall={overall_ep:.4f}"
            )
        self.episode_steps += 1
        # Reset accumulators for terminated envs
        if any_done.any():
            new_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, 0:2].clone()
            self.path_length[any_done] = 0.0
            self.episode_steps[any_done] = 0
            self._energy_accum[any_done] = 0.0
            self._in_ps_steps[any_done] = 0
            self._in_proj_ps_steps[any_done] = 0
            self._min_human_dist[any_done] = float("inf")
            current_pos[any_done] = new_pos[any_done]
            for idx in done_envs.tolist():
                row, col = scene.terrain.terrain_levels[idx], scene.terrain.terrain_types[idx]
                projector = scene.terrain.terrain_centerline_projectors[row][col]
                if projector is not None:
                    local_pos = new_pos[idx] - scene.env_origins[idx, :2]
                    s_start, _ = projector(local_pos[0].item(), local_pos[1].item())
                    goal_local = scene.env_goals[idx, :2] - scene.env_origins[idx, :2]
                    s_goal, _ = projector(goal_local[0].item(), goal_local[1].item())
                    self._cached_arc_start[idx] = s_start
                    self._cached_arc_goal[idx] = s_goal
                    self._cached_arc_length[idx] = s_start
                self._cached_cum_lateral[idx] = 0.0
                self._cached_lat_steps[idx] = 0
                self._off_sidewalk_steps[idx] = 0

        # Update centerline projection using last_pos (pre-reset position)
        if self.last_pos is not None:
            for i in range(self.num_envs):
                row, col = scene.terrain.terrain_levels[i], scene.terrain.terrain_types[i]
                projector = scene.terrain.terrain_centerline_projectors[row][col]
                if projector is None:
                    continue
                local_pos = self.last_pos[i] - scene.env_origins[i, :2]
                s, lat = projector(local_pos[0].item(), local_pos[1].item())
                self._cached_arc_length[i] = s
                self._cached_cum_lateral[i] += abs(lat)
                self._cached_lat_steps[i] += 1
                boundary_dist = scene.terrain.terrain_boundary_checkers[row][col](
                    local_pos[0].item(), local_pos[1].item()
                )
                if float(boundary_dist) > -0.236:
                    self._off_sidewalk_steps[i] += 1

        self.last_pos = current_pos.clone()
        self.last_vel = current_vel.clone()
        return done_info

    def summarize(self, env_config=None, pg_config=None):
        """Aggregate metrics across all envs and episodes.

        Returns:
            results: dict with aggregated metrics, or None if no episodes completed.
        """
        all_success = [s for per_env in self.success for s in per_env]
        all_collision = [c for per_env in self.collision for c in per_env]
        all_oob = [o for per_env in self.oob for o in per_env]
        all_timeout = [t for per_env in self.timeout for t in per_env]
        all_freeze = [f for per_env in self.freeze for f in per_env]
        all_col_static = [c for per_env in self.episode_collision_static for c in per_env]
        all_col_dynamic = [c for per_env in self.episode_collision_dynamic for c in per_env]
        all_route_completion = [r for per_env in self.episode_route_completion for r in per_env]
        all_time_efficiency = [t for per_env in self.episode_time_efficiency for t in per_env]
        all_energy_efficiency = [e for per_env in self.episode_energy_efficiency for e in per_env]
        all_in_ps_frac = [f for per_env in self.episode_in_ps_fraction for f in per_env]
        all_in_proj_ps_frac = [f for per_env in self.episode_in_proj_ps_fraction for f in per_env]
        all_min_human_distance = [d for per_env in self.episode_min_human_distance for d in per_env]
        all_path_length = [p for per_env in self.episode_path_length for p in per_env]
        all_optimal_path = [p for per_env in self.episode_optimal_path for p in per_env]
        all_ep_time = [t for per_env in self.episode_time for t in per_env]

        total = len(all_success)
        if total == 0:
            print("[Eval] No episodes completed.")
            return None

        # Success-only filters
        succ_mask = np.array(all_success, dtype=bool)
        succ_time_efficiency = np.array(all_time_efficiency)[succ_mask] if succ_mask.any() else np.array([])
        succ_energy_efficiency = np.array(all_energy_efficiency)[succ_mask] if succ_mask.any() else np.array([])
        succ_in_ps = np.array(all_in_ps_frac)[succ_mask] if succ_mask.any() else np.array([])
        succ_in_proj_ps = np.array(all_in_proj_ps_frac)[succ_mask] if succ_mask.any() else np.array([])
        succ_min_human_distance = np.array(all_min_human_distance)[succ_mask] if succ_mask.any() else np.array([])
        succ_path = np.array(all_path_length)[succ_mask] if succ_mask.any() else np.array([])
        succ_opt_path = np.array(all_optimal_path)[succ_mask] if succ_mask.any() else np.array([])

        # Path efficiency: optimal / total
        all_path_efficiency = []
        for opt_p, total_p in zip(all_optimal_path, all_path_length):
            if total_p > 1e-6:
                all_path_efficiency.append(min(opt_p / total_p, 1.0))
            else:
                all_path_efficiency.append(0.0)
        all_path_eff = np.array(all_path_efficiency)
        succ_path_eff = all_path_eff[succ_mask] if succ_mask.any() else np.array([])

        # A qualifying gesturing freeze is a success, not a timeout failure.
        all_timeout_combined = np.array([(t or f) and not s for t, f, s in
                                         zip(all_timeout, all_freeze, all_success)], dtype=float)
        # Env collision = oob OR static collision
        all_env_collision = np.array([o or s for o, s in zip(all_oob, all_col_static)], dtype=float)

        # Outside personal space fractions (higher = better)
        all_outside_ps = 1.0 - np.array(all_in_ps_frac)
        succ_outside_ps = 1.0 - succ_in_ps if len(succ_in_ps) > 0 else np.array([])
        all_outside_proj_ps = 1.0 - np.array(all_in_proj_ps_frac)
        succ_outside_proj_ps = 1.0 - succ_in_proj_ps if len(succ_in_proj_ps) > 0 else np.array([])

        success_rate = np.mean(all_success)

        # Overall metric: per-episode success_weighted (path_eff * outside_ps), then averaged
        overall_per_ep = np.array(all_success, dtype=float) * all_path_eff
        overall = np.mean(overall_per_ep)

        results = {
            # a. Task-related
            "success_rate": success_rate,
            "human_collision_rate": np.mean(all_col_dynamic),
            "timeout_rate": np.mean(all_timeout_combined),
            "env_collision_rate": np.mean(all_env_collision),
            "group_intrusion_rate": np.mean([v for row in self.group_intrusion for v in row]),
            "ignore_gesture_rate": np.mean([v for row in self.ignore_gesture for v in row]),
            "goal_reached_rate": np.mean([v for row in self.goal_reached for v in row]),
            "avg_route_completion": np.mean(all_route_completion),
            # b. Efficiency-related (all)
            "time_efficiency": np.mean(all_time_efficiency),
            "path_efficiency": np.mean(all_path_eff),
            "energy_efficiency": np.mean(all_energy_efficiency) if all_energy_efficiency else 0.0,
            # b. Efficiency-related (success-only)
            "time_efficiency_success": np.mean(succ_time_efficiency) if len(succ_time_efficiency) > 0 else float("nan"),
            "path_efficiency_success": np.mean(succ_path_eff) if len(succ_path_eff) > 0 else float("nan"),
            "energy_efficiency_success": (
                np.mean(succ_energy_efficiency) if len(succ_energy_efficiency) > 0 else float("nan")
            ),
            # c. Human-related (all)
            "avg_frac_outside_ps": np.mean(all_outside_ps),
            "avg_frac_outside_proj_ps": np.mean(all_outside_proj_ps),
            "avg_min_human_distance": np.mean(all_min_human_distance) if all_min_human_distance else float("inf"),
            # c. Human-related (success-only)
            "avg_frac_outside_ps_success": np.mean(succ_outside_ps) if len(succ_outside_ps) > 0 else float("nan"),
            "avg_frac_outside_proj_ps_success": (
                np.mean(succ_outside_proj_ps) if len(succ_outside_proj_ps) > 0 else float("nan")
            ),
            "avg_min_human_distance_success": (
                np.mean(succ_min_human_distance) if len(succ_min_human_distance) > 0 else float("inf")
            ),
            # d. Overall
            "overall": overall,
            # Metadata
            "total_episodes": total,
            "num_envs": self.num_envs,
            "gesturing_success_min_progress": GESTURING_MIN_PROGRESS,
            "episode_results": self.episode_results,
        }
        if env_config is not None:
            results.update(
                {
                    "scenario_type": env_config.get("scenario_type", "default"),
                    "model_type": env_config.get("model_type", ""),
                    "terrain_type": env_config.get("terrain_type", ""),
                    "ped_lateral": env_config.get("ped_lateral", "center"),
                    "ped_crossing_direction": env_config.get("ped_crossing_direction", "left_to_right"),
                }
            )
        if pg_config is not None:
            results.update(
                {
                    "num_peds_per_terrain": pg_config.get("num_peds_per_terrain", 0),
                    "num_static_obstacles": pg_config.get("num_walkable_objects_per_terrain", 0),
                    "seed": pg_config.get("seed"),
                    "scene_rng_version": pg_config.get("scene_rng_version"),
                }
            )

        print("\n" + "=" * 70)
        print(f"  EVALUATION RESULTS  ({total} episodes across {self.num_envs} envs)")
        print("=" * 70)
        print("  --- TASK-RELATED ---")
        print(f"  Success Rate              : {results['success_rate']:.4f}")
        print(f"  Human Collision Rate      : {results['human_collision_rate']:.4f}")
        print(f"  Timeout Rate (freeze+to)  : {results['timeout_rate']:.4f}")
        print(f"  Env Collision Rate (oob+s): {results['env_collision_rate']:.4f}")
        print(f"  Between Pedestrians Rate  : {results['group_intrusion_rate']:.4f}")
        print(f"  Ignore Gesture Rate       : {results['ignore_gesture_rate']:.4f}")
        print(f"  Avg Route Completion      : {results['avg_route_completion']:.4f}")
        print("  --- EFFICIENCY-RELATED ---")
        print(
            f"  Time Efficiency           : {results['time_efficiency']:.3f} m/s  (success: {results['time_efficiency_success']:.3f})"
        )
        print(
            f"  Path Efficiency           : {results['path_efficiency']:.4f}  (success: {results['path_efficiency_success']:.4f})"
        )
        print(
            f"  Energy Efficiency         : {results['energy_efficiency']:.4f} m/J  (success: {results['energy_efficiency_success']:.4f})"
        )
        print("  --- HUMAN-RELATED ---")
        print(
            f"  Avg Frac Outside PS       : {results['avg_frac_outside_ps']:.4f}  (success: {results['avg_frac_outside_ps_success']:.4f})"
        )
        print(
            f"  Avg Frac Outside Proj PS  : {results['avg_frac_outside_proj_ps']:.4f}  (success: {results['avg_frac_outside_proj_ps_success']:.4f})"
        )
        print(
            f"  Avg Min Human Distance    : {results['avg_min_human_distance']:.3f} m   (success: {results['avg_min_human_distance_success']:.3f} m)"
        )
        print("  --- OVERALL ---")
        print(f"  Overall (mean(succ * PE * OS)): {results['overall']:.4f}")
        print("=" * 70)
        return results

    def save(self, path, env_config=None, pg_config=None):
        """Summarize and save results to a pickle file."""
        import joblib

        results = self.summarize(env_config, pg_config)
        if results is not None:
            joblib.dump(results, path)
            print(f"Saved results to {path}")
        return results
