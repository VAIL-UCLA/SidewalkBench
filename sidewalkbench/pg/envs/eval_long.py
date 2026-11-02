# Evaluation entry point: long-horizon (recovery) scenarios.
# NOTE: do not import isaaclab/pxr modules before env_setup, which starts AppLauncher.

import os
import time

import cv2
import gymnasium as gym
import numpy as np
import torch

from sidewalkbench.pg.envs.env_setup import (
    BaseEnvCfg,
    args_cli,
    env_config,
    exp_folder,
    model,
    model_input_length,
    model_input_shape,
    pg_config,
    robot_cfg,
    simulation_app,
    vis_requested,
)


# ---------------------------------------------------------------------------
# Recovery evaluation mode
# ---------------------------------------------------------------------------


def teleport_robot_to_recovery(env, env_ids, look_ahead=1.0):
    """Teleport robot to a recovery point ahead on the centerline.

    Projects the robot position onto the centerline, then moves ahead toward
    the goal by `look_ahead` meters along the centerline.
    """
    scene = env.unwrapped.scene
    if scene.terrain.terrain_centerline_splines is None:
        return
    if scene.terrain.terrain_centerline_projectors is None:
        return

    num_cols = scene.terrain.terrain_origins.shape[1]
    robot = scene["robot"]

    for env_id in env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids:
        row = scene.terrain.terrain_levels[env_id]
        col = scene.terrain.terrain_types[env_id]
        tidx = int(row * num_cols + col)

        spline_data = scene.terrain.terrain_centerline_splines[row][col]
        projector = scene.terrain.terrain_centerline_projectors[row][col]
        if spline_data is None or projector is None:
            continue
        arc_to_xy, total_len = spline_data

        world_pos = robot.data.root_pos_w[env_id, :2].cpu().numpy()
        env_origin = scene.env_origins[env_id, :2].cpu().numpy()
        goal_world = scene.env_goals[env_id, :2].cpu().numpy()

        local_x = float(world_pos[0] - env_origin[0])
        local_y = float(world_pos[1] - env_origin[1])
        s_current, _ = projector(local_x, local_y)

        goal_local_x = float(goal_world[0] - env_origin[0])
        goal_local_y = float(goal_world[1] - env_origin[1])
        s_goal, _ = projector(goal_local_x, goal_local_y)

        sign = 1.0 if s_goal > s_current else -1.0
        s_recovery = s_current + sign * look_ahead
        s_recovery = max(0.0, min(s_recovery, total_len))

        xy = arc_to_xy(np.array([s_recovery]))
        new_local_x = float(xy[0, 0])
        new_local_y = float(xy[0, 1])

        if s_recovery < total_len - 0.1:
            xy2 = arc_to_xy(np.array([s_recovery + 0.1]))
            dy = float(xy2[0, 1] - xy[0, 1])
            dx = float(xy2[0, 0] - xy[0, 0])
        else:
            xy2 = arc_to_xy(np.array([s_recovery - 0.1]))
            dy = float(xy[0, 1] - xy2[0, 1])
            dx = float(xy[0, 0] - xy2[0, 0])
        yaw = float(np.arctan2(dy, dx))

        new_world_x = float(new_local_x + env_origin[0])
        new_world_y = float(new_local_y + env_origin[1])

        root_pose = robot.data.default_root_pose[env_id]
        root_vel = robot.data.default_root_vel[env_id]
        root_state = torch.cat([root_pose, root_vel]).clone()
        root_state[0] = new_world_x
        root_state[1] = new_world_y
        root_state[2] = float(robot.data.root_pos_w[env_id, 2])
        root_state[3:7] = torch.tensor(
            [0, 0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=root_state.dtype, device=root_state.device
        )
        root_state[7:13] = 0.0
        robot.write_root_state_to_sim(root_state.unsqueeze(0), env_ids=torch.tensor([env_id], device=root_state.device))

        if hasattr(env.unwrapped, "_freeze_pos_buf"):
            new_pos_xy = torch.tensor(
                [[new_world_x, new_world_y]],
                dtype=env.unwrapped._freeze_pos_buf.dtype,
                device=env.unwrapped._freeze_pos_buf.device,
            )
            env.unwrapped._freeze_pos_buf[env_id] = new_pos_xy.unsqueeze(0)
            env.unwrapped._freeze_step[env_id] = 0

        print(f"[Recovery] Env {env_id} teleported to s={s_recovery:.1f}m")


class RecoveryTracker:
    """Track failure counts and steps for recovery evaluation across multiple envs."""

    def __init__(self, num_envs, max_resets=10, dt=0.05, device="cuda"):
        self.num_envs = num_envs
        self.max_resets = max_resets
        self.dt = dt
        self.device = device
        self.success_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.static_collisions = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.dynamic_collisions = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.oob_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.freeze_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.total_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.total_route_length = torch.zeros(num_envs, device=device)

    @property
    def all_done(self):
        return (self.success_count >= self.max_resets).all().item()

    def record_success(self, env_ids):
        self.success_count[env_ids] += 1

    def record_failures(self, env_ids, fail_type):
        getattr(self, f"{fail_type}_collisions" if fail_type in ("static", "dynamic") else f"{fail_type}_count")[
            env_ids
        ] += 1

    def record_route_length(self, env_ids, route_lengths):
        self.total_route_length[env_ids] += route_lengths

    def summarize(self):
        succ = self.success_count.sum().item()
        s_col = self.static_collisions.sum().item()
        d_col = self.dynamic_collisions.sum().item()
        oob = self.oob_count.sum().item()
        frz = self.freeze_count.sum().item()
        steps = self.total_steps.sum().item()
        route_len = self.total_route_length.sum().item()
        failures = s_col + d_col + oob + frz
        total_trials = succ + failures

        print("\n" + "=" * 60)
        print(f"  RECOVERY EVALUATION RESULTS  ({self.num_envs} envs)")
        print("=" * 60)
        print(f"  Successes               : {succ}")
        print(f"  Static Collisions       : {s_col}")
        print(f"  Dynamic Collisions      : {d_col}")
        print(f"  Out-of-Bounds           : {oob}")
        print(f"  Freezes                 : {frz}")
        print(f"  Total Steps (sum)       : {steps}")
        avg_vel = route_len / (steps * self.dt) if steps > 0 else 0.0
        print(f"  Average Velocity        : {avg_vel:.2f} m/s")
        print(f"  Total Route Length      : {route_len:.1f} m")
        if route_len > 0:
            per_m = 100.0 / route_len
        else:
            per_m = 0.0
        print(f"  Static Collisions /100m : {s_col * per_m:.2f}")
        print(f"  Dynamic Collisions/100m : {d_col * per_m:.2f}")
        print(f"  OOB /100m               : {oob * per_m:.2f}")
        print(f"  Freezes /100m           : {frz * per_m:.2f}")
        if total_trials > 0:
            print(f"  Recovery Rate           : {succ / total_trials:.4f}")
        print("=" * 60)
        return {
            "successes": succ,
            "static_collisions": s_col,
            "dynamic_collisions": d_col,
            "oob_count": oob,
            "freeze_count": frz,
            "total_steps": steps,
            "total_route_length": route_len,
        }


def main_recovery():
    """Recovery evaluation: teleport robot on failure, count failure types."""
    import joblib

    import sidewalkbench.pg.mdp.rewards as custom_rew
    from sidewalkbench.pg.utils.eval import disable_frame_rate_limit, prepare_model_input
    from sidewalkbench.pg.utils.vis import (
        VisState,
        build_state_dict,
        build_vis_dict,
        cleanup,
        save_images,
        save_mcap,
        save_video,
        show_online_viewer,
    )

    env = gym.make("SidewalkBench-PG-Base", cfg=BaseEnvCfg(recovery=True), render_mode="rgb_array")
    if getattr(model, "bind_env", None) is not None:
        model.bind_env(env)

    num_envs = env.unwrapped.num_envs
    dt_step = env.unwrapped.step_dt
    scene = env.unwrapped.scene
    tracker = RecoveryTracker(num_envs, max_resets=args_cli.max_resets, dt=dt_step, device=env.unwrapped.device)
    vis_enabled = vis_requested(args_cli)
    vis_dict = None
    vis_state = VisState()
    if vis_enabled:
        assert num_envs == 1, "Visualization only supports num_envs=1"
        os.makedirs(os.path.join(exp_folder, "recovery_images"), exist_ok=True)

    env.unwrapped.scene.pause_pedestrians = True
    env.reset(seed=args_cli.seed)
    # Long-horizon rows: defer resets so reaching the goal does not autoreset the
    # env; the barrier at the end of the loop advances every env to the next
    # terrain row together (with a single texture swap and fresh pedestrians).
    env.unwrapped.defer_resets = True
    row_done = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)

    # Goal conditioning / visualization: route is split into segments of
    # `goal_segment_length` and the current goal advances along the route.
    goal_type = getattr(model, "goal_type", None)
    goal_manager = None
    if vis_enabled or (pg_config.get("use_goal", False) and goal_type):
        from sidewalkbench.pg.utils.goals import GoalManager

        goal_manager = GoalManager(
            env,
            mode="long",
            segment_length=pg_config.get("goal_segment_length", 30.0),
            advance_margin=pg_config.get("goal_advance_margin", 1.0),
            capture_images=(goal_type == "image") or vis_enabled,
        )
        goal_manager.reset()
        goal_manager.pose_current()

    # Capture initial route progress for display offset
    num_cols = scene.terrain.terrain_origins.shape[1]
    row = scene.terrain.terrain_levels[0]
    col = scene.terrain.terrain_types[0]
    projector = scene.terrain.terrain_centerline_projectors[row][col]
    robot_pos_w = scene["robot"].data.root_pos_w[0, :2].cpu().numpy()
    env_origin = scene.env_origins[0, :2].cpu().numpy()
    local = robot_pos_w - env_origin
    route_start_offset, _ = projector(float(local[0]), float(local[1]))

    # Zero-action warm-up step so the first observation and goal image are valid.
    obs_dict, _, _, _, _ = env.step(torch.zeros((num_envs, 2), device=env.unwrapped.device))
    if goal_manager is not None:
        goal_manager.read_current()
    env.unwrapped.scene.pause_pedestrians = False

    disable_frame_rate_limit()

    for _ in range(1000000):
        start_t = time.time()
        if tracker.all_done:
            break

        obs = obs_dict["sensor"].to(env.unwrapped.device)
        goal_kwargs = {}
        if goal_manager is not None and pg_config.get("use_goal", False) and goal_type:
            if goal_type == "point":
                goal_kwargs["goal_xy"] = goal_manager.goal_xy_local()
            elif goal_type == "image":
                goal_kwargs["goal_image"] = goal_manager.goal_image()
        if goal_manager is not None:
            # Pose the goal camera before the step; its render is read afterwards.
            goal_manager.pose_current()
        with torch.no_grad():
            obs_model = prepare_model_input(obs, model_input_length, model_input_shape, env_config["model_type"])
            action, traj = model.inference_vw(obs_model, robot=scene["robot"], **goal_kwargs)

        # Environments that finished the current row are frozen until the barrier.
        action = action.clone()
        action[row_done] = 0.0

        if vis_enabled:
            state_dict = build_state_dict(env, action, traj, obs, robot_cfg, goal_manager=goal_manager)
            vis_dict = build_vis_dict(state_dict, vis_state)

            # Overlay failure counts + world time on third-person frame (top-right)
            demo = vis_dict["demo"]

            # Goal image thumbnail (top-left) so the conditioning image is visible.
            goal_img = state_dict.get("goal_image")
            if goal_img is not None:
                thumb_h = max(1, int(demo.shape[0] * 0.27))
                thumb_w = max(1, int(goal_img.shape[1] * thumb_h / goal_img.shape[0]))
                thumb = cv2.resize(goal_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                gx0, gy0 = 10, 10
                demo[gy0 : gy0 + thumb_h, gx0 : gx0 + thumb_w] = thumb
                cv2.rectangle(demo, (gx0 - 1, gy0 - 1), (gx0 + thumb_w, gy0 + thumb_h), (255, 255, 255), 1)
                cv2.putText(
                    demo,
                    "goal img",
                    (gx0 + 4, gy0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            col_total = tracker.static_collisions[0].item() + tracker.dynamic_collisions[0].item()
            oob_count = tracker.oob_count[0].item()
            freeze_count = tracker.freeze_count[0].item()
            world_time = tracker.total_steps[0].item() * dt_step

            # Compute current route progress via centerline projector
            robot_pos_w = scene["robot"].data.root_pos_w[0, :2].cpu().numpy()
            local = robot_pos_w - env_origin
            route_progress, _ = projector(float(local[0]), float(local[1]))
            route_progress -= route_start_offset
            route_total = scene.route_lengths[0].item()

            lines = [
                (f"Time: {world_time:.1f}s", (0, 0, 0)),
                (f"Progress: {route_progress:.1f}/{route_total:.0f}m", (0, 0, 0)),
                (f"Collision: {col_total}", (255, 255, 0)),
                (f"Out-of-Bound: {oob_count}", (0, 255, 0)),
                (f"Freeze: {freeze_count}", (0, 0, 255)),
            ]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2
            line_h = int(32 * font_scale)
            x0 = demo.shape[1] - 480
            y0 = 30

            # White semi-transparent background
            bg_h = len(lines) * line_h + 16
            bg_w = 470
            overlay = demo.copy()
            cv2.rectangle(overlay, (x0 - 8, y0 - 8), (x0 + bg_w, y0 + bg_h), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.4, demo, 0.6, 0, demo)

            for li, (line, color) in enumerate(lines):
                y = y0 + (li + 1) * line_h
                cv2.putText(demo, line, (x0, y), font, font_scale, color, thickness, cv2.LINE_AA)

            # --- BEV mini-map of route (top-right, below stats) ---
            spline_data = scene.terrain.terrain_centerline_splines[row][col]
            arc_to_xy_map, total_len = spline_data
            route_total_all = scene.route_lengths[0].item()
            map_s = np.linspace(route_start_offset, route_start_offset + route_total_all, 200)
            map_s = np.clip(map_s, 0, total_len)
            map_xy = arc_to_xy_map(map_s)
            mx_min, my_min = map_xy.min(axis=0) - 1
            mx_max, my_max = map_xy.max(axis=0) + 1
            dx = mx_max - mx_min
            dy = my_max - my_min
            max_side = 440
            if dy > dx:
                map_h = max_side
                map_w = max(60, int(max_side * dx / dy))
            else:
                map_w = max_side
                map_h = max(60, int(max_side * dy / dx))
            scale = min((map_w - 10) / max(dx, 1), (map_h - 10) / max(dy, 1))

            def _to_px(x, y):
                px = 5 + int((x - mx_min) * scale)
                py = map_h - 5 - int((y - my_min) * scale)
                return px, py

            map_canvas = np.full((map_h, map_w, 3), 220, dtype=np.uint8)
            pts = np.array([_to_px(x, y) for x, y in map_xy], dtype=np.int32)
            # Start (green), goal (red), current (blue) — based on route, not full spline
            start_xy_m = arc_to_xy_map(route_start_offset)
            goal_xy_m = arc_to_xy_map(min(route_start_offset + route_total_all, total_len))
            cur_xy_m = arc_to_xy_map(float(np.clip(route_progress + route_start_offset, 0, total_len)))
            # Place map canvas with alpha background, then redraw route at full opacity
            mx0 = demo.shape[1] - map_w - 10
            my0 = demo.shape[0] - map_h - 10
            roi = demo[my0 : my0 + map_h, mx0 : mx0 + map_w].astype(np.float32)
            blended = cv2.addWeighted(map_canvas.astype(np.float32), 0.5, roi, 0.5, 0)
            demo[my0 : my0 + map_h, mx0 : mx0 + map_w] = blended.astype(np.uint8)
            # Redraw route and markers at full opacity on demo
            pts_np = np.array([(px + mx0, py + my0) for px, py in pts], dtype=np.int32)
            cv2.polylines(demo, [pts_np], False, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(demo, (_to_px(*start_xy_m)[0] + mx0, _to_px(*start_xy_m)[1] + my0), 5, (0, 200, 0), -1)
            cv2.circle(demo, (_to_px(*goal_xy_m)[0] + mx0, _to_px(*goal_xy_m)[1] + my0), 5, (0, 0, 255), -1)
            cv2.circle(demo, (_to_px(*cur_xy_m)[0] + mx0, _to_px(*cur_xy_m)[1] + my0), 6, (255, 0, 0), -1)
            cv2.putText(
                demo,
                "Start",
                (_to_px(*start_xy_m)[0] + mx0 + 5, _to_px(*start_xy_m)[1] + my0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 0),
                2,
            )
            cv2.putText(
                demo,
                "Goal",
                (_to_px(*goal_xy_m)[0] + mx0 - 30, _to_px(*goal_xy_m)[1] + my0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            if args_cli.save_images:
                save_images(
                    vis_dict, state_dict, os.path.join(exp_folder, "recovery_images"), tracker.total_steps[0].item()
                )
            if args_cli.save_video:
                save_video(vis_dict, vis_state, exp_folder)
            if args_cli.save_mcap:
                save_mcap(vis_dict, state_dict, vis_state, env, exp_folder)

        obs_dict, reward, terminated, truncated, info = env.step(action)
        if goal_manager is not None:
            goal_manager.read_current()
        # Vectorized failure checks
        is_col_static = custom_rew.custom_collision_static(env.unwrapped)
        is_col_dynamic = custom_rew.custom_collision_dynamic(env.unwrapped)
        is_oob = custom_rew.out_of_region(env.unwrapped, buffer=1.0)
        is_freeze = custom_rew.freeze(env.unwrapped)

        # Process each env (failure priority: success > collision > oob > freeze)
        for env_id in range(num_envs):
            if row_done[env_id].item() or tracker.success_count[env_id] >= tracker.max_resets:
                continue

            if terminated[env_id].item():
                tracker.record_success(torch.tensor([env_id], device=tracker.device))
                tracker.record_route_length(env_id, scene.route_lengths[env_id])
                row_done[env_id] = True
                print(
                    f"[Recovery] Env {env_id} row success #{tracker.success_count[env_id].item()} "
                    f"at step {tracker.total_steps[env_id].item()}"
                )
                vis_state.pending_reset = True
                model.reset(env_id)
                if goal_manager is not None:
                    goal_manager.reset([env_id])
                continue

            tracker.total_steps[env_id] += 1

            fail_type = None
            if is_col_dynamic[env_id].item():
                fail_type = "dynamic"
            elif is_col_static[env_id].item():
                fail_type = "static"
            elif is_oob[env_id].item():
                fail_type = "oob"
            elif is_freeze[env_id].item():
                fail_type = "freeze"

            if fail_type:
                tracker.record_failures(torch.tensor([env_id], device=tracker.device), fail_type)
                cnt = getattr(
                    tracker, f"{fail_type}_collisions" if fail_type in ("static", "dynamic") else f"{fail_type}_count"
                )[env_id].item()
                print(f"[Recovery] Env {env_id} {fail_type} #{cnt}")
                model.reset(env_id)
                teleport_robot_to_recovery(env, [env_id], look_ahead=1.0)
                if goal_manager is not None:
                    goal_manager.update([env_id])
                continue  # only one failure type per env_id per step

        if goal_manager is not None:
            goal_manager.update()

        # ---- Row barrier: advance every env to the next terrain row together ----
        if bool(row_done.all()) and not tracker.all_done:
            types_t = scene.terrain.terrain_types
            old_tidx = (scene.terrain.terrain_levels * num_cols + types_t).unique().tolist()
            next_row = int(scene.terrain.terrain_levels[0].item()) + 1
            scene.terrain.terrain_levels[:] = next_row
            scene.terrain.env_origins[:] = scene.terrain.terrain_origins[next_row, types_t]
            # Drop the finished row's pedestrians / spawned objects before respawning.
            scene.clear_terrains(old_tidx)

            scene.pause_pedestrians = True
            env.unwrapped.force_reset_all()  # new start/goal + resample (new row)
            model.reset(None)
            if goal_manager is not None:
                goal_manager.reset()
                goal_manager.pose_current()
            obs_dict, _, _, _, _ = env.step(torch.zeros((num_envs, 2), device=env.unwrapped.device))
            if goal_manager is not None:
                goal_manager.read_current()
            scene.pause_pedestrians = False
            row_done[:] = False

            # Refresh visualization anchors for the new row.
            row = scene.terrain.terrain_levels[0]
            col = scene.terrain.terrain_types[0]
            projector = scene.terrain.terrain_centerline_projectors[row][col]
            env_origin = scene.env_origins[0, :2].cpu().numpy()
            robot_pos_w = scene["robot"].data.root_pos_w[0, :2].cpu().numpy()
            route_start_offset, _ = projector(
                float(robot_pos_w[0] - env_origin[0]), float(robot_pos_w[1] - env_origin[1])
            )
            print(f"[Recovery] Row barrier: all envs advanced to terrain row {next_row}")

        if args_cli.online_viewer:
            vis_dict["composite"] = vis_dict["demo"]
            show_online_viewer(vis_dict, vis_state, dt_step, start_t)

    results = tracker.summarize()
    results.update(seed=scene.randomization.seed, scene_rng_version=pg_config["scene_rng_version"])
    joblib.dump(results, f"{exp_folder}/results.pkl")
    print(f"Saved recovery results to {exp_folder}/results.pkl")
    if args_cli.save_video and vis_dict is not None:
        vis_state.episode_outcome = (True, None)
        save_video(vis_dict, vis_state, exp_folder)
    cleanup(vis_state)
    env.close()

if __name__ == "__main__":
    main_recovery()
    simulation_app.close()
