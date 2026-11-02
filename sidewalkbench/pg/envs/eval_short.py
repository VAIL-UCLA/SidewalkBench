# Evaluation entry point: unit-test and pedestrian-reactive scenarios.
# NOTE: do not import isaaclab/pxr modules before env_setup, which starts AppLauncher.

import time

import gymnasium as gym
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


def main():
    from sidewalkbench.pg.utils.eval import EvalTracker, disable_frame_rate_limit, prepare_model_input
    from sidewalkbench.pg.utils.scenario_rules import outcome_label
    from sidewalkbench.pg.utils.vis import (
        VisState,
        build_state_dict,
        build_vis_dict,
        cleanup,
        finish_episode,
        save_images,
        save_mcap,
        save_video,
        show_online_viewer,
    )

    env = gym.make("SidewalkBench-PG-Base", cfg=BaseEnvCfg(), render_mode="rgb_array")
    if getattr(model, "bind_env", None) is not None:
        model.bind_env(env)
    vis_enabled = vis_requested(args_cli)
    vis_state = VisState()
    if vis_enabled:
        assert args_cli.num_envs == 1, "Saving images only supports num_envs=1"
        import os

        trial_id = 0
        trial_step = 0
        current_trial_dir = os.path.join(exp_folder, f"log_images/trial_{trial_id}_running")
        os.makedirs(current_trial_dir, exist_ok=True)

    # Pedestrians must not advance or trigger during the initial reset + warm-up.
    env.unwrapped.scene.pause_pedestrians = True
    env.reset()
    # Synchronized episodes: a terminated environment is frozen (zero action) instead
    # of being reset immediately. Once every environment has finished the round, the
    # boundary resets them all together, which re-randomizes the global scene (terrain
    # texture, statics/peds) and the start/goal, then re-captures the goal image(s)
    # before the next round begins.
    env.unwrapped.defer_resets = True
    device = env.unwrapped.device

    # ---- Evaluation tracking ----
    num_envs = env.unwrapped.num_envs
    dt_step = env.unwrapped.step_dt
    tracker = EvalTracker(num_envs, device, dt_step, max_resets=args_cli.max_resets,
                          scenario_type=env_config["scenario_type"])
    tracker.init_pos(env.unwrapped.scene["robot"].data.root_pos_w[:, 0:2], env)

    # Goal conditioning is opt-in via `use_goal: true` for models that declare
    # support. The goal manager is also built for
    # visualization so the goal can be inspected for any model, but goal kwargs are
    # passed only when goal conditioning is actually enabled.
    goal_type = getattr(model, "goal_type", None)
    use_goal_active = bool(pg_config.get("use_goal", False) and goal_type)
    goal_manager = None
    if use_goal_active or vis_enabled:
        from sidewalkbench.pg.utils.goals import GoalManager

        goal_manager = GoalManager(
            env,
            mode="short",
            segment_length=pg_config.get("goal_segment_length", 30.0),
            advance_margin=pg_config.get("goal_advance_margin", 1.0),
            capture_images=(goal_type == "image") or vis_enabled,
        )
        goal_manager.reset()
        goal_manager.pose_current()
    obs_dict, _, _, _, _ = env.step(torch.zeros((num_envs, 2), device=device))
    if goal_manager is not None:
        goal_manager.read_current()
    env.unwrapped.scene.pause_pedestrians = False

    disable_frame_rate_limit()

    round_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    retry_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
    round_num = 0

    while not tracker.all_done:
        round_num += 1
        retry_counts.zero_()
        print(f"[Eval] Round {round_num}/{args_cli.max_resets} started")
        # ---- One synchronized round: step until every environment finishes ----
        while not round_done.all():
            start = time.time()
            # Get action from policy
            obs = obs_dict["sensor"].to(device)

            goal_kwargs = {}
            if use_goal_active:
                if goal_type == "point":
                    goal_kwargs["goal_xy"] = goal_manager.goal_xy_local()
                elif goal_type == "image":
                    goal_kwargs["goal_image"] = goal_manager.goal_image()
            if goal_manager is not None:
                # Pose the goal camera before the step; its render is read afterwards.
                goal_manager.pose_current()

            with torch.no_grad():
                obs_model = prepare_model_input(obs, model_input_length, model_input_shape, env_config["model_type"])
                action, traj = model.inference_vw(obs_model, robot=env.unwrapped.scene["robot"], **goal_kwargs)

            # Environments that already finished this round are frozen.
            action = action.clone()
            action[round_done] = 0.0

            if vis_enabled:
                state_dict = build_state_dict(env, action, traj, obs, robot_cfg, goal_manager=goal_manager)
                vis_dict = build_vis_dict(state_dict, vis_state)
                if args_cli.save_images:
                    save_images(vis_dict, state_dict, current_trial_dir, trial_step)
                if args_cli.save_video:
                    save_video(vis_dict, vis_state, exp_folder)
                if args_cli.save_mcap:
                    save_mcap(vis_dict, state_dict, vis_state, env, exp_folder)

            obs_dict, reward, terminated, truncated, info = env.step(action)
            if goal_manager is not None:
                goal_manager.read_current()

            # Record and freeze only the environments that newly terminated this step.
            term = env.unwrapped.termination_manager._term_dones.any(dim=-1)
            newly_done = term & ~round_done
            done_info = tracker.step(env, record_ids=newly_done.nonzero(as_tuple=False).squeeze(-1))
            recorded = torch.zeros_like(round_done)
            for result in done_info:
                recorded[result["env_idx"]] = True
            round_done |= recorded

            # First-step terminations are intentionally omitted by EvalTracker.
            # Retry them within this round instead of advancing an unrecorded env.
            retry_ids = (newly_done & ~recorded).nonzero(as_tuple=False).squeeze(-1)
            if retry_ids.numel() > 0:
                retry_counts[retry_ids] += 1
                if bool((retry_counts[retry_ids] > 100).any()):
                    raise RuntimeError(f"Too many rejected first-step attempts in round {round_num}")
                for env_id in retry_ids.tolist():
                    print(f"[Eval retry] Env {env_id} episode {round_num}/{args_cli.max_resets}: "
                          f"rejected first-step termination; retry {retry_counts[env_id].item()}")
                obs_dict, _ = env.unwrapped.force_reset_envs(retry_ids)
                for env_id in retry_ids.tolist():
                    model.reset(env_id)
                tracker.reset_attempt(retry_ids, env)
                if goal_manager is not None:
                    goal_manager.reset(retry_ids.tolist())
                    goal_manager.pose_current()
                    env.unwrapped.sim.forward()
                    for _ in range(max(1, env.unwrapped.cfg.num_rerenders_on_reset)):
                        env.unwrapped.sim.render()
                    goal_manager.read_current()
                vis_state.pending_reset = True

            # Signal vis_state to clear history on next frame if episode ended
            if done_info:
                vis_state.pending_reset = True
                for info in done_info:
                    vis_state.episode_outcome = (info["success"], outcome_label(info["failure_reason"]))
                if vis_enabled:
                    finish_episode(vis_dict, vis_state,
                                   image_dir=current_trial_dir if args_cli.save_images else None)

            # Finalize trial image directories for terminated episodes
            if args_cli.save_images:
                trial_step += 1
                for info in done_info:
                    outcome = "success" if info["success"] else info["failure_reason"]
                    final_trial_dir = os.path.join(exp_folder, f"log_images/trial_{trial_id}_{outcome}")
                    os.rename(current_trial_dir, final_trial_dir)
                    print(f"Saved trial images to {final_trial_dir}")
                    trial_id += 1
                    trial_step = 0
                    current_trial_dir = os.path.join(exp_folder, f"log_images/trial_{trial_id}_running")
                    os.makedirs(current_trial_dir, exist_ok=True)

            if args_cli.online_viewer:
                show_online_viewer(vis_dict, vis_state, dt_step, start)

        if not bool((tracker.reset_count == round_num).all()):
            raise RuntimeError(
                f"Evaluation round {round_num} finished with inconsistent episode counts: "
                f"{tracker.reset_count.tolist()}"
            )
        print(f"[Eval] Round {round_num}/{args_cli.max_resets} complete: all {num_envs} environments recorded")
        if tracker.all_done:
            break

        # ---- Round boundary: global retexture + new start/goal + goal images ----
        # Hold pedestrians through the reset and the warm-up so the new round starts
        # with them at their reset pose (no advance, no trigger).
        env.unwrapped.scene.pause_pedestrians = True
        obs_dict, _ = env.unwrapped.force_reset_all()
        model.reset(None)
        if goal_manager is not None:
            goal_manager.reset()
            goal_manager.pose_current()
        # A warm-up step propagates the new texture and the goal-camera pose to the renderer.
        obs_dict, _, _, _, _ = env.step(torch.zeros((num_envs, 2), device=device))
        if goal_manager is not None:
            goal_manager.read_current()
        env.unwrapped.scene.pause_pedestrians = False
        tracker.init_pos(env.unwrapped.scene["robot"].data.root_pos_w[:, 0:2], env)
        round_done[:] = False

    # ---- Save results ----
    tracker.save(f"{exp_folder}/results.pkl", env_config, pg_config)

    cleanup(vis_state)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
