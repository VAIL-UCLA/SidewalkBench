# Copyright (c) 2022-2025, The SidewalkBench-PG Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Visualization utilities for evaluation scripts."""

import math
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Persistent visualization state (replaces main._XXX attributes)
# ---------------------------------------------------------------------------
class VisState:
    def __init__(self):
        self.pos_history = []  # list of np.array([x, y, yaw])
        self.pending_reset = False  # set True by caller on env reset
        self.world_px_per_m = None
        self.world_x_min = None
        self.world_x_max = None
        self.world_y_min = None
        self.world_y_max = None
        # video
        self.video_writer = None
        self.video_reset_idx = 0
        self.episode_outcome = None  # (success: bool, reason: str) set by caller on episode end
        self.prev_demo = None  # last frame of previous episode, for outcome overlay
        # mcap
        self.mcap_writer = None
        self.mcap_channels = None
        self.mcap_reset_idx = 0
        self.mcap_episode_t0 = None
        self.mcap_foxglove = None
        self.mcap_schemas = None
        # online viewer
        self.viewer_root = None
        self.viewer_label = None
        self.viewer_tk_img = None


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------
def project_waypoints_pinhole(waypoints_xy, fx, fy, cx, cy, camera_height):
    """Project ego-frame (x=forward, y=left) waypoints to pixel coords."""
    wp = np.asarray(waypoints_xy, dtype=np.float32)
    Z = wp[:, 0]
    X = -wp[:, 1]
    valid = Z > 1e-3
    Z, X = Z[valid], X[valid]
    u = fx * X / Z + cx
    v = fy * camera_height / Z + cy
    return np.stack([u, v], axis=1), valid


def draw_trajectory_corridor(
    img, waypoints_xy, fx, fy, cx, cy, camera_height, path_width=0.28, color_bgr=(255, 255, 0)
):
    """Draw a navigation-path corridor on *img* (modified in place)."""
    wp = np.asarray(waypoints_xy, dtype=np.float32)
    if wp.ndim != 2 or wp.shape[0] < 2 or wp.shape[1] < 2:
        return img
    wp = wp[:, :2]

    offsets_left, offsets_right = [], []
    for i in range(len(wp) - 1):
        d = wp[i + 1] - wp[i]
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        normal = np.array([-d[1], d[0]], dtype=np.float32) / n
        offsets_left.append(wp[i] + path_width * normal)
        offsets_right.append(wp[i] - path_width * normal)
    d = wp[-1] - wp[-2]
    n = np.linalg.norm(d)
    if n > 1e-6:
        normal = np.array([-d[1], d[0]], dtype=np.float32) / n
        offsets_left.append(wp[-1] + path_width * normal)
        offsets_right.append(wp[-1] - path_width * normal)
    if len(offsets_left) < 2:
        return img
    offsets_left = np.stack(offsets_left)
    offsets_right = np.stack(offsets_right)

    pts_center, _ = project_waypoints_pinhole(wp, fx, fy, cx, cy, camera_height)
    pts_left, _ = project_waypoints_pinhole(offsets_left, fx, fy, cx, cy, camera_height)
    pts_right, _ = project_waypoints_pinhole(offsets_right, fx, fy, cx, cy, camera_height)
    if len(pts_left) < 2 or len(pts_right) < 2:
        return img

    H, W = img.shape[:2]
    if wp[-1, 0] > 1.0:
        polygon = np.vstack([pts_left, pts_right[::-1]]).astype(np.int32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            t = (ys.astype(np.float32) - ys.min()) / (float(ys.max()) - ys.min() + 1e-6)
            alpha_vals = 0.2 + (1.0 - 0.2) * (1.0 - (1.0 - t))
            grad = np.zeros((H, W), dtype=np.float32)
            grad[ys, xs] = alpha_vals
            overlay = img.copy()
            overlay[mask > 0] = color_bgr
            for c in range(3):
                img[:, :, c] = img[:, :, c] * (1 - grad) + overlay[:, :, c] * grad

    # if wp[-1, 0] > 2.0:
    #     bar_color = tuple(int(c * 0.7) for c in color_bgr)
    #     def _in_bounds(pts):
    #         pts = pts.astype(np.int32)
    #         return pts[(pts[:, 0] >= 0) & (pts[:, 0] < W)
    #                    & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
    #     left_ok, right_ok = _in_bounds(pts_left), _in_bounds(pts_right)
    #     if len(left_ok) > 1:
    #         cv2.polylines(img, [left_ok], False, bar_color, 2, cv2.LINE_AA)
    #     if len(right_ok) > 1:
    #         cv2.polylines(img, [right_ok], False, bar_color, 2, cv2.LINE_AA)
    #     start_point = np.array([[W // 2, H - 1]], dtype=np.float32)
    #     center = np.vstack([start_point, pts_center])
    #     center = center[(center[:, 0] >= 0) & (center[:, 0] < W)
    #                     & (center[:, 1] >= 0) & (center[:, 1] < H)]
    #     if len(center) > 1:
    #         center_color = tuple(int(c * 0.8) for c in color_bgr)
    #         for i in range(len(center) - 1):
    #             p1 = tuple(center[i].astype(int))
    #             p2 = tuple(center[i + 1].astype(int))
    #             opacity = 0.1 + 0.2 * ((p1[1] + p2[1]) / 2.0 / H)
    #             ov = img.copy()
    #             cv2.line(ov, p1, p2, center_color, 2, cv2.LINE_AA)
    #             img[:] = cv2.addWeighted(ov, opacity, img, 1 - opacity, 0)
    return img


# ---------------------------------------------------------------------------
# build_state_dict: gather all state info needed for visualization
# ---------------------------------------------------------------------------
def build_state_dict(env, action, traj, obs, robot_cfg, goal_manager=None):
    """Collect current state into a plain dict for visualization.

    Args:
        goal_manager: Optional :class:`~sidewalkbench.pg.utils.goals.GoalManager`
            used to expose the current goal point and rendered goal image.
    """
    from sidewalkbench.pg.utils.math import _quat_to_yaw

    robot = env.unwrapped.scene["robot"]
    cur_xy_w = robot.data.root_pos_w[0, 0:2].detach().cpu().numpy()
    cur_yaw = _quat_to_yaw(robot.data.root_quat_w[0]).item()

    # Goal state. Prefer the GoalManager (long-horizon segments); fall back to the
    # scene's scenario/route goal.
    scene = env.unwrapped.scene
    goal_image = None
    if goal_manager is not None:
        goal_world = goal_manager.goal_world()[0]
        goal_yaw = float(goal_manager.goal_yaw()[0])
        images = goal_manager.goal_image()
        if images is not None:
            frame = images[0].detach().cpu().numpy()  # (3, H, W) in [0, 1]
            goal_image = (frame.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    else:
        goal_world = scene.env_goals[0, :2].detach().cpu().numpy()
        goal_yaw = float(scene.env_goal_yaws[0].item()) if hasattr(scene, "env_goal_yaws") else float("nan")

    # Third-person camera
    tp_rgb = env.unwrapped.scene["camera2"].data.output["rgb"]
    tp_depth = env.unwrapped.scene["camera2"].data.output["distance_to_camera"]
    tp_pos = env.unwrapped.scene["camera2"].data.pos_w
    tp_rot = env.unwrapped.scene["camera2"].data.quat_w_world
    tp_rgb = (tp_rgb[0].cpu().numpy()).astype(np.uint8)
    tp_depth = tp_depth[0].cpu().numpy()
    # tp_rot = tp_rot[[1, 2, 3, 0]]  # convert (w, x, y, z) to (x, y, z, w)
    tp_intrinsics = env.unwrapped.scene["camera2"].data.intrinsic_matrices

    # human overlay
    if env.unwrapped.scene.num_peds_per_terrain > 0:
        tp_human, tp_human_mask, tp_human_depth = env.unwrapped.scene.render_human(
            camera_pos=tp_pos,
            camera_quat=tp_rot,
            camera_intrinsics=tp_intrinsics,
            camera_shape=(1080, 1920),
            return_depth=True,
            first_env_only=True,
        )
        tp_human = tp_human.cpu().numpy()[0]
        tp_human_mask = tp_human_mask.cpu().numpy()[0]
        tp_human_depth = tp_human_depth.cpu().numpy()[0]
        tp_human_mask = np.logical_and(tp_human_mask.astype(np.uint8), tp_human_depth < tp_depth)
        tp_rgb = (tp_human * 255).astype(np.uint8) * tp_human_mask + tp_rgb * (1 - tp_human_mask)
        tp_rgb = tp_rgb.astype(np.uint8)

    # Observation image
    obs_image = obs[0, -1, :3].cpu().numpy().transpose(1, 2, 0)
    obs_image = (obs_image * 255.0).astype(np.uint8)
    # obs_image = cv2.resize(obs_image, (tp_rgb.shape[1], tp_rgb.shape[0]))

    camera_intrinsics = robot_cfg["camera_intrinsics"]
    # terrain size and origin for BEV world view
    try:
        terrain_size = env.unwrapped.cfg.scene.terrain.terrain_generator.size
    except Exception:
        terrain_size = (10.0, 10.0)
    env_origin = env.unwrapped.scene.env_origins[0, :2].detach().cpu().numpy()

    return {
        "obs_image": obs_image,
        "tp_rgb": cv2.resize(
            tp_rgb, (obs_image.shape[1], obs_image.shape[0])
        ),  # resize to match obs for easier overlay
        "tp_rgb_raw": tp_rgb,
        "traj": traj,
        "action": action,
        "vx": action[0, 0].item(),
        "vyaw": action[0, 1].item(),
        "cur_xy_w": cur_xy_w,
        "cur_yaw": cur_yaw,
        "fx": camera_intrinsics[0, 0],
        "fy": camera_intrinsics[1, 1],
        "cx": camera_intrinsics[0, 2],
        "cy": camera_intrinsics[1, 2],
        "half_w": robot_cfg["bbox"][1] / 2.0,
        "camera_height": robot_cfg["camera_height"],
        "robot": robot,
        "terrain_size": terrain_size,
        "env_origin": env_origin,
        "goal_world": goal_world,
        "goal_yaw": goal_yaw,
        "goal_image": goal_image,
    }


# ---------------------------------------------------------------------------
# build_vis_dict: produce all visualization images
# ---------------------------------------------------------------------------
def build_vis_dict(state, vis_state):
    """Generate visualization images from state_dict. Updates vis_state."""
    s = state
    fx, fy, cx, cy = s["fx"], s["fy"], s["cx"], s["cy"]
    cam_h = s["camera_height"]
    half_w = s["half_w"]
    vx, vyaw = s["vx"], s["vyaw"]
    cur_xy_w, cur_yaw = s["cur_xy_w"], s["cur_yaw"]
    traj = s["traj"]
    obs_img = s["obs_image"]
    h_img, w_img = obs_img.shape[:2]
    y_down = cam_h
    third_person = s["tp_rgb"]
    goal_world = s.get("goal_world")
    goal_yaw = s.get("goal_yaw", float("nan"))
    goal_img = s.get("goal_image")

    # -- col_traj: observation + trajectory corridor --
    col_traj = obs_img.copy()
    if traj is not None and len(traj) > 0:
        draw_trajectory_corridor(
            col_traj, traj[0], fx=fx, fy=fy, cx=cx, cy=cy, camera_height=cam_h, color_bgr=(0, 0, 255)
        )

    # -- col_cmd: observation + ground corridor + distance markers + arrow --
    col_cmd = obs_img.copy()
    z_vals = np.array([0.1, 3.0])
    pts_l = np.stack([fx * (-half_w) / z_vals + cx, fy * y_down / z_vals + cy], axis=1)
    pts_r = np.stack([fx * half_w / z_vals + cx, fy * y_down / z_vals + cy], axis=1)
    poly = np.vstack([pts_l, pts_r[::-1]]).astype(np.int32)
    overlay = col_cmd.copy()
    cv2.fillPoly(overlay, [poly], color=(255, 0, 0))
    col_cmd = cv2.addWeighted(overlay, 0.25, col_cmd, 0.75, 0)
    for boundary in (pts_l, pts_r):
        cv2.line(col_cmd, tuple(boundary[0].astype(int)), tuple(boundary[1].astype(int)), (200, 0, 0), 2)
    for dist in (1.0, 2.0):
        ul = int(fx * (-half_w) / dist + cx)
        ur = int(fx * half_w / dist + cx)
        vm = int(fy * y_down / dist + cy)
        cv2.line(col_cmd, (ul, vm), (ur, vm), (255, 255, 0), 2)
        cv2.putText(col_cmd, f"{int(dist)}m", (ur + 4, vm + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    # Command arrow
    cx_a, cy_a = w_img // 2, h_img - 30
    scale = 80 * min(abs(vx) / 1.5, 1.0)
    angle = -vyaw * 2.0
    tip = (cx_a + int(scale * math.sin(angle)), cy_a - int(scale * math.cos(angle)))
    cv2.arrowedLine(col_cmd, (cx_a, cy_a), tip, (0, 0, 255), 3, tipLength=0.3)
    cv2.putText(col_cmd, f"vx:{vx:.2f} vyaw:{vyaw:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # -- BEV ego --
    bev_h, bev_w = h_img, w_img
    x_min, x_max = -1.0, 5.0
    px_per_m = bev_h / (x_max - x_min)
    half_y_m = (bev_w / 2.0) / px_per_m

    def _ego_to_px(xy):
        xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        u = (half_y_m - xy[:, 1]) * px_per_m
        v = (x_max - xy[:, 0]) * px_per_m
        return np.stack([u, v], axis=1).astype(np.int32)

    # Update pos history
    if vis_state.pending_reset:
        vis_state.pos_history = []
        vis_state.pending_reset = False
    vis_state.pos_history.append(np.array([cur_xy_w[0], cur_xy_w[1], cur_yaw], dtype=np.float32))

    bev = np.full((bev_h, bev_w, 3), 30, dtype=np.uint8)
    for r in range(int(np.ceil(x_min)), int(np.floor(x_max)) + 1):
        pts = _ego_to_px([[r, -half_y_m], [r, half_y_m]])
        cv2.line(bev, tuple(pts[0]), tuple(pts[1]), (60, 60, 60), 1)
    for l in range(int(np.ceil(-half_y_m)), int(np.floor(half_y_m)) + 1):
        pts = _ego_to_px([[x_min, l], [x_max, l]])
        cv2.line(bev, tuple(pts[0]), tuple(pts[1]), (60, 60, 60), 1)
    # Past track
    if len(vis_state.pos_history) > 1:
        hist = np.stack(vis_state.pos_history)
        rel = hist[:, :2] - cur_xy_w
        c, s_r = math.cos(-cur_yaw), math.sin(-cur_yaw)
        rot = np.array([[c, -s_r], [s_r, c]], dtype=np.float32)
        cv2.polylines(bev, [_ego_to_px(rel @ rot.T)], False, (255, 255, 255), 2, cv2.LINE_AA)
    # Predicted trajectory
    if traj is not None and len(traj) > 0:
        pred_full = np.vstack([[0.0, 0.0], traj[0][:, :2]])
        cv2.polylines(bev, [_ego_to_px(pred_full)], False, (0, 255, 255), 2, cv2.LINE_AA)
    # (v,w) integration
    ts = np.arange(0.0, 3.0 + 1e-6, 0.1, dtype=np.float32)
    if abs(vyaw) < 1e-4:
        tx, ty = vx * ts, np.zeros_like(ts)
    else:
        tx = (vx / vyaw) * np.sin(vyaw * ts)
        ty = (vx / vyaw) * (1.0 - np.cos(vyaw * ts))
    cv2.polylines(bev, [_ego_to_px(np.stack([tx, ty], axis=1))], False, (255, 0, 255), 2, cv2.LINE_AA)
    # Ego arrow + legend
    cv2.arrowedLine(
        bev, tuple(_ego_to_px([[0, 0]])[0]), tuple(_ego_to_px([[0.4, 0]])[0]), (255, 128, 0), 2, tipLength=0.4
    )
    cv2.putText(bev, "past", (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bev, "pred", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bev, "(v,w)", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
    # Goal (ego frame)
    if goal_world is not None:
        delta = np.asarray(goal_world, dtype=np.float32) - cur_xy_w.astype(np.float32)
        c_g, s_g = math.cos(-cur_yaw), math.sin(-cur_yaw)
        ego_goal = np.array([c_g * delta[0] - s_g * delta[1], s_g * delta[0] + c_g * delta[1]], dtype=np.float32)
        goal_px = _ego_to_px([ego_goal])[0]
        dist = float(np.hypot(delta[0], delta[1]))
        ang = float(np.degrees(np.arctan2(ego_goal[1], ego_goal[0])))
        if 0 <= goal_px[0] < bev_w and 0 <= goal_px[1] < bev_h:
            cv2.drawMarker(bev, tuple(goal_px), (0, 0, 255), cv2.MARKER_STAR, 16, 2, cv2.LINE_AA)
        cv2.putText(
            bev, f"goal {dist:.1f}m {ang:+.0f}deg", (10, 64),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA,
        )

    # -- BEV world --
    terrain_size = s.get("terrain_size", (10.0, 10.0))
    env_origin = s.get("env_origin", cur_xy_w)
    new_episode = len(vis_state.pos_history) == 1
    if vis_state.world_px_per_m is None or new_episode:
        # Show the full terrain block with env_origin at bottom-left corner
        vis_state.world_x_min = env_origin[0]
        vis_state.world_x_max = env_origin[0] + terrain_size[0]
        vis_state.world_y_min = env_origin[1]
        vis_state.world_y_max = env_origin[1] + terrain_size[1]
        vis_state.world_px_per_m = min(bev_w / terrain_size[0], bev_h / terrain_size[1])
    wppm = vis_state.world_px_per_m

    def _world_to_px(xy):
        xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        u = (xy[:, 0] - vis_state.world_x_min) * wppm
        v = bev_h - 1 - (xy[:, 1] - vis_state.world_y_min) * wppm
        return np.stack([u, v], axis=1).astype(np.int32)

    col_world = np.full((bev_h, bev_w, 3), 30, dtype=np.uint8)
    grid_step = max(5, int(max(terrain_size) / 10))
    for gx in range(int(np.floor(vis_state.world_x_min)), int(np.ceil(vis_state.world_x_max)) + 1, grid_step):
        pts = _world_to_px([[gx, vis_state.world_y_min], [gx, vis_state.world_y_max]])
        cv2.line(col_world, tuple(pts[0]), tuple(pts[1]), (60, 60, 60), 1)
    for gy in range(int(np.floor(vis_state.world_y_min)), int(np.ceil(vis_state.world_y_max)) + 1, grid_step):
        pts = _world_to_px([[vis_state.world_x_min, gy], [vis_state.world_x_max, gy]])
        cv2.line(col_world, tuple(pts[0]), tuple(pts[1]), (60, 60, 60), 1)
    if len(vis_state.pos_history) > 1:
        hist_xy = np.stack([h[:2] for h in vis_state.pos_history])
        cv2.polylines(col_world, [_world_to_px(hist_xy)], False, (255, 255, 255), 2, cv2.LINE_AA)
    if traj is not None and len(traj) > 0:
        ego_pred = traj[0][:, :2].astype(np.float32)
        c_w, s_w = math.cos(cur_yaw), math.sin(cur_yaw)
        R_ew = np.array([[c_w, -s_w], [s_w, c_w]], dtype=np.float32)
        pred_world = np.vstack([cur_xy_w.astype(np.float32), ego_pred @ R_ew.T + cur_xy_w.astype(np.float32)])
        cv2.polylines(col_world, [_world_to_px(pred_world)], False, (0, 255, 255), 2, cv2.LINE_AA)
    robot_px = _world_to_px([cur_xy_w])[0]
    tip_w = cur_xy_w + np.array([math.cos(cur_yaw), math.sin(cur_yaw)], dtype=np.float32)
    cv2.arrowedLine(col_world, tuple(robot_px), tuple(_world_to_px([tip_w])[0]), (255, 128, 0), 2, tipLength=0.4)
    cv2.putText(col_world, "world", (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    # Goal (world frame): marker, heading arrow, and a dashed robot->goal line.
    if goal_world is not None:
        goal_px = _world_to_px([goal_world])[0]
        n_dash = 12
        line = np.linspace(robot_px, goal_px, n_dash * 2, dtype=np.int32)
        for k in range(0, len(line) - 1, 2):
            cv2.line(col_world, tuple(line[k]), tuple(line[k + 1]), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(col_world, tuple(goal_px), (0, 0, 255), cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)
        if not math.isnan(goal_yaw):
            goal_tip = np.asarray(goal_world, dtype=np.float32) + np.array(
                [math.cos(goal_yaw), math.sin(goal_yaw)], dtype=np.float32
            )
            cv2.arrowedLine(
                col_world, tuple(goal_px), tuple(_world_to_px([goal_tip])[0]), (0, 0, 255), 2, tipLength=0.4
            )
        cv2.putText(
            col_world, "goal", (goal_px[0] + 6, goal_px[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
        )

    # -- Goal image panel --
    if goal_img is not None:
        col_goal = cv2.resize(goal_img, (bev_w, bev_h), interpolation=cv2.INTER_AREA)
    else:
        col_goal = np.full((bev_h, bev_w, 3), 30, dtype=np.uint8)
    cv2.putText(col_goal, "goal img", (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # -- Composite --
    composite = np.concatenate([third_person, col_traj, col_cmd, bev, col_world, col_goal], axis=1)
    third_person_raw = s.get("tp_rgb_raw", third_person)
    PIP_SCALE = 0.3
    pip_h = int(third_person_raw.shape[0] * PIP_SCALE)
    pip_w = int(col_traj.shape[1] * pip_h / col_traj.shape[0])
    col_traj_pip = cv2.resize(col_traj, (pip_w, pip_h))
    demo = third_person_raw.copy()
    demo[-pip_h:, :pip_w] = col_traj_pip

    return {
        "third_person": third_person,
        "obs_raw": obs_img,
        "col_traj": col_traj,
        "col_cmd": col_cmd,
        "bev_ego": bev,
        "bev_world": col_world,
        "col_goal": col_goal,
        "composite": composite,
        "demo": demo,
    }


# ---------------------------------------------------------------------------
# Output functions
# ---------------------------------------------------------------------------
def save_images(vis_dict, state_dict, output_dir, step):
    """Save observation, composite, and demo images to disk."""
    cv2.imwrite(f"{output_dir}/observation_{step:04d}.jpg", state_dict["obs_image"][..., ::-1])
    cv2.imwrite(f"{output_dir}/rgb_{step:04d}.jpg", vis_dict["composite"][..., ::-1])
    if "demo" in vis_dict:
        cv2.imwrite(f"{output_dir}/demo_{step:04d}.jpg", vis_dict["demo"][..., ::-1])


def save_video(vis_dict, vis_state, exp_folder):
    """Append composite frame to video. Handles episode boundaries."""
    composite = vis_dict["demo"]
    new_episode = len(vis_state.pos_history) == 1 or vis_state.pending_reset
    if new_episode and vis_state.video_writer is not None:
        _finish_video(vis_state)
    if vis_state.video_writer is None:
        import os

        video_path = os.path.join(exp_folder, f"visualize_reset{vis_state.video_reset_idx}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 20.0
        writer = cv2.VideoWriter(video_path, fourcc, fps, (composite.shape[1], composite.shape[0]))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"Cannot open MP4 video writer for {video_path}. "
                "Restore the FFmpeg-enabled OpenCV package with: uv sync --reinstall-package opencv-python"
            )
        vis_state.video_writer = writer
        print(f"[Video] Writing {composite.shape[1]}x{composite.shape[0]} @ {fps} FPS to {video_path}")
    vis_state.video_writer.write(composite[..., ::-1])
    vis_state.prev_demo = composite.copy()


def _finish_video(vis_state):
    """Flush the outcome to the current video, including the final episode."""
    if vis_state.video_writer is None:
        return
    if vis_state.episode_outcome is not None and vis_state.prev_demo is not None:
        last_frame = vis_state.prev_demo.copy()
        _overlay_outcome(last_frame, *vis_state.episode_outcome)
        for _ in range(20):  # hold for one second at the video rate of 20 fps
            vis_state.video_writer.write(last_frame[..., ::-1])
    vis_state.video_writer.release()
    vis_state.video_writer = None
    vis_state.video_reset_idx += 1
    vis_state.episode_outcome = None


def finish_episode(vis_dict, vis_state, image_dir=None):
    """Show the recorded outcome in the viewer, saved images, and video."""
    if vis_state.episode_outcome is None:
        return
    for key in ("demo", "composite"):
        if key in vis_dict:
            frame = vis_dict[key].copy()
            _overlay_outcome(frame, *vis_state.episode_outcome)
            vis_dict[key] = frame
            if image_dir is not None:
                cv2.imwrite(f"{image_dir}/{key}_outcome.jpg", frame[..., ::-1])
    _finish_video(vis_state)
    vis_state.episode_outcome = None


def _overlay_outcome(frame, success, reason):
    """Overlay episode outcome text with semi-transparent background on *frame* (modified in place)."""
    h, w = frame.shape[:2]
    if success:
        text = "SUCCESS"
        color = (0, 255, 0)
        bg_color = (0, 80, 0)
    else:
        text = f"FAILURE: {reason}" if reason else "FAILURE"
        color = (255, 0, 0)
        bg_color = (80, 0, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(w, h) / 400  # 2× larger
    thickness = max(3, int(font_scale * 2))
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = int(th * 0.8)
    x = (w - tw) // 2
    y = (h + th) // 2  # centered vertically
    y1 = max(0, y - th - margin)
    y2 = min(h, y + margin)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), bg_color, -1)
    cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)


def show_online_viewer(vis_dict, vis_state, dt_step, loop_start_time):
    """Display composite image in a tkinter window."""
    import tkinter as tk

    from PIL import Image, ImageTk

    composite = vis_dict["composite"]
    if vis_state.viewer_root is None:
        vis_state.viewer_root = tk.Tk()
        vis_state.viewer_root.title("SidewalkBench-PG Viewer")
        screen_w = vis_state.viewer_root.winfo_screenwidth()
        img_w = composite.shape[1]
        init_w = min(img_w, screen_w - 40)
        init_h = int(round(composite.shape[0] * init_w / img_w))
        vis_state.viewer_root.geometry(f"{init_w}x{init_h}")
        vis_state.viewer_root.pack_propagate(False)
        vis_state.viewer_label = tk.Label(
            vis_state.viewer_root, borderwidth=0, highlightthickness=0, padx=0, pady=0, bd=0
        )
        vis_state.viewer_label.pack(fill=tk.BOTH, expand=True)

    pil_img = Image.fromarray(composite)
    w = vis_state.viewer_root.winfo_width()
    h = vis_state.viewer_root.winfo_height()
    if w > 1 and h > 1:
        pil_img = pil_img.resize((w, h), Image.Resampling.BILINEAR)
    vis_state.viewer_tk_img = ImageTk.PhotoImage(pil_img)
    vis_state.viewer_label.configure(image=vis_state.viewer_tk_img)
    vis_state.viewer_root.update()
    while time.time() - loop_start_time < dt_step:
        time.sleep(0.001)


def save_mcap(vis_dict, state_dict, vis_state, env, exp_folder):
    """Record visualization data to a Foxglove MCAP file."""
    if vis_state.mcap_channels is None:
        import foxglove
        from foxglove.channels import (
            PoseChannel,
            PosesInFrameChannel,
            RawImageChannel,
            Vector2Channel,
            Vector3Channel,
        )
        from foxglove.schemas import (
            Pose,
            PosesInFrame,
            Quaternion,
            RawImage,
            Timestamp,
            Vector2,
            Vector3,
        )

        vis_state.mcap_foxglove = foxglove
        vis_state.mcap_schemas = {
            "RawImage": RawImage,
            "Pose": Pose,
            "PosesInFrame": PosesInFrame,
            "Quaternion": Quaternion,
            "Timestamp": Timestamp,
            "Vector2": Vector2,
            "Vector3": Vector3,
        }
        vis_state.mcap_channels = {
            "panel_follow": RawImageChannel(topic="/image/panel/follow"),
            "panel_traj": RawImageChannel(topic="/image/panel/visualized_traj"),
            "panel_cmd": RawImageChannel(topic="/image/panel/command_distance"),
            "panel_bev_ego": RawImageChannel(topic="/image/panel/bev_ego"),
            "panel_topdown": RawImageChannel(topic="/image/panel/topdown_world"),
            "best_path": PosesInFrameChannel(topic="/model/output/best_path"),
            "target_vw": Vector2Channel(topic="/control/target_vw"),
            "actual_vw": Vector3Channel(topic="/control/actual_vw"),
            "robot_world": PoseChannel(topic="/robot/pose_world"),
        }

    import os

    new_episode = len(vis_state.pos_history) == 1
    if new_episode and vis_state.mcap_writer is not None:
        vis_state.mcap_writer.close()
        vis_state.mcap_reset_idx += 1
        vis_state.mcap_writer = None
    if vis_state.mcap_writer is None:
        mcap_path = os.path.join(exp_folder, f"visualize_reset{vis_state.mcap_reset_idx}.mcap")
        vis_state.mcap_writer = vis_state.mcap_foxglove.open_mcap(mcap_path, allow_overwrite=True)
        vis_state.mcap_episode_t0 = time.time()
        print(f"[MCAP] Writing to {mcap_path}")

    t_rel = time.time() - vis_state.mcap_episode_t0
    sec_i = int(t_rel)
    nsec_i = int((t_rel - sec_i) * 1e9)
    log_time_ns = int(sec_i * 1e9 + nsec_i)
    S = vis_state.mcap_schemas
    _ts = S["Timestamp"](sec=sec_i, nsec=nsec_i)
    chs = vis_state.mcap_channels

    def _log_panel(key, img_rgb):
        if img_rgb is None:
            return
        img_bgr = np.ascontiguousarray(img_rgb[..., ::-1])
        chs[key].log(
            S["RawImage"](
                timestamp=_ts,
                frame_id="base_link",
                width=img_bgr.shape[1],
                height=img_bgr.shape[0],
                encoding="bgr8",
                step=img_bgr.shape[1] * 3,
                data=img_bgr.tobytes(),
            ),
            log_time=log_time_ns,
        )

    _log_panel("panel_follow", vis_dict["third_person"])
    _log_panel("panel_traj", vis_dict["col_traj"])
    _log_panel("panel_cmd", vis_dict["col_cmd"])
    _log_panel("panel_bev_ego", vis_dict["bev_ego"])
    _log_panel("panel_topdown", vis_dict["bev_world"])

    traj = state_dict["traj"]
    if traj is not None and len(traj) > 0:
        poses = [
            S["Pose"](position=S["Vector3"](x=float(p[0]), y=float(p[1]), z=0.0), orientation=S["Quaternion"](w=1.0))
            for p in traj[0]
        ]
        chs["best_path"].log(S["PosesInFrame"](timestamp=_ts, frame_id="base_link", poses=poses), log_time=log_time_ns)

    chs["target_vw"].log(S["Vector2"](x=state_dict["vx"], y=state_dict["vyaw"]), log_time=log_time_ns)

    robot = state_dict["robot"]
    v_xy = robot.data.root_vel_w[0, :2]
    v_speed = float(v_xy.norm().item())
    w_yaw = float(robot.data.root_ang_vel_w[0, 2].item())
    chs["actual_vw"].log(S["Vector3"](x=v_speed, y=w_yaw, z=v_speed), log_time=log_time_ns)

    cur_yaw = state_dict["cur_yaw"]
    half = cur_yaw / 2.0
    chs["robot_world"].log(
        S["Pose"](
            position=S["Vector3"](x=float(state_dict["cur_xy_w"][0]), y=float(state_dict["cur_xy_w"][1]), z=0.0),
            orientation=S["Quaternion"](x=0.0, y=0.0, z=float(math.sin(half)), w=float(math.cos(half))),
        ),
        log_time=log_time_ns,
    )


def cleanup(vis_state):
    """Release video writer and mcap writer."""
    if vis_state.video_writer is not None:
        _finish_video(vis_state)
        print("[Video] Saved.")
    if vis_state.mcap_writer is not None:
        vis_state.mcap_writer.close()
        print("[MCAP] Saved.")
