"""PG-format visualization for the GS eval (``visualize_reset<k>.mp4``, ``log_images/``).

The layout, writers and episode handling are PG's own -- ``build_vis_dict``,
``save_video``, ``save_images``, ``VisState`` from ``sidewalkbench.pg.utils.vis`` --
so the output is the same format PG produces (composite = third-person | obs +
trajectory corridor | obs + command corridor | BEV ego | BEV world; ``demo`` =
third-person frame with an obs+corridor PIP; 20 fps;
a SUCCESS / FAILURE overlay held ~1 s at each episode end; one file per episode).

The GS scene has no RTX cameras, so the two PG inputs it lacks are supplied here:
  * the third-person view is a gsplat chase-camera render at PG's ``camera2``
    geometry (1920x1080, K = [790, 0, 960; 0, 790, 540], mounted (-3, 0, 2) m in
    ``base_link``, looking at a point ahead of the robot), and
  * the robot itself (absent from a splat) is its articulated USD mesh, posed per link
    from the sim's body poses and depth-composited into that render
    (``mesh_render.RobotMesh``).
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch

from sidewalkbench.pg.utils.vis import VisState, _overlay_outcome, build_vis_dict, cleanup, save_images, save_video

from . import checks
from .camera import rotmat_from_quat_xyzw
from sidewalkbench.pg.primitives.robot.registry import get_robot_profile

# PG camera2 (sidewalkbench/pg/primitives/navigation/random_env_cfg.py)
TP_W, TP_H = 1920, 1080
TP_K = np.array([[790.0, 0.0, 960.0], [0.0, 790.0, 540.0], [0.0, 0.0, 1.0]])
TP_MOUNT = (-3.0, 0.0, 2.0)      # chase camera position in base_link
TP_LOOK_AT = (1.5, 0.0, 0.3)     # base_link point the chase camera looks at
OBS_W, OBS_H = 512, 288
PG_OUTCOME = {  # sbgs status -> (success, reason) for the video overlay, and trial-dir suffix
    checks.SUCCESS: ((True, None), "success"),
    checks.COLLISION: ((False, "Obstacle Collision"), "collision"),
    checks.OFF_SIDEWALK: ((False, "Out of Bounds"), "out_of_bound"),
    checks.FREEZE: ((False, "Freeze"), "freeze"),
    checks.TIMEOUT: ((False, "Timeout"), "timeout"),
}


def chase_camera_pose(pos_w: torch.Tensor, quat_xyzw: torch.Tensor):
    """Camera position (3,) and camera->world rotation (3, 3) (columns right, down,
    forward) for the chase camera rigidly mounted on base_link, like PG's camera2."""
    r = rotmat_from_quat_xyzw(quat_xyzw.reshape(1, 4))[0]
    cam = pos_w + r @ torch.tensor(TP_MOUNT, dtype=pos_w.dtype, device=pos_w.device)
    target = pos_w + r @ torch.tensor(TP_LOOK_AT, dtype=pos_w.dtype, device=pos_w.device)
    fwd = target - cam
    fwd = fwd / torch.linalg.norm(fwd)
    up = r[:, 2]
    right = torch.linalg.cross(fwd, up)
    right = right / torch.linalg.norm(right)
    down = torch.linalg.cross(fwd, right)
    down = down / torch.linalg.norm(down)
    return cam, torch.stack([right, down, fwd], dim=1)


class PGRecorder:
    """Drives PG's visualization/writers for one env of the GS eval (one recorder per env)."""

    def __init__(self, exp_folder: str, *, save_video: bool, save_images: bool,
                 obs_intrinsics: np.ndarray, robot_mesh, robot_cfg=None):
        self.exp_folder = exp_folder
        self.robot_mesh = robot_mesh  # mesh_render.LinkedMesh, shared by every env's recorder
        self.save_video = save_video
        self.save_images = save_images
        self.obs_k = np.asarray(obs_intrinsics, dtype=float)
        self.robot_cfg = get_robot_profile("coco").as_dict() if robot_cfg is None else robot_cfg
        self.vis_state = VisState()
        self.trial_id = 0
        self.trial_step = 0
        self.trial_dir = None

    def start_episode(self, index: int) -> None:
        """Name this episode's outputs by its route index k (``visualize_reset<k>.mp4``,
        ``trial_<k>_*``), so parallel envs never collide. save_video bumps the index once
        when it rolls over from a previous episode's writer."""
        self.vis_state.video_reset_idx = index - (self.vis_state.video_writer is not None)
        self.trial_id, self.trial_step = index, 0
        if self.save_images:
            self.trial_dir = os.path.join(self.exp_folder, f"log_images/trial_{self.trial_id}_running")
            os.makedirs(self.trial_dir, exist_ok=True)

    def frame(self, *, renderer, env_origin: torch.Tensor, pos_w: torch.Tensor, quat_xyzw: torch.Tensor,
              body_pos_w: torch.Tensor, body_quat_xyzw: torch.Tensor,
              obs_rgb: torch.Tensor, action: torch.Tensor, traj, world_min, world_size, human=None,
              goal_world=None, goal_yaw=None, goal_image=None) -> None:
        """One step of this env: obs the policy saw + the action/trajectory it produced (1-row batches);
        ``human`` is its pedestrians' Gaussians (``humans.HumanCrowd.gaussians``) or None."""
        device = pos_w.device
        cam, rot = chase_camera_pose(pos_w, quat_xyzw)
        robot = self.robot_mesh.posed(body_pos_w - env_origin, body_quat_xyzw)
        tp = renderer.render((cam - env_origin).unsqueeze(0), rot.unsqueeze(0),
                             torch.tensor(TP_K, dtype=torch.float32, device=device).unsqueeze(0),
                             width=TP_W, height=TP_H, meshes=[robot], humans=[human])[0]
        tp_rgb = (tp.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        tp_rgb = np.ascontiguousarray(tp_rgb)

        obs_image = (obs_rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        q = quat_xyzw.tolist()
        cur_yaw = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]))
        cur_xy = (pos_w[:2] - env_origin[:2]).cpu().numpy()
        state = {
            "obs_image": obs_image,
            "tp_rgb": cv2.resize(tp_rgb, (obs_image.shape[1], obs_image.shape[0])),
            "tp_rgb_raw": tp_rgb,
            "traj": traj,
            "action": action,
            "vx": float(action[0, 0]),
            "vyaw": float(action[0, 1]),
            "cur_xy_w": cur_xy,
            "cur_yaw": cur_yaw,
            "fx": self.obs_k[0, 0],
            "fy": self.obs_k[1, 1],
            "cx": self.obs_k[0, 2],
            "cy": self.obs_k[1, 2],
            "half_w": self.robot_cfg["bbox"][1] / 2.0,
            "camera_height": self.robot_cfg["camera_height"],
            "goal_world": goal_world,
            "goal_yaw": float("nan") if goal_yaw is None else float(goal_yaw),
            "goal_image": None if goal_image is None else (
                goal_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            "robot": None,
            "terrain_size": tuple(float(v) for v in world_size),
            "env_origin": np.asarray(world_min, dtype=np.float32),
        }
        vis = build_vis_dict(state, self.vis_state)
        if self.save_images:
            save_images(vis, state, self.trial_dir, self.trial_step)
            self.trial_step += 1
        if self.save_video:
            save_video(vis, self.vis_state, self.exp_folder)

    def episode_end(self, status: str) -> None:
        """Mark the episode boundary (PG: outcome overlay on the next video call, new file)."""
        outcome, suffix = PG_OUTCOME[status]
        self.vis_state.episode_outcome = outcome
        self.vis_state.pending_reset = True
        if self.save_images:
            final_dir = os.path.join(self.exp_folder, f"log_images/trial_{self.trial_id}_{suffix}")
            os.rename(self.trial_dir, final_dir)
            print(f"Saved trial images to {final_dir}", flush=True)
            self.trial_dir = None

    def close(self) -> None:
        """Flush the last episode's outcome overlay and release the writers.

        PG does this with one more ``save_video`` call, which also opens an extra
        1-frame file; here the overlay hold is written directly instead.
        """
        vs = self.vis_state
        if self.save_video and vs.video_writer is not None and vs.episode_outcome is not None and vs.prev_demo is not None:
            success, reason = vs.episode_outcome
            last_frame = vs.prev_demo.copy()
            _overlay_outcome(last_frame, success, reason)
            for _ in range(20):
                vs.video_writer.write(last_frame[..., ::-1])
            vs.episode_outcome = None
        cleanup(vs)
        if self.save_images and self.trial_dir and os.path.isdir(self.trial_dir) and not os.listdir(self.trial_dir):
            os.rmdir(self.trial_dir)
