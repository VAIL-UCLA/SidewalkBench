"""Per-environment annotated point/image goals for scanned scenes."""

import math

import numpy as np
import torch

from .camera import camera_pose


class GSGoalManager:
    """Cache target views without moving the robot or stepping the simulation.

    Short/cocoa routes use their annotated endpoint. Long routes use the same
    segment length and advancement margin as PG. All coordinates here are in
    the shared scan frame, with env origins removed.
    """

    def __init__(self, num_envs, *, mode, ground_z, camera_height, intrinsics,
                 width=512, height=288, segment_length=30.0, advance_margin=1.0,
                 capture_images=False, device="cpu"):
        if mode not in ("short_horizon", "long_horizon", "cocoa"):
            raise ValueError(f"Unknown goal mode: {mode}")
        if not math.isfinite(segment_length) or segment_length <= 0:
            raise ValueError("segment_length must be finite and positive")
        if not math.isfinite(advance_margin) or not 0 <= advance_margin < segment_length:
            raise ValueError("advance_margin must be nonnegative and less than segment_length")
        self.mode, self.ground_z = mode, ground_z
        self.camera_height = camera_height
        self.camera_heights = np.full(num_envs, camera_height, dtype=np.float64)
        self.width, self.height = width, height
        self.segment_length, self.advance_margin = segment_length, advance_margin
        self.device = device
        self.intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32, device=device).reshape(1, 3, 3)
        self.runtimes = [None] * num_envs
        self.xy = np.zeros((num_envs, 2), np.float32)
        self.yaw = np.zeros(num_envs, np.float32)
        self.target_s = np.zeros(num_envs, np.float64)
        self.images = torch.zeros(num_envs, 3, height, width, device=device) if capture_images else None
        self.dirty = set()

    def reset(self, env_id, runtime):
        self.runtimes[env_id] = runtime
        self.camera_heights[env_id] = self.camera_height
        target = runtime.goal_progress_m
        if self.mode == "long_horizon":
            target = min(runtime.start_progress_m + self.segment_length, target)
        self._set_target(env_id, target)

    def set_camera_height(self, env_id, height):
        """Use the settled camera height above the local surface, as PG does."""
        if not math.isfinite(height) or height <= 0:
            raise ValueError("Goal camera height must be finite and positive")
        if not np.isclose(self.camera_heights[env_id], height):
            self.camera_heights[env_id] = height
            self.dirty.add(env_id)

    def _set_target(self, env_id, target):
        runtime = self.runtimes[env_id]
        xy, yaw = runtime.centerline.point_at(target)
        # Keep the exact annotation for the final goal, including a lateral offset.
        if target >= runtime.goal_progress_m - 1e-6:
            xy = runtime.route.goal_xy
        self.xy[env_id], self.yaw[env_id], self.target_s[env_id] = xy, yaw, target
        self.dirty.add(env_id)

    def update(self, robot_xy):
        if self.mode != "long_horizon":
            return
        for i, runtime in enumerate(self.runtimes):
            if runtime is None:
                continue
            progress = runtime.robot_progress_m(*robot_xy[i])
            target = self.target_s[i]
            while target < runtime.goal_progress_m - 1e-6 and progress >= target - self.advance_margin:
                target = min(target + self.segment_length, runtime.goal_progress_m)
            if target != self.target_s[i]:
                self._set_target(i, target)

    def goal_xy_local(self, robot_xy, robot_quat_xyzw):
        xy = torch.as_tensor(robot_xy)
        q = torch.as_tensor(robot_quat_xyzw, device=xy.device)
        yaw = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                          1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        delta = torch.as_tensor(self.xy, device=xy.device) - xy
        c, s = torch.cos(yaw), torch.sin(yaw)
        return torch.stack((c * delta[:, 0] + s * delta[:, 1],
                            -s * delta[:, 0] + c * delta[:, 1]), -1).cpu().numpy().astype(np.float32)

    def render_pending(self, renderer, extras=None):
        """Static target images include route cones; dynamic pedestrians are omitted, as in PG."""
        if self.images is None or not self.dirty:
            return
        ids = sorted(self.dirty)
        pos = torch.tensor([[*self.xy[i], self.ground_z(*self.xy[i]) + self.camera_heights[i]] for i in ids],
                           dtype=torch.float32, device=self.device)
        yaw = torch.as_tensor(self.yaw[ids], device=self.device)
        quat = torch.zeros(len(ids), 4, device=self.device)
        quat[:, 2], quat[:, 3] = torch.sin(yaw / 2), torch.cos(yaw / 2)
        cam, rot = camera_pose(pos, quat)
        self.images[ids] = renderer.render(cam, rot, self.intrinsics.expand(len(ids), -1, -1),
                                           width=self.width, height=self.height,
                                           humans=None if extras is None else [extras(i) for i in ids])
        self.dirty.clear()
