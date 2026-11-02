#!/usr/bin/env python3
"""S2E inference utilities.

S2E takes 11 frames at 256x256 (no ImageNet normalization, [0,1]),
a goal vector [norm_dist, cos(angle), sin(angle)], and outputs
10 waypoints x 3 (x, y, yaw). Waypoints are scaled by 1/0.25 = 4x.

User provides goal_xy in standard frame (x=forward, y=left, meters).

    nav = S2ENavigator(device="cuda")
    traj, scores = nav.inference_trajectory(obs, goal_xy=np.array([5.0, 0.2]))
    vw, best = nav.inference_vw(obs, goal_xy=np.array([5.0, 0.2]))
"""

import os

import numpy as np
import torch

from sidewalkbench.models.base_model import BaseModel, ControllerCfg

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(MODEL_DIR, "s2e.onnx")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.yaml")
IMG_SIZE = 256
WP_SCALE = 0.25  # model outputs are multiplied by this to get meters


class S2ENavigator(BaseModel):
    """S2E point-goal / goal-free navigator.

    All user-facing coordinates: x=forward, y=left, meters.
    """

    context_size = 11
    multimodal = False
    goal_type = "point"

    def __init__(
        self,
        onnx_path: str = ONNX_PATH,
        device: str = "cuda",
        controller: ControllerCfg | None = None,
    ):
        super().__init__(device=device, controller=controller, info_path=INFO_PATH)
        self._session = self._make_session(onnx_path)

    @staticmethod
    def _goal_to_input(goal_xy):
        """Standard goal(s) [x_fwd, y_left] meters → model input [norm_dist, cos(θ), sin(θ)].

        Accepts a single ``(2,)`` goal (returns ``(3,)``) or a batch ``(B, 2)``
        (returns ``(B, 3)``).
        """
        goal_xy = np.asarray(goal_xy, dtype=np.float32)
        single = goal_xy.ndim == 1
        goal_xy = goal_xy.reshape(-1, 2)
        x, y = goal_xy[:, 0], goal_xy[:, 1]
        dist = np.hypot(x, y)
        norm_dist = np.clip(dist, 0.1, 200.0) / 200.0
        angle = np.arctan2(y, x)
        goal = np.stack([norm_dist, np.cos(angle), np.sin(angle)], axis=-1).astype(np.float32)
        return goal[0] if single else goal

    def inference_trajectory(self, obs, goal_xy=None):
        """Run model → (trajectory, scores).

        Args:
            obs: (B, 11, 3, H, W) float32 in [0,1]. 11 frames.
                 Images are resized internally to 256x256.
            goal_xy: ``(B, 2)`` or ``(2,)`` goal(s) in standard frame, or None → [5.0, 0.0].

        Returns:
            trajectory: np.ndarray (B, 1, 10, 2) meters
            scores:     np.ndarray (B, 1)
        """
        obs_np = self._obs_to_numpy(obs)
        B = obs_np.shape[0]

        # Resize to 256x256 if needed (obs is CHW)
        if obs_np.shape[-2:] != (IMG_SIZE, IMG_SIZE):
            import cv2

            resized = np.empty((B, 11, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            for b in range(B):
                for c in range(11):
                    frame_hwc = obs_np[b, c].transpose(1, 2, 0)  # HWC
                    frame_rsz = cv2.resize(frame_hwc, (IMG_SIZE, IMG_SIZE))
                    resized[b, c] = frame_rsz.transpose(2, 0, 1)
            obs_np = resized

        if goal_xy is None:
            goal_xy = np.random.uniform(low=[2.0, -2.0], high=[10.0, 2.0], size=(B, 2)).astype(np.float32)

        goal_input = self._goal_to_input(goal_xy)
        if goal_input.ndim == 1 or goal_input.shape[0] == 1:
            goal_batch = np.tile(goal_input.reshape(1, -1), (B, 1))
        else:
            goal_batch = goal_input

        all_wp_raw = []
        for i in range(B):
            out = self._session.run(
                None,
                {
                    "obs_images": obs_np[i : i + 1],  # (1, 11, 3, 256, 256)
                    "goal": goal_batch[i : i + 1],  # (1, 3)
                },
            )
            all_wp_raw.append(out[0][:, :, :2])

        wp_raw = np.concatenate(all_wp_raw, axis=0)  # (B, 10, 2) — x, y (scaled)
        wp_meters = wp_raw * WP_SCALE  # un-scale to meters

        trajectory = wp_meters[:, np.newaxis].astype(np.float32)  # (B, 1, 10, 2)
        scores = np.ones((B, 1), dtype=np.float32)
        return trajectory, scores
