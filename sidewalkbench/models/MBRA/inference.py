#!/usr/bin/env python3
"""MBRA inference utilities.

MBRA takes 6 ImageNet-normalized 96x96 frames (stacked as 18 channels)
and a goal pose [x, y, sin(yaw), cos(yaw)] normalized by metric_spacing=0.8m.

User provides goal_xy in standard frame (x=forward, y=left, meters).

    nav = MBRAPGNavigator(device="cuda")
    traj, scores = nav.inference_trajectory(obs, goal_xy=np.array([5.0, 0.2]))
    vw, best = nav.inference_vw(obs, goal_xy=np.array([5.0, 0.2]))
"""

import os

import numpy as np
import torch
from torchvision import transforms

from sidewalkbench.models.base_model import BaseModel, ControllerCfg

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(MODEL_DIR, "mbra.onnx")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.yaml")
METRIC_SPACING = 0.8  # waypoint normalization factor

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
IMG_SIZE = 96


class MBRAPGNavigator(BaseModel):
    """MBRA point-goal navigator.

    All user-facing coordinates: x=forward, y=left, meters.
    """

    context_size = 6
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

    def _preprocess_images(self, obs_rgb):
        """obs_rgb: (B, C, 3, H, W) float32 [0,1] → (B, C*3, 96, 96) ImageNet-norm."""
        from PIL import Image as PILImage
        B, C = obs_rgb.shape[:2]
        frames = []
        for b in range(B):
            ch_list = []
            for c in range(C):
                frame = (obs_rgb[b, c].transpose(1, 2, 0) * 255).astype(np.uint8)
                pil = PILImage.fromarray(frame).resize((IMG_SIZE, IMG_SIZE))
                ch_list.append(IMAGENET_TRANSFORM(pil).numpy())  # (3, 96, 96)
            frames.append(np.concatenate(ch_list, axis=0))  # (C*3, 96, 96)
        return np.stack(frames)  # (B, 18, 96, 96)

    @staticmethod
    def _goal_to_pose(goal_xy):
        """Convert standard goal(s) [x_fwd, y_left] (meters) → MBRA goal_pose (..., 4).

        MBRA expects ``[x/spacing, y/spacing, cos(angle), sin(angle)]``. Accepts a
        single ``(2,)`` goal (returns ``(4,)``) or a batch ``(B, 2)`` (returns ``(B, 4)``).
        """
        goal_xy = np.asarray(goal_xy, dtype=np.float32)
        single = goal_xy.ndim == 1
        goal_xy = goal_xy.reshape(-1, 2)
        x, y = goal_xy[:, 0], goal_xy[:, 1]
        angle = np.arctan2(y, x)
        pose = np.stack([x / METRIC_SPACING, y / METRIC_SPACING, np.cos(angle), np.sin(angle)], axis=-1)
        return pose[0].astype(np.float32) if single else pose.astype(np.float32)

    def inference_trajectory(self, obs, goal_xy=None):
        """Run model → (trajectory, scores).

        Args:
            obs: (B, 6, 3, H, W) float32 in [0,1]. 6 frames.
            goal_xy: ``(B, 2)`` or ``(2,)`` goal(s) in standard frame
                ``[x_fwd, y_left]`` meters. If None, uses random goals.

        Returns:
            trajectory: np.ndarray (B, 1, 8, 2) waypoints [x, y] meters
            scores:     np.ndarray (B, 1)
        """
        obs_np = self._obs_to_numpy(obs)
        B = obs_np.shape[0]

        obs_norm = self._preprocess_images(obs_np)

        if goal_xy is None:
            goal_xy = np.random.uniform(low=[2.0, -2.0], high=[10.0, 2.0], size=(B, 2)).astype(np.float32)
        goal_pose = self._goal_to_pose(goal_xy)
        if goal_pose.ndim == 1 or goal_pose.shape[0] == 1:
            goal_batch = np.tile(goal_pose.reshape(1, -1), (B, 1))
        else:
            goal_batch = goal_pose

        out = self._session.run(None, {
            "obs_images": obs_norm,
            "goal_pose": goal_batch,
        })
        wp_raw = out[0]  # (B, 8, 4) — [x, y, sin(yaw), cos(yaw)]

        # Extract x, y and un-normalize
        wp_xy = wp_raw[:, :, :2] * METRIC_SPACING  # (B, 8, 2) meters
        trajectory = wp_xy[:, np.newaxis]  # (B, 1, 8, 2)
        scores = np.ones((B, 1), dtype=np.float32)
        return trajectory, scores
