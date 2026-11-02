#!/usr/bin/env python3
"""MIMIC inference utilities.

MIMIC ONNX (``mimic.onnx``) — point-goal, 16-frame context with an explicit
past-trajectory + past-mask input.

The ONNX has four inputs and three outputs::

    obs       : (B, 16, 3, 256, 256)  float32 in [0, 1]
    goal      : (B, 3)                float32 = [dist/200, cos(a), sin(a)]
    past_traj : (B, 15, 2)             float32 — past XY in camera frame x 0.25
                                       (usually zeros at inference)
    past_mask : (B, 16)                float32 — 1.0 for valid frames

    best_reg_mid         : (B, 10, 3)       — best single trajectory (model-picked)
    poses_reg_mid_denorm : (B, 62, 10, 3)   — 62 candidates
    poses_cls_mid        : (B, 62)          — classification scores

    nav = MIMICNavigator(device="cuda")
    traj, scores = nav.inference_trajectory(obs)
    vw, best = nav.inference_vw(obs)
    nav.reset()
"""

import os

import numpy as np
import torch

from sidewalkbench.models.base_model import BaseModel, ControllerCfg

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(MODEL_DIR, "mimic.onnx")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.yaml")
HORIZON_IDX = 13
Scaler = 0.25
MIN_SPEED_MPS = 0.1


class MIMICNavigator(BaseModel):
    """MIMIC point-goal navigator."""

    context_size = 16
    multimodal = False

    def __init__(
        self,
        onnx_path: str = ONNX_PATH,
        device: str = "cuda",
        controller: ControllerCfg | None = None,
        horizon_idx: int = HORIZON_IDX,
    ):
        """
        Args:
            onnx_path: Path to the ONNX model file.
            device: "cuda" or "cpu".
            controller: Controller selection and parameters (defaults to PD).
            horizon_idx: Number of output waypoints to keep (up to 4s).
        """
        super().__init__(device=device, controller=controller, info_path=INFO_PATH)
        self.horizon_idx = horizon_idx
        # t_mid tracks the (possibly truncated) trajectory length, not the full YAML horizon.
        self.t_mid = self._pd_time_divisor(self.horizon_idx)
        self._session = self._make_session(onnx_path)

    def inference_trajectory(self, obs):
        """Run model → (trajectory, scores).

        Args:
            obs: (B, 16, 3, 288, 512) float32 in [0,1].
                 Note: batch dim is fixed to 1 for this model.

        Returns:
            trajectory: np.ndarray (B, 1, K, 2) where K = horizon_idx
            scores:     np.ndarray (B, 1)
        """
        obs_np = self._obs_to_numpy(obs)
        B = obs_np.shape[0]

        if B == 1:
            out = self._session.run(None, {"input": obs_np})
            raw = self._select_moving(out)
        else:
            raw = np.concatenate(
                [self._select_moving(self._session.run(None, {"input": obs_np[i:i+1]})) for i in range(B)],
                axis=0,
            )

        # Take x, y columns up to the 4-second horizon
        traj_xy = raw[:, :self.horizon_idx, :2].astype(np.float32)
        # Reshape to (B, 1, K, 2) for uniform interface
        trajectory = traj_xy[:, np.newaxis] * Scaler
        scores = np.ones((B, 1), dtype=np.float32)
        return trajectory, scores

    def _select_moving(self, out):
        """The highest-scoring candidate trajectory (B, 15, 3) with speed >= MIN_SPEED_MPS (the model's best when none
        is). ``out`` = the session outputs: the best candidate, all candidates (B, 64, 15, 3) and their scores (B, 64)."""
        _, candidates, scores = out
        k = self._pd_speed_index(self.horizon_idx)
        speed = candidates[:, :, k, 0] * Scaler / self._pd_time_divisor(self.horizon_idx)  # (B, 64)
        moving = np.where(speed >= MIN_SPEED_MPS, scores, -np.inf)
        idx = np.where(np.isfinite(moving).any(axis=1), moving.argmax(axis=1), scores.argmax(axis=1))
        return candidates[np.arange(len(idx)), idx]
