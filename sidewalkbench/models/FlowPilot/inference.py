#!/usr/bin/env python3
"""FlowPilot inference utilities.

obs: torch.Tensor or np.ndarray, shape (B, 21, 3, 288, 512), float32 in [0,1].
     21 RGB frames [t-20, ..., t] @ 5 Hz.

    nav = FlowPilotNavigator(device="cuda")
    traj, scores = nav.inference_trajectory(obs)   # (B, 64, 20, 2), (B, 64)
    vw, best = nav.inference_vw(obs)               # (B, 2), (B, 20, 2)
    nav.reset()
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from sidewalkbench.models.base_model import BaseModel, ControllerCfg

# ── Defaults from model_info.yaml ─────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(MODEL_DIR, "flowpilot_base_goal_less.onnx")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.yaml")
DEFAULT_METRIC_SPACING = np.array([[0.0, 0.51, -0.16, 0.16]], dtype=np.float32)


def select_best_mode(trajectory, scores):
    """Pick the highest-scoring forward-moving mode per batch element.

    Returns:
        best_traj: (B, 20, 2)
        best_idx:  (B,)
    """
    mode_order = np.argsort(-scores, axis=-1)
    x_disp = trajectory[..., -1, 0] - trajectory[..., 0, 0]
    valid_move = x_disp >= 0.0
    valid_sorted = np.take_along_axis(valid_move, mode_order, axis=1)
    has_valid = valid_sorted.any(axis=1)
    first_valid_pos = valid_sorted.argmax(axis=1)
    first_valid_idx = np.take_along_axis(
        mode_order, first_valid_pos[:, None], axis=1).squeeze(1)
    best_idx = np.where(has_valid, first_valid_idx, mode_order[:, 0])
    best_traj = trajectory[np.arange(trajectory.shape[0]), best_idx]
    return best_traj, best_idx


class FlowPilotNavigator(BaseModel):
    """FlowPilot goal-less navigation model for continuous inference."""

    context_size = 21
    multimodal = True

    def __init__(
        self,
        onnx_path: str = ONNX_PATH,
        device: str = "cuda",
        metric_spacing: list | None = None,
        embodiment_id: int = 0,
        controller: ControllerCfg | None = None,
    ):
        """
        Args:
            onnx_path: Path to the ONNX model file.
            device: "cuda" or "cpu" for velocity output tensors.
            metric_spacing: [x_min, x_max, y_min, y_max] normalization bounds.
                Default: [0.0, 0.51, -0.16, 0.16].
            embodiment_id: Robot embodiment identifier (default 0).
            controller: Controller selection and parameters (defaults to PD).
        """
        super().__init__(device=device, controller=controller, info_path=INFO_PATH)
        self.embodiment_id = embodiment_id
        if metric_spacing is None:
            self.metric_spacing = DEFAULT_METRIC_SPACING.copy()
        else:
            self.metric_spacing = np.array([metric_spacing], dtype=np.float32)
        self._session = self._load_session(onnx_path)

    def _load_session(self, onnx_path):
        if torch.device(self.device).type == 'cpu':
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            return ort.InferenceSession(onnx_path, sess_options=options, providers=['CPUExecutionProvider'])
        return self._make_session(onnx_path)

    def _select_best(self, trajectory, scores=None):
        best_traj, _ = select_best_mode(trajectory, scores)
        return best_traj

    def inference_trajectory(self, obs, metric_spacing=None, embodiment_id=None):
        """Run model → (trajectory, scores) with register token removed.

        Args:
            obs: (B, 21, 3, 288, 512) float32 in [0,1].
                 Accepts np.ndarray or torch.Tensor.
            metric_spacing: (B, 4) override, or None for default.
            embodiment_id: int override, or None for default.

        Returns:
            trajectory: np.ndarray (B, 64, 20, 2)
            scores:     np.ndarray (B, 64)
        """
        obs_np = self._obs_to_numpy(obs)
        if obs_np.ndim != 5 or obs_np.shape[1:3] != (self.context_size, 3):
            raise ValueError('FlowPilot expects (B, 21, 3, H, W) RGB observations.')
        B, T = obs_np.shape[:2]
        if obs_np.shape[-2:] != (288, 512):
            frames = F.interpolate(torch.from_numpy(obs_np).flatten(0, 1),
                                   size=(288, 512), mode='bilinear', align_corners=False)
            obs_np = frames.unflatten(0, (B, T)).numpy()

        if metric_spacing is None:
            ms = np.tile(self.metric_spacing, (B, 1))
        elif isinstance(metric_spacing, torch.Tensor):
            ms = metric_spacing.cpu().numpy()
        else:
            ms = np.asarray(metric_spacing, dtype=np.float32)

        eid_val = embodiment_id if embodiment_id is not None else self.embodiment_id
        eid = np.full((B, 1), eid_val, dtype=np.int64)

        raw_traj, raw_scores = self._session.run(None, {
            "observation": obs_np,
            "metric_spacing": ms,
            "embodiment_id": eid,
        })

        trajectory = raw_traj[:, 1:]   # (B, 64, 20, 2)
        scores = raw_scores[:, 1:]     # (B, 64)
        if not np.isfinite(trajectory).all() or not np.isfinite(scores).all():
            raise RuntimeError('FlowPilot returned non-finite predictions.')
        return trajectory, scores
