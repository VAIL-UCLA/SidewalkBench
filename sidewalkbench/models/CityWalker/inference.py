"""CityWalker-PG-Official inference utilities.

CityWalker uses a NON-STANDARD coordinate system internally:
    model-internal:  y = forward, x = right
    our standard:    x = forward, y = left

This navigator accepts standard-frame inputs (goal_xy in x=forward, y=left)
and returns standard-frame outputs. The coordinate swap is handled internally.

    nav = CityWalkerNavigator(device="cuda")
    traj, scores = nav.inference_trajectory(obs, goal_xy=np.array([5.0, 0.2]))
    vw = nav.inference_vw(obs, goal_xy=np.array([5.0, 0.2]))
"""
from sidewalkbench.models.base_model import BaseModel, ControllerCfg
from .motion_history import MotionHistory
from .stationary_graph import safe_stationary_graph
import os
import numpy as np
import torch
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(MODEL_DIR, 'citywalker.onnx')
INFO_PATH = os.path.join(MODEL_DIR, 'model_info.yaml')
IMG_H, IMG_W = (350, 630)

def _std_to_cw(xy):
    """Convert [forward, left] metres to CityWalker's [right, forward]."""
    out = np.empty_like(xy)
    out[..., 0] = -xy[..., 1]
    out[..., 1] = xy[..., 0]
    return out

def _cw_to_std(xy):
    """Convert CityWalker's [right, forward] to [forward, left] metres."""
    out = np.empty_like(xy)
    out[..., 0] = xy[..., 1]
    out[..., 1] = -xy[..., 0]
    return out

class CityWalkerNavigator(BaseModel):
    """CityWalker point-goal navigator.

    All user-facing coordinates are standard frame: x=forward, y=left, meters.
    """
    context_size = 5
    control_dt = 1.0  # Policy, RGB context, and pose history use one-second steps.
    multimodal = False

    def _preprocess_images(self, obs_rgb):
        """Resize RGB and normalize once for the current public export."""
        from PIL import Image as PILImage
        B, C = obs_rgb.shape[:2]
        out = np.empty((B, C, 3, IMG_H, IMG_W), dtype=np.float32)
        for b in range(B):
            for c in range(C):
                rgb = np.clip(obs_rgb[b, c].transpose(1, 2, 0), 0.0, 1.0)
                frame = np.rint(rgb * 255).astype(np.uint8)
                pil = PILImage.fromarray(frame)
                pil = pil.resize((IMG_W, IMG_H))
                out[b, c] = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if self._info.get('obs_normalize', False):
            mean = np.array([.485, .456, .406], np.float32)[None, None, :, None, None]
            std = np.array([.229, .224, .225], np.float32)[None, None, :, None, None]
            out = (out - mean) / std
        return out

    def inference_trajectory(self, obs, goal_xy=None, past_traj_std=None):
        """Run model → (trajectory, scores).

        Args:
            obs: (B, context_size, 3, H, W) float32 in [0,1]. 5 frames.
            goal_xy: (2,) or (B, 2) goal in standard frame [x_fwd, y_left] meters.
                     If None, samples a dummy goal with x in [2,10] and y in [-2,2].
            past_traj_std: (B, 6, 2) history in the current standard robot frame,
                           oldest first. The first five rows are measured poses;
                           the last row is replaced by the goal. If None, use
                           poses from bind_env, or zeros for a standalone call.

        Returns:
            trajectory: np.ndarray (B, 1, 5, 2) in standard frame
            scores:     np.ndarray (B, 1)
        """
        if isinstance(obs, torch.Tensor):
            obs_np = obs.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs, dtype=np.float32)
        B = obs_np.shape[0]
        obs_rgb = self._preprocess_images(obs_np)
        if past_traj_std is None:
            past_traj_std = self._measured_history()
        if past_traj_std is None:
            past_std = np.zeros((B, 6, 2), dtype=np.float32)
        else:
            if isinstance(past_traj_std, torch.Tensor):
                past_traj_std = past_traj_std.detach().cpu().numpy()
            past_std = np.asarray(past_traj_std, dtype=np.float32).copy()
            if past_std.ndim == 2:
                past_std = past_std[np.newaxis]
        past_std = np.broadcast_to(past_std, (B, 6, 2)).copy()
        if goal_xy is None:
            past_std[:, -1] = np.random.uniform(
                low=[2.0, -2.0], high=[10.0, 2.0], size=(B, 2)
            ).astype(np.float32)
        else:
            if isinstance(goal_xy, torch.Tensor):
                goal_xy = goal_xy.detach().cpu().numpy()
            past_std[:, -1] = np.broadcast_to(np.asarray(goal_xy, dtype=np.float32).reshape(-1, 2), (B, 2))
        traj_cw = _std_to_cw(past_std)
        wp_cw = np.concatenate([self._session.run(None, {'obs_images': obs_rgb[i:i+1], 'trajectory': traj_cw[i:i+1]})[0] for i in range(B)], axis=0)
        if not np.isfinite(wp_cw).all():
            raise RuntimeError('CityWalker returned non-finite waypoints.')
        wp_std = _cw_to_std(wp_cw)
        trajectory = wp_std[:, np.newaxis]
        scores = np.ones((B, 1), dtype=np.float32)
        return (trajectory, scores)
    goal_type = 'point'

    def __init__(self, onnx_path=ONNX_PATH, device='cuda', controller=None,
                 reset_history_speed_mps=None):
        super().__init__(device=device, controller=controller, info_path=INFO_PATH)
        self._env = None
        if reset_history_speed_mps is None:
            reset_history_speed_mps = self._info.get("reset_history_speed_mps", 1.0)
        self.configure_reset_history(reset_history_speed_mps)
        self._session = self._load_session(onnx_path)

    def configure_reset_history(self, reset_history_speed_mps=1.0):
        """Set a forward-motion prior for missing reset poses; clear old history.

        Defaults to a 1 m/s forward-motion prior; zero keeps stationary padding.
        Measured poses replace the prior over the next context_size - 1
        policy updates after either a full or per-environment reset.
        The configured speed changes model input history, never robot commands.
        """
        sample_dt_s = (float(self._env.step_dt) if self._env is not None
                       else 1.0 / float(self._info.get('input_fps', 1.0)))
        self._motion_history = MotionHistory(
            self.context_size, reset_speed_mps=reset_history_speed_mps,
            sample_dt_s=sample_dt_s)

    def _load_session(self, onnx_path):
        graph = safe_stationary_graph(onnx_path)
        if torch.device(self.device).type == 'cpu':
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            return ort.InferenceSession(graph, sess_options=options, providers=['CPUExecutionProvider'])
        return self._make_session(graph)

    def bind_env(self, env):
        """Attach the evaluator's measured-pose source and clear old history."""
        self._env = env.unwrapped
        # Match the actual evaluator policy period, including changed decimation.
        self.configure_reset_history(self._motion_history.reset_speed_mps)
        self.reset()

    def _measured_history(self, robot=None):
        if robot is None and self._env is not None:
            robot = self._env.scene['robot']
        if robot is None:
            return None
        tick = getattr(self._env, 'common_step_counter', None)
        poses = self._motion_history.sample(robot, tick=None if tick is None else int(tick))
        # Keep the public six-row input contract: five poses plus one goal row.
        return np.concatenate([poses, np.zeros((len(poses), 1, 2), np.float32)], axis=1)

    def _reset_state(self, env_idx=None):
        self._motion_history.reset(env_idx)

    def inference_vw(self, obs, robot=None, **kwargs):
        """Use measured robot history before the shared waypoint controller."""
        if kwargs.get('past_traj_std') is None:
            kwargs['past_traj_std'] = self._measured_history(robot)
        return super().inference_vw(obs, robot=robot, **kwargs)
