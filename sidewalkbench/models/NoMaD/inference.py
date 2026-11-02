#!/usr/bin/env python3
"""NoMaD inference utilities (3-component diffusion model).

NoMaD uses iterative DDPM diffusion (10 steps) with 3 ONNX components:
  1. vision_encoder: obs + goal_img + goal_mask → conditioning vector
  2. noise_pred: single denoising step
  3. dist_pred: distance estimation

Goal-free mode: goal_mask=1, goal_img=random noise.

Action decode (NavDP / official NoMaD convention):
  Diffusion output is normalized delta-actions clipped to [-1, 1]. To recover
  metric waypoints in the robot frame:
    1. unnormalize:  d = (norm + 1)/2 * (max - min) + min     (dataset units)
    2. cumsum over time → cumulative positions                (dataset units)
    3. multiply by `metric_waypoint_spacing` (m) → meters     (robot frame)

    nav = NoMaDGLNavigator(device="cuda")
    traj, scores = nav.inference_trajectory(obs)     # (B, S, 8, 2) meters
    vw, best = nav.inference_vw(obs)
"""

import math
import os

import numpy as np
import torch
from torchvision import transforms

from sidewalkbench.models.base_model import BaseModel, ControllerCfg

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
VE_PATH = os.path.join(MODEL_DIR, "nomad_vision_encoder.onnx")
NP_PATH = os.path.join(MODEL_DIR, "nomad_noise_pred.onnx")
DP_PATH = os.path.join(MODEL_DIR, "nomad_dist_pred.onnx")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.yaml")
IMG_SIZE = 96
NUM_DIFFUSION_STEPS = 10
NUM_SAMPLES = 8
IMAGE_ASPECT_RATIO = 4 / 3  # NavDP optional center-crop target

# Training camera intrinsics (NoMaD recon dataset, from NavDP data_config.yaml).
TRAIN_CAM_FX = 272.547
TRAIN_CAM_FY = 266.358
TRAIN_CAM_CX = 320.0
TRAIN_CAM_CY = 220.0
TRAIN_CAM_W = 640   # = 2 * cx
TRAIN_CAM_H = 440   # = 2 * cy

DEFAULT_SRC_INTRINSICS = {
    "fx": 210.667, "fy": 210.667,
    "cx": 256.0, "cy": 144.0,
    "w": 512, "h": 288,
}

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class NoMaDGLNavigator(BaseModel):
    """NoMaD goal-free diffusion navigator.

    Runs DDPM diffusion loop (10 steps) to generate trajectory samples.
    All coordinates: x=forward, y=left, meters.
    """

    context_size = 4  # 4 obs frames → 12 channels
    multimodal = True
    goal_type = "image"

    def __init__(
        self,
        device: str = "cuda",
        controller: ControllerCfg | None = None,
        num_samples: int = NUM_SAMPLES,
        num_diffusion_steps: int = NUM_DIFFUSION_STEPS,
        center_crop: bool = True,
        use_train_intrinsics: bool = True,
        src_intrinsics: dict | None = None,
    ):
        super().__init__(device=device, controller=controller, info_path=INFO_PATH)
        self.num_samples = num_samples
        self.num_steps = num_diffusion_steps
        # NavDP default is center_crop=False (direct resize). Flag exposed for
        # A/B testing: True crops landscape input to 4:3 before resizing.
        self.center_crop = center_crop

        # Optional virtual rectification to the NoMaD training camera (recon).
        self.use_train_intrinsics = use_train_intrinsics
        self.src_intrinsics = dict(src_intrinsics or DEFAULT_SRC_INTRINSICS)
        self._train_affine = None  # built lazily in _ensure_train_affine

        self._ve = self._make_session(VE_PATH)
        self._np = self._make_session(NP_PATH)
        self._dp = self._make_session(DP_PATH)

        # Action decoder params (NavDP convention)
        action_stats = self._info.get("action_stats", {})
        self._act_min = np.asarray(
            action_stats.get("min", [0.0, -4.0]), dtype=np.float32)
        self._act_max = np.asarray(
            action_stats.get("max", [5.0, 4.0]), dtype=np.float32)
        self._normalize_actions = bool(self._info.get("normalize", True))
        self._waypoint_spacing = float(
            self._info.get("metric_waypoint_spacing", 0.25))
        # Model's expected observation frame rate (Hz).
        self.model_fps = float(self._info.get("model_fps", 4.0))

        # Build DDPM schedule (squaredcos_cap_v2)
        self._build_schedule()

    def _build_schedule(self):
        """Pre-compute DDPM alpha/beta schedule (matches diffusers squaredcos_cap_v2)."""
        T = self.num_steps

        def alpha_bar(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(T):
            t1 = i / T
            t2 = (i + 1) / T
            b = min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999)
            betas.append(b)
        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alpha_cumprod = np.cumprod(alphas)

        self._betas = betas
        self._alpha_cumprod = alpha_cumprod
        self._sqrt_alpha = np.sqrt(alphas)
        self._sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - alpha_cumprod)
        # For x0 prediction clipping
        self._sqrt_recip_alpha_cumprod = np.sqrt(1.0 / alpha_cumprod)
        self._sqrt_recip_alpha_cumprod_m1 = np.sqrt(1.0 / alpha_cumprod - 1.0)

    def _ddpm_step(self, sample, noise_pred, t):
        """Single DDPM reverse step: x_{t-1} from x_t and predicted noise."""
        alpha_prod_t = self._alpha_cumprod[t]
        beta_t = self._betas[t]

        # Predict x0
        x0_pred = (sample - math.sqrt(1.0 - alpha_prod_t) * noise_pred) / math.sqrt(alpha_prod_t)
        x0_pred = np.clip(x0_pred, -1.0, 1.0)

        # Compute mean of q(x_{t-1} | x_t, x_0)
        if t > 0:
            alpha_prod_prev = self._alpha_cumprod[t - 1]
            beta_prod_t = 1.0 - alpha_prod_t
            beta_prod_prev = 1.0 - alpha_prod_prev
            coeff_x0 = math.sqrt(alpha_prod_prev) * beta_t / beta_prod_t
            coeff_xt = math.sqrt(1.0 - beta_t) * beta_prod_prev / beta_prod_t
            mean = coeff_x0 * x0_pred + coeff_xt * sample
            # Variance
            variance = beta_t * beta_prod_prev / beta_prod_t
            noise = np.random.randn(*sample.shape).astype(np.float32)
            return (mean + math.sqrt(variance) * noise).astype(np.float32)
        else:
            return x0_pred.astype(np.float32)

    def _ensure_train_affine(self):
        """Build PIL affine coeffs mapping output (96x96) pixels → source pixels."""
        if self._train_affine is not None:
            return
        fx_t = TRAIN_CAM_FX * IMG_SIZE / TRAIN_CAM_W
        fy_t = TRAIN_CAM_FY * IMG_SIZE / TRAIN_CAM_H
        cx_t = TRAIN_CAM_CX * IMG_SIZE / TRAIN_CAM_W
        cy_t = TRAIN_CAM_CY * IMG_SIZE / TRAIN_CAM_H

        src = self.src_intrinsics
        a = src["fx"] / fx_t
        e = src["fy"] / fy_t
        c = src["cx"] - cx_t * a
        f = src["cy"] - cy_t * e
        self._train_affine = (a, 0.0, c, 0.0, e, f)

    def _to_model_input_uint8(self, frame_uint8):
        """Apply preprocessing to one (H, W, 3) uint8 RGB frame → (96, 96, 3) uint8."""
        from PIL import Image as PILImage
        pil = PILImage.fromarray(frame_uint8)
        if self.use_train_intrinsics:
            self._ensure_train_affine()
            pil = pil.transform(
                (IMG_SIZE, IMG_SIZE), PILImage.AFFINE, self._train_affine,
                resample=PILImage.BILINEAR, fillcolor=0)
            return np.array(pil)
        if self.center_crop:
            w, h = pil.size
            if w > h:
                crop_w = int(h * IMAGE_ASPECT_RATIO)
                x0 = max((w - crop_w) // 2, 0)
                pil = pil.crop((x0, 0, x0 + crop_w, h))
            else:
                crop_h = int(w / IMAGE_ASPECT_RATIO)
                y0 = max((h - crop_h) // 2, 0)
                pil = pil.crop((0, y0, w, y0 + crop_h))
        pil = pil.resize((IMG_SIZE, IMG_SIZE))
        return np.array(pil)

    def _preprocess_images(self, obs_rgb):
        """obs_rgb: (B, T=4, 3, H, W) float32 [0,1] → (B, 12, 96, 96) ImageNet-norm."""
        from PIL import Image as PILImage
        B, T = obs_rgb.shape[:2]
        frames = []
        for b in range(B):
            ch_list = []
            for t in range(T):
                frame = (obs_rgb[b, t].transpose(1, 2, 0) * 255).astype(np.uint8)
                model_in = self._to_model_input_uint8(frame)
                pil = PILImage.fromarray(model_in)
                ch_list.append(IMAGENET_TRANSFORM(pil).numpy())
            frames.append(np.concatenate(ch_list, axis=0))
        return np.stack(frames)  # (B, 12, 96, 96)

    def _decode_actions(self, sample):
        """Decode normalized deltas → metric waypoints (NavDP recipe)."""
        # 1. Unnormalize from [-1, 1] → [min, max] (dataset units).
        deltas = (sample + 1.0) / 2.0
        deltas = deltas * (self._act_max - self._act_min) + self._act_min
        # 2. Cumulative sum over time.
        positions = np.cumsum(deltas, axis=1)
        # 3. Scale by metric_waypoint_spacing (only when training-time normalize=True).
        if self._normalize_actions:
            positions = positions * self._waypoint_spacing
        return positions.astype(np.float32)

    def inference_trajectory(self, obs, goal_image=None):
        """Run diffusion model → (trajectories, scores).

        Args:
            obs: (B, 4, 3, H, W) float32 in [0,1]. 4 frames.
            goal_image: Optional (B, 3, H, W) float32 egocentric goal images in
                [0,1]. When None, runs goal-free with a random goal (mask=1).

        Returns:
            trajectory: np.ndarray (B, num_samples, 8, 2) — METERS, robot frame
            scores:     np.ndarray (B, num_samples) — uniform
        """
        obs_np = self._obs_to_numpy(obs)
        B = obs_np.shape[0]

        obs_norm = self._preprocess_images(obs_np)  # (B, 12, 96, 96)

        if goal_image is not None:
            # Goal provided: preprocess the frame like an observation and unmask.
            goal_np = self._obs_to_numpy(goal_image)[:, None]  # (B, 1, 3, H, W)
            goal_norm = self._preprocess_images(goal_np)  # (B, 3, 96, 96)
            mask = np.zeros(B, dtype=np.int64)
        else:
            # Goal-free: random goal image + mask=1
            goal_norm = np.random.randn(B, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
            mask = np.ones(B, dtype=np.int64)

        # 1. Vision encoder
        cond = self._ve.run(None, {
            "obs_img": obs_norm,
            "goal_img": goal_norm,
            "goal_mask": mask,
        })[0]  # (B, 256)

        # Replicate for num_samples
        cond_rep = np.repeat(cond, self.num_samples, axis=0)  # (B*S, 256)

        # 2. Diffusion denoising loop (output: normalized deltas in [-1,1]).
        BS = B * self.num_samples
        sample = np.random.randn(BS, 8, 2).astype(np.float32)

        for t in reversed(range(self.num_steps)):
            t_arr = np.array(t, dtype=np.int64)
            # Suppress stderr from ORT during inference
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stderr = os.dup(2)
            os.dup2(devnull, 2)
            try:
                noise_pred = self._np.run(None, {
                    "sample": sample,
                    "timestep": t_arr,
                    "global_cond": cond_rep,
                })[0]
            finally:
                os.dup2(old_stderr, 2)
                os.close(devnull)
                os.close(old_stderr)

            sample = self._ddpm_step(sample, noise_pred, t)

        # 3. Decode normalized deltas → metric waypoints.
        waypoints = self._decode_actions(sample)  # (BS, 8, 2) meters

        trajectory = waypoints.reshape(B, self.num_samples, 8, 2)
        scores = np.ones((B, self.num_samples), dtype=np.float32)
        return trajectory, scores
