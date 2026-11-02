#!/usr/bin/env python3
"""ViNT inference wrapper (NavDP image-goal ViNT in goal-free mode).

The ONNX has two inputs (obs_img, goal_img) and two outputs (dist, action).
Goal-free mode: pass random noise as goal_img — same trick NavDP uses in
its ``ViNTAgent.step_nogoal``.

    nav = VintGLOfficialNavigator(device="cuda")
    traj, scores = nav.inference_trajectory(obs)   # (B, 1, 5, 2) meters
    vw, best = nav.inference_vw(obs)

Action decode (NavDP convention):
  The model bakes ``torch.cumsum`` into its forward, so the ONNX output
  ``action_pred[:, :, :2]`` is *already* cumulative — we just scale by
  ``metric_waypoint_spacing`` to get meters in the robot frame (x=fwd, y=left).
  No unnormalize step (unlike NoMaD).
"""

import os

import numpy as np
import torch
from torchvision import transforms

from sidewalkbench.models.base_model import BaseModel, ControllerCfg

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(MODEL_DIR, "vint_imagegoal.onnx")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.yaml")

# NavDP training image size for ViNT (width, height).
IMG_W, IMG_H = 85, 64
NUM_WAYPOINTS = 5
CTX_FRAMES = 6  # context_size=5 past + 1 current

IMAGE_ASPECT_RATIO = 4 / 3

# Recon training camera (for the optional virtual rectification path).
TRAIN_CAM_FX = 272.547
TRAIN_CAM_FY = 266.358
TRAIN_CAM_CX = 320.0
TRAIN_CAM_CY = 220.0
TRAIN_CAM_W = 640
TRAIN_CAM_H = 440

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


class VintGLOfficialNavigator(BaseModel):
    """ViNT navigator (NavDP image-goal ViNT, goal-free mode)."""

    context_size = CTX_FRAMES   # 6 frames in obs tensor
    multimodal = False
    num_waypoints = NUM_WAYPOINTS
    goal_type = "image"

    def __init__(
        self,
        device: str = "cuda",
        controller: ControllerCfg | None = None,
        center_crop: bool = True,
        use_train_intrinsics: bool = True,
        src_intrinsics: dict | None = None,
    ):
        super().__init__(device=device, controller=controller, info_path=INFO_PATH)

        # NavDP default is center_crop=False (direct resize). Flag exposed for
        # A/B testing: True crops landscape input to 4:3 before resizing.
        self.center_crop = center_crop

        # Optional virtual rectification to the NoMaD/ViNT training camera
        # (recon dataset). When True, the source pinhole frame is warped
        # (affine, same math as a homography between two pinhole cameras).
        self.use_train_intrinsics = use_train_intrinsics
        self.src_intrinsics = dict(src_intrinsics or DEFAULT_SRC_INTRINSICS)
        self._train_affine = None

        self._sess = self._make_session(ONNX_PATH)

        # Action decoder params (NavDP convention — cumsum already baked
        # into the network, only the metric scale remains here).
        self._normalize_actions = bool(self._info.get("normalize", True))
        self._waypoint_spacing = float(
            self._info.get("metric_waypoint_spacing", 0.8 / 3.0))
        self.model_fps = float(self._info.get("model_fps", 4.0))

    # ------------------------------------------------------------------
    # Preprocessing — mirrors NoMaD, only the target size differs.
    # ------------------------------------------------------------------

    def _ensure_train_affine(self):
        if self._train_affine is not None:
            return
        # Recon intrinsics scaled to (IMG_W, IMG_H) = (85, 64).
        fx_t = TRAIN_CAM_FX * IMG_W / TRAIN_CAM_W
        fy_t = TRAIN_CAM_FY * IMG_H / TRAIN_CAM_H
        cx_t = TRAIN_CAM_CX * IMG_W / TRAIN_CAM_W
        cy_t = TRAIN_CAM_CY * IMG_H / TRAIN_CAM_H

        src = self.src_intrinsics
        a = src["fx"] / fx_t
        e = src["fy"] / fy_t
        c = src["cx"] - cx_t * a
        f = src["cy"] - cy_t * e
        self._train_affine = (a, 0.0, c, 0.0, e, f)

    def _to_model_input_uint8(self, frame_uint8):
        from PIL import Image as PILImage
        pil = PILImage.fromarray(frame_uint8)
        if self.use_train_intrinsics:
            self._ensure_train_affine()
            pil = pil.transform(
                (IMG_W, IMG_H), PILImage.AFFINE, self._train_affine,
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
        pil = pil.resize((IMG_W, IMG_H))
        return np.array(pil)

    def _preprocess_images(self, obs_rgb):
        """obs_rgb: (B, T=6, 3, H, W) float32 [0,1] → (B, 18, 64, 85) ImageNet-norm."""
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
        return np.stack(frames)  # (B, 3*T, IMG_H, IMG_W)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def inference_trajectory(self, obs, goal_image=None):
        """Run ViNT → metric waypoints + uniform scores.

        Args:
            obs: (B, 6, 3, H, W) float32 in [0,1]. 6 frames.
            goal_image: Optional (B, 3, H, W) float32 egocentric goal images in
                [0,1]. When None, runs goal-free with a random goal (NavDP trick).

        Returns:
            trajectory: np.ndarray (B, 1, 5, 2) — meters in robot frame
            scores:     np.ndarray (B, 1) — uniform (1.0)
        """
        obs_np = self._obs_to_numpy(obs)
        B = obs_np.shape[0]

        obs_in = self._preprocess_images(obs_np)  # (B, 18, 64, 85)
        if goal_image is not None:
            # Preprocess the goal frame exactly like an observation frame.
            goal_np = self._obs_to_numpy(goal_image)[:, None]  # (B, 1, 3, H, W)
            goal_in = self._preprocess_images(goal_np)  # (B, 3, IMG_H, IMG_W)
        else:
            # Goal-free: random fake goal (matches NavDP ViNTAgent.step_nogoal).
            goal_in = np.random.randn(B, 3, IMG_H, IMG_W).astype(np.float32)

        # Suppress ORT stderr noise during this call.
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        all_action_pred = []
        try:
            for i in range(B):
                dist_pred, action_pred = self._sess.run(
                    None, {"obs_img": obs_in[i:i+1], "goal_img": goal_in[i:i+1]})
                all_action_pred.append(action_pred)
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)
        action_pred = np.concatenate(all_action_pred, axis=0)  # (B, 5, 4)
        # action_pred shape (B, 5, 4): xy already cumsum'd, last two are sin/cos.
        xy = action_pred[:, :, :2]
        if self._normalize_actions:
            xy = xy * self._waypoint_spacing  # meters

        trajectory = xy[:, None, :, :]  # (B, 1, 5, 2)
        scores = np.ones((B, 1), dtype=np.float32)
        return trajectory.astype(np.float32), scores
