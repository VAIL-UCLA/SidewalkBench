#!/usr/bin/env python3
"""InternVLA-N1 DualVLN with the benchmark's historical always-S1 pipeline.

Every control step runs the current RGB frame through S2, generates the
trajectory latent from the complete S2 output, and runs S1 diffusion. This
also applies to STOP, arrow tokens, and other text. S1 receives two copies
of the current image and zero depth, matching the earlier wrapper.

The trajectory decoder returns the origin followed by predicted XY points.
BaseModel converts that path to commands using the shared PD controller.
The untimed spatial path uses an explicit 2-second benchmark horizon,
preserving the previous 1-second PD divisor.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from sidewalkbench.models.base_model import BaseModel, ControllerCfg

# ────────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
# Checkpoints live next to this file (see download_checkpoints.sh), not in the
# repository root. The absolute path also lets InternNav's CWD-relative
# ``MODEL_PATH_TO`` be redirected here at runtime.
CKPTS_DIR = os.path.join(MODEL_DIR, "ckpts")
DEFAULT_MODEL_PATH = os.path.join(CKPTS_DIR, "InternVLA-N1-DualVLN")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.yaml")


def _ensure_internnav_on_path(repo_root: Optional[str] = None) -> Path:
    """Locate the vendored InternNav checkout so its model classes are importable.

    Search order: explicit arg → ``$INTERNNAV_ROOT`` → ``<repo>/third_party/InternNav``.
    """
    candidates: list[Path] = []
    if repo_root:
        candidates.append(Path(repo_root))
    if os.environ.get("INTERNNAV_ROOT"):
        candidates.append(Path(os.environ["INTERNNAV_ROOT"]))
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "third_party" / "InternNav")
    for c in candidates:
        c = c.expanduser().resolve()
        if (c / "internnav").is_dir():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c
    raise RuntimeError(f"InternNav repo not found. Tried: {candidates}")


# ────────────────────────────────────────────────────────────────────────
# Official prompts / action vocabulary (internvla_n1_policy.py:init_prompts)
# ────────────────────────────────────────────────────────────────────────
_DEFAULT_IMAGE_TOKEN = "<image>"
_PROMPT = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully "
    "completed the task."
)
_CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]
# S1 (diffusion) input resolution.
_S1_RESIZE = 224


def _patch_internnav_checkpoint_dir() -> None:
    """Point InternNav's CWD-relative checkpoint dir at this model's ``ckpts``.

    ``internnav/.../internvla_n1_arch.py`` hardcodes ``MODEL_PATH_TO = "checkpoints"``
    and loads the DepthAnything backbone from it relative to the current working
    directory. Redirect it to the absolute ``<model_dir>/ckpts`` so checkpoints
    stay with the model and runs are CWD-independent.
    """
    try:
        from internnav.model.basemodel.internvla_n1 import internvla_n1_arch as _arch
    except Exception:  # noqa: BLE001 - best effort
        return
    _arch.MODEL_PATH_TO = CKPTS_DIR


def _patch_nextdit_gradient_checkpointing() -> None:
    """Make InternNav's NextDiT compatible with the pinned diffusers ModelMixin.

    InternNav's ``LuminaNextDiT2DModel._set_gradient_checkpointing(self, module, value=False)``
    predates diffusers' ``enable_gradient_checkpointing()`` calling
    ``_set_gradient_checkpointing(enable=..., gradient_checkpointing_func=...)``. Inference
    does not need gradient checkpointing, so accept and record the flag.
    """
    try:
        from internnav.model.basemodel.internvla_n1.nextdit_traj import LuminaNextDiT2DModel
    except Exception:  # noqa: BLE001 - best effort; only needed for this model
        return

    def _set_gradient_checkpointing(self, enable: bool = False, gradient_checkpointing_func=None, value=None):
        self.gradient_checkpointing = enable if value is None else value

    LuminaNextDiT2DModel._set_gradient_checkpointing = _set_gradient_checkpointing


def _patch_lumina_feed_forward() -> None:
    """Restore the SwiGLU FFN scaling used to train InternVLA-N1's NextDiT.

    ``internnav/.../nextdit_traj.py`` passes ``inner_dim=4*dim`` to
    ``diffusers.models.attention.LuminaFeedForward``. The diffusers version used
    to author the checkpoint applied an additional ``2/3`` SwiGLU factor
    (``int(2 * 4 * dim / 3)``), which this pinned diffusers no longer does. Without
    the factor the FFN is 1536 wide instead of the checkpoint's 1024.
    """
    try:
        import diffusers.models.attention as _attention
    except Exception:  # noqa: BLE001
        return
    original = _attention.LuminaFeedForward.__init__
    if getattr(original, "_internvla_patched", False):
        return

    def _init(self, dim, inner_dim, multiple_of=256, ffn_dim_multiplier=None):
        inner_dim = int(2 * inner_dim / 3)
        original(self, dim, inner_dim, multiple_of, ffn_dim_multiplier)

    _init._internvla_patched = True
    _attention.LuminaFeedForward.__init__ = _init


class DualVLNNavigator(BaseModel):
    """InternVLA-N1 (DualVLN) navigator, goal-conditioned by a language instruction."""

    context_size = 8
    multimodal = False
    goal_type = None

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        repo_root: Optional[str] = None,
        device: str = "cuda:0",
        attn_impl: str = "sdpa",
        max_new_tokens: int = 128,
        resize: int = 384,
        controller: ControllerCfg | None = None,
    ):
        super().__init__(device=device, controller=controller, info_path=INFO_PATH)

        _ensure_internnav_on_path(repo_root)
        _patch_internnav_checkpoint_dir()
        _patch_nextdit_gradient_checkpointing()
        _patch_lumina_feed_forward()

        from transformers import AutoProcessor, AutoTokenizer
        from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
        from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions

        self._split_and_clean = split_and_clean
        self._traj_to_actions = traj_to_actions

        self.device = torch.device(device)
        self.max_new_tokens = int(max_new_tokens)
        self.resize_w = int(resize)
        self.resize_h = int(resize)

        self._model = InternVLAN1ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map={"": self.device},
        ).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._processor.tokenizer = self._tokenizer
        self._processor.tokenizer.padding_side = "left"

        self._instruction: Optional[str] = None
        self._env = None

    # ── bindings / lifecycle ────────────────────────────────────────
    def bind_env(self, env) -> None:
        """Keep the benchmark binding API; the restored S1 path uses zero depth."""
        self._env = env

    def set_instruction(self, instruction: str) -> None:
        self._instruction = instruction

    def _obs_last_frame(self, obs) -> torch.Tensor:
        """Reduce zoo obs to ``(B, 3, H, W)`` float in ``[0, 1]`` (last frame)."""
        if isinstance(obs, np.ndarray):
            x = torch.from_numpy(obs)
        elif isinstance(obs, torch.Tensor):
            x = obs
        elif isinstance(obs, Image.Image):
            x = torch.from_numpy(np.asarray(obs.convert("RGB"), dtype=np.float32) / 255.0)
            x = x.permute(2, 0, 1)[None]
        else:
            raise TypeError(f"Unsupported obs type {type(obs).__name__}")
        if x.ndim == 5:
            x = x[:, -1]
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"Cannot reduce obs of shape {tuple(x.shape)} to (B, 3, H, W)")
        x = x.float()
        if x.max() > 1.0 + 0.001:
            x = x / 255.0
        return x.clamp(0.0, 1.0)

    def _to_pil(self, frame_chw: torch.Tensor) -> Image.Image:
        arr = frame_chw.detach().cpu().permute(1, 2, 0).numpy()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
        if img.size != (self.resize_w, self.resize_h):
            img = img.resize((self.resize_w, self.resize_h))
        return img

    def _build_inputs(self, pil: Image.Image, instruction: str):
        """Replicates the first-frame path of InternVLAN1AsyncAgent.step_s2."""
        text = _PROMPT.replace("<instruction>.", instruction)
        text += f" {_CONJUNCTIONS[0]}{_DEFAULT_IMAGE_TOKEN}."
        parts = self._split_and_clean(text)
        content = []
        for part in parts:
            if part == _DEFAULT_IMAGE_TOKEN:
                content.append({"type": "image", "image": pil})
            else:
                content.append({"type": "text", "text": part})
        conversation = [{"role": "user", "content": content}]
        chat = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        return self._processor(text=[chat], images=[pil], return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def _forward_single(self, pil: Image.Image, instruction: str) -> np.ndarray:
        """One DualVLN step on a single PIL frame → (N, 2) trajectory."""
        pil = pil.resize((self.resize_w, self.resize_h))
        inputs = self._build_inputs(pil, instruction)
        out = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )
        image_grid_thw = inputs.image_grid_thw
        if image_grid_thw.dim() > 2:
            image_grid_thw = torch.cat([t.unsqueeze(0) for t in image_grid_thw], dim=0)
        latent = self._model.generate_latents(out.sequences, inputs.pixel_values, image_grid_thw)
        img_224 = np.asarray(pil.resize((_S1_RESIZE, _S1_RESIZE)), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_224)
        rgbs = torch.stack([img_t, img_t]).unsqueeze(0).to(self.device)
        depths = torch.zeros((1, 2, _S1_RESIZE, _S1_RESIZE, 1), device=self.device)
        trajectories = self._model.generate_traj(latent, rgbs, depths)
        return self._traj_to_actions(trajectories, use_discrate_action=False)

    # ── zoo API ─────────────────────────────────────────────────────
    def inference_trajectory(self, obs, instruction: Optional[str] = None):
        """Run the DualVLN pipeline → ``((B, 1, N, 2), (B, 1))``.

        Args:
            obs: ``(B, context, 3, H, W)`` float in ``[0, 1]`` (last frame used).
            instruction: optional per-call override of :meth:`set_instruction`.
        """
        inst = instruction if instruction is not None else self._instruction
        if inst is None:
            raise RuntimeError("call set_instruction(...) before inference")
        self._instruction = inst

        frames = self._obs_last_frame(obs)  # (B, 3, H, W)
        per_batch: list[np.ndarray] = []
        for b in range(frames.shape[0]):
            rgb_pil = self._to_pil(frames[b])
            per_batch.append(np.asarray(self._forward_single(rgb_pil, inst), dtype=np.float32))

        n = max(t.shape[0] for t in per_batch)
        out = np.zeros((len(per_batch), 1, n, 2), dtype=np.float32)
        for b, t in enumerate(per_batch):
            out[b, 0, : t.shape[0]] = t
            if t.shape[0] < n:
                out[b, 0, t.shape[0]:] = t[-1]
        scores = np.ones((len(per_batch), 1), dtype=np.float32)
        return out, scores
