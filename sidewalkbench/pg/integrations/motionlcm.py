import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# MotionLCM is vendored as a git submodule at ``third_party/MotionLCM``.
# Resolve it relative to this file so the loader is independent of the CWD.
MOTIONLCM_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "MotionLCM"
sys.path.append(str(MOTIONLCM_ROOT))

from mld.config import get_module_config  # noqa: E402

from sidewalkbench.pg.integrations.motionlcm_mld import MLD  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_motion_lcm(device="cuda"):
    cfg_path = str(MOTIONLCM_ROOT / "configs" / "motionlcm_t2m.yaml")
    cfg = OmegaConf.load(cfg_path)
    cfg_root = str(MOTIONLCM_ROOT / "configs")
    cfg_model = get_module_config(cfg.model, cfg.model.target, cfg_root)
    cfg = OmegaConf.merge(cfg, cfg_model)
    cfg.DATASET.HUMANML3D.ROOT = str(MOTIONLCM_ROOT / "datasets" / "humanml3d")
    cfg.DATASET.NFEATS = 263
    cfg.DATASET.NJOINTS = 22
    cfg.model.t5_path = str(MOTIONLCM_ROOT / "deps" / "sentence-t5-large")
    state_dict = torch.load(str(MOTIONLCM_ROOT / cfg.TEST.CHECKPOINTS), map_location="cpu")["state_dict"]

    # Check if the checkpoint is LCM-based.
    is_lcm = False
    lcm_key = "denoiser.time_embedding.cond_proj.weight"  # unique key for CFG
    if lcm_key in state_dict:
        is_lcm = True
        time_cond_proj_dim = state_dict[lcm_key].shape[1]
        cfg.model.denoiser.params.time_cond_proj_dim = time_cond_proj_dim

    # Step 4: Check if the checkpoint is Controlnet-based.
    cn_key = "controlnet.controlnet_cond_embedding.0.weight"
    is_controlnet = True if cn_key in state_dict else False
    cfg.model.is_controlnet = is_controlnet
    cfg.output_dir = "temp"

    model = MLD(cfg)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    new_state_dict = {}
    for key in state_dict.keys():
        if "t2m_moveencoder" in key or "t2m_textencoder" in key or "t2m_motionencoder" in key:
            continue
        else:
            new_state_dict[key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    return model
