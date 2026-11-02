"""A pretrained HUGS human avatar, posed with SMPL parameters in a z-up world, and rendered.

Run in the ``hugs`` conda env; HUGS is the ml-hugs checkout next to SidewalkBench.

Coordinate conventions
  * SMPL / HUGS canonical space and HumanML3D (MotionLCM) motion are y-up, metres.
  * Our world (sidewalkbench scenes) is z-up, metres. ``Y_UP_TO_Z_UP`` = Rx(+90 deg):
    (x, y, z) -> (x, -z, y), so +y (up) becomes +z and +z (SMPL facing) becomes -y.
  * SMPL applies ``global_orient`` about the rest pelvis joint J0(betas) and adds ``transl``
    after, so the posed pelvis is J0 + transl. A y-up motion with pelvis path p(t) and root
    rotation R(t) is expressed in the z-up world, for THIS avatar's body shape, as
        global_orient' = Y_UP_TO_Z_UP @ R(t),   transl' = Y_UP_TO_Z_UP @ p(t) - J0(betas)
    which is exactly HUGS's own posing path (no post-hoc transform of the Gaussians).
  * Scale: ``smpl_scale`` = 1 (metric SMPL), as HUGS's own animation path uses.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

HUGS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "ml-hugs"))
sys.path.insert(0, HUGS_ROOT)

from omegaconf import OmegaConf  # noqa: E402
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle  # noqa: E402

from hugs.cfg.config import cfg as default_cfg  # noqa: E402
from hugs.models.hugs_trimlp import HUGS_TRIMLP  # noqa: E402
from hugs.renderer.gs_renderer import render_human_scene  # noqa: E402
from hugs.utils.graphics import get_projection_matrix  # noqa: E402

Y_UP_TO_Z_UP = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


class Avatar:
    def __init__(self, seq: str = "lab", device: str = "cuda"):
        cwd = os.getcwd()
        os.chdir(HUGS_ROOT)  # HUGS resolves data/smpl relative to its repo root
        try:
            out = f"output/pretrained_models/{seq}"
            cfg = OmegaConf.merge(default_cfg, OmegaConf.load(f"{out}/config_train.yaml"))
            params = np.load(f"data/neuman/dataset/{seq}/4d_humans/smpl_optimized_aligned_scale.npz")
            betas = torch.tensor(params["betas"][0][:10], dtype=torch.float32, device=device)
            h = cfg.human
            self.model = HUGS_TRIMLP(
                sh_degree=h.sh_degree, n_subdivision=h.n_subdivision, use_surface=h.use_surface,
                init_2d=h.init_2d, rotate_sh=h.rotate_sh, isotropic=h.isotropic,
                init_scale_multiplier=h.init_scale_multiplier, n_features=32, use_deformer=h.use_deformer,
                disable_posedirs=h.disable_posedirs, triplane_res=h.triplane_res, betas=betas,
            )
            self.model.create_betas(betas, False)
            self.model.setup_optimizer(cfg=h.lr)
            self.model.load_state_dict(torch.load(f"{out}/human_final.pth"))
            self.model.eval()
        finally:
            os.chdir(cwd)
        self.seq, self.betas, self.device = seq, betas, device
        with torch.no_grad():  # rest pelvis of this body shape
            self.j0 = self.model.smpl(betas=betas[None]).joints[0, 0]

    def zup_params(self, poses_yup: torch.Tensor, pelvis_yup: torch.Tensor):
        """y-up SMPL motion (T, 72 axis-angle; T, 3 pelvis positions) -> z-up global_orient, body_pose, transl."""
        r = Y_UP_TO_Z_UP.to(poses_yup)
        go = matrix_to_axis_angle(r @ axis_angle_to_matrix(poses_yup[:, :3]))
        transl = pelvis_yup @ r.T - self.j0.to(poses_yup)
        return go, poses_yup[:, 3:], transl

    @torch.no_grad()
    def gaussians(self, global_orient, body_pose, transl) -> dict:
        """The posed avatar (one frame) as 3D Gaussians in the z-up world (HUGS forward output)."""
        return self.model.forward(global_orient=global_orient, body_pose=body_pose, betas=self.betas, transl=transl,
                                  smpl_scale=torch.ones(1, device=self.device), dataset_idx=-1, is_train=False)

    @torch.no_grad()
    def smpl_vertices(self, global_orient, body_pose, transl) -> torch.Tensor:
        """The posed SMPL body mesh (6890, 3) for the same parameters (z-up world)."""
        out = self.model.smpl(betas=self.betas[None], body_pose=body_pose[None], global_orient=global_orient[None])
        return out.vertices[0] + transl


def look_at_camera(eye, target, width=1280, height=720, fovy_deg=40.0, device="cuda") -> dict:
    """A HUGS renderer camera (z-up world, +z up) at ``eye`` looking at ``target``."""
    eye, target = torch.tensor(eye, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    f = target - eye
    f = f / f.norm()
    r = torch.linalg.cross(f, torch.tensor([0.0, 0.0, 1.0]))
    r = r / r.norm()
    d = torch.linalg.cross(f, r)  # camera axes: x right, y down, z forward (OpenCV / 3DGS)
    w2c = torch.eye(4)
    w2c[:3, :3] = torch.stack([r, d, f])
    w2c[:3, 3] = -w2c[:3, :3] @ eye
    fovy = math.radians(fovy_deg)
    fovx = 2 * math.atan(math.tan(fovy / 2) * width / height)
    world_view = w2c.T  # 3DGS stores row-vector (transposed) matrices
    proj = get_projection_matrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).T
    return {"fovx": fovx, "fovy": fovy, "image_height": height, "image_width": width,
            "world_view_transform": world_view.to(device), "full_proj_transform": (world_view @ proj).to(device),
            "camera_center": eye.to(device)}


@torch.no_grad()
def render(camera: dict, gs: dict, bg=(1.0, 1.0, 1.0)) -> torch.Tensor:
    """Rasterize a Gaussian set (HUGS output dict keys) with HUGS's renderer -> (3, H, W) in [0, 1]."""
    pkg = render_human_scene(data=camera, human_gs_out=gs, scene_gs_out=None,
                             bg_color=torch.tensor(bg, device=camera["camera_center"].device), render_mode="human")
    return pkg["render"].clamp(0, 1)
