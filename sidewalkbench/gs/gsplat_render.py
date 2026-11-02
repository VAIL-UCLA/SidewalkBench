"""Gaussian-splat renderer (gsplat's experimental inference renderer).

A clean rewrite of coco-gs's ``GaussianPointCloudRenderer`` stable path: load a
Gaussian ``.ply`` once and pack it for gsplat's inference renderer (one camera per
call, ~40x faster than ``rasterization`` on the long-horizon scenes). A view with a
view with pedestrians renders the packed scene plus their Gaussians in one pass.

Camera convention matches the reference: given a camera position and a
camera->world rotation ``rot_c2w`` (camera axes X=right, Y=down, Z=forward as
world columns), the view matrix is ``[R_w2c | -R_w2c t]`` with ``R_w2c =
rot_c2w^T``. Output is ``(B, 3, H, W)`` float32 in [0, 1]. ``meshes`` (the robot, in the video;
``mesh_render``) are depth-composited against the scene's collision mesh (the inference renderer
has no depth output). Static extras (the cones) are packed with the scene.

Needs gsplat 1.6 with ``gsplat.experimental.render`` (not on PyPI yet), a project dependency compiled by
``uv sync`` for the machine's GPU (a few minutes; see the README).
The packing kernel (gsplat_scene) compiles on first use (~12 s, cached).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_SH_C0 = 0.28209479177387814  # SH DC -> RGB constant


def load_ply(path, device) -> tuple:
    """A 3DGS ``.ply`` -> activated (means, scales, wxyz quats, opacities, rgb) float32 tensors (finite ones)."""
    from plyfile import PlyData

    el = PlyData.read(str(path))["vertex"]

    def col(name: str) -> np.ndarray:
        return np.asarray(el[name], dtype=np.float32)

    xyz = np.stack([col("x"), col("y"), col("z")], axis=1)
    scales = np.exp(np.stack([col("scale_0"), col("scale_1"), col("scale_2")], axis=1))
    quats = np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], axis=1)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-9
    opacities = 1.0 / (1.0 + np.exp(-col("opacity")))
    rgb = np.clip(np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=1) * _SH_C0 + 0.5, 0.0, 1.0)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(scales).all(axis=1)
    return tuple(torch.from_numpy(np.ascontiguousarray(a[finite])).to(device, dtype=torch.float32)
                 for a in (xyz, scales, quats, opacities, rgb))


class GaussianRenderer:
    def __init__(self, ply_path: str | Path, *, device: str = "cuda", extra=None, depth_mesh=None):
        """``extra``: more static Gaussians (means, quats, scales, opacities, rgb; e.g. the cones) packed with
        the scene; ``depth_mesh``: the scene's collision trimesh, depth for compositing ``meshes`` (``render``)."""
        from gsplat.experimental.render.components import GaussianInferenceRenderer  # deferred (CUDA import)
        from gsplat.scene import GaussianInferenceScene

        self._scene_cls, self._renderer_cls = GaussianInferenceScene, GaussianInferenceRenderer
        self.device = torch.device(device)
        means, scales, quats, opacities, colors = load_ply(ply_path, self.device)
        self.set_scene(*(torch.cat([a, b.to(self.device, torch.float32)]) for a, b in
                         zip((means, quats, scales, opacities, colors), extra)) if extra is not None
                       else (means, quats, scales, opacities, colors))
        self._depth_mesh, self._depth = depth_mesh, None  # trimesh, its mesh_render.MeshDepth (lazy)

    def set_scene(self, means, quats, scales, opacities, colors) -> None:
        """Pack these (activated) Gaussians as the scene."""
        self._scene = self._pack(means, quats, scales, opacities, colors, "scene")
        self._renderer = self._renderer_cls(self._scene)
        self._human = None  # (scene + human slot, its renderer, slot size), built on the first humans

    def _pack(self, means, quats, scales, opacities, colors, name):
        return self._scene_cls.from_gaussian_tensors(*(t.contiguous() for t in (means, quats, scales, opacities, colors)),
                                                     None, "none", id=name)

    def _with_human(self, h: dict):
        """A renderer for the packed scene plus a set of human Gaussians (``humans.HumanCrowd.gaussians``): the
        scene + a human slot is built once (rebuilt larger when a bigger set comes); each set is padded with
        transparent Gaussians to the slot's size, packed and copied into the slot."""
        n, cap = len(h["means"]), 0 if self._human is None else self._human[2]
        if n < cap:
            pad = {"means": torch.zeros(3), "quats": torch.tensor([1.0, 0, 0, 0]), "scales": torch.full((3,), 1e-6),
                   "opacities": torch.zeros(()), "colors": torch.zeros(3)}
            h = {k: torch.cat([h[k], pad[k].to(h[k]).expand(cap - n, *h[k].shape[1:])]) for k in pad}
        hp = self._pack(h["means"], h["quats"], h["scales"], h["opacities"], h["colors"], "human")
        keys = ("means_planar", "qso_packed", "colors_packed")
        if n > cap:
            self._human = None  # free the old slot first
            scene = self._scene_cls("view")
            for name, s in (("scene", self._scene), ("human", hp)):
                scene.put(name, {**{k: getattr(s, k) for k in keys}, "sh_degree": s.sh_degree,
                                 "sh_compression_mode": s.sh_compression_mode})
            self._human = (scene, self._renderer_cls(scene), n)
        else:
            m = self._scene.num_gaussians
            for k in keys:
                dst = getattr(self._human[0], k)
                (dst[:, m:] if k == "means_planar" else dst[m:]).copy_(getattr(hp, k))
        return self._human[1]

    @torch.no_grad()
    def render(
        self,
        cam_pos_w: torch.Tensor,   # (B, 3)
        rot_c2w: torch.Tensor,     # (B, 3, 3)
        intrinsics: torch.Tensor,  # (B, 3, 3)
        *,
        width: int,
        height: int,
        meshes=(),
        humans=None,
    ) -> torch.Tensor:
        """Render B camera views -> (B, 3, H, W) float32 in [0, 1]; ``humans`` is a per-view list of
        extra Gaussian sets (``humans.BakedWalker.gaussians``) or None, rendered with the scene in
        one pass; ``meshes`` are ``(MeshCompositor, verts_w, normals_w)`` layers depth-composited in order."""
        cam_pos_w = cam_pos_w.to(self.device, dtype=torch.float32)
        rot_c2w = rot_c2w.to(self.device, dtype=torch.float32)
        ks = intrinsics.to(self.device, dtype=torch.float32)
        rot_w2c = rot_c2w.transpose(-1, -2)
        viewmats = torch.eye(4, device=self.device).repeat(len(cam_pos_w), 1, 1)
        viewmats[:, :3, :3] = rot_w2c
        viewmats[:, :3, 3] = -torch.bmm(rot_w2c, cam_pos_w.unsqueeze(-1)).squeeze(-1)
        humans = list(humans) if humans is not None else [None] * len(cam_pos_w)
        colors = torch.stack([
            (self._renderer if h is None else self._with_human(h)).render(
                viewmat=viewmats[i].contiguous(), K=ks[i].contiguous(), width=width, height=height).frame[0, ..., :3].float()
            for i, h in enumerate(humans)])
        if meshes:
            if self._depth is None:
                from .mesh_render import MeshDepth

                self._depth = MeshDepth(self._depth_mesh, self.device)
            depth = self._depth.depth(cam_pos_w, rot_c2w, ks, width, height)
            for comp, verts_w, normals_w in meshes:
                colors, depth = comp.composite(colors, depth, cam_pos_w, rot_c2w, ks, width, height,
                                               verts_w=verts_w, normals_w=normals_w)
        return torch.clamp(colors, 0.0, 1.0).permute(0, 3, 1, 2).contiguous()
