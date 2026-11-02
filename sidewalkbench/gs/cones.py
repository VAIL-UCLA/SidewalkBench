"""Annotated traffic cones: the cone mesh and its placement at the settled poses.

Cones are scene-level (on every route) and static during evaluation. Each annotated cone
(``xy`` + ``yaw_deg``) is dropped once onto the scene mesh by ``sidewalkbench/gs/annotate/settle_cones.py``
(``settle_cones.sh``), which stores the resting pose in the annotation. The eval puts each
cone at that settled pose as a kinematic collider whose contact with the robot is a collision. They are rendered as 3D Gaussians added to the scene splat
(``cone_gaussians``): CONE_SPLAT is the cone mesh converted by mesh2splat
(github.com/electronicarts/mesh2splat @c3d3d52 + sidewalkbench/gs/annotate/mesh2splat_headless.patch, Linux build):
    Mesh2Splat gs_assets/cone/traffic_cone.glb gs_assets/cone/traffic_cone_splat.ply 64
(sampling resolution 64 -> ~11k Gaussians per cone; 512 is ~730k), lit like the meshes (``mesh_render.shade``) but
with the sky dome made the same from every horizontal direction (``CONE_SIDE_IRRADIANCE``): PG's sky HDR is lopsided
(a cone's north face got ~7x the sky light of its south face), so a cone's colour depended on which way the camera
looked along the route; now every side gets the dim side's light (the deep orange seen in dark), the sun is unchanged.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import trimesh

CONE_GLB = os.path.join(os.path.dirname(__file__), "..", "..", "gs_assets", "cone", "traffic_cone.glb")
CONE_SPLAT = os.path.join(os.path.dirname(CONE_GLB), "traffic_cone_splat.ply")
# glTF is Y-up (base at y=0, tip at y=0.713 m): +90 deg about X stands the cone up in z-up
_Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
CONE_SIDE_IRRADIANCE = 770.0  # sky irradiance on a cone's side face (~18 deg up), every azimuth (picked by eye, 2026-10-10)


def cone_mesh() -> trimesh.Trimesh:
    """The cone mesh (collider, RVO2 footprint), upright in its own frame: base centered on the origin, tip up +z."""
    mesh = trimesh.load(CONE_GLB, force="mesh")
    mesh.vertices = mesh.vertices @ _Y_UP_TO_Z_UP.T
    return mesh


def cone_gaussians(poses: np.ndarray, device) -> tuple:
    """Activated Gaussians (means, wxyz quats, scales, opacities, rgb) of cones at ``poses`` (K, 7: settled
    x, y, z, qx, qy, qz, qw), their albedo lit by the azimuth-symmetric sky + PG's overhead sun (module docstring)."""
    from plyfile import PlyData
    from scipy.spatial.transform import Rotation

    from .gsplat_render import _SH_C0
    from .mesh_render import LIGHT_INTENSITY, _sh9, shade, sky_irradiance_sh

    v = PlyData.read(CONE_SPLAT)["vertex"]
    col = lambda *n: np.stack([np.asarray(v[k], dtype=np.float64) for k in n], 1)  # noqa: E731
    up = Rotation.from_matrix(_Y_UP_TO_Z_UP)
    g = Rotation.from_quat(col("rot_1", "rot_2", "rot_3", "rot_0"))  # stored wxyz
    out = []
    for p in np.asarray(poses, dtype=np.float64).reshape(-1, 7):
        r = Rotation.from_quat(p[3:]) * up
        q = (r * g).as_quat()
        out.append((r.apply(col("x", "y", "z")) + p[:3], np.concatenate([q[:, 3:], q[:, :3]], 1), r.apply(col("nx", "ny", "nz"))))
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)  # noqa: E731
    means, quats, normals = (t(np.concatenate(x)) for x in zip(*out))
    k = len(out)
    albedo = t(np.tile(np.clip(col("f_dc_0", "f_dc_1", "f_dc_2") * _SH_C0 + 0.5, 0, 1) ** 2.2, (k, 1)))
    sky = sky_irradiance_sh()
    sky[[1, 3, 4, 5, 7, 8]] = 0.0  # keep only the azimuth-independent SH terms (1, z, 3z^2 - 1)
    sky *= CONE_SIDE_IRRADIANCE / (_sh9(np.array([0.95, 0.0, 0.31])) @ sky).mean()
    irr = _sh9(normals) @ t(sky) + LIGHT_INTENSITY * normals[..., 2:].clamp(min=0.0)  # + PG's sun straight down
    return (means, quats, t(np.tile(np.exp(col("scale_0", "scale_1", "scale_2")), (k, 1))),
            t(np.tile(1 / (1 + np.exp(-np.asarray(v["opacity"], dtype=np.float64))), k)), shade(albedo, irr))
