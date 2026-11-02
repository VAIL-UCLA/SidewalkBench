"""Shared-gsplat observation term (imported only after AppLauncher).

One shared GaussianRenderer renders all N robot cameras in one batched call. Each
robot's LIVE camera pose is derived from its configured camera-link state and mapped
into the (single, shared) scene frame by subtracting its env origin.

Camera model (COCO default offset ``(0.38243, 0, 0.184)`` with rot
``(0.5, -0.5, 0.5, -0.5)`` in the "ros" convention): a forward-looking optical
frame rigidly mounted on the selected link. In the body frame the
optical axes are forward=+X, right=-Y, down=-Z.

NOTE: ``root_quat_w`` is (x, y, z, w) in this Isaac Lab build (``wp.quatf``).
Feeding it to ``matrix_from_quat`` (which expects w, x, y, z) produced a ~90 deg
camera roll, so the rotation matrix is built here explicitly from xyzw.
Returns (N, 3, H, W) float32 in [0, 1], like an RGB camera observation.
"""

from __future__ import annotations

import torch

from .gsplat_render import GaussianRenderer
from .camera import camera_pose, rotmat_from_quat_xyzw

_RENDERER: GaussianRenderer | None = None
_INTRINSICS: torch.Tensor | None = None
_OFFSET_POS: tuple = (0.38243, 0.0, 0.184)  # camera mount in base_link (m)
_W = 512
_H = 288
_PLY = ""
_DEPTH_MESH = None  # the scene's collision trimesh: depth for compositing the robot mesh (video)
_EXTRAS = None  # .gaussians(env) -> that env's extra Gaussians (its route's cones + pedestrians), rendered into its view
_CAMERA_LINK = None
_CAMERA_BODY_ID = None


def configure_gsplat(ply_path, width, height, intrinsics_3x3, *, offset_pos=None, depth_mesh=None, camera_link=None) -> None:
    global _INTRINSICS, _W, _H, _RENDERER, _PLY, _OFFSET_POS, _DEPTH_MESH, _CAMERA_LINK, _CAMERA_BODY_ID
    _W, _H = int(width), int(height)
    _INTRINSICS = torch.as_tensor(intrinsics_3x3, dtype=torch.float32).reshape(1, 3, 3)
    _RENDERER = None  # lazily (re)built on first use, on the env device
    _PLY = ply_path
    _DEPTH_MESH = depth_mesh
    _CAMERA_LINK, _CAMERA_BODY_ID = camera_link, None
    if offset_pos is not None:
        _OFFSET_POS = tuple(offset_pos)


def set_extras(extras) -> None:
    """Render ``extras.gaussians(env)`` (each env's route cones + pedestrians, or None) into every observation."""
    global _EXTRAS
    _EXTRAS = extras


def get_renderer() -> GaussianRenderer | None:
    """The shared renderer (built on the first observation render), for extra views."""
    return _RENDERER


def camera_pose_from_root(pos_w: torch.Tensor, quat_w_xyzw: torch.Tensor):
    """Camera position (N, 3) and camera->world rotation (N, 3, 3) from the root state.

    Columns of the rotation are the optical axes [right, down, forward] in world.
    """
    return camera_pose(pos_w, quat_w_xyzw, _OFFSET_POS)


def camera_poses(env):
    """Current optical poses in the shared scan frame (without env origins)."""
    global _CAMERA_BODY_ID
    device = env.device
    robot = env.scene["robot"]
    pos_w = robot.data.root_pos_w.to(device)     # (N, 3)
    quat_w = robot.data.root_quat_w.to(device)   # (N, 4) xyzw (wp.quatf)
    if _CAMERA_LINK is not None:
        if _CAMERA_BODY_ID is None:
            ids, _ = robot.find_bodies(_CAMERA_LINK)
            if len(ids) != 1:
                raise ValueError(f"Expected one camera body {_CAMERA_LINK!r}, found {ids}")
            _CAMERA_BODY_ID = ids[0]
        pos_w = robot.data.body_link_pos_w[:, _CAMERA_BODY_ID].to(device)
        quat_w = robot.data.body_link_quat_w[:, _CAMERA_BODY_ID].to(device)
    cam_pos, rots = camera_pose_from_root(pos_w, quat_w)
    return cam_pos - env.scene.env_origins.to(device), rots


def get_gsplat_image(env) -> torch.Tensor:
    """Render the shared gsplat from each robot's live camera-link pose."""
    global _RENDERER
    device = env.device
    if _RENDERER is None:
        _RENDERER = GaussianRenderer(_PLY, device=str(device), depth_mesh=_DEPTH_MESH)

    local_pos, rots = camera_poses(env)
    n = local_pos.shape[0]
    ks = _INTRINSICS.to(device).expand(n, 3, 3)
    humans = None if _EXTRAS is None else [_EXTRAS.gaussians(i) for i in range(n)]
    return _RENDERER.render(local_pos, rots, ks, width=_W, height=_H, humans=humans)
