"""Camera geometry in the xyzw quaternion convention used by GS/Isaac Lab."""

import torch


def rotmat_from_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack((1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
                        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
                        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)), -1).reshape(-1, 3, 3)


def camera_pose(pos: torch.Tensor, quat_xyzw: torch.Tensor, offset=(0.0, 0.0, 0.0)):
    """Body pose to forward-looking optical pose: right=-Y, down=-Z, forward=+X."""
    rotation = rotmat_from_quat_xyzw(quat_xyzw)
    offset = torch.as_tensor(offset, dtype=pos.dtype, device=pos.device)
    return pos + torch.einsum("nij,j->ni", rotation, offset), torch.stack(
        (-rotation[:, :, 1], -rotation[:, :, 2], rotation[:, :, 0]), dim=2)
