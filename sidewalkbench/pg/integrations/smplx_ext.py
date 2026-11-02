"""Additional SMPL-X helpers that are not present in the pinned ``smplx`` release.

Vendored from the project's former ``replacement/smplx/lbs.py`` so the installed
``smplx`` package does not need to be patched at import time. The public entry
point is :func:`posed_joints_to_axis_angle`, used by the scene code to recover
minimum-angle axis-angle pose parameters from joint positions.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from smplx.lbs import transform_mat
from smplx.utils import Tensor


def _minimum_arc_rotation(a: Tensor, b: Tensor, dtype=torch.float32) -> Tensor:
    """Compute the minimum-angle rotation matrix that maps unit vector *a*
    to unit vector *b* (batched).  The result has **zero twist** around the
    bone axis, which makes it the smallest possible rotation.

    Uses the Rodrigues closed-form:
        v = a × b,  c = a · b
        R = I + [v]× + [v]×² / (1 + c)

    Near the anti-parallel case (c ≈ -1) a fallback via an arbitrary
    perpendicular axis is used.

    Parameters
    ----------
    a : Tensor  B × 3  (unit vectors)
    b : Tensor  B × 3  (unit vectors)

    Returns
    -------
    R : Tensor  B × 3 × 3  rotation matrices
    """
    B = a.shape[0]
    device = a.device
    eps = 1e-7

    v = torch.cross(a, b, dim=-1)  # B × 3
    c = (a * b).sum(dim=-1, keepdim=True)  # B × 1

    # Skew-symmetric [v]×  –  B × 3 × 3
    vx = torch.zeros(B, 3, 3, dtype=dtype, device=device)
    vx[:, 0, 1] = -v[:, 2]
    vx[:, 0, 2] = v[:, 1]
    vx[:, 1, 0] = v[:, 2]
    vx[:, 1, 2] = -v[:, 0]
    vx[:, 2, 0] = -v[:, 1]
    vx[:, 2, 1] = v[:, 0]

    eye = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)  # 1×3×3

    # Normal case: R = I + [v]× + [v]×² / (1 + c)
    denom = (1.0 + c).unsqueeze(-1)  # B × 1 × 1
    R = eye + vx + torch.bmm(vx, vx) / denom.clamp(min=eps)

    # --- Fallback for near-anti-parallel vectors (c ≈ -1) ---------------
    anti = c.squeeze(-1) < -1.0 + 1e-4  # B  (bool)
    if anti.any():
        # Pick an arbitrary axis perpendicular to a
        perp = torch.zeros_like(a[anti])
        abs_a = a[anti].abs()
        # Choose the coordinate axis most perpendicular to a
        min_dim = abs_a.argmin(dim=-1)  # B'
        perp[torch.arange(perp.shape[0], device=device), min_dim] = 1.0
        perp = perp - (perp * a[anti]).sum(-1, keepdim=True) * a[anti]
        perp = F.normalize(perp, dim=-1)
        # 180° rotation around perp:  R = 2 * perp perp^T - I
        R_anti = 2.0 * torch.bmm(perp.unsqueeze(-1), perp.unsqueeze(-2)) - eye
        R[anti] = R_anti

    return R


def inverse_batch_rigid_transform(
    posed_joints: Tensor, joints: Tensor, parents: Tensor, dtype=torch.float32
) -> Tuple[Tensor, Tensor]:
    """
    Inverse of batch_rigid_transform. Given posed (transformed) joint
    locations, rest-pose joint locations, and the kinematic tree, recover
    the per-joint local rotation matrices and the relative rigid
    transformations (A).

    Each local rotation is made **as small as possible**:

    * **Multi-child joints** (≥ 2 children): the local rotation is uniquely
      determined by SVD-based Procrustes alignment (computed in the parent
      frame).
    * **Single-child joints**: only the bone direction is constrained; the
      twist around the bone is free.  We choose the **minimum-arc rotation**
      (zero twist), which is the rotation with the smallest angle.
    * **Leaf joints** (no children): completely unconstrained – set to
      **identity** (the smallest possible rotation).

    Parameters
    ----------
    posed_joints : torch.Tensor BxNx3
        The posed (transformed) joint locations, i.e. J_transformed.
    joints : torch.Tensor BxNx3
        The rest-pose joint locations (before any pose is applied).
    parents : torch.Tensor N
        The kinematic tree (parent index for each joint; root's parent is -1).
    dtype : torch.dtype, optional
        The data type of the created tensors, default is torch.float32.

    Returns
    -------
    rot_mats : torch.Tensor BxNx3x3
        The recovered per-joint local rotation matrices (minimum-angle).
    rel_transforms : torch.Tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints (same semantics as the A output of
        batch_rigid_transform).
    """
    batch_size = posed_joints.shape[0]
    num_joints = posed_joints.shape[1]
    device = posed_joints.device

    # ------------------------------------------------------------------
    # 1. Build a children list from the parents tensor
    # ------------------------------------------------------------------
    children: List[List[int]] = [[] for _ in range(num_joints)]
    for i in range(1, num_joints):
        children[parents[i].item()].append(i)

    # ------------------------------------------------------------------
    # 2. Estimate local rotations (top-down), accumulating global
    #    rotations along the chain.
    #
    #    Constraint at joint i with children {c1, c2, …}:
    #        G_parent @ L_i @ rest_bone_c  =  posed_bone_c
    #        ⟹  L_i @ rest_bone_c  =  G_parent^T @ posed_bone_c
    #
    #    Working in the parent frame lets us directly recover L_i.
    #    For single-child joints we use the minimum-arc rotation
    #    (smallest angle, zero twist) instead of SVD.
    # ------------------------------------------------------------------
    eye3 = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)  # B × 3 × 3
    global_rot = torch.zeros(batch_size, num_joints, 3, 3, dtype=dtype, device=device)
    local_rot = torch.zeros_like(global_rot)

    for i in range(num_joints):
        # Parent's accumulated global rotation
        if i == 0:
            G_parent = eye3
        else:
            G_parent = global_rot[:, parents[i]]

        if len(children[i]) > 0:
            child_idx = children[i]

            # Rest-pose bones (in global rest frame):  B × 3 × K
            bones_rest = (joints[:, child_idx] - joints[:, i : i + 1]).permute(0, 2, 1)
            # Posed bones (in global posed frame):     B × 3 × K
            bones_posed_global = (posed_joints[:, child_idx] - posed_joints[:, i : i + 1]).permute(0, 2, 1)

            # Rotate posed bones into the parent frame so that
            # the Procrustes / minimum-arc solution directly yields L_i
            bones_posed_local = torch.bmm(G_parent.transpose(1, 2), bones_posed_global)  # B × 3 × K

            if len(child_idx) == 1:
                # ----- Single child  →  minimum-arc rotation (zero twist) -----
                a = F.normalize(bones_rest[:, :, 0], dim=-1)  # B × 3
                b = F.normalize(bones_posed_local[:, :, 0], dim=-1)  # B × 3
                local_rot[:, i] = _minimum_arc_rotation(a, b, dtype=dtype)
            else:
                # ----- Multiple children  →  SVD Procrustes (unique) ----------
                H = torch.bmm(bones_posed_local, bones_rest.transpose(1, 2))  # B × 3 × 3
                U, S, Vh = torch.linalg.svd(H)

                # Ensure proper rotation (det = +1)
                det = torch.det(torch.bmm(U, Vh))  # B
                correction = torch.ones(batch_size, 3, dtype=dtype, device=device)
                correction[:, -1] = det.sign()
                U_corrected = U * correction.unsqueeze(-1)
                local_rot[:, i] = torch.bmm(U_corrected, Vh)
        else:
            # Leaf joint: no constraint → identity (smallest rotation)
            local_rot[:, i] = eye3

        # Accumulate global rotation: G_i = G_parent @ L_i
        global_rot[:, i] = torch.bmm(G_parent, local_rot[:, i])

    # ------------------------------------------------------------------
    # 3. Build transforms & A (rel_transforms) in the same format as
    #    batch_rigid_transform, using the recovered local rotations.
    # ------------------------------------------------------------------
    joints_col = joints.unsqueeze(-1)  # B × N × 3 × 1

    rel_joints = joints_col.clone()
    rel_joints[:, 1:] -= joints_col[:, parents[1:]]

    transforms_mat = transform_mat(local_rot.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(
        batch_size, num_joints, 4, 4
    )

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # A: relative transforms (same formula as batch_rigid_transform)
    joints_homogen = F.pad(joints_col, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return local_rot, rel_transforms


def _rotmat_to_axis_angle(rot_mats: Tensor) -> Tensor:
    """Convert a batch of rotation matrices to axis-angle representation.

    Handles all three regimes robustly:
      * θ ≈ 0  (near identity)  – first-order Taylor expansion
      * 0 < θ < π              – standard Rodrigues inverse
      * θ ≈ π  (near 180°)     – extract axis from R + I

    Parameters
    ----------
    rot_mats : Tensor  B × 3 × 3

    Returns
    -------
    axis_angle : Tensor  B × 3
    """
    # trace(R) = 1 + 2 cos θ
    trace = rot_mats[:, 0, 0] + rot_mats[:, 1, 1] + rot_mats[:, 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)  # B

    # Skew-symmetric part → r = 2 sin(θ) · axis
    r = torch.stack(
        [
            rot_mats[:, 2, 1] - rot_mats[:, 1, 2],
            rot_mats[:, 0, 2] - rot_mats[:, 2, 0],
            rot_mats[:, 1, 0] - rot_mats[:, 0, 1],
        ],
        dim=-1,
    )  # B × 3

    # axis_angle = axis · θ = r · θ / (2 sin θ)
    # For θ → 0: θ/(2 sin θ) → 0.5  (Taylor)
    sin_angle = torch.sin(angle)
    scale = torch.where(sin_angle.abs() > 1e-6, angle / (2.0 * sin_angle), torch.ones_like(angle) * 0.5)
    axis_angle = r * scale.unsqueeze(-1)

    # --- Near-π fallback: sin θ → 0, skew part unreliable ---------------
    # Use R + I = 2 · axis · axisᵀ  to recover the axis direction.
    near_pi = angle > (np.pi - 1e-4)
    if near_pi.any():
        R_pi = rot_mats[near_pi]
        eye = torch.eye(3, dtype=rot_mats.dtype, device=rot_mats.device)
        RpI = R_pi + eye  # B' × 3 × 3

        # Pick the column of R+I with the largest squared norm
        col_norms_sq = (RpI * RpI).sum(dim=1)  # B' × 3
        best_col = col_norms_sq.argmax(dim=-1)  # B'

        idx = best_col.unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 1)
        n = torch.gather(RpI, 2, idx).squeeze(-1)  # B' × 3
        n = F.normalize(n, dim=-1)

        # Consistent sign: make the largest absolute component positive
        max_abs_idx = n.abs().argmax(dim=-1)
        sign_fix = torch.gather(n, 1, max_abs_idx.unsqueeze(-1)).sign().squeeze(-1)
        n = n * sign_fix.unsqueeze(-1)

        axis_angle[near_pi] = n * angle[near_pi].unsqueeze(-1)

    return axis_angle


def posed_joints_to_axis_angle(
    posed_joints: Tensor,
    rest_joints: Tensor,
    parents: Tensor,
    dtype=torch.float32,
) -> Tensor:
    """Recover minimum-angle axis-angle pose parameters from joint positions.

    This is the high-level utility that combines
    :func:`inverse_batch_rigid_transform` (to get per-joint local rotation
    matrices with the smallest possible rotations) with a rotation-matrix →
    axis-angle conversion.

    Parameters
    ----------
    posed_joints : torch.Tensor  B × J × 3
        The posed (transformed) joint locations (i.e. J_transformed).
    rest_joints : torch.Tensor  B × J × 3
        The rest-pose joint locations.  These can be obtained from shape
        parameters via ``J = vertices2joints(J_regressor, v_shaped)`` where
        ``v_shaped = v_template + blend_shapes(betas, shapedirs)``.
    parents : torch.Tensor  J
        The kinematic tree (parent indices; root's parent is -1).
    dtype : torch.dtype, optional
        Data type for intermediate tensors, default ``torch.float32``.

    Returns
    -------
    pose : torch.Tensor  B × J × 3
        Per-joint axis-angle rotations with the smallest possible angle at
        every joint.  Can be fed directly back into ``lbs(pose=...,
        pose2rot=True)``.
    """
    rot_mats, _ = inverse_batch_rigid_transform(posed_joints, rest_joints, parents, dtype=dtype)

    B, J = rot_mats.shape[:2]
    axis_angle = _rotmat_to_axis_angle(rot_mats.reshape(-1, 3, 3))
    return axis_angle.reshape(B, J, 3)
