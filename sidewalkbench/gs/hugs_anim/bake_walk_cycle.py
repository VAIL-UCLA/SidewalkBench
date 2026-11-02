"""Bake a HUGS avatar's walk cycle into per-frame 3D Gaussians for the eval (no HUGS at runtime).

    python bake_walk_cycle.py [--seq lab] [--motion walk_forward_smpl.npz] [--frame K] [--out OUT]   (hugs env)
    (default OUT: gs_assets/pedestrians/<seq>/walk.npz, or stand.npz with --frame)

Takes the most seamless single gait cycle of a fitted MotionLCM walk (fit_smpl.py output),
plays it in place -- forward travel and heading drift removed, the vertical bob kept --
facing +x in the z-up world, with the lowest point of the cycle on z = 0 (avatar_bottom.py),
and stores the posed Gaussians of every frame (all of them: MIN_OPACITY = 0; opacity and colour are
pose-independent in HUGS, so one index set would serve every frame if pruning were turned back on).

The eval places a frame by rotating it to the walker's heading and translating it to its
ground position, and plays the cycle continuously: frame = (time / cycle_s) * num_frames.

Output npz: xyz (F, N, 3), rotq_wxyz (F, N, 4), scales (F, N, 3) [float16, mm precision over the
avatar's ~2 m extent], opacity (N,), rgb (N, 3),
stride_m, cycle_s, seq.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from avatar_bottom import gaussian_bottom
from hugs_avatar import Y_UP_TO_Z_UP, Avatar, look_at_camera, render

SH_C0 = 0.28209479177387814
MIN_OPACITY = 0.0  # keep every Gaussian: pruning at 0.02 dropped 30-63% of them (up to 31% of an avatar's opacity)
FACE_PLUS_X = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # Rz(+90): -y (SMPL facing) -> +x


def best_cycle(poses: np.ndarray, lengths=range(17, 24)) -> tuple[int, int]:
    """(start, length) of the segment whose last+1 frame best matches its first (body pose only)."""
    p = poses[:, 3:]
    return min(((np.abs(p[s + n] - p[s]).mean(), s, n) for n in lengths for s in range(20, len(p) - n - 1)))[1:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="lab")
    ap.add_argument("--motion", default="walk_forward_smpl.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frame", type=int, default=None,
                    help="bake this single frame of --motion (a standing pose; heading = the body's facing)")
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "gs_assets", "pedestrians",
                                   args.seq, f"{'walk' if args.frame is None else 'stand'}.npz")

    m = np.load(args.motion)
    if args.frame is None:
        s, n = best_cycle(m["poses"])
        poses, root = m["poses"][s:s + n + 1].copy(), m["root"][s:s + n + 1].copy()
        step = root[-1, [0, 2]] - root[0, [0, 2]]  # travel over the cycle (y-up: x, z ground plane)
        stride = float(np.linalg.norm(step))
        yaw = math.atan2(step[0], step[1])  # heading of travel, from +z toward +x
    else:  # one standing frame, facing where the body faces (SMPL faces +z, y-up)
        s, n, stride = args.frame, 1, 0.0
        poses, root = m["poses"][s:s + 2].copy(), m["root"][s:s + 2].copy()
        f = axis_angle_to_matrix(torch.tensor(poses[0, :3], dtype=torch.float32)) @ torch.tensor([0.0, 0.0, 1.0])
        yaw = math.atan2(float(f[0]), float(f[2]))
    # in place: undo the heading (rotate about y by -yaw) and the horizontal travel, keep the bob
    ry = torch.tensor([[math.cos(-yaw), 0.0, math.sin(-yaw)], [0.0, 1.0, 0.0], [-math.sin(-yaw), 0.0, math.cos(-yaw)]])
    t = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
    go = matrix_to_axis_angle(ry @ axis_angle_to_matrix(t(poses[:n, :3])))
    pelvis = t(root[:n]) * t([0.0, 1.0, 0.0])
    a = Avatar(args.seq)
    dev = a.device
    r = (FACE_PLUS_X @ Y_UP_TO_Z_UP).to(dev)  # y-up, facing +z  ->  z-up, facing +x
    go = matrix_to_axis_angle(r @ axis_angle_to_matrix(go.to(dev)))
    transl = pelvis.to(dev) @ r.T - a.j0
    bp = t(poses[:n, 3:]).to(dev)

    frames = [a.gaussians(go[i], bp[i], transl[i]) for i in range(n)]
    lift = -min(gaussian_bottom(g) for g in frames)  # lowest point of the cycle on the ground
    op = frames[0]["opacity"].squeeze(-1)
    keep = op >= MIN_OPACITY

    # pruning check: frame 0 rendered with all Gaussians vs the kept ones
    cam = look_at_camera((3.0, -3.0, 1.2), (0.0, 0.0, 0.9 - lift), width=512, height=512, fovy_deg=35)
    full = render(cam, frames[0], bg=(1.0, 1.0, 1.0))
    pruned = render(cam, {**frames[0], **{k: frames[0][k][keep] for k in ("xyz", "scales", "rotq", "opacity", "shs")}},
                    bg=(1.0, 1.0, 1.0))
    psnr = float(10 * torch.log10(1.0 / torch.mean((full - pruned) ** 2)))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out,
        xyz=np.stack([(g["xyz"][keep] + torch.tensor([0.0, 0.0, lift], device=dev)).cpu().numpy() for g in frames]).astype(np.float16),
        rotq_wxyz=np.stack([g["rotq"][keep].cpu().numpy() for g in frames]).astype(np.float16),
        scales=np.stack([g["scales"][keep].cpu().numpy() for g in frames]).astype(np.float16),
        opacity=op[keep].cpu().numpy(),
        rgb=np.clip(frames[0]["shs"][keep, 0].cpu().numpy() * SH_C0 + 0.5, 0, 1),
        stride_m=stride, cycle_s=n / float(m["fps"]), seq=args.seq,
    )
    print(f"[bake] {args.seq}: cycle frames {s}..{s + n - 1} ({n} @ {float(m['fps']):.0f} fps), stride {stride:.3f} m "
          f"({stride / (n / float(m['fps'])):.2f} m/s); kept {int(keep.sum())}/{len(op)} Gaussians "
          f"(opacity >= {MIN_OPACITY}), pruned-vs-full PSNR {psnr:.1f} dB; ground lift {lift:+.3f} m -> {out}")


if __name__ == "__main__":
    main()
