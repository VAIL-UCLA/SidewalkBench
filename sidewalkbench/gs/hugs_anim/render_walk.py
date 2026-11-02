"""Render a HUGS avatar performing a fitted MotionLCM motion on flat ground, to an mp4.

    python render_walk.py [--seq lab] [--motion walk_circle_smpl.npz] [--out walk_circle_lab.mp4]   (hugs env)

The y-up SMPL motion is put in the z-up world for this avatar's body shape (hugs_avatar),
lifted by the avatar_bottom ground offset so the lowest point of the sequence is on z = 0,
and rendered together with a checkerboard ground plane (flat Gaussians, so occlusion is
correct) from a fixed camera looking at the middle of the path.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from avatar_bottom import sequence_bottom
from hugs_avatar import Avatar, look_at_camera, render

SH_C0 = 0.28209479177387814


def ground_gaussians(center_xy, half=6.0, cell=0.5, spacing=0.04, device="cuda") -> dict:
    """A z = 0 checkerboard (light/dark ``cell`` squares) of flat, opaque Gaussians."""
    g = torch.arange(-half, half + 1e-6, spacing, device=device)
    x, y = torch.meshgrid(g, g, indexing="ij")
    xyz = torch.stack([x.flatten() + center_xy[0], y.flatten() + center_xy[1], torch.zeros(x.numel(), device=device)], 1)
    dark = ((torch.floor(x / cell) + torch.floor(y / cell)) % 2).flatten()[:, None]
    rgb = 0.78 - 0.28 * dark
    shs = torch.zeros(len(xyz), 16, 3, device=device)
    shs[:, 0] = (rgb - 0.5) / SH_C0
    n = len(xyz)
    return {"xyz": xyz, "scales": torch.tensor([spacing * 0.75, spacing * 0.75, 1e-4], device=device).repeat(n, 1),
            "rotq": torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1),
            "opacity": torch.ones(n, 1, device=device), "shs": shs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="lab")
    ap.add_argument("--motion", default="walk_circle_smpl.npz")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"walk_circle_{args.seq}.mp4"

    a = Avatar(args.seq)
    m = np.load(args.motion)
    t = lambda x: torch.tensor(x, dtype=torch.float32, device=a.device)  # noqa: E731
    go, bp, tr = a.zup_params(t(m["poses"]), t(m["root"]))
    bottom, _ = sequence_bottom(a, args.motion)
    tr[:, 2] -= float(bottom.min())  # lowest point of the sequence on the ground (avatar_bottom.py)
    path = (tr + a.j0)[:, :2].cpu().numpy()
    c = (path.min(0) + path.max(0)) / 2
    print(f"[walk] {len(tr)} frames, path centre {c.round(2)}, extent {np.ptp(path, 0).round(2)} m, ground offset "
          f"{-bottom.min():+.4f} m")

    ground = ground_gaussians(c)
    cam = look_at_camera((c[0], c[1] - 4.5, 2.2), (c[0], c[1], 0.7))
    fps = float(m["fps"]) if "fps" in m else 20.0
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (cam["image_width"], cam["image_height"]))
    for i in range(len(tr)):
        h = a.gaussians(go[i], bp[i], tr[i])
        gs = {k: torch.cat([h[k], ground[k]]) for k in ("xyz", "scales", "rotq", "opacity", "shs")}
        gs["active_sh_degree"] = h["active_sh_degree"]
        img = (render(cam, gs, bg=(0.92, 0.95, 1.0)).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        vw.write(img[..., ::-1])
    vw.release()
    print(f"[walk] wrote {out}")


if __name__ == "__main__":
    main()
