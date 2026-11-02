"""Where is the bottom of a HUGS human avatar?

    python avatar_bottom.py [--seq lab] [--motion walk_circle_smpl.npz]   (hugs env)

The bottom is measured on the avatar's 3D Gaussians, not the SMPL mesh: HUGS learns offsets
from SMPL (shoes, clothes), and for ``lab`` the boot soles reach ~5-6 cm below the SMPL soles.
Each Gaussian's visible lower edge is its centre minus k standard deviations along +z, with k
where its alpha (opacity * exp(-k^2/2)) drops below the renderer's 1/255 cutoff. The bottom
is a robust low quantile (``QUANTILE``) of those edges over Gaussians with opacity >=
``MIN_OPACITY``, which ignores the many near-transparent ones. Works for any pose/frame.

Without --motion: the avatar standing in its rest pose (z-up, pelvis at the origin).
With --motion (fit_smpl.py output, y-up): the per-frame bottom of the whole sequence and the
height offset that puts its lowest point on the ground (z = 0), as render_walk.py uses it.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from hugs_avatar import Avatar

MIN_OPACITY = 0.1
QUANTILE = 1e-3


def gaussian_bottom(gs: dict) -> float:
    """z of the avatar's visible bottom for one posed Gaussian set (HUGS forward output, z-up)."""
    op = gs["opacity"].squeeze(-1)
    sig_z = torch.sqrt(((gs["rotmat"][:, 2, :] * gs["scales"]) ** 2).sum(-1))  # std along world z
    k = torch.sqrt(torch.clamp(2 * torch.log(255 * op), min=0))                 # sigmas to alpha 1/255
    edge = gs["xyz"][:, 2] - k * sig_z
    return torch.quantile(edge[op >= MIN_OPACITY], QUANTILE).item()


def sequence_bottom(avatar: Avatar, motion_npz: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame Gaussian bottom and SMPL sole height of a y-up SMPL motion placed in the z-up world."""
    m = np.load(motion_npz)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=avatar.device)  # noqa: E731
    go, bp, tr = avatar.zup_params(t(m["poses"]), t(m["root"]))
    gb, sole = [], []
    for i in range(len(tr)):
        gb.append(gaussian_bottom(avatar.gaussians(go[i], bp[i], tr[i])))
        sole.append(avatar.smpl_vertices(go[i], bp[i], tr[i])[:, 2].min().item())
    return np.array(gb), np.array(sole)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="lab", help="NeuMan sequence / pretrained HUGS avatar")
    ap.add_argument("--motion", default=None, help="fit_smpl.py output (.npz, y-up)")
    args = ap.parse_args()
    a = Avatar(args.seq)
    if args.motion is None:
        go = torch.tensor([np.pi / 2, 0.0, 0.0], device=a.device)  # rest pose stood up in z-up
        bp, tr = torch.zeros(69, device=a.device), -a.j0  # pelvis at the origin
        b = gaussian_bottom(a.gaussians(go, bp, tr))
        sole = a.smpl_vertices(go, bp, tr)[:, 2].min().item()
        print(f"[bottom] {args.seq} rest pose, pelvis at z=0: avatar bottom z={b:.4f} m, SMPL sole z={sole:.4f} m "
              f"(avatar reaches {(sole - b) * 100:.1f} cm below the SMPL soles); pelvis height above ground {-b:.4f} m")
        return
    gb, sole = sequence_bottom(a, args.motion)
    print(f"[bottom] {args.seq} + {args.motion}: {len(gb)} frames")
    print(f"[bottom]   avatar bottom z: min {gb.min():.4f}  median {np.median(gb):.4f}  max {gb.max():.4f}")
    print(f"[bottom]   SMPL sole z:     min {sole.min():.4f}  median {np.median(sole):.4f}  max {sole.max():.4f}")
    print(f"[bottom]   ground offset: add {-gb.min():+.4f} m to z so the lowest point of the sequence is on z=0")


if __name__ == "__main__":
    main()
