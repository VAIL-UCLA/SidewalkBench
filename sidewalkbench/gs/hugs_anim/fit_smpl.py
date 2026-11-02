"""Fit SMPL pose parameters to a MotionLCM joint sequence (MotionLCM's SMPLify3D, as in its fit.py).

    python sidewalkbench/gs/hugs_anim/fit_smpl.py <joints.pkl> <out.npz>   (uses third_party/MotionLCM)

MotionLCM outputs HumanML3D joint positions (T, 22, 3): y-up, metres, floor at y = 0. fit.py
fits SMPL to them but keeps only the mesh vertices; this keeps the parameters HUGS needs:
  poses (T, 72)  axis-angle: global_orient (3) + body_pose (69), in the same y-up frame
  root  (T, 3)   pelvis (joint 0) positions -- the SMPL ``transl`` for a body with shape
                 ``betas`` is root - J0(betas), J0 = that body's rest pelvis position
Uses the chumpy-free SMPL_NEUTRAL from SidewalkBench/assets (MotionLCM's copy needs chumpy).
"""

import os
import pickle
import sys

import h5py
import numpy as np
import smplx
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MOTIONLCM = os.path.join(REPO, "third_party", "MotionLCM")  # its SMPLify config paths are relative to it
sys.path.insert(0, MOTIONLCM)

from mld.transforms.joints2rots import config  # noqa: E402
from mld.transforms.joints2rots.smplify import SMPLify3D  # noqa: E402

SMPL_PKL = os.path.join(REPO, "assets", "smplx", "SMPL_NEUTRAL.pkl")


def main(src: str, dst: str, iters: int = 150) -> None:
    joints = pickle.load(open(src, "rb"))["joints"]  # (T, 22, 3)
    n, dev = len(joints), torch.device("cuda")
    model = smplx.SMPL(model_path=SMPL_PKL, batch_size=n).to(dev)
    mean = h5py.File(config.SMPL_MEAN_FILE, "r")
    pose0 = torch.from_numpy(mean["pose"][:]).float().repeat(n, 1).to(dev)
    shape0 = torch.from_numpy(mean["shape"][:]).float().repeat(n, 1).to(dev)
    smplify = SMPLify3D(smplxmodel=model, batch_size=n, joints_category="AMASS", num_iters=iters, device=dev)
    kp = torch.tensor(joints, dtype=torch.float32, device=dev)
    _, fit_joints, pose, _, _, loss = smplify(pose0, shape0, torch.zeros(1, 3, device=dev), kp,
                                              conf_3d=torch.ones(22, device=dev))
    err = (fit_joints[:, :22] - (fit_joints[:, :1] - kp[:, :1]) - kp).norm(dim=-1).mean().item()
    np.savez(dst, poses=pose.detach().cpu().numpy(), root=joints[:, 0], fps=20.0, joints=joints)
    print(f"fitted {n} frames: mean joint error {err * 100:.1f} cm (root-aligned) -> {dst}")


if __name__ == "__main__":
    src, dst = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    os.chdir(MOTIONLCM)
    main(src, dst)
