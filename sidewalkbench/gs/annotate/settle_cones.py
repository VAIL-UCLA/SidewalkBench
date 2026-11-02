#!/usr/bin/env python3
"""Drop a scene's annotated cones onto its mesh.ply once and store where they come to rest.

    sidewalkbench/gs/_python.sh-activated python sidewalkbench/gs/annotate/settle_cones.py <scene> \
        [--max_shift_m 0.2] [--drop_m 0.02]

Each cone (annotation.json ``cones[k]``: ``xy``, ``yaw_deg``) is spawned as a dynamic rigid
body (the cone mesh's convex hull, 3 kg) ``--drop_m`` above the highest mesh point under its
base and simulated until every cone is at rest. Its resting pose (base-center frame) is written back as
``cones[k].settled = {"xyz": [...], "quat_xyzw": [...]}``; the eval places the cones there,
fixed. Nothing is written, and the exit code is 1, if a cone ends more than ``--max_shift_m``
(horizontally) from its marked xy, or the cones never come to rest.
"""

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("scene", help="scene folder name under gs_assets/")
parser.add_argument("--max_shift_m", type=float, default=0.2, help="max horizontal drift from the marked xy")
parser.add_argument("--drop_m", type=float, default=0.02, help="spawn height above the mesh under the base")
parser.add_argument("--max_time_s", type=float, default=10.0, help="give up if not at rest by then")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402

from sidewalkbench.gs.collision_mesh import load_collision_mesh  # noqa: E402
from sidewalkbench.gs.isaac_helpers import ConeSpawnerCfg, author_collision_mesh  # noqa: E402

CONE_MASS_KG = 3.0  # a 28" traffic cone
DT = 0.005
REST_S = 0.5  # settled = over the last REST_S every cone moved < 0.5 mm and turned < 0.1 deg
BASE_HALF_M = 0.25  # the cone's square base plate is 0.5 x 0.5 m


def main() -> int:
    scene_dir = os.path.join(REPO_ROOT, "gs_assets", args.scene)
    ann_path = os.path.join(scene_dir, "annotation.json")
    with open(ann_path, encoding="utf-8") as fh:
        data = json.load(fh)
    cones = data.get("cones") or []
    if not cones:
        print(f"[settle] {args.scene}: no cones")
        return 0
    cmesh = load_collision_mesh(os.path.join(scene_dir, "mesh.ply"))

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT))
    author_collision_mesh("/World/Scene", cmesh.mesh)
    cfg = ConeSpawnerCfg(
        # damping only makes the cones come to rest faster (no rocking on the mesh triangles)
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, max_depenetration_velocity=0.5,
                                                     linear_damping=1.0, angular_damping=5.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=CONE_MASS_KG),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    g = np.linspace(-BASE_HALF_M, BASE_HALF_M, 5)
    for k, c in enumerate(cones):
        x, y = c["xy"]
        yaw = math.radians(float(c.get("yaw_deg") or 0.0))
        cy, sy = math.cos(yaw), math.sin(yaw)
        top = max(cmesh.surface_z(x + u * cy - v * sy, y + u * sy + v * cy) for u in g for v in g)
        cfg.func(f"/World/Cone_{k}", cfg, translation=(x, y, top + args.drop_m),
                 orientation=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
    view = RigidObject(RigidObjectCfg(prim_path="/World/Cone_.*", spawn=None))
    sim.reset()

    window = int(REST_S / DT)
    history = []
    for step in range(int(args.max_time_s / DT)):
        sim.step()
        view.update(DT)
        history = (history + [(torch.as_tensor(view.data.root_link_pos_w).clone(),
                               torch.as_tensor(view.data.root_link_quat_w).clone())])[-window:]
        if len(history) == window:
            (p0, q0), (p1, q1) = history[0], history[-1]
            moved = torch.linalg.norm(p1 - p0, dim=-1).max().item()
            turned = math.degrees(2 * math.acos(min(1.0, (q0 * q1).sum(-1).abs().min().item())))
            if moved < 5e-4 and turned < 0.1:
                break
    else:
        print(f"[settle] ERROR {args.scene}: cones not at rest after {args.max_time_s} s "
              f"(still moving {moved * 1000:.1f} mm / turning {turned:.2f} deg per {REST_S} s); nothing written")
        return 1
    print(f"[settle] {args.scene}: {len(cones)} cone(s) at rest after {(step + 1) * DT:.2f} s")

    pos = torch.as_tensor(view.data.root_link_pos_w).cpu().numpy().astype(float)
    quat = torch.as_tensor(view.data.root_link_quat_w).cpu().numpy().astype(float)  # (x, y, z, w)
    errors = []
    for k, c in enumerate(cones):
        shift = float(np.hypot(*(pos[k, :2] - np.asarray(c["xy"], dtype=float))))
        qx, qy = quat[k, 0], quat[k, 1]
        tilt = math.degrees(math.acos(float(np.clip(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0))))
        label = c.get("id", f"C{k + 1}")
        print(f"[settle]   {label}: shift {shift * 100:.1f} cm, tilt {tilt:.1f} deg")
        if shift > args.max_shift_m:
            errors.append(f"{label} moved {shift * 100:.1f} cm from its marked xy (limit {args.max_shift_m * 100:.1f} cm)")
        c["settled"] = {"xyz": pos[k].tolist(), "quat_xyzw": quat[k].tolist()}
    if errors:
        print(f"[settle] ERROR {args.scene}: " + "; ".join(errors) + "; nothing written")
        return 1
    with open(ann_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    print(f"[settle] wrote settled poses -> {ann_path}")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        app.close(exit_code=code)
