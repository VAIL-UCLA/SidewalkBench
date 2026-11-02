"""Isaac-side helpers for sidewalkbench.gs (imported only after AppLauncher).

Copied/adapted from the PG code:
* ``author_collision_mesh`` — authors the scene collision ``trimesh`` as a USD Mesh prim with
  collision enabled and made invisible (physics-only proxy). Adapted from
  sidewalkbench.pg.utils.math.create_prim_from_mesh.
* ``ConeSpawnerCfg`` — the annotated traffic cone (``cones.cone_mesh``) as a convex-hull rigid
  body: dynamic in ``sidewalkbench/gs/annotate/settle_cones.py``, kinematic (fixed at its settled pose) in the eval.
"""

from __future__ import annotations

from collections.abc import Callable

import isaaclab.sim as sim_utils
import numpy as np
import trimesh
from isaaclab.utils import configclass


def author_collision_mesh(prim_path: str, mesh: trimesh.Trimesh, *, physics_material=None):
    """Author ``mesh`` at ``prim_path`` as an invisible physics-only collision prim.

    Faithful to PG's ``create_prim_from_mesh`` (the PG terrain path) and to
    coco-gs's default ``--collision_mode triangle``: the ground is an EXACT
    triangle-mesh collider (no ``meshSimplification``, which decimates the sidewalk
    into a coarse hull), with a default ``RigidBodyMaterialCfg`` bound (static / dynamic
    friction 0.5; PG release's ground uses 1.0 / 1.0, multiply). Visuals come from the
    gsplat, so the prim is hidden.
    """
    from pxr import UsdGeom
    import isaaclab.sim.utils.prims as prim_utils

    top_prim = prim_utils.create_prim(prim_path, "Xform")
    prim = prim_utils.create_prim(
        f"{prim_path}/mesh",
        "Mesh",
        attributes={
            "points": np.asarray(mesh.vertices, dtype=np.float64),
            "faceVertexIndices": np.asarray(mesh.faces, dtype=np.int32).flatten(),
            "faceVertexCounts": np.asarray([3] * len(mesh.faces), dtype=np.int32),
            "subdivisionScheme": "none",
        },
    )
    # collision: exact triangle mesh (default approximation), like create_prim_from_mesh
    sim_utils.define_collision_properties(
        prim.GetPrimPath(), sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    )
    # friction material (default RigidBodyMaterialCfg: static / dynamic friction 0.5)
    if physics_material is None:
        physics_material = sim_utils.RigidBodyMaterialCfg()
    physics_material.func(f"{prim_path}/physicsMaterial", physics_material)
    sim_utils.bind_physics_material(prim.GetPrimPath(), f"{prim_path}/physicsMaterial")
    # invisible (gsplat provides the visuals)
    UsdGeom.Imageable(prim).MakeInvisible()
    return top_prim


def spawn_cone(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the cone mesh at ``prim_path``; the collider is its convex hull."""
    from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
    from isaaclab.sim.utils.stage import get_current_stage

    from .cones import cone_mesh

    _spawn_mesh_geom_from_mesh(prim_path, cfg, cone_mesh(), translation, orientation)
    if cfg.activate_contact_sensors:  # (the mesh spawner itself ignores this flag)
        sim_utils.schemas.activate_contact_sensors(prim_path)
    return get_current_stage().GetPrimAtPath(prim_path)


@configclass
class ConeSpawnerCfg(sim_utils.MeshCfg):
    func: Callable = spawn_cone
