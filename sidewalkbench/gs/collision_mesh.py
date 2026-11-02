"""Scene collision mesh + robot spawn height (pure trimesh/NumPy, no Isaac).

The scene's ``mesh.ply`` is the physics ground: it is authored later as one
invisible collision prim per env (see ``isaac_helpers.author_collision_mesh``),
and the robot is spawned on its surface, found by a downward raycast at the
route's start XY, plus the base-link height.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

ROBOT_BASE_LINK_HEIGHT_M = 0.27668

class CollisionMeshError(ValueError):
    pass

@dataclass
class CollisionMesh:
    """The scene collision mesh + the spawn-height lookup on it."""

    mesh: trimesh.Trimesh
    _ray: object = field(default=None, init=False, repr=False, compare=False)
    _top: float = field(default=0.0, init=False, repr=False, compare=False)  # ray start height, set with _ray

    def surface_z(self, x: float, y: float) -> float:
        """Collision-surface height under (x, y): the LOWEST downward ray hit.
        """
        if self._ray is None:
            self._ray = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)
            self._top = float(self.mesh.vertices[:, 2].max()) + 1.0
        locs, _, _ = self._ray.intersects_location([[float(x), float(y), self._top]], [[0.0, 0.0, -1.0]])
        if len(locs) == 0:
            raise CollisionMeshError(f"no collision mesh under xy=({x:.3f}, {y:.3f}): downward ray missed")
        return float(np.min(locs[:, 2]))

    def spawn_z(self, x: float, y: float, *, base_link_height_m: float = ROBOT_BASE_LINK_HEIGHT_M) -> float:
        """Robot base spawn Z = mesh surface under (x, y) + base-link height.
        """
        return self.surface_z(x, y) + float(base_link_height_m)


def load_collision_mesh(path: str | Path) -> CollisionMesh:
    p = Path(path).expanduser()
    if not p.is_file():
        raise CollisionMeshError(f"Collision mesh does not exist: {p}")
    mesh = trimesh.load(str(p), process=False, force="mesh")  # a multi-part file is flattened to one mesh
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise CollisionMeshError(f"Collision mesh is empty or not a mesh: {p}")
    return CollisionMesh(mesh=mesh)
