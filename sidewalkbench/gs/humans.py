"""Pedestrians on a route: baked HUGS avatars (gs_assets/pedestrians/<identity>/), the walking ones
steered by RVO2 (ORCA).

A route's pedestrians (annotation.json): a static one stands where it was placed (anywhere), facing its
annotated yaw; a dynamic one spawns on the route centerline and walks the route centerline back and forth, first toward the start,
turning around immediately at each end. Its preferred velocity points LOOKAHEAD_M ahead along
the centerline; ORCA turns that into a collision-free velocity against the walkable-area boundary
(gs_assets/<scene>/walkable.npz, sidewalkbench/gs/annotate/bev_walkable.py, within CORRIDOR_M of the routes), the
scene's cones (their footprints), the static pedestrians, the other walkers and the robot. The
robot is an RVO2 agent of radius ROBOT_RADIUS_M (as PG) overwritten from the sim every tick; it does not
avoid, and RVO2 is reciprocal (each side takes half), so the humans may pass closer than the radii sum.

The walk cycle (sidewalkbench/gs/hugs_anim/bake_walk_cycle.py) plays continuously at its own pace whatever the
human's speed, turned to the walking heading and put on the mesh ground.
RVO2 bindings: the pyrvo2 dependency (Python-RVO2, pinned in pyproject.toml).
"""

from __future__ import annotations

import math

import numpy as np
import rvo2
import shapely
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from shapely.geometry.polygon import orient

from .annotation import HUMAN_RADIUS_M, pedestrian_npz
from .cones import cone_mesh
from .sampler import CenterlineTracker

MAX_SPEED = 1.8
LOOKAHEAD_M = 2.5
TURN_M = 0.3            # turn around this close to a route end (arc length)
ROBOT_RADIUS_M = 0.5    # Coco footprint (0.76 x 0.61 m) as a circle covering its corners (as PG)
ROBOT_FOOTPRINT_M = (0.76, 0.61)  # Coco length x width (pg_vis ROBOT_CFG bbox), centred on the root
MIN_AREA_M2 = 1.0       # walkable areas smaller than this are dropped (unreachable specks)
CORRIDOR_M = 2.0        # the walkable area is kept only this close to the routes' centerlines
NEIGHBOR_DIST_M, MAX_NEIGHBORS, TIME_HORIZON_S, TIME_HORIZON_OBST_S = 6.0, 10, 3.0, 1.0


def walkable_obstacles(npz_path: str, centerlines) -> list[np.ndarray]:
    """RVO2 obstacles of a walkable grid near the routes: the walkable area within CORRIDOR_M of
    ``centerlines`` (the humans never go further; the rest would only cost RVO2 preprocessing),
    as each area's outline clockwise (keeps agents inside) and its unwalkable holes
    counter-clockwise (solid), simplified to within one grid cell (drops the pixel staircase)."""
    d = np.load(npz_path)
    (ox, oy), cell = d["origin"], float(d["cell"])
    e = np.diff(np.pad(d["walkable"].astype(np.int8), ((0, 0), (1, 1))), axis=1)  # one box per run of walkable cells
    (j, a), (_, b) = np.nonzero(e == 1), np.nonzero(e == -1)                       # keeps the union cheap
    boxes = shapely.box(ox + a * cell, oy + j * cell, ox + b * cell, oy + (j + 1) * cell)
    corridor = shapely.union_all([shapely.buffer(shapely.linestrings(c), CORRIDOR_M) for c in centerlines])
    out = []
    for p in shapely.get_parts(shapely.intersection(shapely.union_all(boxes, grid_size=cell / 100), corridor)):
        if p.geom_type == "Polygon" and p.area >= MIN_AREA_M2:
            p = orient(p.simplify(cell, preserve_topology=True), sign=-1.0)  # exterior CW, holes CCW
            out += [np.asarray(r.coords)[:-1] for r in (p.exterior, *p.interiors)]
    return out


def touches_robot(human_xy, robot_xy, robot_yaw: float, footprint=ROBOT_FOOTPRINT_M) -> bool:
    """Does a human (circle of HUMAN_RADIUS_M) overlap the robot's footprint rectangle?"""
    dx, dy = human_xy[0] - robot_xy[0], human_xy[1] - robot_xy[1]
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    u, v = abs(c * dx + s * dy), abs(-s * dx + c * dy)  # human centre in the robot frame (folded)
    return math.hypot(max(u - footprint[0] / 2, 0.0), max(v - footprint[1] / 2, 0.0)) < HUMAN_RADIUS_M


def cone_footprints(cones: np.ndarray) -> list[np.ndarray]:
    """(K, 7) settled cone poses (x, y, z, qx, qy, qz, qw) -> K counter-clockwise xy polygons."""
    v = cone_mesh().vertices
    out = []
    for c in np.asarray(cones).reshape(-1, 7):
        xy = (Rotation.from_quat(c[3:]).apply(v) + c[:3])[:, :2]
        out.append(xy[ConvexHull(xy).vertices])  # scipy 2D hulls are counter-clockwise
    return out


def merge_gaussians(parts) -> dict | None:
    """Several Gaussian sets (``BakedWalker.gaussians`` dicts) as one, or None for none."""
    parts = [p for p in parts if p is not None]
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]} if parts else None


class BakedWalker:
    """A baked HUGS walk cycle (bake_walk_cycle.py output), placed as world-frame Gaussians."""

    def __init__(self, npz_path: str, device="cuda"):
        d = np.load(npz_path)
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)  # noqa: E731
        self.xyz, self.quat = t(d["xyz"]), t(d["rotq_wxyz"])
        self.scales = t(d["scales"]).abs().clamp(min=1e-6)  # the bake has signed / zero scales; only |s| shapes a Gaussian
        self.quat = self.quat / self.quat.norm(dim=-1, keepdim=True)
        self.opacity, self.rgb = t(d["opacity"]), t(d["rgb"])
        self.cycle_s = float(d["cycle_s"])
        self.speed = float(d["stride_m"]) / self.cycle_s  # the clip's own walking speed

    def gaussians(self, t_s: float, pos: np.ndarray, yaw: float) -> dict:
        """The walker ``t_s`` seconds into its walk, standing at ``pos`` (x, y, ground z) facing ``yaw``."""
        f = int(t_s / self.cycle_s * len(self.xyz)) % len(self.xyz)
        q, c, s = self.quat[f], math.cos(yaw), math.sin(yaw)
        rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=self.xyz.device)
        qw, qz = math.cos(yaw / 2), math.sin(yaw / 2)  # world rotation = yaw about z * baked (wxyz)
        quat = torch.stack([qw * q[:, 0] - qz * q[:, 3], qw * q[:, 1] - qz * q[:, 2],
                            qw * q[:, 2] + qz * q[:, 1], qw * q[:, 3] + qz * q[:, 0]], -1)
        return {"means": self.xyz[f] @ rz.T + torch.tensor(pos, dtype=torch.float32, device=rz.device),
                "quats": quat, "scales": self.scales[f], "opacities": self.opacity, "colors": self.rgb}


class HumanCrowd:
    """Each env's pedestrians on its route: the static (standing) ones and the dynamic (walking) ones,
    one RVO2 simulator per env, rebuilt for each route."""

    def __init__(self, walkable_npz: str | None, num_envs: int, dt: float, ground_z, cones: np.ndarray, routes,
                 device="cuda", robot_radius=ROBOT_RADIUS_M):
        """``walkable_npz``: the scene's walkable grid (None: no dynamic pedestrians, no walkable boundary);
        ``cones`` (K, 7): the settled cone poses (scene-level); ``routes``: the routes of the run (the walkable
        area is kept within CORRIDOR_M of them)."""
        self.obstacles = walkable_obstacles(walkable_npz, [r.centerline_xy for r in routes]) if walkable_npz else []
        self.cone_obstacles = cone_footprints(cones)  # one per cone; a route avoids only the cones it uses
        self.device = device
        self.robot_radius = robot_radius
        self.ground_z, self.dt, self.n = ground_z, dt, num_envs
        self._avatars: dict = {}  # npz path -> BakedWalker
        self.sims, self.robot = [None] * num_envs, [None] * num_envs  # per env: RVO2 simulator, its robot agent
        self.walkers = [[] for _ in range(num_envs)]   # per env: one dict per dynamic pedestrian
        self.static = [[] for _ in range(num_envs)]    # per env: (x, y) of the static pedestrians
        self._static_g = [None] * num_envs              # per env: the static pedestrians' Gaussians

    def avatar(self, identity: str, kind: str) -> BakedWalker:
        path = pedestrian_npz(identity, kind)
        if path not in self._avatars:
            self._avatars[path] = BakedWalker(path, device=self.device)
        return self._avatars[path]

    def reset(self, i: int, route, cones_on=None) -> None:
        """New route on env i: its pedestrians at their spawn points (the dynamic ones walk toward the start)."""
        statics = [np.asarray(shapely.Point(*p.xy).buffer(HUMAN_RADIUS_M, 4).exterior.coords)[-2::-1]  # CCW: solid
                   for p in route.static_pedestrians]
        sim = rvo2.PyRVOSimulator(self.dt, NEIGHBOR_DIST_M, MAX_NEIGHBORS, TIME_HORIZON_S, TIME_HORIZON_OBST_S,
                                  HUMAN_RADIUS_M, MAX_SPEED)
        cones = [c for c, on in zip(self.cone_obstacles, np.ones(len(self.cone_obstacles), bool) if cones_on is None
                                       else cones_on) if on]  # the route's cones (SceneAnnotation.route_cones)
        for poly in self.obstacles + cones + statics:
            sim.addObstacle([tuple(map(float, q)) for q in poly])
        sim.processObstacles()
        self.robot[i] = sim.addAgent((1e4, 1e4))  # overwritten from the sim every tick
        sim.setAgentRadius(self.robot[i], self.robot_radius)
        tr = CenterlineTracker(route.centerline_xy)
        self.walkers[i] = []
        for p in route.dynamic_pedestrians:
            self.walkers[i].append({"agent": sim.addAgent(tuple(map(float, p.xy))), "track": tr,
                                    "ends": (0.0, tr.length), "heading": -1.0, "clock": 0.0, "yaw": p.yaw_rad,
                                    "walker": self.avatar(p.identity, "walk"), "z": self.ground_z(*p.xy)})
        self.sims[i] = sim
        self.static[i] = [tuple(p.xy) for p in route.static_pedestrians]
        self._static_g[i] = merge_gaussians(
            [self.avatar(p.identity, "stand").gaussians(0.0, np.array([*p.xy, self.ground_z(*p.xy)]), p.yaw_rad)
             for p in route.static_pedestrians])

    def step(self, robot_xy: np.ndarray, robot_vel: np.ndarray, paused=None) -> None:
        """Advance every env's dynamic pedestrians one tick; robot_xy / robot_vel (N, 2) in the scene frame.
        ``paused`` (N bools): those envs' pedestrians are held still and their clock does not advance."""
        for i, sim in enumerate(self.sims):
            if sim is None or not self.walkers[i] or (paused is not None and paused[i]):
                continue
            sim.setAgentPosition(self.robot[i], tuple(map(float, robot_xy[i])))
            sim.setAgentVelocity(self.robot[i], tuple(map(float, robot_vel[i])))
            for w in self.walkers[i]:
                tr, (s0, s1) = w["track"], w["ends"]
                p = np.array(sim.getAgentPosition(w["agent"]))
                s = tr.project(*p)
                if (w["heading"] < 0 and s <= s0 + TURN_M) or (w["heading"] > 0 and s >= s1 - TURN_M):
                    w["heading"] = -w["heading"]  # reached an end: turn around immediately
                target, _ = tr.point_at(min(max(s + w["heading"] * LOOKAHEAD_M, s0), s1))
                d = target - p
                n = float(np.linalg.norm(d))
                sim.setAgentPrefVelocity(w["agent"], tuple(map(float, d / n * w["walker"].speed)) if n > 1e-6 else (0.0, 0.0))
            sim.doStep()
            for w in self.walkers[i]:
                v = sim.getAgentVelocity(w["agent"])
                if math.hypot(*v) > 0.1:
                    w["yaw"] = math.atan2(v[1], v[0])
                w["clock"] += self.dt
                w["z"] = self.ground_z(*sim.getAgentPosition(w["agent"]))

    def xy(self, i: int) -> list[tuple[float, float]]:
        """Env i's pedestrians' positions (scene frame): the dynamic ones, then the static ones."""
        return [self.sims[i].getAgentPosition(w["agent"]) for w in self.walkers[i]] + self.static[i]

    def gaussians(self, i: int) -> dict | None:
        """Env i's pedestrians as world Gaussians (scene frame), or None when it has none."""
        if self.sims[i] is None:
            return None
        return merge_gaussians([self._static_g[i]] + [
            w["walker"].gaussians(w["clock"], np.array([*self.sims[i].getAgentPosition(w["agent"]), w["z"]]), w["yaw"])
            for w in self.walkers[i]])
