"""Scene annotation loader (scene format v3).

``gs_assets/<scene>/annotation.json`` holds scene-level **cones** and **routes**. A route is a
start pose and a goal on the scene's centerlines (centerlines.json), with its own centerline (the way
from the start to the goal along the centerlines and their connections, so arc-length progress
increases start -> goal) and its pedestrians: static (standing) and dynamic (walking), each a
spawn point and an identity (gs_assets/pedestrians/<identity>/): a static one stands where it was placed
(anywhere), facing its yaw_deg; a dynamic one spawns on the route centerline. A route may carry its
walkable region (``region_xy``, a polygon drawn in the annotator): the surface the route runs on; off-sidewalk
is then measured against it (sidewalkbench/gs/sampler.py RouteRegion).
Written by sidewalkbench/gs/annotate/annotate.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 3
MODES = ("short_horizon", "long_horizon", "cocoa")  # cocoa: generated on a cocoa capture's trajectory (sidewalkbench/gs/annotate/import_cocoa.py)
HUMAN_RADIUS_M = 0.3  # a pedestrian's footprint (circle)
CONE_ROUTE_RADIUS_M = 3.0  # a route uses the (scene-level) cones within this of its centerline
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENES_DIR = os.path.join(REPO, "gs_assets")
PEDESTRIANS_DIR = os.path.join(SCENES_DIR, "pedestrians")  # <identity>/{walk,stand}.npz (sidewalkbench/gs/hugs_anim/bake_walk_cycle.py)


class AnnotationError(ValueError):
    """Raised when an annotation file is missing or malformed."""


def _xy(value: Any, *, label: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise AnnotationError(f"{label} must be an [x, y] pair, got {value!r}")
    return np.array([float(value[0]), float(value[1])], dtype=float)


@dataclass(frozen=True)
class Pedestrian:
    xy: np.ndarray   # (2,) where it stands (static: anywhere) / spawns (dynamic: on the route centerline)
    identity: str    # gs_assets/pedestrians/<identity>/
    yaw_rad: float   # static: the annotated facing; dynamic: down the route centerline toward the start


@dataclass(frozen=True)
class Route:
    route_id: str
    mode: str                  # "short_horizon" | "long_horizon" | "cocoa"
    centerline_ids: tuple[str, ...]  # the scene centerlines the route passes along, in order
    start_xy: np.ndarray       # (2,)
    start_yaw_rad: float
    goal_xy: np.ndarray        # (2,)
    centerline_xy: np.ndarray  # (N, 2) start -> goal
    static_pedestrians: tuple[Pedestrian, ...] = ()
    dynamic_pedestrians: tuple[Pedestrian, ...] = ()
    region_xy: np.ndarray | None = None  # (M, 2) polygon: the route's walkable region (None = not annotated)


@dataclass(frozen=True)
class SceneAnnotation:
    scene: str
    routes: tuple[Route, ...]
    cones: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))  # (K, 7) settled x, y, z, qx, qy, qz, qw

    def route_cones(self, route: Route) -> np.ndarray:
        """(K,) bool: the cones ``route`` uses -- stored scene-level, but at runtime only those whose settled
        position is within CONE_ROUTE_RADIUS_M of the route centerline are present (rendered, colliders, obstacles)."""
        cl = np.asarray(route.centerline_xy, dtype=float)
        a, ab = cl[:-1], np.diff(cl, axis=0)
        out = np.zeros(len(self.cones), dtype=bool)
        for k, c in enumerate(self.cones[:, :2]):
            t = np.clip(np.einsum("ij,ij->i", c - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
            out[k] = np.linalg.norm(a + t[:, None] * ab - c, axis=1).min() <= CONE_ROUTE_RADIUS_M
        return out


def pedestrian_npz(identity: str, kind: str) -> str:
    """The baked avatar of ``identity``: kind "walk" (dynamic) or "stand" (static)."""
    return os.path.join(PEDESTRIANS_DIR, identity, f"{kind}.npz")


def _pedestrians(items, cl: np.ndarray, kind: str, label: str) -> tuple[Pedestrian, ...]:
    out = []
    for k, p in enumerate(items or []):
        ident = str(p.get("identity") or "")
        if not os.path.isfile(pedestrian_npz(ident, kind)):
            raise AnnotationError(f"{label}[{k}]: no baked {kind} for identity {ident!r} ({pedestrian_npz(ident, kind)})")
        xy = _xy(p.get("xy"), label=f"{label}[{k}].xy")
        if kind == "stand":
            if p.get("yaw_deg") is None:
                raise AnnotationError(f"{label}[{k}]: yaw_deg is required")
            yaw = float(np.radians(float(p["yaw_deg"])))
        else:  # down the route centerline toward the start (the oncoming robot)
            i = int(np.clip(np.argmin(np.linalg.norm(cl - xy, axis=1)), 1, len(cl) - 1))
            yaw = float(np.arctan2(*(cl[i - 1] - cl[i])[::-1]))
        out.append(Pedestrian(xy=xy, identity=ident, yaw_rad=yaw))
    return tuple(out)


def _route_from_json(item: dict[str, Any], k: int) -> Route:
    rid = str(item.get("id") or f"route_{k + 1}")
    mode = str(item.get("mode") or "short_horizon")
    if mode not in MODES:
        raise AnnotationError(f"route {rid}: mode {mode!r} not in {MODES}")
    start, goal = item.get("start") or {}, item.get("goal") or {}
    if start.get("yaw_deg") is None:
        raise AnnotationError(f"route {rid}: start.yaw_deg is required")
    cl = np.asarray(item.get("centerline_xy") or [], dtype=float).reshape(-1, 2)
    if len(cl) < 2:
        raise AnnotationError(f"route {rid}: centerline_xy needs >= 2 points")
    region = item.get("region_xy")
    if region is not None:
        region = np.asarray(region, dtype=float).reshape(-1, 2)
        if len(region) < 3:
            raise AnnotationError(f"route {rid}: region_xy needs >= 3 points")
    return Route(
        route_id=rid, mode=mode,
        centerline_ids=tuple(item.get("centerline_ids") or [item.get("centerline_id")]),  # (cocoa routes: one id)
        start_xy=_xy(start.get("xy"), label=f"route {rid}.start.xy"),
        start_yaw_rad=float(np.radians(float(start["yaw_deg"]))),
        goal_xy=_xy(goal.get("xy"), label=f"route {rid}.goal.xy"),
        centerline_xy=cl,
        static_pedestrians=_pedestrians(item.get("static_pedestrians"), cl, "stand", f"route {rid}.static_pedestrians"),
        dynamic_pedestrians=_pedestrians(item.get("dynamic_pedestrians"), cl, "walk", f"route {rid}.dynamic_pedestrians"),
        region_xy=region,
    )


def load_scene_annotation(annotation_path: str | Path, *, scene_name: str | None = None) -> SceneAnnotation:
    """Parse a scene format v3 ``annotation.json`` into a :class:`SceneAnnotation`."""
    path = Path(annotation_path).expanduser()
    if not path.is_file():
        raise AnnotationError(f"Annotation file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("version", 0)) != FORMAT_VERSION:
        raise AnnotationError(f"{path}: format version {data.get('version')} (expected {FORMAT_VERSION})")
    routes = tuple(_route_from_json(r, k) for k, r in enumerate(data.get("routes") or []))
    if not routes:
        raise AnnotationError(f"Annotation has no routes: {path}")
    if len({r.route_id for r in routes}) != len(routes):
        raise AnnotationError(f"Annotation has duplicate route ids: {path}")
    cones = data.get("cones") or []
    unsettled = [c.get("id", k) for k, c in enumerate(cones) if "settled" not in c]
    if unsettled:
        raise AnnotationError(f"{path}: cone(s) {unsettled} have no settled pose; run "
                              "sidewalkbench/gs/annotate/settle_cones.sh on this scene")
    cones = np.array([c["settled"]["xyz"] + c["settled"]["quat_xyzw"] for c in cones], dtype=float).reshape(-1, 7)
    return SceneAnnotation(scene=str(scene_name or data.get("scene") or path.parent.name), routes=routes, cones=cones)
