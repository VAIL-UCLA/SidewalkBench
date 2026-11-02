"""Route sampler + per-route runtime helpers.

* :class:`RouteQueue` — hands out a scene's routes in annotation order, each
  exactly once (the route list IS the evaluation; no sampling).
* :class:`RouteRegion` — off-sidewalk detection against the route's annotated walkable region
  (``region_xy``, a polygon): the robot is off-sidewalk more than the margin (1 m) outside it.
* :class:`SemanticBounds` — the fallback for a route without a region: its walkable region on
  the semantic grid (flood-filled from the route centerline), with a margin; the robot is also
  off-sidewalk when more than ``OFF_ROUTE_M`` (2.5 m) from the route centerline.
* :class:`CenterlineTracker` — projects the robot XY onto the annotated centerline
  to give progress / remaining distance (drives the arrive + freeze checks).

These are pure NumPy/SciPy; they carry no Isaac dependency so they can be unit
tested standalone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from scipy import ndimage
from scipy.spatial import cKDTree

from .annotation import Route

OFF_ROUTE_M = 2.5  # off-sidewalk also when the robot is more than this from the route centerline


class RouteQueue:
    """Hands out a scene's routes in annotation order, each exactly once."""

    def __init__(self, routes: tuple[Route, ...] | list[Route]):
        if not routes:
            raise ValueError("RouteQueue requires at least one route.")
        self._routes = tuple(routes)
        self._next = 0

    def __len__(self) -> int:
        return len(self._routes)

    @property
    def remaining(self) -> int:
        return len(self._routes) - self._next

    def next(self) -> Route | None:
        """The next route in order, or None once every route has been handed out."""
        if self._next >= len(self._routes):
            return None
        route = self._routes[self._next]
        self._next += 1
        return route


class RouteRegion:
    """Off-sidewalk = robot XY more than ``margin_m`` outside the route's annotated walkable region (a polygon;
    PG's out_of_region(buffer=1.0) on the route's own surface)."""

    def __init__(self, region_xy: np.ndarray, *, margin_m: float = 1.0):
        poly = shapely.Polygon(np.asarray(region_xy, dtype=float))
        if not poly.is_valid:
            raise ValueError(f"route region polygon is invalid ({shapely.is_valid_reason(poly)})")
        shapely.prepare(poly)
        self.polygon = poly
        self.margin_m = float(margin_m)

    def is_off_sidewalk(self, x: float, y: float) -> bool:
        return self.polygon.distance(shapely.Point(float(x), float(y))) > self.margin_m


class SemanticBounds:
    """Off-sidewalk = robot XY more than ``margin_m`` from the nearest cell centre of the route's walkable
    region: the semantic grid's walkable cells (semantic_grid_0p1m.npz, coco-gs) flood-filled
    (4-connected) from the cells under the route centerline."""

    def __init__(self, grid_npz: str, centerline_xy: np.ndarray, *, margin_m: float = 1.0):
        d = np.load(grid_npz)
        walk, org, res = d["walkable"].astype(bool), d["origin_xy"].astype(float), float(d["resolution_m"][0])
        ij = np.floor((np.asarray(centerline_xy) - org) / res).astype(int)[:, ::-1]  # (row = y, col = x)
        ij = ij[(ij >= 0).all(1) & (ij < walk.shape).all(1)]
        labels, _ = ndimage.label(walk)
        seeds = np.unique(labels[ij[:, 0], ij[:, 1]])
        region = np.isin(labels, seeds[seeds > 0])
        if not region.any():
            raise ValueError(f"no walkable semantic-grid cell under the route centerline ({grid_npz})")
        self._tree = cKDTree(org + (np.argwhere(region)[:, ::-1] + 0.5) * res)
        self.margin_m = float(margin_m)

    def is_off_sidewalk(self, x: float, y: float) -> bool:
        return self._tree.query((float(x), float(y)), distance_upper_bound=self.margin_m)[0] > self.margin_m


class CenterlineTracker:
    """Projects XY onto the annotated centerline to give arc-length progress.

    Progress is arc length along the polyline from its first sample; route-level
    quantities (remaining to goal, completion) are defined by :class:`RouteRuntime`
    relative to the start/goal projections, as in coco-gs.
    """

    def __init__(self, centerline_xy: np.ndarray):
        pts = np.asarray(centerline_xy, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
            raise ValueError("centerline must be an (N>=2, 2) array")
        self.points = pts
        seg = np.diff(pts, axis=0)
        self._seg_len = np.linalg.norm(seg, axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(self._seg_len)])

    @property
    def length(self) -> float:
        return float(self._cum[-1])

    def project(self, x: float, y: float) -> float:
        """Arc-length progress of the nearest point on the polyline (clamped per segment)."""
        return self._nearest(x, y)[0]

    def distance(self, x: float, y: float) -> float:
        """Distance from XY to the nearest point on the polyline."""
        return self._nearest(x, y)[1]

    def _nearest(self, x: float, y: float) -> tuple[float, float]:
        p = np.array([float(x), float(y)], dtype=float)
        a = self.points[:-1]
        b = self.points[1:]
        ab = b - a
        denom = np.einsum("ij,ij->i", ab, ab)
        denom = np.where(denom > 1e-12, denom, 1.0)
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(proj - p, axis=1)
        i = int(np.argmin(d))
        return float(self._cum[i] + t[i] * self._seg_len[i]), float(d[i])

    def point_at(self, s: float) -> tuple[np.ndarray, float]:
        """XY and heading (rad, along increasing arc length) at arc length ``s`` (clamped to the polyline)."""
        i = int(np.clip(np.searchsorted(self._cum, s, side="right") - 1, 0, len(self._seg_len) - 1))
        d = self.points[i + 1] - self.points[i]
        t = float(np.clip((s - self._cum[i]) / max(self._seg_len[i], 1e-9), 0.0, 1.0))
        return self.points[i] + t * d, float(np.arctan2(d[1], d[0]))


@dataclass
class RouteRuntime:
    """A route bundled with the per-env helpers the checks consume.

    Success is defined by projecting the start/end points AND the robot onto the
    annotated centerline: the robot has arrived when its centerline-progress is
    within ``success_tolerance_m`` (default 0.1 m) of the goal's centerline-progress.
    """

    route: Route
    bounds: RouteRegion | SemanticBounds | None  # None = no off-sidewalk check (always in bounds)
    centerline: CenterlineTracker
    start_progress_m: float
    goal_progress_m: float
    success_tolerance_m: float = 0.1

    @classmethod
    def from_route(
        cls,
        route: Route,
        semantic_grid: str | None,
        *,
        off_sidewalk_margin_m: float = 1.0,
        success_tolerance_m: float = 0.1,
    ) -> "RouteRuntime":
        """Off-sidewalk against the route's annotated region (``route.region_xy``), else the semantic grid
        (``semantic_grid``: the scene's semantic_grid_0p1m.npz); ``semantic_grid`` None = no off-sidewalk check."""
        centerline = CenterlineTracker(route.centerline_xy)
        start_progress_m = centerline.project(*route.start_xy)
        goal_progress_m = centerline.project(*route.goal_xy)
        return cls(
            route=route,
            bounds=None if semantic_grid is None
            else RouteRegion(route.region_xy, margin_m=off_sidewalk_margin_m) if route.region_xy is not None
            else SemanticBounds(semantic_grid, route.centerline_xy, margin_m=off_sidewalk_margin_m),
            centerline=centerline,
            start_progress_m=float(start_progress_m),
            goal_progress_m=float(goal_progress_m),
            success_tolerance_m=float(success_tolerance_m),
        )

    def robot_progress_m(self, x: float, y: float) -> float:
        """Robot XY projected onto the centerline (arc-length progress)."""
        return self.centerline.project(x, y)

    def reached_goal(self, x: float, y: float) -> bool:
        """Arrived iff robot progress >= goal progress - success_tolerance_m (PG ``arrive``: one-sided, so a
        check that comes after the robot passed the goal still counts it)."""
        return self.robot_progress_m(x, y) >= self.goal_progress_m - self.success_tolerance_m

    def is_off_sidewalk(self, x: float, y: float) -> bool:
        """More than the margin outside the route's annotated region (RouteRegion); without one, off the route's
        walkable region on the semantic grid (SemanticBounds) or more than OFF_ROUTE_M from its centerline."""
        if self.bounds is None:
            return False
        if isinstance(self.bounds, RouteRegion):
            return self.bounds.is_off_sidewalk(x, y)
        return self.bounds.is_off_sidewalk(x, y) or self.centerline.distance(x, y) > OFF_ROUTE_M
