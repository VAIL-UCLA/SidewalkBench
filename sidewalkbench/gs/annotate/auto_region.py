"""Automatic first guess at a route's walkable region (the annotator's "Auto-fill" seed; route_regions.py overlays).

Every STEP_M along the route centerline a ray is cast to each side (perpendicular to the local direction) through
the semantic grid's walkable cells (small enclosed non-walkable specks filled) and stops at the first non-walkable
cell; each side's half-width is the median of those within +-SMOOTH_M along the route, capped at CAP_M; the region
is the walkable cells on the rays out to those half-widths. Known weakness: a side branch (driveway, the curb
sidewalk at a crosswalk's end) wider than ~SMOOTH_M along the route is swept in -- the annotator edits that out.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import shapely
from scipy import ndimage

from sidewalkbench.gs.route_types import resample

STEP_M = 0.1         # ray spacing along the centerline
RAY_STEP_M = 0.05    # march step along a ray (half a grid cell)
SMOOTH_M = 1.5       # median window half-length along the route
CAP_M = 5.0          # max half-width per side
SPECK_M2 = 1.0       # enclosed non-walkable specks smaller than this are filled (manholes, grid noise)
TANGENT_M = 0.5      # local direction from the centerline +- this
BUFFER_M = 1.0       # off-sidewalk = more than this from the region (PG out_of_region buffer)


def walkable_grid(scene_dir: str) -> tuple[np.ndarray, np.ndarray, float]:
    d = np.load(os.path.join(scene_dir, "semantic_grid_0p1m.npz"))
    walk, org, res = d["walkable"].astype(bool), d["origin_xy"].astype(float), float(d["resolution_m"][0])
    lab, n = ndimage.label(~walk)
    if n:
        size = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * res * res
        border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        small = np.zeros(n + 1, dtype=bool)
        small[1:] = size < SPECK_M2
        small[border] = False
        walk = walk | small[lab]
    return walk, org, res


def route_region(cl: np.ndarray, walk: np.ndarray, org: np.ndarray, res: float) -> dict:
    """-> {region (bool grid), samples (K,2), raw (K,2) [left, right], smooth (K,2), s (K,)}."""
    p = resample(cl, STEP_M)
    p = np.vstack([p, cl[-1]])
    k = len(p)
    s = np.arange(k) * STEP_M
    lag = max(1, int(round(TANGENT_M / STEP_M)))
    t = p[np.minimum(np.arange(k) + lag, k - 1)] - p[np.maximum(np.arange(k) - lag, 0)]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    n = np.column_stack([-t[:, 1], t[:, 0]])  # left normal
    h, w = walk.shape

    def cell(xy):
        ij = np.floor((xy - org) / res).astype(int)
        ok = (ij[..., 0] >= 0) & (ij[..., 0] < w) & (ij[..., 1] >= 0) & (ij[..., 1] < h)
        return ij, ok

    d = np.arange(0.0, CAP_M + 3 * SMOOTH_M + RAY_STEP_M, RAY_STEP_M)  # rays a bit past the cap (for the median)
    raw = np.zeros((k, 2))
    for side, sgn in enumerate((1.0, -1.0)):
        pts = p[:, None, :] + sgn * d[None, :, None] * n[:, None, :]          # (K, D, 2)
        ij, ok = cell(pts)
        wk = np.zeros(ok.shape, dtype=bool)
        wk[ok] = walk[ij[..., 1][ok], ij[..., 0][ok]]
        first_bad = np.where((~wk).any(1), (~wk).argmax(1), len(d))
        raw[:, side] = d[np.maximum(first_bad - 1, 0)] * (first_bad > 0)
    win = max(1, int(round(SMOOTH_M / STEP_M)))
    smooth = np.stack([np.minimum(ndimage.median_filter(raw[:, j], size=2 * win + 1, mode="nearest"), CAP_M)
                       for j in range(2)], 1)
    region = np.zeros_like(walk)
    for side, sgn in enumerate((1.0, -1.0)):
        for i in range(k):
            dd = d[d <= smooth[i, side]]
            ij, ok = cell(p[i] + sgn * dd[:, None] * n[i])
            ij = ij[ok]
            keep = walk[ij[:, 1], ij[:, 0]]
            region[ij[keep, 1], ij[keep, 0]] = True
    return {"region": region, "samples": p, "raw": raw, "smooth": smooth, "s": s}


def region_polygon(region: np.ndarray, org: np.ndarray, res: float, simplify_m: float = 0.15) -> list[list[float]]:
    """The largest connected part of a region grid as a simple polygon (world xy, outer boundary, simplified)."""
    cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)  # (col, row) of boundary cells
    def largest(g):  # the largest polygon of a (possibly multi-part / collection) geometry
        if g.geom_type == "Polygon":
            return g
        if not g.geom_type.startswith(("Multi", "GeometryCollection")):
            return None  # lines / points left by make_valid
        parts = [q for p in shapely.get_parts(g) for q in [largest(p)] if q is not None]
        return max(parts, key=lambda q: q.area) if parts else None

    poly = largest(shapely.make_valid(shapely.Polygon(org + (c + 0.5) * res)))  # a contour can touch itself
    poly = largest(shapely.make_valid(shapely.Polygon(poly.exterior).simplify(simplify_m, preserve_topology=True)))
    return np.round(np.asarray(poly.exterior.coords)[:-1], 4).tolist()


def auto_region(scene_dir: str, centerline_xy) -> list[list[float]]:
    walk, org, res = walkable_grid(scene_dir)
    return region_polygon(route_region(np.asarray(centerline_xy, dtype=float), walk, org, res)["region"], org, res)
