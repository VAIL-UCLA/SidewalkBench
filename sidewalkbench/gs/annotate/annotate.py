#!/usr/bin/env python3
"""Scene annotator: routes, cones and pedestrians on a top-down view of a scene's splat (sidewalkbench env).

    python sidewalkbench/gs/annotate/annotate.py [--port 8765] [--no_browser]

Serves a local web page (sidewalkbench/gs/annotate/annotate_ui.html) and opens it in the browser. Every edit is saved
to gs_assets/<scene>/annotation.json at once.

The view is an orthographic top-down render of gs_assets/<scene>/gs.ply: every Gaussian centre in its
colour, per PX pixel the highest one below HEIGHT_CLIP_M above the ground (the mesh), so tree canopies
do not hide the sidewalk. Over it: the scene centerlines (centerlines.json), the routes, cones and the
selected route's pedestrians.

* route: click the start, then the direction the robot faces, then the goal. Start and goal stay where clicked;
  each is projected onto the nearest centerline or bridge (the connector between two centerlines, e.g. a
  crosswalk ramp), and the route centerline is the shortest way between the projections along the centerlines and
  bridges (``Network``), on a loop setting off the way the robot faces (that picks the way round). Its length is
  the route length; progress and success are measured along it.
* static pedestrian: click its spot (anywhere, kept as clicked), then the direction it faces. dynamic
  pedestrian: click its spot (snapped to the selected route's centerline); it faces down it toward the start.
* route region: the walkable surface the selected route runs on (a polygon, ``region_xy``); the eval's off-sidewalk
  check is "more than 1 m outside it". Click its corners (click the first one or press Enter to close), or
  Auto-fill it from the semantic grid (annotate/auto_region.py) and fix it: drag a corner, click an edge to add one,
  Shift+click a corner to remove it.
After adding cones, settle them: sidewalkbench/gs/annotate/settle_cones.sh <scene> (or prepare_scene.sh).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData
from scipy import ndimage

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCENES = os.path.join(REPO, "gs_assets")
PEDESTRIANS = os.path.join(SCENES, "pedestrians")
UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotate_ui.html")
PX = 0.05             # view pixel (m); coarser for scenes wider than MAX_PX pixels
MAX_PX = 4096
HEIGHT_CLIP_M = 2.0   # Gaussians higher than this above the ground are not drawn
GROUND_CELL_M = 1.0   # ground = the lowest mesh vertex per cell this size
MIN_OPACITY = 0.3
FILL_PX = 2           # empty pixels this close to a drawn one take its colour
SH_C0 = 0.28209479177387814
CONE_GLB = os.path.join(REPO, "gs_assets", "cone", "traffic_cone.glb")  # the eval's cone (sidewalkbench/gs/cones.py)


def rnd(a) -> list:
    return np.round(np.asarray(a, dtype=float), 4).tolist()


def bev_image(sdir: str) -> tuple[np.ndarray, tuple]:
    """The top-down view of the splat: (H, W, 3) image (row 0 = min y) and its extent."""
    v = PlyData.read(os.path.join(sdir, "gs.ply"))["vertex"]
    xyz = np.stack([np.asarray(v[k], dtype=np.float32) for k in "xyz"], 1)
    rgb = np.clip(np.stack([np.asarray(v[f"f_dc_{k}"], dtype=np.float32) for k in range(3)], 1) * SH_C0 + 0.5, 0, 1)
    keep = np.isfinite(xyz).all(1) & (1 / (1 + np.exp(-np.asarray(v["opacity"], dtype=np.float32))) > MIN_OPACITY)
    xyz, rgb = xyz[keep], rgb[keep]
    mv = np.asarray(trimesh.load(os.path.join(sdir, "mesh.ply"), process=False).vertices)
    lo, hi = mv[:, :2].min(0), mv[:, :2].max(0)
    gw, gh = (np.ceil((hi - lo) / GROUND_CELL_M).astype(int) + 1)
    ground = np.full((gh, gw), np.inf)
    gi = ((mv[:, :2] - lo) // GROUND_CELL_M).astype(int)
    np.minimum.at(ground, (gi[:, 1], gi[:, 0]), mv[:, 2])
    ground = ground[tuple(ndimage.distance_transform_edt(np.isinf(ground), return_distances=False, return_indices=True))]
    gi = ((xyz[:, :2] - lo) // GROUND_CELL_M).astype(int)
    keep = (gi >= 0).all(1) & (gi[:, 0] < gw) & (gi[:, 1] < gh)
    xyz, rgb, gi = xyz[keep], rgb[keep], gi[keep]
    keep = xyz[:, 2] < ground[gi[:, 1], gi[:, 0]] + HEIGHT_CLIP_M
    xyz, rgb = xyz[keep], rgb[keep]
    px = max(PX, float((hi - lo).max()) / MAX_PX)
    w, h = (np.ceil((hi - lo) / px).astype(int) + 1)
    pi = ((xyz[:, :2] - lo) / px).astype(int)
    keep = (pi[:, 0] < w) & (pi[:, 1] < h)
    xyz, rgb, pi = xyz[keep], rgb[keep], pi[keep]
    order = np.argsort(-xyz[:, 2])  # highest first: np.unique keeps each pixel's first
    lin, first = np.unique((pi[order, 1] * w + pi[order, 0]), return_index=True)
    img = np.full((h * w, 3), 0.15, dtype=np.float32)
    img[lin] = rgb[order][first]
    img, empty = img.reshape(h, w, 3), np.ones(h * w, dtype=bool)
    empty[lin] = False
    dist, idx = ndimage.distance_transform_edt(empty.reshape(h, w), return_indices=True)
    fill = (dist > 0) & (dist <= FILL_PX)  # an empty pixel takes its nearest drawn pixel's colour
    img[fill] = img[idx[0][fill], idx[1][fill]]
    return img, (lo[0], lo[0] + w * px, lo[1], lo[1] + h * px)


TOUCH_M = 0.5  # a centerline end this close to another centerline is joined to it


class Network:
    """A scene's centerlines as one graph: every vertex a node, edges along each line (closing edge on a closed
    one) and, at each connection in centerlines.json (e.g. a crosswalk and a sidewalk), a straight connector from
    the end of one line to the nearest vertex of the other (connected lines stop short of each other)."""

    def __init__(self, scene: str):
        d = json.load(open(os.path.join(SCENES, scene, "centerlines.json")))
        self.xy, self.line, segs, first, self.closed = [], [], [], {}, set()  # closed: vertices of loop centerlines
        for c in d["centerlines"]:
            xy, n0 = np.asarray(c["xy"], dtype=float), len(self.line)
            first[c["id"]] = (n0, len(xy), c["closed"])
            if c["closed"]:
                self.closed.update(range(n0, n0 + len(xy)))
            self.xy.append(xy)
            self.line += [c["id"]] * len(xy)
            segs += [(n0 + i, n0 + i + 1) for i in range(len(xy) - 1)] + ([(n0 + len(xy) - 1, n0)] if c["closed"] else [])
        self.xy = np.concatenate(self.xy)
        self.segs = np.array(segs)
        self.joins = []
        for a, b in d.get("connections") or []:  # the closest (end of one line, vertex of the other) pair
            cands = []
            for e, o in ((a, b), (b, a)):
                n0, n, closed = first[e]
                m0, m, _ = first[o]
                for j in (() if closed else (n0, n0 + n - 1)):
                    k = m0 + int(np.argmin(np.linalg.norm(self.xy[m0:m0 + m] - self.xy[j], axis=1)))
                    cands.append((float(np.linalg.norm(self.xy[k] - self.xy[j])), j, k))
            if cands:
                self.joins.append(min(cands)[1:])
        for e, (n0, n, closed) in first.items():  # line ends that touch another line join it (coco-gs lists only some)
            for j in (() if closed else (n0, n0 + n - 1)):
                for o, (m0, m, _) in first.items():
                    if o != e:
                        dd = np.linalg.norm(self.xy[m0:m0 + m] - self.xy[j], axis=1)
                        if dd.min() <= TOUCH_M:
                            self.joins.append((j, m0 + int(np.argmin(dd))))
        self.all_segs = np.vstack([self.segs, np.array(self.joins, dtype=int).reshape(-1, 2)])  # centerlines + connectors

    def project(self, p):
        """p's projection: the nearest point on a centerline or a bridge (connector) -> (segment (u, v), the point)."""
        A, B = self.xy[self.all_segs[:, 0]], self.xy[self.all_segs[:, 1]]
        ab = B - A
        t = np.clip(np.einsum("ij,ij->i", p - A, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
        q = A + t[:, None] * ab
        i = int(np.argmin(np.linalg.norm(q - p, axis=1)))
        return tuple(int(x) for x in self.all_segs[i]), q[i]

    def route(self, start, yaw: float, goal):
        """The shortest way along the network from the start's projection to the goal's. On a loop (the start on a
        closed centerline) it sets off the way ``yaw`` points when that can reach the goal: that picks the way round.
        -> (polyline from the start's projection to the goal's, the centerline ids passed)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        (u0, v0), a = self.project(np.asarray(start, dtype=float))
        (u1, v1), b = self.project(np.asarray(goal, dtype=float))
        n = len(self.xy)
        S, G = n, n + 1
        pts = np.vstack([self.xy, a, b])
        same = {u0, v0} == {u1, v1}  # start and goal on one segment
        ahead = np.dot(b - a, self.xy[v0] - self.xy[u0]) > 0  # the goal lies toward v0 from the start
        tang = self.xy[v0] - self.xy[u0]
        fwd = math.cos(yaw - math.atan2(tang[1], tang[0])) >= 0  # facing toward v0
        loop = u0 in self.closed and v0 in self.closed
        for strict in ((True, False) if loop else (False,)):
            e = [(u, v) for u, v in self.all_segs] + [(v, u) for u, v in self.all_segs] + [(u1, G), (v1, G)]
            if strict:  # set off the way the robot faces, without turning back along the start segment
                e.remove((u0, v0) if not fwd else (v0, u0))
            e += [(S, v0)] if (fwd or not strict) else []
            e += [(S, u0)] if (not fwd or not strict) else []
            if same and (not strict or ahead == fwd):
                e.append((S, G))
            e = np.unique(np.array(e), axis=0)  # coo_matrix would add up duplicate edges
            w = np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1) + 1e-9
            dist, pred = dijkstra(coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n + 2, n + 2)).tocsr(),
                                  indices=S, return_predecessors=True)
            if np.isfinite(dist[G]):
                break
        else:
            return None, []
        path, k = [], G
        while k != S:
            path.append(k)
            k = pred[k]
        path = [S] + path[::-1]
        xy = pts[path]
        keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
        ids = [self.line[k] for k in path if k < n]
        return xy[keep], [c for i, c in enumerate(ids) if i == 0 or c != ids[i - 1]]


def cone_radius() -> float:
    """The cone's footprint radius (m): the farthest point of its base from its axis (the glTF is Y-up)."""
    v = np.asarray(trimesh.load(CONE_GLB, force="mesh").vertices)
    return float(np.hypot(v[:, 0], v[:, 2]).max())


def scene_names() -> list[str]:
    return sorted(d for d in os.listdir(SCENES) if os.path.isfile(os.path.join(SCENES, d, "annotation.json")))


def identities() -> list[str]:
    return sorted(d for d in os.listdir(PEDESTRIANS) if all(
        os.path.isfile(os.path.join(PEDESTRIANS, d, f"{kind}.npz")) for kind in ("walk", "stand")))


_BEV: dict = {}  # scene -> (png bytes, extent)
_LOCK = threading.Lock()


def bev(scene: str):
    with _LOCK:
        if scene not in _BEV:
            img, extent = bev_image(os.path.join(SCENES, scene))
            alpha = (np.abs(img - 0.15).sum(-1) > 1e-6)  # empty pixels: transparent
            rgba = np.concatenate([img, alpha[..., None]], -1)[::-1]  # row 0 = max y
            buf = io.BytesIO()
            Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(buf, "PNG", compress_level=1)
            _BEV[scene] = (buf.getvalue(), [float(e) for e in extent])
        return _BEV[scene]


def walkable_png(scene: str) -> tuple[bytes, list]:
    """The semantic grid's walkable cells (what the off-sidewalk check is measured against) as a translucent green
    RGBA PNG (row 0 = max y) and its extent [x0, x1, y0, y1]."""
    d = np.load(os.path.join(SCENES, scene, "semantic_grid_0p1m.npz"))
    walk, org, res = d["walkable"].astype(bool)[::-1], d["origin_xy"].astype(float), float(d["resolution_m"][0])
    rgba = np.zeros((*walk.shape, 4), dtype=np.uint8)
    rgba[walk] = (74, 222, 128, 110)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    h, w = walk.shape
    return buf.getvalue(), [org[0], org[0] + w * res, org[1], org[1] + h * res]


def centerlines(scene: str) -> list:
    return json.load(open(os.path.join(SCENES, scene, "centerlines.json")))["centerlines"]


def make_route(scene: str, start, face, goal) -> dict:
    """A route from three clicks: start and goal kept where clicked, the robot facing the second click; the route
    centerline is the shortest way between their projections onto the nearest centerline or bridge
    (``Network.route``), and every calculation (length, progress, success) uses it."""
    net = Network(scene)
    p0, p1, p2 = (np.asarray(p, dtype=float) for p in (start, face, goal))
    yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    cl, ids = net.route(p0, yaw, p2)
    if cl is None:
        return {"error": "The goal's centerline is not connected to the start's."}
    if np.linalg.norm(np.diff(cl, axis=0), axis=1).sum() < 0.5:
        return {"error": "Start and goal project to the same spot."}
    return {"centerline_ids": ids, "start": {"xy": rnd(p0), "yaw_deg": round(math.degrees(yaw), 2)},
            "goal": {"xy": rnd(p2)}, "centerline_xy": rnd(cl)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def reply(self, body, ctype="application/json", code=200):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def scene(self, q) -> str:
        name = q.get("name", [""])[0]
        if name not in scene_names():
            raise ValueError(f"unknown scene {name!r}")
        return name

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self.reply(open(UI, "rb").read(), "text/html; charset=utf-8")
            elif u.path == "/api/scenes":
                self.reply({"scenes": scene_names(), "identities": identities(), "cone_radius_m": cone_radius()})
            elif u.path == "/api/scene":
                name = self.scene(q)
                self.reply({"annotation": json.load(open(os.path.join(SCENES, name, "annotation.json"))),
                            "centerlines": centerlines(name), "extent": bev(name)[1],
                            "connectors": [rnd(net.xy[[j, k]]) for net in [Network(name)] for j, k in net.joins]})
            elif u.path == "/api/bev.png":
                self.reply(bev(self.scene(q))[0], "image/png")
            elif u.path == "/api/walkable.png":
                self.reply(walkable_png(self.scene(q))[0], "image/png")
            elif u.path == "/api/walkable":
                self.reply({"extent": walkable_png(self.scene(q))[1]})
            else:
                self.reply({"error": "not found"}, code=404)
        except Exception as e:  # noqa: BLE001
            self.reply({"error": str(e)}, code=400)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            name = self.scene(q)
            if u.path == "/api/save":
                with open(os.path.join(SCENES, name, "annotation.json"), "w") as fh:
                    json.dump(body, fh, indent=1)
                self.reply({"ok": True})
            elif u.path == "/api/route":
                self.reply(make_route(name, body["start"], body["face"], body["goal"]))
            elif u.path == "/api/auto_region":
                if REPO not in sys.path:
                    sys.path.insert(0, REPO)
                from sidewalkbench.gs.annotate.auto_region import auto_region
                self.reply({"region_xy": auto_region(os.path.join(SCENES, name), body["centerline_xy"])})
            else:
                self.reply({"error": "not found"}, code=404)
        except Exception as e:  # noqa: BLE001
            self.reply({"error": str(e)}, code=400)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no_browser", action="store_true")
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[annotate] {url}  (Ctrl+C to quit)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
