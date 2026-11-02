"""Crosswalk transitions built from the road union, including incidental crossings."""

import numpy as np
import shapely
import trimesh


def _polygon_parts(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _polygon_parts(part)


def _refine_triangles(vertices, faces, max_edge=0.25):
    """Split long edges on both adjacent faces, keeping a conforming triangulation."""
    for _ in range(16):
        edges, inverse = np.unique(
            np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1),
            axis=0,
            return_inverse=True,
        )
        split = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1) > max_edge
        if not split.any():
            return vertices, faces
        midpoint = np.full(len(edges), -1, dtype=np.int64)
        midpoint[split] = len(vertices) + np.arange(split.sum())
        vertices = np.concatenate((vertices, vertices[edges[split]].mean(axis=1)))
        mids = midpoint[inverse].reshape(-1, 3)
        counts = (mids >= 0).sum(axis=1)
        pieces = [faces[counts == 0]]
        for count in (1, 2):
            selected = counts == count
            tri = faces[selected]
            mid = mids[selected]
            # Rotate so the split edge is AB (one split), or the intact edge is CA (two).
            shift = np.argmax(mid >= 0, axis=1) if count == 1 else (np.argmax(mid < 0, axis=1) + 1) % 3
            columns = (np.arange(3)[None, :] + shift[:, None]) % 3
            tri = np.take_along_axis(tri, columns, axis=1)
            mid = np.take_along_axis(mid, columns, axis=1)
            a, b, c = tri.T
            ab, bc, _ = mid.T
            if count == 1:
                pieces.extend((np.column_stack((a, ab, c)), np.column_stack((ab, b, c))))
            else:
                pieces.extend(
                    (
                        np.column_stack((b, bc, ab)),
                        np.column_stack((a, ab, c)),
                        np.column_stack((ab, bc, c)),
                    )
                )
        selected = counts == 3
        a, b, c = faces[selected].T
        ab, bc, ca = mids[selected].T
        pieces.extend(
            (
                np.column_stack((a, ab, ca)),
                np.column_stack((ab, b, bc)),
                np.column_stack((ca, bc, c)),
                np.column_stack((ab, bc, ca)),
            )
        )
        faces = np.concatenate(pieces)
    raise ValueError("Crosswalk ramp triangulation did not converge")


def _profile_meshes(footprint, sidewalk_height, height_at):
    meshes = []
    for polygon in _polygon_parts(footprint):
        if polygon.area < 1e-8:
            continue
        # GEOS intersections can leave effectively duplicate vertices (~1e-12 m
        # apart). Remove those before triangulation to avoid zero-area faces.
        polygon = polygon.simplify(1e-7, preserve_topology=True)
        vertices, faces = trimesh.creation.triangulate_polygon(polygon, engine="manifold")
        # Interior vertices are needed for the crest between nearby roads and for
        # the side wings. Boundary-only triangulation can flatten a short island.
        vertices, faces = _refine_triangles(vertices, faces)
        mesh = trimesh.creation.extrude_triangulation(vertices, faces, height=sidewalk_height)
        top = mesh.vertices[:, 2] > sidewalk_height * 0.5
        mesh.vertices[top, 2] = height_at(mesh.vertices[top, :2])
        meshes.append(mesh)
    return meshes


def build_crosswalk_ramps(
    sidewalk,
    road_zones,
    sidewalk_height,
    ground_height,
    ramp_length=2.0,
    blend_width=1.0,
    curb_width=0.15,
):
    """Return ramp footprint/meshes, all crossings, and sloped opening curbs.

    Heights use distance to the *combined* crosswalk roads. Overlapping roads
    therefore have no internal ramps, and nearby ramps share one continuous
    surface. The original one-metre side wings remain raised at sidewalk edges.
    """
    crossings = sidewalk.intersection(road_zones)
    if crossings.area < 1e-8:
        return shapely.Polygon(), [], shapely.Polygon(), []
    ramp_footprint = sidewalk.difference(road_zones).intersection(
        road_zones.buffer(ramp_length, join_style=2)
    )
    # A road that passes near a different sidewalk leg without cutting it does
    # not need a ramp on that leg.
    ramp_footprint = shapely.union_all(
        [part for part in _polygon_parts(ramp_footprint) if part.distance(crossings) < 1e-7]
    )

    rise = sidewalk_height - ground_height
    # Limit distance queries to the local sidewalk edges.
    edges = sidewalk.boundary.intersection(crossings.envelope.buffer(ramp_length + blend_width))

    def shoulder(xy):
        if edges.is_empty:
            return np.zeros(len(xy))
        distance = shapely.distance(shapely.points(xy), edges)
        return np.clip(1.0 - distance / blend_width, 0.0, 1.0)

    def ramp_height(xy):
        distance = shapely.distance(shapely.points(xy), road_zones)
        slope = np.clip(distance / ramp_length, 0.0, 1.0)
        return ground_height + rise * np.maximum(slope, shoulder(xy))

    ramps = _profile_meshes(ramp_footprint, sidewalk_height, ramp_height)
    # Clear the full-height road curb at every actual crossing, then restore its
    # side wings and a 1 mm lip across the low opening.
    curb_footprint = crossings.intersection(
        road_zones.boundary.buffer(curb_width, cap_style=2, join_style=2)
    )
    curbs = _profile_meshes(
        curb_footprint,
        sidewalk_height,
        lambda xy: ground_height + np.maximum(0.001, rise * shoulder(xy)),
    )
    for mesh in curbs:
        mesh.visual.vertex_colors = (128, 128, 128, 255)
    return ramp_footprint, ramps, crossings, curbs
