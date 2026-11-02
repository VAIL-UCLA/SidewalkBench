from __future__ import annotations

from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
from shapely.geometry import Point, box
from shapely.ops import unary_union

from ._assets import load_building_infos as _load_building_infos
from ._assets import load_car_infos as _load_car_infos
from .base import (
    BaseTerrainCfg,
    Terrain,
    Polygon,
    _build_centerline_projector,
    configclass,
    np,
    trimesh,
)

_cut_in_params = {}  # {id(sampler_fn): {'px', 'py', 'sw'}}
_overtake_params = {}  # {id(sampler_fn): {'forward', 'toward', 'ext_cl', 'ext_cum', 'total_len'}}


def _sample_building_locations(terrain_width, terrain_length, rl, rr, rb, rt, sw, sidewalk_height):
    building_infos = _load_building_infos()
    if not building_infos:
        return None

    # Inner corners of non-walkable patches (closest to intersection center)
    # (ix, iy) = inner corner, (sx, sy) = direction from inner corner into the patch
    corners = [
        # TL: extends left (-1) and up (+1) from inner
        {"ix": rl - sw - 2, "iy": rt + sw + 2, "sx": -1, "sy": +1},
        # TR: extends right (+1) and up (+1)
        {"ix": rr + sw + 2, "iy": rt + sw + 2, "sx": +1, "sy": +1},
        # BR: extends right (+1) and down (-1)
        {"ix": rr + sw + 2, "iy": rb - sw - 2, "sx": +1, "sy": -1},
        # BL: extends left (-1) and down (-1)
        {"ix": rl - sw - 2, "iy": rb - sw - 2, "sx": -1, "sy": -1},
    ]

    placed = []  # (cx, cy, rotated, length, width)
    locations = []  # [x, y, z, yaw, b_idx]
    max_attempts = 50
    max_per_arm = 10

    for corner in corners:
        ix, iy = corner["ix"], corner["iy"]
        sx, sy = corner["sx"], corner["sy"]

        # --- place anchor at inner corner ---
        anchor_placed = False
        for _ in range(max_attempts):
            b_idx = np.random.randint(0, len(building_infos))
            b_info = building_infos[b_idx]
            b_len = b_info["length"]
            b_wid = b_info["width"]

            for rotated in [False, True]:
                l = b_len if not rotated else b_wid
                w = b_wid if not rotated else b_len
                cx = ix + sx * l / 2.0
                cy = iy + sy * w / 2.0
                bx0, bx1 = cx - l / 2.0, cx + l / 2.0
                by0, by1 = cy - w / 2.0, cy + w / 2.0
                if bx0 < 0 or bx1 > terrain_width or by0 < 0 or by1 > terrain_length:
                    continue
                anchor_placed = True
                placed.append((cx, cy, rotated, l, w))
                yaw = np.pi / 2 if rotated else 0.0
                locations.append([float(cx), float(cy), float(sidewalk_height), float(yaw), b_idx])
                break
            if anchor_placed:
                break
        if not anchor_placed:
            continue

        # --- extend along X arm (horizontal) ---
        front_x = cx + sx * l / 2.0
        for _ in range(max_per_arm):
            arm_placed = False
            for _ in range(max_attempts):
                b_idx = np.random.randint(0, len(building_infos))
                b_info = building_infos[b_idx]
                b_len = b_info["length"]
                b_wid = b_info["width"]
                rotated = np.random.choice([False, True])

                l2 = b_len if not rotated else b_wid
                w2 = b_wid if not rotated else b_len
                cx2 = front_x + sx * l2 / 2.0
                cy2 = iy + sy * w2 / 2.0
                bx0, bx1 = cx2 - l2 / 2.0, cx2 + l2 / 2.0
                by0, by1 = cy2 - w2 / 2.0, cy2 + w2 / 2.0
                if bx0 < 0 or bx1 > terrain_width or by0 < 0 or by1 > terrain_length:
                    continue
                arm_placed = True
                placed.append((cx2, cy2, rotated, l2, w2))
                yaw = np.pi / 2 if rotated else 0.0
                locations.append([float(cx2), float(cy2), float(sidewalk_height), float(yaw), b_idx])
                front_x = cx2 + sx * l2 / 2.0
                if arm_placed:
                    break

            if not arm_placed:
                break

        # --- extend along Y arm (vertical) ---
        front_y = cy + sy * w / 2.0
        for _ in range(max_per_arm):
            arm_placed = False
            for _ in range(max_attempts):
                b_idx = np.random.randint(0, len(building_infos))
                b_info = building_infos[b_idx]
                b_len = b_info["length"]
                b_wid = b_info["width"]
                rotated = np.random.choice([False, True])
                l2 = b_len if not rotated else b_wid
                w2 = b_wid if not rotated else b_len
                cx2 = ix + sx * l2 / 2.0
                cy2 = front_y + sy * w2 / 2.0
                bx0, bx1 = cx2 - l2 / 2.0, cx2 + l2 / 2.0
                by0, by1 = cy2 - w2 / 2.0, cy2 + w2 / 2.0
                if bx0 < 0 or bx1 > terrain_width or by0 < 0 or by1 > terrain_length:
                    continue
                arm_placed = True
                placed.append((cx2, cy2, rotated, l2, w2))
                yaw = np.pi / 2 if rotated else 0.0
                locations.append([float(cx2), float(cy2), float(sidewalk_height), float(yaw), b_idx])
                front_y = cy2 + sy * w2 / 2.0
                if arm_placed:
                    break
            if not arm_placed:
                break

    if not locations:
        return None
    return np.array(locations, dtype=float)


def _sample_car_locations(
    terrain_width, terrain_length, cx, cy, road_width, ground_height, crosswalk_offset, crosswalk_width
):
    car_infos = _load_car_infos()
    if not car_infos:
        return None

    road_hw = road_width / 2.0
    road_hw_inner = road_hw
    center_avoid = road_hw + crosswalk_offset + crosswalk_width

    road_arms = [
        {
            "name": "north",
            "x0": cx - road_hw_inner,
            "x1": cx + road_hw_inner,
            "y0": cy + center_avoid,
            "y1": terrain_length - 1.0,
            "yaw": np.pi / 2,
        },
        {
            "name": "south",
            "x0": cx - road_hw_inner,
            "x1": cx + road_hw_inner,
            "y0": 1.0,
            "y1": cy - center_avoid,
            "yaw": -np.pi / 2,
        },
        {
            "name": "east",
            "x0": cx + center_avoid,
            "x1": terrain_width - 1.0,
            "y0": cy - road_hw_inner,
            "y1": cy + road_hw_inner,
            "yaw": 0.0,
        },
        {
            "name": "west",
            "x0": 1.0,
            "x1": cx - center_avoid,
            "y0": cy - road_hw_inner,
            "y1": cy + road_hw_inner,
            "yaw": np.pi,
        },
    ]

    placed = []
    locations = []

    for arm in road_arms:
        arm_w = arm["x1"] - arm["x0"]
        arm_h = arm["y1"] - arm["y0"]
        if arm_w <= 0 or arm_h <= 0:
            continue
        num_cars = np.random.randint(1, 3)
        attempts = 0
        cars_placed = 0
        while cars_placed < num_cars and attempts < 30:
            attempts += 1
            c_idx = np.random.randint(0, len(car_infos))
            c_info = car_infos[c_idx]
            c_len = c_info["length"]
            c_wid = c_info["width"]
            c_rad = max(c_len, c_wid) / 2.0 + 0.1

            cx_c = np.random.uniform(arm["x0"] + c_rad, arm["x1"] - c_rad)
            cy_c = np.random.uniform(arm["y0"] + c_rad, arm["y1"] - c_rad)

            overlap = False
            for pc in placed:
                if np.hypot(cx_c - pc[0], cy_c - pc[1]) < (c_rad + pc[2]):
                    overlap = True
                    break
            if overlap:
                continue

            rotated = np.random.choice([False, True])
            placed.append((float(cx_c), float(cy_c), c_rad))
            locations.append(
                [float(cx_c), float(cy_c), float(ground_height), float(arm["yaw"]) + float(rotated) * np.pi, c_idx]
            )
            cars_placed += 1
    if not locations:
        return None
    return np.array(locations, dtype=float)


def _polys_from_geom(geom):
    """Return a list of Polygons from a (Multi)Polygon geometry."""
    if geom.is_empty:
        return []
    if isinstance(geom, ShapelyMultiPolygon):
        return [g for g in geom.geoms if not g.is_empty and g.area > 1e-9]
    return [geom]


def _build_ramp_wedge(x0, y0, x1, y1, h_low, h_high, low_side, raise_height=0):
    """Build a sloped wedge mesh for a curb ramp.

    Top surface is raised 0.001m to stay visible above surrounding geometry.
    """
    h_low += raise_height
    h_high += raise_height
    # Map low_side to top heights at each footprint corner
    if low_side == "south":
        top_z = [(x0, y0, h_low), (x1, y0, h_low), (x1, y1, h_high), (x0, y1, h_high)]
    elif low_side == "north":
        top_z = [(x0, y0, h_high), (x1, y0, h_high), (x1, y1, h_low), (x0, y1, h_low)]
    elif low_side == "east":
        top_z = [(x0, y0, h_high), (x1, y0, h_low), (x1, y1, h_low), (x0, y1, h_high)]
    elif low_side == "west":
        top_z = [(x0, y0, h_low), (x1, y0, h_high), (x1, y1, h_high), (x0, y1, h_low)]
    else:
        raise ValueError(f"Unknown low_side: {low_side}")

    vertices = [
        (x0, y0, 0),
        (x1, y0, 0),
        (x1, y1, 0),
        (x0, y1, 0),  # base 0-3
        top_z[0],
        top_z[1],
        top_z[2],
        top_z[3],  # top 4-7
    ]
    faces = [
        (0, 2, 1),
        (0, 3, 2),  # bottom
        (4, 5, 6),
        (4, 6, 7),  # sloped top
        (0, 1, 5),
        (0, 5, 4),  # y0 wall
        (1, 2, 6),
        (1, 6, 5),  # x1 wall
        (2, 3, 7),
        (2, 7, 6),  # y1 wall
        (3, 0, 4),
        (3, 4, 7),  # x0 wall
    ]
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def _build_sloped_quad(x0, y0, x1, y1, h00, h10, h11, h01, raise_height=0.0, flip=False):
    """Build a quadrilateral wedge. Top surface raised 0.001m for visibility."""
    h00 += raise_height
    h10 += raise_height
    h11 += raise_height
    h01 += raise_height
    vertices = [
        (x0, y0, 0),
        (x1, y0, 0),
        (x1, y1, 0),
        (x0, y1, 0),  # base 0-3
        (x0, y0, h00),
        (x1, y0, h10),
        (x1, y1, h11),
        (x0, y1, h01),  # top 4-7
    ]
    if flip:
        faces = [
            (0, 2, 1),
            (0, 3, 2),  # bottom
            (4, 5, 7),
            (5, 6, 7),  # sloped top
            (0, 1, 5),
            (0, 5, 4),  # y0 wall
            (1, 2, 6),
            (1, 6, 5),  # x1 wall
            (2, 3, 7),
            (2, 7, 6),  # y1 wall
            (3, 0, 4),
            (3, 4, 7),  # x0 wall
        ]
    else:
        faces = [
            (0, 2, 1),
            (0, 3, 2),  # bottom
            (4, 5, 6),
            (4, 6, 7),  # sloped top
            (0, 1, 5),
            (0, 5, 4),  # y0 wall
            (1, 2, 6),
            (1, 6, 5),  # x1 wall
            (2, 3, 7),
            (2, 7, 6),  # y1 wall
            (3, 0, 4),
            (3, 4, 7),  # x0 wall
        ]
    return trimesh.Trimesh(vertices=vertices, faces=faces)


class IntersectionTerrain(Terrain):
    """Procedural terrain builder for IntersectionTerrainCfg."""

    def build(self) -> tuple:
        cfg = self.cfg
        generator_cfg = self.generator_cfg
        """Generate an intersection with 4 L-shaped corner sidewalks.

        Each L-shape is built from two rectangle arms (shortened by corner_radius
        at the road-side end) + a quarter-disk bridging them at the concave corner.
        The terrain-side (convex) corner is a sharp right angle. At the road-side
        end of each arm, a 1m sloped curb ramp replaces the sidewalk surface.
        """
        sidewalk_width = generator_cfg.sidewalk_width
        sidewalk_height = generator_cfg.sidewalk_height
        ground_height = generator_cfg.ground_height
        use_ramp = cfg.use_ramp
        terrain_segment = cfg.terrain_segment

        terrain_width = cfg.size[0]
        terrain_length = cfg.size[1]

        cx = terrain_width / 2.0
        cy = terrain_length / 2.0
        half_w = sidewalk_width / 2.0
        road_width = 10.0
        curb_width = 0.15 if use_ramp else 2.0
        rh = road_width / 2.0
        rl = cx - rh - curb_width
        rr = cx + rh + curb_width
        rb = cy - rh - curb_width
        rt = cy + rh + curb_width
        corner_radius = 2.0
        assert corner_radius <= sidewalk_width, "corner_radius must be <= sidewalk_width to avoid blank"
        crosswalk_offset = 0.0
        crosswalk_width = sidewalk_width + 1.0

        meshes_list: list[trimesh.Trimesh] = []

        # ---- 4 L-shaped sidewalks: shortened arms + quarter-disk fillet ----
        # Each L = vertical arm + horizontal arm (each shortened by corner_radius at the
        # road-side end) + a quarter-disk bridging them with a smooth concave curve.
        # The disk is centered at the sidewalk-interior corner of the gap square; the arc
        # curves toward the road corner. The terrain-side corner is a sharp right angle.
        sw = sidewalk_width
        cr = corner_radius
        n_arc = 32
        sidewalk_polys = []

        def _quarter_disk(center_x, center_y, radius, a0, a1, n=n_arc):
            """Pie-slice polygon: arc points + center."""
            angles = np.linspace(a0, a1, n)
            arc_pts = np.column_stack(
                [
                    center_x + radius * np.cos(angles),
                    center_y + radius * np.sin(angles),
                ]
            )
            pts = np.vstack([arc_pts, [[center_x, center_y]]])
            return Polygon(pts)

        # Top-left: pie centered at (rl-cr, rt+cr), arc curves SE toward road corner
        v = box(rl - sw, rt + cr, rl, terrain_length)
        h = box(0, rt, rl - cr, rt + sw)
        qd = _quarter_disk(rl - cr, rt + cr, cr, -np.pi / 2, 0.0)
        sidewalk_polys.append(unary_union([v, h, qd]))

        # Top-right: pie centered at (rr+cr, rt+cr), arc curves SW toward road corner
        v = box(rr, rt + cr, rr + sw, terrain_length)
        h = box(rr + cr, rt, terrain_width, rt + sw)
        qd = _quarter_disk(rr + cr, rt + cr, cr, -np.pi, -np.pi / 2)
        sidewalk_polys.append(unary_union([v, h, qd]))

        # Bottom-right: pie centered at (rr+cr, rb-cr), arc curves NW toward road corner
        v = box(rr, 0, rr + sw, rb - cr)
        h = box(rr + cr, rb - sw, terrain_width, rb)
        qd = _quarter_disk(rr + cr, rb - cr, cr, np.pi / 2, np.pi)
        sidewalk_polys.append(unary_union([v, h, qd]))

        # Bottom-left: pie centered at (rl-cr, rb-cr), arc curves NE toward road corner
        v = box(rl - sw, 0, rl, rb - cr)
        h = box(0, rb - sw, rl - cr, rb)
        qd = _quarter_disk(rl - cr, rb - cr, cr, 0.0, np.pi / 2)
        sidewalk_polys.append(unary_union([v, h, qd]))

        # ---- Curb ramps: cut sidewalk arms, add sloped wedges + buffer transitions ----
        ramp_length = 3.0 if use_ramp else 0.0
        ramp_depth = 2.0  # ramp slope must not exceed a maximum 1:12 ratio (8.33% grade)
        assert ramp_depth <= corner_radius, "ramp_depth must be <= corner_radius"
        buf = 0.5
        sh = sidewalk_height
        gh = ground_height
        sidewalk_polys_original = sidewalk_polys.copy()
        if use_ramp:
            # Full ramp footprints (for cutting the sidewalk polygon)
            ramp_specs = [
                (box(rl - ramp_depth, rt + cr, rl, rt + cr + ramp_length), "east", 0),  # TL top
                (box(rl - cr - ramp_length, rt, rl - cr, rt + ramp_depth), "south", 0),  # TL left
                (box(rr, rt + cr, rr + ramp_depth, rt + cr + ramp_length), "west", 1),  # TR top
                (box(rr + cr, rt, rr + cr + ramp_length, rt + ramp_depth), "south", 1),  # TR right
                (box(rr, rb - cr - ramp_length, rr + ramp_depth, rb - cr), "west", 2),  # BR bottom
                (box(rr + cr, rb - ramp_depth, rr + cr + ramp_length, rb), "north", 2),  # BR right
                (box(rl - ramp_depth, rb - cr - ramp_length, rl, rb - cr), "east", 3),  # BL bottom
                (box(rl - cr - ramp_length, rb - ramp_depth, rl - cr, rb), "north", 3),  # BL left
            ]

            # Inner ramp wedge footprints (with buffer trimmed from arm-direction ends)
            ramp_specs_rect = [
                (box(rl - ramp_depth, rt + cr + buf, rl, rt + cr + ramp_length - buf), "east", 0),
                (box(rl - cr - ramp_length + buf, rt, rl - cr - buf, rt + ramp_depth), "south", 0),
                (box(rr, rt + cr + buf, rr + ramp_depth, rt + cr + ramp_length - buf), "west", 1),
                (box(rr + cr + buf, rt, rr + cr + ramp_length - buf, rt + ramp_depth), "south", 1),
                (box(rr, rb - cr - ramp_length + buf, rr + ramp_depth, rb - cr - buf), "west", 2),
                (box(rr + cr + buf, rb - ramp_depth, rr + cr + ramp_length - buf, rb), "north", 2),
                (box(rl - ramp_depth, rb - cr - ramp_length + buf, rl, rb - cr - buf), "east", 3),
                (box(rl - cr - ramp_length + buf, rb - ramp_depth, rl - cr - buf, rb), "north", 3),
            ]

            # Cut ramp footprints from sidewalk polygons
            ramp_union_per_corner = [None] * 4
            for fp, _, idx in ramp_specs:
                if ramp_union_per_corner[idx] is None:
                    ramp_union_per_corner[idx] = fp
                else:
                    ramp_union_per_corner[idx] = ramp_union_per_corner[idx].union(fp)
            for i in range(4):
                if ramp_union_per_corner[i] is not None:
                    sidewalk_polys[i] = sidewalk_polys[i].difference(ramp_union_per_corner[i])

        sidewalk_meshes = []
        for poly in sidewalk_polys:
            for sub in _polys_from_geom(poly):
                m = trimesh.creation.extrude_polygon(polygon=sub, height=sh, engine="manifold")
                sidewalk_meshes.append(m)

        if use_ramp:
            # Append ramp wedges
            for fp, low_side, _ in ramp_specs_rect:
                minx, miny, maxx, maxy = fp.bounds
                ramp = _build_ramp_wedge(minx, miny, maxx, maxy, gh, sh, low_side)
                sidewalk_meshes.append(ramp)

            # Build 16 buffer transition wedges (2 per ramp)
            for (fp, low_side, _), (rpf, _, _) in zip(ramp_specs, ramp_specs_rect):
                fx0, fy0, fx1, fy1 = fp.bounds
                rx0, ry0, rx1, ry1 = rpf.bounds
                if low_side in ("east", "west"):
                    # Vertical arm — buffers above/below ramp (Y direction)
                    ramp_hi_x, ramp_lo_x = (fx0, fx1) if low_side == "east" else (fx1, fx0)
                    # Bottom buffer (ramp ↔ quarter-disk): qdisk edge at fy0 (all sh), ramp edge at ry0
                    h_qdisk = sh
                    h_ramp_w = sh if fx0 == ramp_hi_x else gh  # west side of ramp
                    h_ramp_e = sh if fx1 == ramp_hi_x else gh  # east side of ramp
                    sidewalk_meshes.append(
                        _build_sloped_quad(
                            fx0, fy0, fx1, ry0, h_qdisk, h_qdisk, h_ramp_e, h_ramp_w, flip=not (h_qdisk == h_ramp_e)
                        )
                    )
                    # Top buffer (ramp ↔ flat arm): ramp edge at ry1, arm edge at fy1 (all sh)
                    sidewalk_meshes.append(
                        _build_sloped_quad(
                            fx0, ry1, fx1, fy1, h_ramp_w, h_ramp_e, h_qdisk, h_qdisk, flip=not (h_qdisk == h_ramp_w)
                        )
                    )
                else:
                    # Horizontal arm — buffers left/right of ramp (X direction)
                    ramp_hi_y, ramp_lo_y = (fy1, fy0) if low_side == "south" else (fy0, fy1)
                    h_flat = sh
                    h_ramp_s = sh if fy0 == ramp_hi_y else gh  # south side of ramp
                    h_ramp_n = sh if fy1 == ramp_hi_y else gh  # north side of ramp
                    # Left buffer (flat arm ↔ ramp): arm edge at fx0 (all sh), ramp edge at rx0
                    sidewalk_meshes.append(
                        _build_sloped_quad(
                            fx0, fy0, rx0, fy1, h_flat, h_ramp_s, h_ramp_n, h_flat, flip=not (h_flat == h_ramp_n)
                        )
                    )
                    # Right buffer (ramp ↔ quarter-disk): ramp edge at rx1, qdisk edge at fx1 (all sh)
                    sidewalk_meshes.append(
                        _build_sloped_quad(
                            rx1, fy0, fx1, fy1, h_ramp_s, h_flat, h_flat, h_ramp_n, flip=not (h_flat == h_ramp_s)
                        )
                    )

        meshes_list.append(trimesh.util.concatenate(sidewalk_meshes))

        # ---- Curb at each smoothed road-side concave corner (grey, curb_width wide) ----
        curb_color = (128, 128, 128, 255)
        # The curb annulus is centered at the same sidewalk-interior corner as the pie,
        # with radii [corner_radius, corner_radius + curb_width] extending into the road.
        r_inner = corner_radius
        r_outer = corner_radius + curb_width
        n_curb = 30
        # Each entry: (center_x, center_y, start_angle, end_angle) — same as pie centers
        curb_corners = [
            (rl - cr, rt + cr, -np.pi / 2, 0.0),  # TL: SE toward road
            (rr + cr, rt + cr, -np.pi, -np.pi / 2),  # TR: SW toward road
            (rr + cr, rb - cr, np.pi / 2, np.pi),  # BR: NW toward road
            (rl - cr, rb - cr, 0.0, np.pi / 2),  # BL: NE toward road
        ]
        curb_meshes = []
        for cx_c, cy_c, a0, a1 in curb_corners:
            angles = np.linspace(a0, a1, n_curb)
            outer = np.column_stack([cx_c + r_outer * np.cos(angles), cy_c + r_outer * np.sin(angles)])
            inner = np.column_stack([cx_c + r_inner * np.cos(angles), cy_c + r_inner * np.sin(angles)])
            curb_poly = Polygon(np.concatenate([outer, inner[::-1]], axis=0))
            if not curb_poly.is_valid:
                curb_poly = curb_poly.buffer(0)
            for sub in _polys_from_geom(curb_poly):
                cm = trimesh.creation.extrude_polygon(polygon=sub, height=sidewalk_height, engine="manifold")
                cm.visual.vertex_colors = curb_color
                curb_meshes.append(cm)

        # Arm curbs along inner edges (road side) of each straight sidewalk arm
        h = sidewalk_height
        cw = curb_width
        arm_curb_polys = [
            # TL vertical arm: inner edge at x=rl, y from ramp end to top
            box(rl, rt + cr + ramp_length, rl + cw, terrain_length),
            # TL horizontal arm: inner edge at y=rt, x from left to ramp start
            box(0, rt - cw, rl - cr - ramp_length, rt),
            # TR vertical arm: inner edge at x=rr, y from ramp end to top
            box(rr - cw, rt + cr + ramp_length, rr, terrain_length),
            # TR horizontal arm: inner edge at y=rt, x from ramp start to right
            box(rr + cr + ramp_length, rt - cw, terrain_width, rt),
            # BR vertical arm: inner edge at x=rr, y from bottom to ramp start
            box(rr - cw, 0, rr, rb - cr - ramp_length),
            # BR horizontal arm: inner edge at y=rb, x from ramp start to right
            box(rr + cr + ramp_length, rb, terrain_width, rb + cw),
            # BL vertical arm: inner edge at x=rl, y from bottom to ramp start
            box(rl, 0, rl + cw, rb - cr - ramp_length),
            # BL horizontal arm: inner edge at y=rb, x from left to ramp start
            box(0, rb, rl - cr - ramp_length, rb + cw),
        ]
        for poly in arm_curb_polys:
            cm = trimesh.creation.extrude_polygon(polygon=poly, height=h, engine="manifold")
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)

        if use_ramp:
            # ---- TL top curb ramp (east-low V-arm) ----
            # Bottom buffer (quarter-disk → ramp)
            cm = _build_sloped_quad(rl, rt + cr, rl + curb_width, rt + cr + buf, sh, sh, gh, gh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            # Ramp rect (constant X-slope)
            cm = _build_ramp_wedge(
                rl, rt + cr + buf, rl + curb_width, rt + cr + ramp_length - buf, gh, gh, "east", raise_height=0.001
            )
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            # Top buffer (ramp → flat arm)
            cm = _build_sloped_quad(rl, rt + cr + ramp_length - buf, rl + curb_width, rt + cr + ramp_length, gh, gh, sh, sh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)

            # ---- TL left curb ramp (south-low H-arm) ----
            # Left buffer
            cm = _build_sloped_quad(rl - cr - ramp_length, rt - curb_width, rl - cr - ramp_length + buf, rt, sh, gh, gh, sh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            # Ramp rect
            cm = _build_ramp_wedge(
                rl - cr - ramp_length + buf, rt - curb_width, rl - cr - buf, rt, gh, gh, "south", raise_height=0.001
            )
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            # Right buffer
            cm = _build_sloped_quad(rl - cr - buf, rt - curb_width, rl - cr, rt, gh, sh, sh, gh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)

            # ---- TR top curb ramp (west-low V-arm) ----
            cm = _build_sloped_quad(rr - curb_width, rt + cr, rr, rt + cr + buf, sh, sh, gh, gh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            cm = _build_ramp_wedge(
                rr - curb_width, rt + cr + buf, rr, rt + cr + ramp_length - buf, gh, gh, "west", raise_height=0.001
            )
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            cm = _build_sloped_quad(rr - curb_width, rt + cr + ramp_length - buf, rr, rt + cr + ramp_length, gh, gh, sh, sh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)

            # ---- TR right curb ramp (south-low H-arm) ----
            cm = _build_sloped_quad(rr + cr, rt - curb_width, rr + cr + buf, rt, sh, gh, gh, sh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            cm = _build_ramp_wedge(
                rr + cr + buf, rt - curb_width, rr + cr + ramp_length - buf, rt, gh, gh, "south", raise_height=0.001
            )
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            cm = _build_sloped_quad(rr + cr + ramp_length - buf, rt - curb_width, rr + cr + ramp_length, rt, gh, sh, sh, gh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)

            # ---- BR bottom curb ramp (west-low V-arm) ----
            cm = _build_sloped_quad(rr - curb_width, rb - cr - ramp_length, rr, rb - cr - ramp_length + buf, sh, sh, gh, gh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            cm = _build_ramp_wedge(
                rr - curb_width, rb - cr - ramp_length + buf, rr, rb - cr - buf, gh, gh, "west", raise_height=0.001
            )
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            cm = _build_sloped_quad(rr - curb_width, rb - cr - buf, rr, rb - cr, gh, gh, sh, sh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)

            # ---- BR right curb ramp (north-low H-arm) ----
            cm = _build_sloped_quad(rr + cr, rb, rr + cr + buf, rb + curb_width, sh, gh, gh, sh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            cm = _build_ramp_wedge(
                rr + cr + buf, rb, rr + cr + ramp_length - buf, rb + curb_width, gh, gh, "north", raise_height=0.001
            )
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            cm = _build_sloped_quad(rr + cr + ramp_length - buf, rb, rr + cr + ramp_length, rb + curb_width, gh, sh, sh, gh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)

            # ---- BL bottom curb ramp (east-low V-arm) ----
            cm = _build_sloped_quad(rl, rb - cr - ramp_length, rl + curb_width, rb - cr - ramp_length + buf, sh, sh, gh, gh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            cm = _build_ramp_wedge(
                rl, rb - cr - ramp_length + buf, rl + curb_width, rb - cr - buf, gh, gh, "east", raise_height=0.001
            )
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            cm = _build_sloped_quad(rl, rb - cr - buf, rl + curb_width, rb - cr, gh, gh, sh, sh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)

            # ---- BL left curb ramp (north-low H-arm) ----
            cm = _build_sloped_quad(rl - cr - ramp_length, rb, rl - cr - ramp_length + buf, rb + curb_width, sh, gh, gh, sh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            cm = _build_ramp_wedge(
                rl - cr - ramp_length + buf, rb, rl - cr - buf, rb + curb_width, gh, gh, "north", raise_height=0.001
            )
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)
            cm = _build_sloped_quad(rl - cr - buf, rb, rl - cr, rb + curb_width, gh, sh, sh, gh)
            cm.visual.vertex_colors = curb_color
            curb_meshes.append(cm)

        # ---- Road (full rectangle at ground level, bottom layer) ----
        road_poly = Polygon([[0, 0], [terrain_width, 0], [terrain_width, terrain_length], [0, terrain_length]])
        road_mesh = trimesh.creation.extrude_polygon(polygon=road_poly, height=ground_height, engine="manifold")
        meshes_list.append(road_mesh)

        stripe_geom = Point(0, 0).buffer(0.0)
        if use_ramp:
            # ---- Crosswalk stripes (4 crossing zones: top, bottom, left, right) ----
            stripe_width = 0.3
            stripe_count = max(4, int(round(road_width / (stripe_width * 2.0))) - 1)
            stripe_period_x = (road_width - stripe_width * 2) / stripe_count
            stripe_period_y = (road_width - stripe_width * 2) / stripe_count
            stripe_polys = []

            # Top (between TL & TR sidewalks, x ∈ [rl, rr], y ∈ [rt+offset, rt+offset+cw])
            y0 = rt + crosswalk_offset
            y1 = y0 + crosswalk_width
            for i in range(stripe_count):
                sx = rl + i * stripe_period_x + stripe_width + curb_width
                stripe_polys.append(
                    Polygon(
                        [
                            (sx, y0),
                            (sx, y1),
                            (sx + stripe_width, y1),
                            (sx + stripe_width, y0),
                        ]
                    )
                )
            # Bottom (between BL & BR sidewalks, x ∈ [rl, rr], y ∈ [rb-offset-cw, rb-offset])
            y1 = rb - crosswalk_offset
            y0 = y1 - crosswalk_width
            for i in range(stripe_count):
                sx = rl + i * stripe_period_x + stripe_width + curb_width
                stripe_polys.append(
                    Polygon(
                        [
                            (sx, y0),
                            (sx, y1),
                            (sx + stripe_width, y1),
                            (sx + stripe_width, y0),
                        ]
                    )
                )
            # Left (between TL & BL sidewalks, x ∈ [rl-offset-cw, rl-offset], y ∈ [rb, rt])
            x1 = rl - crosswalk_offset
            x0 = x1 - crosswalk_width
            for i in range(stripe_count):
                sy = rb + i * stripe_period_y + stripe_width + curb_width
                stripe_polys.append(
                    Polygon(
                        [
                            (x0, sy),
                            (x1, sy),
                            (x1, sy + stripe_width),
                            (x0, sy + stripe_width),
                        ]
                    )
                )
            # Right (between TR & BR sidewalks, x ∈ [rr+offset, rr+offset+cw], y ∈ [rb, rt])
            x0 = rr + crosswalk_offset
            x1 = x0 + crosswalk_width
            for i in range(stripe_count):
                sy = rb + i * stripe_period_y + stripe_width + curb_width
                stripe_polys.append(
                    Polygon(
                        [
                            (x0, sy),
                            (x1, sy),
                            (x1, sy + stripe_width),
                            (x0, sy + stripe_width),
                        ]
                    )
                )

            stripe_geom = unary_union(stripe_polys)
            if not stripe_geom.is_valid:
                stripe_geom = stripe_geom.buffer(0.0)
            stripe_meshes = []
            for sub in _polys_from_geom(stripe_geom):
                sm = trimesh.creation.extrude_polygon(polygon=sub, height=ground_height + 0.001, engine="manifold")
                sm.visual.vertex_colors = (255, 255, 255, 255)
                stripe_meshes.append(sm)
            meshes_list.append(trimesh.util.concatenate(curb_meshes + stripe_meshes))
        else:
            dummy = trimesh.creation.box(extents=(0.01, 0.01, 0.001))
            meshes_list.append(dummy)

        # ---- Dummy placeholder (walkable group 2) ----
        dummy = trimesh.creation.box(extents=(0.01, 0.01, 0.001))
        meshes_list.append(dummy)

        # ---- Non-walkable corner terrain (elevated, just below sidewalk) ----
        sw = sidewalk_width
        terrain_corners = [
            box(0.0, rt + sw, rl - sw, terrain_length),  # top-left
            box(rr + sw, rt + sw, terrain_width, terrain_length),  # top-right
            box(rr + sw, 0.0, terrain_width, rb - sw),  # bottom-right
            box(0.0, 0.0, rl - sw, rb - sw),  # bottom-left
        ]
        terrain_geom = unary_union(terrain_corners)
        terrain_meshes = []
        terrain_height = sidewalk_height - 0.001
        for sub in _polys_from_geom(terrain_geom):
            m = trimesh.creation.extrude_polygon(polygon=sub, height=terrain_height, engine="manifold")
            terrain_meshes.append(m)
        if use_ramp:
            meshes_list.append(trimesh.util.concatenate(terrain_meshes))
        else:
            meshes_list.append(trimesh.util.concatenate(terrain_meshes + curb_meshes))

        # ---- Border mesh ----
        border = trimesh.creation.extrude_polygon(
            polygon=Polygon([[0, 0], [terrain_width, 0], [terrain_width, terrain_length], [0, terrain_length]]),
            height=0.001,
            engine="manifold",
        )
        meshes_list.append(border)

        # Combined walkable geometry for boundary checks
        sidewalk_geom = unary_union(sidewalk_polys_original)
        if not sidewalk_geom.is_valid:
            sidewalk_geom = sidewalk_geom.buffer(0.0)
        walkable_geom = unary_union([sidewalk_geom, stripe_geom])
        if not walkable_geom.is_valid:
            walkable_geom = walkable_geom.buffer(0.0)

        origin = np.array([0.0, 0.0, 0.0])

        # ---- Sample building and car locations ----
        building_locations = _sample_building_locations(terrain_width, terrain_length, rl, rr, rb, rt, sw, sidewalk_height)
        car_locations = _sample_car_locations(
            terrain_width, terrain_length, cx, cy, road_width, ground_height, crosswalk_offset, crosswalk_width
        )

        sidewalk_union = unary_union(sidewalk_polys)
        sidewalk_interior = sidewalk_union.buffer(-1.0)

        sw_minx, sw_miny, sw_maxx, sw_maxy = sidewalk_union.bounds

        def sample_pos(walkable=True, num_samples=1, rng=None):
            rng = np.random if rng is None else rng
            xs = np.zeros(num_samples)
            ys = np.zeros(num_samples)
            for i in range(num_samples):
                while True:
                    x = rng.uniform(sw_minx, sw_maxx)
                    y = rng.uniform(sw_miny, sw_maxy)
                    in_sw = sidewalk_union.contains(Point(x, y))
                    # Only the straight unit test uses the full sidewalk width.
                    if terrain_segment != "straight" or cfg.scenario_type != "empty":
                        in_sw = in_sw and not sidewalk_interior.contains(Point(x, y))
                    if walkable and in_sw:
                        xs[i] = x
                        ys[i] = y
                        break
                    if not walkable and not in_sw:
                        xs[i] = x
                        ys[i] = y
                        break
            return xs, ys

        def sample_trajectory(num_points, num_samples=1, delta_distance=0.2, scenario_type="default", rng=None, **kwargs):

            rng = np.random if rng is None else rng
            def _build_trajs_from_path(path, num_samples, spacing=0.5):
                """Build (num_samples, n_pts, 3) trajs from a path with lateral offset."""
                n_pts = len(path)
                tangents = np.zeros((n_pts, 2))
                tangents[:-1] = path[1:] - path[:-1]
                tangents[-1] = tangents[-2]
                norms = np.linalg.norm(tangents, axis=1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                tangents /= norms
                perps = np.column_stack([-tangents[:, 1], tangents[:, 0]])
                yaws = np.zeros(n_pts)
                yaws[:-1] = np.arctan2(tangents[:-1, 1], tangents[:-1, 0])
                yaws[-1] = yaws[-2]
                trajs = np.zeros((num_samples, n_pts, 3))
                for k in range(num_samples):
                    lateral = (k - (num_samples - 1) / 2.0) * spacing
                    trajs[k, :, 0] = path[:, 0] + lateral * perps[:, 0]
                    trajs[k, :, 1] = path[:, 1] + lateral * perps[:, 1]
                    trajs[k, :, 2] = yaws
                return trajs

            def _resample_path(points, cumlen, start_s, end_s):
                """Resample points at delta_distance intervals from start_s to end_s."""
                n_pts = max(2, int((end_s - start_s) / delta_distance) + 1)
                target_s = np.linspace(start_s, end_s, n_pts)
                rx = np.interp(target_s, cumlen, points[:, 0])
                ry = np.interp(target_s, cumlen, points[:, 1])
                return np.column_stack([rx, ry])

            if scenario_type in ["frontal_approaching", "ped_crossing", "active_collision"] and _cached_centerline[0] is not None:
                cl = _cached_centerline[0]
                rcl = cl[::-1]  # goal→start
                diffs = np.diff(rcl, axis=0)
                seg_lens = np.linalg.norm(diffs, axis=1)
                cumlen = np.insert(np.cumsum(seg_lens), 0, 0)
                total_len = cumlen[-1]
                path = _resample_path(rcl, cumlen, 0, total_len)
                return _build_trajs_from_path(path, num_samples)

            if scenario_type == "human_following" and _cached_centerline[0] is not None:
                lead = 3.0
                cl = _cached_centerline[0]
                # Compute arc-lengths
                diffs = np.diff(cl, axis=0)
                seg_lens = np.linalg.norm(diffs, axis=1)
                cumlen_native = np.insert(np.cumsum(seg_lens), 0, 0)
                total_len = cumlen_native[-1]
                # Extend centerline past goal along final tangent
                final_tg = cl[-1] - cl[-2]
                final_tg /= np.linalg.norm(final_tg, keepdims=True) or 1.0
                ext_cl = np.vstack([cl, cl[-1] + lead * final_tg])
                ext_cum = np.append(cumlen_native, total_len + lead)
                path = _resample_path(ext_cl, ext_cum, lead, total_len + lead)
                # Yaw is along path direction (from start toward goal), not reversed
                path_yaws = _build_trajs_from_path(path, num_samples)
                # Tangent-based yaw already correct (path goes start→goal direction)
                return path_yaws

            if (
                scenario_type in ["lateral_approaching", "obstructing", "human_lying", "gesturing"]
                and _cached_centerline[0] is not None
            ):
                cl = _cached_centerline[0]
                n_cl = len(cl)
                mid_idx = n_cl // 2
                mx, my = cl[mid_idx]
                # Tangent at midpoint
                if n_cl > 2:
                    prev_idx = max(0, mid_idx - 5)
                    next_idx = min(n_cl - 1, mid_idx + 5)
                    dx = cl[next_idx, 0] - cl[prev_idx, 0]
                    dy = cl[next_idx, 1] - cl[prev_idx, 1]
                    norm = np.hypot(dx, dy) or 1.0
                    tx, ty = dx / norm, dy / norm
                else:
                    tx, ty = 1.0, 0.0
                px, py = -ty, tx  # lateral unit vector

                if scenario_type == "gesturing" or scenario_type == "human_lying":
                    # Single human at midpoint, face start (toward approaching robot)
                    hdg = np.arctan2(-ty, -tx)
                    trajs = np.full((num_samples, num_points, 3), [mx, my, hdg])
                    return trajs
                elif scenario_type == "lateral_approaching":
                    # Single human at one sidewalk edge, not spread laterally
                    offset = sw / 2.0 - 0.3
                    hx = mx + offset * px
                    hy = my + offset * py
                    _cut_in_params[id(sample_trajectory)] = {"px": px, "py": py, "sw": sw}
                    hdg = float(np.arctan2(-py, -px))
                    trajs = np.full((num_samples, num_points, 3), [hx, hy, hdg])
                    return trajs
                else:
                    # obstructing / human_lying: spread laterally
                    trajs = np.zeros((num_samples, num_points, 3))
                    for k in range(num_samples):
                        lateral = (k - (num_samples - 1) / 2.0) * 0.5
                        hx = mx + lateral * px
                        hy = my + lateral * py
                        hdg = rng.uniform(0, 2 * np.pi)
                        trajs[k] = np.full((num_points, 3), [hx, hy, hdg])
                    return trajs

            if scenario_type == "overtaking" and _cached_centerline[0] is not None:
                cl = _cached_centerline[0]
                # Tangent at start (avg over first 10 points for stability)
                idx = min(10, len(cl) - 1)
                tx = cl[idx, 0] - cl[0, 0]
                ty = cl[idx, 1] - cl[0, 1]
                nd = np.hypot(tx, ty) or 1.0
                tx /= nd
                ty /= nd
                px, py = -ty, tx  # perpendicular
                # Random side: ±1m lateral from start
                sign = rng.choice([-1.0, 1.0])
                sx = cl[0, 0] + sign * 1.0 * px
                sy = cl[0, 1] + sign * 1.0 * py
                # Toward centerline direction
                toward_x, toward_y = -sign * px, -sign * py
                hdg = np.arctan2(ty, tx)
                # Build Phase 2 extended centerline (like human_following: start → goal+3m)
                lead = 3.0
                diffs = np.diff(cl, axis=0)
                seg_lens = np.linalg.norm(diffs, axis=1)
                cumlen = np.insert(np.cumsum(seg_lens), 0, 0)
                total_len = cumlen[-1]
                final_tg = cl[-1] - cl[-2]
                final_tg /= np.linalg.norm(final_tg, keepdims=True) or 1.0
                ext_cl = np.vstack([cl, cl[-1] + lead * final_tg])
                ext_cum = np.append(cumlen, total_len + lead)
                # Build Phase 2 trajectory as ped_traj (goal+3m → start, resampled)
                path = _resample_path(ext_cl, ext_cum, 0, total_len + lead)
                trajs = _build_trajs_from_path(path, num_samples)
                # Override first point with Phase 1 start position
                trajs[:, 0, 0] = sx
                trajs[:, 0, 1] = sy
                trajs[:, 0, 2] = hdg
                _overtake_params[id(sample_trajectory)] = {
                    "forward": (tx, ty),
                    "toward": (toward_x, toward_y),
                }
                return trajs

            if scenario_type == "conversing" and _cached_centerline[0] is not None:
                cl = _cached_centerline[0]
                mid_idx = len(cl) // 2
                cx_, cy_ = cl[mid_idx]
                # Tangent and perpendicular at midpoint
                prev_idx = max(0, mid_idx - 5)
                next_idx = min(len(cl) - 1, mid_idx + 5)
                dx = cl[next_idx, 0] - cl[prev_idx, 0]
                dy = cl[next_idx, 1] - cl[prev_idx, 1]
                nd = np.hypot(dx, dy) or 1.0
                tx, ty = dx / nd, dy / nd
                px, py = -ty, tx  # perpendicular (CCW)
                # Place peds on opposite sides of centerline, facing each other
                offset = 0.5
                trajs = np.zeros((num_samples, num_points, 3))
                for k in range(num_samples):
                    if num_samples == 2:
                        pos_x = cx_ + (1 if k == 0 else -1) * offset * px
                        pos_y = cy_ + (1 if k == 0 else -1) * offset * py
                        yaw = np.arctan2(-py if k == 0 else py, -px if k == 0 else px)
                    else:
                        radius = 0.5
                        angle = 2 * np.pi * k / num_samples
                        pos_x = cx_ + radius * np.cos(angle)
                        pos_y = cy_ + radius * np.sin(angle)
                        yaw = (angle + np.pi) % (2 * np.pi)
                    trajs[k] = np.full((num_points, 3), [pos_x, pos_y, yaw])
                return trajs

            if scenario_type == "queueing" and _cached_centerline[0] is not None:
                cl = _cached_centerline[0]
                n_cl = len(cl)
                mid_idx = n_cl // 2
                mx, my = cl[mid_idx]
                # Tangent at midpoint
                if n_cl > 2:
                    prev_idx = max(0, mid_idx - 5)
                    next_idx = min(n_cl - 1, mid_idx + 5)
                    dx = cl[next_idx, 0] - cl[prev_idx, 0]
                    dy = cl[next_idx, 1] - cl[prev_idx, 1]
                    nd = np.hypot(dx, dy) or 1.0
                    tx, ty = dx / nd, dy / nd
                else:
                    tx, ty = 1.0, 0.0
                px, py = -ty, tx  # CCW perpendicular
                # Determine terrain side: vector from midpoint toward terrain edge
                road_vec = np.array([cx - mx, cy - my])
                if road_vec[0] * px + road_vec[1] * py > 0:
                    terrain_px, terrain_py = -px, -py
                else:
                    terrain_px, terrain_py = px, py
                # Queue from terrain edge inward, 1m spacing
                offset_start = sw / 2.0 - 0.5
                trajs = np.zeros((num_samples, num_points, 3))
                for k in range(num_samples):
                    offset = offset_start - k * 1.0
                    hx = mx + offset * terrain_px
                    hy = my + offset * terrain_py
                    yaw = np.arctan2(terrain_py, terrain_px)
                    trajs[k] = np.full((num_points, 3), [hx, hy, yaw])
                return trajs

            traj_y = np.linspace(0.1 * terrain_length, 0.9 * terrain_length, num_points)
            traj_x = np.full(num_points, cx)
            yaw = np.full(num_points, np.pi / 2)
            traj = np.stack([traj_x, traj_y, yaw], axis=-1)
            return np.tile(traj[None], (num_samples, 1, 1))

        _cached_centerline = [None]
        _cached_start_xy = [0.0, 0.0]
        _cached_goal_xy = [0.0, 0.0]

        def sample_start_and_goal(rng=None):
            rng = np.random if rng is None else rng
            ramp_len = ramp_length
            ramp_dep = ramp_depth
            sw = sidewalk_width
            cr = corner_radius
            tw = terrain_width
            tl = terrain_length

            # 8 sidewalk arm centerlines (excluding ramp cuts)
            # For h-arms: x varies in [x0, x1], y fixed. For v-arms: y varies, x fixed.
            arms = [
                # TL horizontal (westbound)
                {"x": None, "y": rt + sw / 2, "x0": 0, "x1": rl - cr - ramp_len, "heading": 0},
                # TL vertical (northbound)
                {"x": rl - sw / 2, "y": None, "y0": rt + cr + ramp_len, "y1": tl, "heading": np.pi / 2},
                # TR horizontal (eastbound)
                {"x": None, "y": rt + sw / 2, "x0": rr + cr + ramp_len, "x1": tw, "heading": 0},
                # TR vertical (northbound)
                {"x": rr + sw / 2, "y": None, "y0": rt + cr + ramp_len, "y1": tl, "heading": np.pi / 2},
                # BR horizontal (eastbound)
                {"x": None, "y": rb - sw / 2, "x0": rr + cr + ramp_len, "x1": tw, "heading": 0},
                # BR vertical (southbound)
                {"x": rr + sw / 2, "y": None, "y0": 0, "y1": rb - cr - ramp_len, "heading": np.pi / 2},
                # BL horizontal (westbound)
                {"x": None, "y": rb - sw / 2, "x0": 0, "x1": rl - cr - ramp_len, "heading": 0},
                # BL vertical (southbound)
                {"x": rl - sw / 2, "y": None, "y0": 0, "y1": rb - cr - ramp_len, "heading": np.pi / 2},
            ]

            def _make_projector(points):
                diffs = np.diff(points, axis=0)
                seg_lens = np.linalg.norm(diffs, axis=1)
                cl_arc = np.insert(np.cumsum(seg_lens), 0, 0.0)
                return _build_centerline_projector(points, cl_arc)

            if terrain_segment == "straight":
                # Filter arms with length >= 20
                valid = []
                for i, a in enumerate(arms):
                    L = (a["y1"] - a["y0"]) if a["x"] is not None else (a["x1"] - a["x0"])
                    if L >= 20:
                        valid.append((i, L))
                idx, _ = valid[rng.randint(len(valid))]
                a = arms[idx]
                hdg = a["heading"]
                margin = 5.0
                if a["x"] is not None:
                    # vertical arm
                    x = a["x"]
                    y = rng.uniform(a["y0"] + margin, a["y1"] - 20 - margin)
                else:
                    # horizontal arm
                    y = a["y"]
                    x = rng.uniform(a["x0"] + margin, a["x1"] - 20 - margin)
                gx = x + 20 * np.cos(hdg)
                gy = y + 20 * np.sin(hdg)
                flip = rng.rand() < 0.5
                if flip:
                    x, y, gx, gy = gx, gy, x, y
                    hdg = (hdg + np.pi) % (2 * np.pi)
                start = np.array([x, y, hdg])
                goal = np.array([gx, gy, hdg])
                _path = np.linspace([x, y], [gx, gy], 500)
                _cached_centerline[0] = _path
                _cached_start_xy[0], _cached_start_xy[1] = x, y
                _cached_goal_xy[0], _cached_goal_xy[1] = gx, gy
                projector = _make_projector(_path)
                return start, goal, 20.0, projector

            if terrain_segment == "corner":
                d1 = d2 = 10.0
                arc_radius = sw / 2.0
                arc_len = np.pi / 2 * arc_radius
                total_len = d1 + arc_len + d2 - 2 * (sw - cr)
                corner_turns = [
                    # TL H→V
                    {
                        "sx": (rl - cr - d1, rt + sw / 2),
                        "gx": (rl - sw / 2, rt + cr + d2),
                        "yaw": 0,
                        "entry": (rl - sw, rt + sw / 2),
                        "exit": (rl - sw / 2, rt + sw),
                        "center": (rl - sw, rt + sw),
                    },
                    # TL V→H
                    {
                        "sx": (rl - sw / 2, rt + cr + d1),
                        "gx": (rl - cr - d2, rt + sw / 2),
                        "yaw": -np.pi / 2,
                        "entry": (rl - sw / 2, rt + sw),
                        "exit": (rl - sw, rt + sw / 2),
                        "center": (rl - sw, rt + sw),
                    },
                    # TR H→V
                    {
                        "sx": (rr + cr + d1, rt + sw / 2),
                        "gx": (rr + sw / 2, rt + cr + d2),
                        "yaw": np.pi,
                        "entry": (rr + sw, rt + sw / 2),
                        "exit": (rr + sw / 2, rt + sw),
                        "center": (rr + sw, rt + sw),
                    },
                    # TR V→H
                    {
                        "sx": (rr + sw / 2, rt + cr + d1),
                        "gx": (rr + cr + d2, rt + sw / 2),
                        "yaw": -np.pi / 2,
                        "entry": (rr + sw / 2, rt + sw),
                        "exit": (rr + sw, rt + sw / 2),
                        "center": (rr + sw, rt + sw),
                    },
                    # BR H→V
                    {
                        "sx": (rr + cr + d1, rb - sw / 2),
                        "gx": (rr + sw / 2, rb - cr - d2),
                        "yaw": np.pi,
                        "entry": (rr + sw, rb - sw / 2),
                        "exit": (rr + sw / 2, rb - sw),
                        "center": (rr + sw, rb - sw),
                    },
                    # BR V→H
                    {
                        "sx": (rr + sw / 2, rb - cr - d1),
                        "gx": (rr + cr + d2, rb - sw / 2),
                        "yaw": np.pi / 2,
                        "entry": (rr + sw / 2, rb - sw),
                        "exit": (rr + sw, rb - sw / 2),
                        "center": (rr + sw, rb - sw),
                    },
                    # BL H→V
                    {
                        "sx": (rl - cr - d1, rb - sw / 2),
                        "gx": (rl - sw / 2, rb - cr - d2),
                        "yaw": 0,
                        "entry": (rl - sw, rb - sw / 2),
                        "exit": (rl - sw / 2, rb - sw),
                        "center": (rl - sw, rb - sw),
                    },
                    # BL V→H
                    {
                        "sx": (rl - sw / 2, rb - cr - d1),
                        "gx": (rl - cr - d2, rb - sw / 2),
                        "yaw": np.pi / 2,
                        "entry": (rl - sw / 2, rb - sw),
                        "exit": (rl - sw, rb - sw / 2),
                        "center": (rl - sw, rb - sw),
                    },
                ]
                turn = corner_turns[rng.randint(len(corner_turns))]
                sx, sy = turn["sx"]
                gx, gy = turn["gx"]
                start_xy = np.array(turn["sx"])
                entry_xy = np.array(turn["entry"])
                exit_xy = np.array(turn["exit"])
                goal_xy = np.array(turn["gx"])
                center_xy = np.array(turn["center"])

                n_pts = 150
                pts1 = np.linspace(start_xy, entry_xy, n_pts)
                pts2 = np.linspace(exit_xy, goal_xy, n_pts)

                if arc_radius > 0.0:
                    ang_e = np.arctan2(entry_xy[1] - center_xy[1], entry_xy[0] - center_xy[0])
                    ang_x = np.arctan2(exit_xy[1] - center_xy[1], exit_xy[0] - center_xy[0])
                    diff = (ang_x - ang_e + np.pi) % (2 * np.pi) - np.pi
                    angles = np.linspace(ang_e, ang_e + diff, n_pts)
                    pts_arc = np.column_stack(
                        [
                            center_xy[0] + arc_radius * np.cos(angles),
                            center_xy[1] + arc_radius * np.sin(angles),
                        ]
                    )
                    centerline = np.concatenate([pts1, pts_arc, pts2])
                else:
                    centerline = np.concatenate([pts1, pts2])
                _cached_centerline[0] = centerline
                _cached_start_xy[0], _cached_start_xy[1] = sx, sy
                _cached_goal_xy[0], _cached_goal_xy[1] = gx, gy
                projector = _make_projector(centerline)
                # Forward heading at the goal = direction of the final arm.
                goal_yaw = float(np.arctan2(goal_xy[1] - exit_xy[1], goal_xy[0] - exit_xy[0]))
                return np.array([sx, sy, turn["yaw"]]), np.array([gx, gy, goal_yaw]), total_len, projector

            if terrain_segment == "crosswalk":
                cw = [
                    {"sx": (rl, rt + cr + ramp_len / 2), "gx": (rr, rt + cr + ramp_len / 2)},  # top
                    {"sx": (rl, rb - cr - ramp_len / 2), "gx": (rr, rb - cr - ramp_len / 2)},  # bottom
                    {"sx": (rl - cr - ramp_len / 2, rt), "gx": (rl - cr - ramp_len / 2, rb)},  # left
                    {"sx": (rr + cr + ramp_len / 2, rt), "gx": (rr + cr + ramp_len / 2, rb)},  # right
                ]
                pair = cw[rng.randint(len(cw))]
                sx, sy = pair["sx"]
                gx, gy = pair["gx"]
                flip = rng.rand() < 0.5
                if flip:
                    sx, sy, gx, gy = gx, gy, sx, sy
                dy = gy - sy
                dx = gx - sx
                dist = float(np.hypot(dx, dy))
                start = np.array([sx, sy, np.arctan2(dy, dx)])
                goal = np.array([gx, gy, np.arctan2(dy, dx)])
                _path = np.linspace([sx, sy], [gx, gy], 500)
                _cached_centerline[0] = _path
                _cached_start_xy[0], _cached_start_xy[1] = sx, sy
                _cached_goal_xy[0], _cached_goal_xy[1] = gx, gy
                projector = _make_projector(_path)
                return start, goal, dist, projector

            raise ValueError(f"Unknown terrain_segment: {terrain_segment}")

        def check_outside_boundary(x, y):
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            scalar = x.ndim == 0
            x, y = np.atleast_1d(x), np.atleast_1d(y)
            from shapely.geometry import Point as _Pt

            dists = np.array(
                [
                    (
                        walkable_geom.distance(_Pt(float(xi), float(yi)))
                        if not walkable_geom.contains(_Pt(float(xi), float(yi)))
                        else -walkable_geom.boundary.distance(_Pt(float(xi), float(yi)))
                    )
                    for xi, yi in zip(x, y)
                ]
            )
            return dists[0] if scalar else dists

        # Static route-map metadata in the same local frame as the boundary checker.
        # Include complete crossing regions, including the gaps between painted stripes.
        route_crosswalks = unary_union([
            box(rl, rt + crosswalk_offset, rr, rt + crosswalk_offset + crosswalk_width),
            box(rl, rb - crosswalk_offset - crosswalk_width, rr, rb - crosswalk_offset),
            box(rl - crosswalk_offset - crosswalk_width, rb, rl - crosswalk_offset, rt),
            box(rr + crosswalk_offset, rb, rr + crosswalk_offset + crosswalk_width, rt),
        ]) if use_ramp else Point(0, 0).buffer(0.0)
        check_outside_boundary.route_geometries = (sidewalk_geom, route_crosswalks)

        # Centerline: diagonal from top-left sidewalk toward bottom-right sidewalk,
        # passing through the intersection center.
        start_pt = sidewalk_polys[0].representative_point()
        goal_pt = sidewalk_polys[2].representative_point()
        _cl_xy = np.linspace(
            np.array([start_pt.x, start_pt.y]),
            np.array([goal_pt.x, goal_pt.y]),
            500,
        )
        _cl_arc = np.linspace(0, np.hypot(goal_pt.x - start_pt.x, goal_pt.y - start_pt.y), 500)
        project_to_centerline = _build_centerline_projector(_cl_xy, _cl_arc)

        return (
            meshes_list,
            origin,
            sample_pos,
            sample_trajectory,
            sample_start_and_goal,
            check_outside_boundary,
            project_to_centerline,
            building_locations,
            car_locations,
            None,
            None,
        )


@configclass
class IntersectionTerrainCfg(BaseTerrainCfg):
    """Configuration for an intersection terrain with 4 L-shaped corner sidewalks."""

    use_ramp: bool = True
    scenario_type: str = "empty"
    terrain_segment: str = "straight"  # one of ["straight", "corner", "crosswalk"]
    class_type = IntersectionTerrain
