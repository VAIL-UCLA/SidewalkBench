from __future__ import annotations

from shapely.ops import unary_union

from ._assets import load_building_infos as _load_building_infos
from ._assets import load_car_infos as _load_car_infos
from ._crosswalk_ramps import build_crosswalk_ramps
from .base import (
    BaseTerrainCfg,
    Terrain,
    Polygon,
    _build_centerline_projector,
    _generate_block_centerline,
    configclass,
    interpolate,
    np,
    trimesh,
)


def _sample_block_building_locations(
    terrain_width,
    terrain_length,
    sidewalk_poly,
    spline_x,
    spline_y,
    half_w,
    sidewalk_height,
    total_length,
    crosswalk_zones=None,
):
    building_infos = _load_building_infos()
    if not building_infos:
        return None

    sidewalk_buffer = sidewalk_poly.buffer(1.95)
    sidewalk_buffer_large = sidewalk_poly.buffer(19.95)

    if not sidewalk_buffer.is_valid:
        sidewalk_buffer = sidewalk_buffer.buffer(0)

    n_samples = max(5, int(total_length / 10))
    s_vals = np.linspace(0.05 * total_length, 0.95 * total_length, n_samples)

    placed_polys = []
    locations = []
    max_attempts = 30
    ds = total_length * 0.01
    margin = half_w + 2.0

    for s in s_vals:
        xy = np.array([float(spline_x(s)), float(spline_y(s))])
        s_deriv = max(s, ds)
        sx1 = float(spline_x(s_deriv, 1))
        sy1 = float(spline_y(s_deriv, 1))
        nd = np.hypot(sx1, sy1) or 1.0
        tx, ty = sx1 / nd, sy1 / nd
        tangent_yaw = np.arctan2(ty, tx)

        # Try both sides: building side (close), car side (beyond road)
        for side_sign, side_margin in [(-1, margin), (+1, half_w + 20.0)]:
            nx = -ty * side_sign
            ny = tx * side_sign

            for _ in range(max_attempts // 2):
                b_idx = np.random.randint(0, len(building_infos))
                b_info = building_infos[b_idx]
                b_len = b_info["length"]
                b_wid = b_info["width"]
                rotated = np.random.choice([False, True])

                l = b_len if not rotated else b_wid
                w = b_wid if not rotated else b_len

                dir_lx, dir_ly = (nx, ny)
                dir_wx, dir_wy = (tx, ty)

                cx = xy[0] + nx * (side_margin + l / 2.0)
                cy = xy[1] + ny * (side_margin + l / 2.0)

                corners = np.array(
                    [
                        [cx + l / 2 * dir_lx + w / 2 * dir_wx, cy + l / 2 * dir_ly + w / 2 * dir_wy],
                        [cx - l / 2 * dir_lx + w / 2 * dir_wx, cy - l / 2 * dir_ly + w / 2 * dir_wy],
                        [cx - l / 2 * dir_lx - w / 2 * dir_wx, cy - l / 2 * dir_ly - w / 2 * dir_wy],
                        [cx + l / 2 * dir_lx - w / 2 * dir_wx, cy + l / 2 * dir_ly - w / 2 * dir_wy],
                    ]
                )

                if (
                    np.any(corners[:, 0] < 0)
                    or np.any(corners[:, 0] > terrain_width)
                    or np.any(corners[:, 1] < 0)
                    or np.any(corners[:, 1] > terrain_length)
                ):
                    continue

                bbox_poly = Polygon(corners)
                if sidewalk_buffer.intersects(bbox_poly) and side_sign == -1:
                    continue
                if sidewalk_buffer_large.intersects(bbox_poly) and side_sign == +1:
                    continue

                if crosswalk_zones is not None:
                    hit_cw = False
                    for cw_zone in crosswalk_zones:
                        if bbox_poly.intersects(cw_zone):
                            hit_cw = True
                            break
                    if hit_cw:
                        continue

                overlap = False
                for pp in placed_polys:
                    if bbox_poly.intersects(pp):
                        overlap = True
                        break
                if overlap:
                    continue

                placed_polys.append(bbox_poly)
                building_yaw = tangent_yaw + (np.pi / 2 if not rotated else 0.0)
                locations.append([float(cx), float(cy), float(sidewalk_height), float(building_yaw), b_idx])
                break

    if not locations:
        return None
    return np.array(locations, dtype=float)


def _sample_block_car_locations(
    terrain_width,
    terrain_length,
    sidewalk_poly,
    spline_x,
    spline_y,
    half_w,
    ground_height,
    total_length,
    crosswalk_zones=None,
):
    car_infos = _load_car_infos()
    if not car_infos:
        return None

    sidewalk_buffer = sidewalk_poly.buffer(2.15)
    if not sidewalk_buffer.is_valid:
        sidewalk_buffer = sidewalk_buffer.buffer(0)

    n_samples = max(3, int(total_length / 20))
    s_vals = np.linspace(0.05 * total_length, 0.95 * total_length, n_samples)

    placed_polys = []
    locations = []
    ds = total_length * 0.01

    for s in s_vals:
        xy = np.array([float(spline_x(s)), float(spline_y(s))])
        s_deriv = max(s, ds)
        sx1 = float(spline_x(s_deriv, 1))
        sy1 = float(spline_y(s_deriv, 1))
        nd = np.hypot(sx1, sy1) or 1.0
        tx, ty = sx1 / nd, sy1 / nd
        tangent_yaw = np.arctan2(ty, tx)

        sign = +1  # cars on the other side only
        nx = -ty * sign
        ny = tx * sign

        for _ in range(10):
            margin = np.random.uniform(half_w + 2.2, half_w + 3.0)

            c_idx = np.random.randint(0, len(car_infos))
            c_info = car_infos[c_idx]
            c_len = c_info["length"]
            c_wid = c_info["width"]
            rotated = np.random.choice([False, True])

            l = c_len if not rotated else c_wid
            w = c_wid if not rotated else c_len

            dir_lx, dir_ly = (nx, ny)
            dir_wx, dir_wy = (tx, ty)

            cx = xy[0] + nx * (margin + l / 2.0)
            cy = xy[1] + ny * (margin + l / 2.0)

            corners = np.array(
                [
                    [cx + l / 2 * dir_lx + w / 2 * dir_wx, cy + l / 2 * dir_ly + w / 2 * dir_wy],
                    [cx - l / 2 * dir_lx + w / 2 * dir_wx, cy - l / 2 * dir_ly + w / 2 * dir_wy],
                    [cx - l / 2 * dir_lx - w / 2 * dir_wx, cy - l / 2 * dir_ly - w / 2 * dir_wy],
                    [cx + l / 2 * dir_lx - w / 2 * dir_wx, cy + l / 2 * dir_ly - w / 2 * dir_wy],
                ]
            )

            if (
                np.any(corners[:, 0] < 0)
                or np.any(corners[:, 0] > terrain_width)
                or np.any(corners[:, 1] < 0)
                or np.any(corners[:, 1] > terrain_length)
            ):
                continue

            bbox_poly = Polygon(corners)
            if sidewalk_buffer.intersects(bbox_poly):
                continue

            if crosswalk_zones is not None:
                hit_cw = False
                for cw_zone in crosswalk_zones:
                    if bbox_poly.intersects(cw_zone):
                        hit_cw = True
                        break
                if hit_cw:
                    continue

            overlap = False
            for pp in placed_polys:
                if bbox_poly.intersects(pp):
                    overlap = True
                    break
            if overlap:
                continue

            placed_polys.append(bbox_poly)
            car_yaw = tangent_yaw + (np.pi if rotated else 0.0)
            locations.append([float(cx), float(cy), float(ground_height), float(car_yaw), c_idx])
            break

    if not locations:
        return None
    return np.array(locations, dtype=float)


class BlockSidewalkTerrain(Terrain):
    """Procedural terrain builder for BlockSidewalkTerrainCfg."""

    def build(self) -> tuple:
        cfg = self.cfg
        generator_cfg = self.generator_cfg
        """Generate a procedural sidewalk by chaining straight/turn blocks."""
        sidewalk_width = generator_cfg.sidewalk_width
        terrain_width = cfg.size[0]
        terrain_length = cfg.size[1]
        half_w = sidewalk_width / 2
        sidewalk_height = generator_cfg.sidewalk_height
        ground_height = generator_cfg.ground_height
        num_blocks = 50
        block_length = 10.0

        # Generate centerline
        centerline, _ = _generate_block_centerline(num_blocks, block_length, sidewalk_width)

        # Center the path in the terrain
        cl_min = centerline.min(axis=0)
        cl_max = centerline.max(axis=0)
        cl_center = (cl_min + cl_max) / 2
        terrain_center = np.array([terrain_width / 2, terrain_length / 2])
        offset = terrain_center - cl_center
        centerline = centerline + offset

        # Compute arc-length parameterization
        diffs = np.diff(centerline, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        arc_length = np.concatenate([[0], np.cumsum(seg_lens)])
        total_length = arc_length[-1]

        # Detect straight segments (low curvature regions for crosswalk placement)
        # Compute heading at each point
        dx_ds = np.gradient(centerline[:, 0])
        dy_ds = np.gradient(centerline[:, 1])
        heading = np.arctan2(dy_ds, dx_ds)
        # Curvature = |d_heading/d_s| with smoothing
        curvature = np.abs(np.gradient(np.unwrap(heading)) / (seg_lens.mean()))
        curvature_smooth = np.convolve(curvature, np.ones(5) / 5, mode="same")
        is_straight = curvature_smooth < 0.05  # radians per meter threshold
        # Find contiguous straight arc-length ranges long enough for crosswalks
        straight_ranges = []
        in_straight = False
        range_start = 0.0
        for i in range(len(is_straight)):
            s = arc_length[i]
            if is_straight[i] and not in_straight:
                range_start = s
                in_straight = True
            elif not is_straight[i] and in_straight:
                if s - range_start > 20.0:  # need at least 20m for a crosswalk
                    straight_ranges.append((range_start, s))
                in_straight = False
        if in_straight:
            s = arc_length[-1]
            if s - range_start > 20.0:
                straight_ranges.append((range_start, s))
        print(f"[crosswalk] straight segments: {straight_ranges}")

        # Fit splines parameterized by arc length
        spline_x = interpolate.CubicSpline(arc_length, centerline[:, 0])
        spline_y = interpolate.CubicSpline(arc_length, centerline[:, 1])

        # Build sidewalk polygon from centerline + width
        tangents = np.zeros_like(centerline)
        tangents[:-1] = diffs / (seg_lens[:, None] + 1e-8)
        tangents[-1] = tangents[-2]
        # Smooth tangents
        from scipy.ndimage import uniform_filter1d

        tangents[:, 0] = uniform_filter1d(tangents[:, 0], size=3)
        tangents[:, 1] = uniform_filter1d(tangents[:, 1], size=3)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / (norms + 1e-8)
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)

        left_edge = centerline + half_w * normals
        right_edge = centerline - half_w * normals
        sidewalk_polygon = np.concatenate([left_edge, right_edge[::-1]], axis=0)
        sidewalk_poly = Polygon(sidewalk_polygon)

        if not sidewalk_poly.is_valid:
            sidewalk_poly = sidewalk_poly.buffer(0)

        # Save original sidewalk polygon (pre-culling) for ground cutout and ramp intersection
        sidewalk_poly_full = sidewalk_poly
        if hasattr(sidewalk_poly_full, "geoms") and not hasattr(sidewalk_poly_full, "exterior"):
            sidewalk_poly_full = unary_union(sidewalk_poly_full)
        if not sidewalk_poly_full.is_valid:
            sidewalk_poly_full = sidewalk_poly_full.buffer(0)

        # Ground polygon (full terrain)
        ground_polygon = Polygon(
            [
                [0, 0],
                [terrain_width, 0],
                [terrain_width, terrain_length],
                [0, terrain_length],
            ]
        )

        # Road strip on the car side
        road_buffer = 2.0
        road_width = 10.0
        out_normals = normals
        road_inner = left_edge + road_buffer * out_normals
        road_outer = left_edge + (road_buffer + road_width) * out_normals
        road_polygon_points = np.concatenate([road_inner, road_outer[::-1]], axis=0)
        road_poly = Polygon(road_polygon_points)
        if not road_poly.is_valid:
            road_poly = road_poly.buffer(0)
        road_poly = road_poly.intersection(ground_polygon)

        # --- Crosswalk stripes: placed every ~100m along the centerline ---
        crosswalk_spacing = 100.0
        stripe_gap = 0.3
        num_stripes_per_crosswalk = 15
        stripe_width = stripe_gap
        crosswalk_stripe_polygons = []
        crosswalk_zone_polygons = []  # 100 m-wide zones for building exclusion/road cutout

        # Place crosswalks only on straight segments
        cw_positions = []
        for r_start, r_end in straight_ranges:
            margin = crosswalk_spacing * 0.5
            pos = np.arange(r_start + margin, r_end - margin + 1e-6, crosswalk_spacing)
            cw_positions.extend(pos.tolist())
        cw_positions = np.array(cw_positions)
        # Ensure at least 2 crosswalks — one in each of the top 2 longest straight segments
        if len(cw_positions) < 2 and len(straight_ranges) > 0:
            sorted_segs = sorted(straight_ranges, key=lambda r: r[1] - r[0], reverse=True)
            cw_positions = []
            for r_start, r_end in sorted_segs[:2]:
                cw_positions.append((r_start + r_end) / 2)
            cw_positions = np.array(cw_positions)
        print(f"[crosswalk] total_length={total_length:.1f}, spacing={crosswalk_spacing}, positions={len(cw_positions)}")
        for cw_s in cw_positions:
            cw_xy = np.array([float(spline_x(cw_s)), float(spline_y(cw_s))])
            dx = float(spline_x(cw_s, 1))
            dy = float(spline_y(cw_s, 1))
            n = np.hypot(dx, dy) or 1.0
            tx, ty = dx / n, dy / n
            nx, ny = -ty, tx
            stripe_period = stripe_gap + stripe_width
            total_stripe_span = num_stripes_per_crosswalk * stripe_period
            start_offset = -total_stripe_span / 2.0

            # 100 m-wide road zone along the normal, stripe-span along the tangent.
            zone_half_w = 50.0
            cz_p1 = [cw_xy[0] + start_offset * tx - zone_half_w * nx, cw_xy[1] + start_offset * ty - zone_half_w * ny]
            cz_p2 = [cw_xy[0] + (-start_offset) * tx - zone_half_w * nx, cw_xy[1] + (-start_offset) * ty - zone_half_w * ny]
            cz_p3 = [cw_xy[0] + (-start_offset) * tx + zone_half_w * nx, cw_xy[1] + (-start_offset) * ty + zone_half_w * ny]
            cz_p4 = [cw_xy[0] + start_offset * tx + zone_half_w * nx, cw_xy[1] + start_offset * ty + zone_half_w * ny]
            cz_zone = Polygon([cz_p1, cz_p2, cz_p3, cz_p4])
            if not cz_zone.is_valid:
                cz_zone = cz_zone.buffer(0)
            cz_zone = cz_zone.intersection(ground_polygon)
            if not cz_zone.is_empty:
                crosswalk_zone_polygons.append(cz_zone)

            for si in range(num_stripes_per_crosswalk):
                t = start_offset + si * stripe_period
                t1 = t
                t2 = t + stripe_width
                # Extend each stripe across the road zone before clipping to the
                # sidewalk: another nearby leg may cross this same road.
                p1 = [cw_xy[0] + t1 * tx - zone_half_w * nx, cw_xy[1] + t1 * ty - zone_half_w * ny]
                p2 = [cw_xy[0] + t2 * tx - zone_half_w * nx, cw_xy[1] + t2 * ty - zone_half_w * ny]
                p3 = [cw_xy[0] + t2 * tx + zone_half_w * nx, cw_xy[1] + t2 * ty + zone_half_w * ny]
                p4 = [cw_xy[0] + t1 * tx + zone_half_w * nx, cw_xy[1] + t1 * ty + zone_half_w * ny]
                stripe_poly = Polygon([p1, p2, p3, p4])
                if not stripe_poly.is_valid:
                    stripe_poly = stripe_poly.buffer(0)
                stripe_poly = stripe_poly.intersection(sidewalk_poly_full).intersection(cz_zone)
                if not stripe_poly.is_empty:
                    crosswalk_stripe_polygons.append(stripe_poly)

        # A wide road zone can also cut a nearby sidewalk leg. Build transitions
        # from the union of actual cuts, including these incidental crossings.
        cw_cut = unary_union(crosswalk_zone_polygons)
        ramp_footprint, ramp_meshes, crosswalk_regions, blend_curb_meshes = build_crosswalk_ramps(
            sidewalk_poly_full, cw_cut, sidewalk_height, ground_height
        )
        print(f"[crosswalk] ramps={len(ramp_meshes)}")

        # --- Curb: narrow strip along the boundary of the combined road area ---
        combined_road = road_poly.union(cw_cut)
        if not combined_road.is_valid:
            combined_road = combined_road.buffer(0)

        curb_width = 0.15
        curb_height = 0.0
        road_boundary = combined_road.boundary
        curb_poly = road_boundary.buffer(curb_width, cap_style=2, join_style=2)
        if not curb_poly.is_valid:
            curb_poly = curb_poly.buffer(0)
        curb_poly = curb_poly.intersection(combined_road)
        # Open the curb at all sidewalk/road crossings, including incidental ones.
        if not crosswalk_regions.is_empty:
            curb_poly = curb_poly.difference(crosswalk_regions.buffer(0.01))
        if not curb_poly.is_valid:
            curb_poly = curb_poly.buffer(0)

        curb_mesh = None
        if not curb_poly.is_empty:
            if hasattr(curb_poly, "exterior"):
                parts = [curb_poly]
            elif hasattr(curb_poly, "geoms"):
                parts = [g for g in curb_poly.geoms if hasattr(g, "exterior")]
            else:
                parts = []
            curb_pieces = []
            for cp in parts:
                if not cp.is_empty:
                    curb_pieces.append(
                        trimesh.creation.extrude_polygon(cp, height=sidewalk_height + curb_height, engine="manifold")
                    )
            if curb_pieces:
                curb_mesh = trimesh.util.concatenate(curb_pieces)
                # Add grey vertex colors
                curb_mesh.visual.vertex_colors = (128, 128, 128, 255)
                print(f"[curb] pieces={len(curb_pieces)}, verts={len(curb_mesh.vertices)}")
        else:
            curb_mesh = None
            print("[curb] FAIL: empty")

        # Add blend curbs to curb mesh (they share the same grey color)
        if blend_curb_meshes:
            if curb_mesh is not None:
                curb_mesh = trimesh.util.concatenate([curb_mesh] + blend_curb_meshes)
            else:
                curb_mesh = trimesh.util.concatenate(blend_curb_meshes)

        # --- Add crosswalk zones to road polygon, cull sidewalk and ground ---
        print(f"[crosswalk] zones={len(crosswalk_zone_polygons)}, stripes={len(crosswalk_stripe_polygons)}")
        if crosswalk_zone_polygons:
            print(f"[crosswalk] cw_cut type={type(cw_cut).__name__}, area={cw_cut.area:.1f}")
            road_poly = combined_road
            if not road_poly.is_valid:
                road_poly = road_poly.buffer(0)
            sw_area_before = sidewalk_poly.area
            sidewalk_poly = sidewalk_poly.difference(cw_cut)
            print(f"[crosswalk] sidewalk area: {sw_area_before:.1f} → {sidewalk_poly.area:.1f}")
            if not sidewalk_poly.is_valid:
                sidewalk_poly = sidewalk_poly.buffer(0)
            if hasattr(sidewalk_poly, "geoms") and not hasattr(sidewalk_poly, "exterior"):
                sidewalk_poly = unary_union(sidewalk_poly)

        # Also subtract ramps from sidewalk
        if not ramp_footprint.is_empty:
            sidewalk_poly = sidewalk_poly.difference(ramp_footprint)
            if not sidewalk_poly.is_valid:
                sidewalk_poly = sidewalk_poly.buffer(0)
            if hasattr(sidewalk_poly, "geoms") and not hasattr(sidewalk_poly, "exterior"):
                sidewalk_poly = unary_union(sidewalk_poly)

        # Cull ground: subtract sidewalk, road, and crosswalk zones
        ground_cut = ground_polygon.difference(sidewalk_poly_full)
        for cw_z in crosswalk_zone_polygons:
            ground_cut = ground_cut.difference(cw_z)
        ground_cut = ground_cut.difference(road_poly)
        print(f"[ground] full area={ground_polygon.area:.0f}, after cull={ground_cut.area:.1f}")
        if not ground_cut.is_valid:
            ground_cut = ground_cut.buffer(0)
        if hasattr(ground_cut, "geoms") and not hasattr(ground_cut, "exterior"):
            pieces = []
            for g in ground_cut.geoms:
                if not g.is_empty and hasattr(g, "exterior"):
                    pieces.append(trimesh.creation.extrude_polygon(g, height=sidewalk_height, engine="manifold"))
            ground_mesh = trimesh.util.concatenate(pieces) if pieces else trimesh.creation.box(extents=(0.01, 0.01, 0.001))
        elif not ground_cut.is_empty:
            ground_mesh = trimesh.creation.extrude_polygon(ground_cut, height=sidewalk_height, engine="manifold")
        else:
            ground_mesh = trimesh.creation.box(extents=(0.01, 0.01, 0.001))

        # Extrude sidewalk — handle MultiPolygon from culling
        if hasattr(sidewalk_poly, "geoms") and not hasattr(sidewalk_poly, "exterior"):
            pieces = []
            for g in sidewalk_poly.geoms:
                if not g.is_empty and hasattr(g, "exterior"):
                    pieces.append(trimesh.creation.extrude_polygon(g, height=sidewalk_height, engine="manifold"))
            sidewalk_mesh = (
                trimesh.util.concatenate(pieces) if pieces else trimesh.creation.box(extents=(0.01, 0.01, 0.001))
            )
        elif not sidewalk_poly.is_empty:
            sidewalk_mesh = trimesh.creation.extrude_polygon(sidewalk_poly, height=sidewalk_height, engine="manifold")
        else:
            sidewalk_mesh = trimesh.creation.box(extents=(0.01, 0.01, 0.001))

        if not road_poly.is_empty:
            if hasattr(road_poly, "geoms") and not hasattr(road_poly, "exterior"):
                pieces = []
                for g in road_poly.geoms:
                    if not g.is_empty and hasattr(g, "exterior"):
                        pieces.append(trimesh.creation.extrude_polygon(g, height=ground_height, engine="manifold"))
                road_mesh = (
                    trimesh.util.concatenate(pieces) if pieces else trimesh.creation.box(extents=(0.01, 0.01, 0.001))
                )
            else:
                road_mesh = trimesh.creation.extrude_polygon(road_poly, height=ground_height, engine="manifold")
        else:
            road_mesh = trimesh.creation.box(extents=(0.01, 0.01, 0.001))

        crosswalk_mesh = None
        if crosswalk_stripe_polygons:
            stripe_union = unary_union(crosswalk_stripe_polygons)
            if not stripe_union.is_valid:
                stripe_union = stripe_union.buffer(0)
            if hasattr(stripe_union, "geoms") and not hasattr(stripe_union, "exterior"):
                stripe_parts = [g for g in stripe_union.geoms if hasattr(g, "exterior")]
            else:
                stripe_parts = [stripe_union] if hasattr(stripe_union, "exterior") else []
            stripe_meshes = []
            for sp in stripe_parts:
                if not sp.is_empty and hasattr(sp, "exterior"):
                    sm = trimesh.creation.extrude_polygon(polygon=sp, height=ground_height + 0.001, engine="manifold")
                    sm.visual.vertex_colors = (255, 255, 255, 255)
                    stripe_meshes.append(sm)
            if stripe_meshes:
                crosswalk_mesh = trimesh.util.concatenate(stripe_meshes)

        # Concatenate ramp meshes with sidewalk (curb is separate in other slot)
        if ramp_meshes:
            sidewalk_mesh = trimesh.util.concatenate([sidewalk_mesh] + ramp_meshes)

        # Build meshes_list following the 3-mesh-per-group convention:
        # Group 1: [sidewalk, ground, crosswalk] — walkable, non-walkable[0]=grass, other=white stripes
        # Group 2: [placeholder, road, curb] — walkable, non-walkable[1]=asphalt, other=grey curb
        # Last element: border placeholder
        placeholder = trimesh.creation.box(extents=(0.01, 0.01, 0.001))

        cw = crosswalk_mesh if crosswalk_mesh is not None else placeholder
        cb = curb_mesh if curb_mesh is not None else placeholder

        meshes_list = [sidewalk_mesh, ground_mesh, cw, placeholder, road_mesh, cb]
        print(
            f"[final] meshes: sw={len(sidewalk_mesh.vertices)}v swZ=[{sidewalk_mesh.vertices[:, 2].min():.2f},{sidewalk_mesh.vertices[:, 2].max():.2f}], "
            f"gnd={len(ground_mesh.vertices)}v gndZ=[{ground_mesh.vertices[:, 2].min():.2f},{ground_mesh.vertices[:, 2].max():.2f}], "
            f"rd={len(road_mesh.vertices)}v rdZ=[{road_mesh.vertices[:, 2].min():.2f},{road_mesh.vertices[:, 2].max():.2f}], "
            f"cw={len(cw.vertices)}v"
        )

        origin = np.array([0.0, 0.0, 0.0])

        # --- Sampling functions ---

        def _arc_to_xy(s):
            s = np.clip(s, 0, total_length)
            return np.stack([spline_x(s), spline_y(s)], axis=-1)

        def _arc_to_yaw(s):
            s = np.clip(s, 1e-3, total_length - 1e-3)
            dx = spline_x(s, 1)
            dy = spline_y(s, 1)
            return np.arctan2(dy, dx)

        def sample_pos(walkable=True, num_samples=1, rng=None):
            rng = np.random if rng is None else rng
            xs = np.zeros(num_samples)
            ys = np.zeros(num_samples)
            for i in range(num_samples):
                for _ in range(50):  # max retries
                    s = rng.uniform(0.05 * total_length + 3, 0.77 * total_length - 3)
                    xy = _arc_to_xy(s)
                    if walkable:
                        lat = rng.uniform(0, half_w)
                        if lat < half_w / 2:
                            lat -= half_w
                        yaw = _arc_to_yaw(s)
                        normal_dir = np.array([-np.sin(yaw), np.cos(yaw)])
                        xy = xy + float(lat) * normal_dir
                    # Reject if inside crosswalk zone
                    if crosswalk_zone_polygons:
                        from shapely.geometry import Point as _Pt

                        p = _Pt(float(xy[0]), float(xy[1]))
                        hit = False
                        for cw in crosswalk_zone_polygons:
                            if cw.contains(p):
                                hit = True
                                break
                        if not hit:
                            xs[i], ys[i] = xy
                            break
                    else:
                        xs[i], ys[i] = xy
                        break
            return xs, ys

        def sample_trajectory(
            num_points, num_samples, delta_distance, scenario_type="default", ped_lateral="center", rng=None, **kwargs
        ):
            rng = np.random if rng is None else rng
            ped_trajs = []
            for _ in range(num_samples):
                lat = rng.uniform(-half_w, half_w)
                s_vals = np.arange(0.05 * total_length + 3, 0.77 * total_length - 3, delta_distance)
                xy = _arc_to_xy(s_vals)
                yaw = _arc_to_yaw(s_vals)
                normal_dir = np.stack([-np.sin(yaw), np.cos(yaw)], axis=-1)
                xy = xy + lat * normal_dir
                ped_trajs.append(np.stack([xy[:, 0], xy[:, 1], yaw], axis=-1))
            return np.stack(ped_trajs, axis=0)

        def sample_start_and_goal(rng=None):
            s_start = 0.05 * total_length
            s_goal = 0.77 * total_length
            start_xy = _arc_to_xy(s_start)
            goal_xy = _arc_to_xy(s_goal)
            start_yaw = _arc_to_yaw(s_start)
            route_length = float(s_goal - s_start)
            return np.array([start_xy[0], start_xy[1], start_yaw]), np.array([goal_xy[0], goal_xy[1]]), route_length

        def check_outside_boundary(x, y):
            """Return signed distance to boundary (positive=outside, negative=inside)."""
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            scalar = x.ndim == 0
            x = np.atleast_1d(x)
            y = np.atleast_1d(y)
            s_samples = np.linspace(0, total_length, int(total_length * 10))
            cl_samples = _arc_to_xy(s_samples)
            result = np.zeros(len(x), dtype=np.float64)
            for i in range(len(x)):
                dists = np.sqrt((cl_samples[:, 0] - x[i]) ** 2 + (cl_samples[:, 1] - y[i]) ** 2)
                result[i] = dists.min() - half_w
            if scalar:
                return result[0]
            return result

        # Static route map: crosswalk regions take precedence over sidewalk labels.
        check_outside_boundary.route_geometries = (sidewalk_poly_full.difference(cw_cut), crosswalk_regions)

        # Block sidewalk centerline projector — use arc-length parameterized splines
        _cl_s = np.linspace(0, total_length, int(total_length * 10))
        _cl_xy_block = _arc_to_xy(_cl_s)
        project_to_centerline = _build_centerline_projector(_cl_xy_block, _cl_s)

        building_locations = _sample_block_building_locations(
            terrain_width,
            terrain_length,
            sidewalk_poly,
            spline_x,
            spline_y,
            half_w,
            sidewalk_height,
            total_length,
            crosswalk_zones=crosswalk_zone_polygons,
        )
        car_locations = _sample_block_car_locations(
            terrain_width,
            terrain_length,
            sidewalk_poly,
            spline_x,
            spline_y,
            half_w,
            ground_height,
            total_length,
            crosswalk_zones=crosswalk_zone_polygons,
        )

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
            _arc_to_xy,
            total_length,
        )


@configclass
class BlockSidewalkTerrainCfg(BaseTerrainCfg):
    """Configuration for a block-based procedural sidewalk terrain."""

    class_type = BlockSidewalkTerrain
