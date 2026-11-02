from __future__ import annotations

from .base import (
    BaseTerrainCfg,
    Terrain,
    Polygon,
    _build_centerline_projector,
    configclass,
    np,
    trimesh,
)


class CrosswalkTerrain(Terrain):
    """Procedural terrain builder for CrosswalkTerrainCfg."""

    def build(self) -> tuple:
        cfg = self.cfg
        generator_cfg = self.generator_cfg
        """Generate a crosswalk terrain: center road/sidewalk with terrain on left and right.

        Y layout: bottom sidewalk → road → top sidewalk (no gaps).
        X layout: terrain_left → walkable (sidewalk_width) → terrain_right.
        """
        sidewalk_width = generator_cfg.sidewalk_width
        terrain_width = cfg.size[0]
        terrain_length = cfg.size[1]
        center_x = terrain_width / 2.0
        half_w = sidewalk_width / 2.0

        meshes_list: list[trimesh.Trimesh] = []

        sidewalk_height = generator_cfg.sidewalk_height
        ground_height = generator_cfg.ground_height

        # Y boundaries — sidewalk depth = fixed 4m
        sidewalk_depth = 4.0
        road_bottom = sidewalk_depth
        road_top = terrain_length - sidewalk_depth

        # X boundaries for the center walkable/road area
        walk_left = center_x - half_w
        walk_right = center_x + half_w

        # Crosswalk stripe params (dynamic count, fixed stripe width)
        road_height = road_top - road_bottom
        stripe_width_y = 0.3
        stripe_count = max(4, int(round(road_height / (stripe_width_y * 2.0))))
        stripe_period = road_height / stripe_count
        stripe_gap = stripe_period - stripe_width_y
        crosswalk_total_width = road_height
        crosswalk_center_y = (road_bottom + road_top) / 2.0
        crosswalk_start_y = road_bottom

        # ==================== Group 1 ====================
        # [0] Walkable: sidewalks (bottom + top, center X band)
        bottom_sidewalk_polygon = Polygon(
            [
                [walk_left, 0],
                [walk_right, 0],
                [walk_right, road_bottom],
                [walk_left, road_bottom],
            ]
        )
        top_sidewalk_polygon = Polygon(
            [
                [walk_left, road_top],
                [walk_right, road_top],
                [walk_right, terrain_length],
                [walk_left, terrain_length],
            ]
        )

        from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
        from shapely.ops import unary_union

        sidewalk_geom = unary_union([bottom_sidewalk_polygon, top_sidewalk_polygon])
        if not sidewalk_geom.is_valid:
            sidewalk_geom = sidewalk_geom.buffer(0.0)
        if isinstance(sidewalk_geom, ShapelyMultiPolygon):
            sidewalk_parts = list(sidewalk_geom.geoms)
        else:
            sidewalk_parts = [sidewalk_geom]
        sidewalk_meshes = []
        for poly in sidewalk_parts:
            m = trimesh.creation.extrude_polygon(polygon=poly, height=sidewalk_height, engine="manifold")
            sidewalk_meshes.append(m)
        meshes_list.append(trimesh.util.concatenate(sidewalk_meshes))

        # [1] Non-walkable: road (full X, center Y) — gets road material
        road_polygon = Polygon(
            [
                [0, road_bottom],
                [terrain_width, road_bottom],
                [terrain_width, road_top],
                [0, road_top],
            ]
        )
        meshes_list.append(trimesh.creation.extrude_polygon(polygon=road_polygon, height=ground_height, engine="manifold"))

        # [2] Other: crosswalk stripes (center X band, center Y)
        crosswalk_stripe_polygons = []
        for si in range(stripe_count):
            sy = crosswalk_start_y + si * stripe_period
            stripe = Polygon(
                [
                    [walk_left, sy],
                    [walk_right, sy],
                    [walk_right, sy + stripe_width_y],
                    [walk_left, sy + stripe_width_y],
                ]
            )
            crosswalk_stripe_polygons.append(stripe)
        stripe_geom = unary_union(crosswalk_stripe_polygons)
        if not stripe_geom.is_valid:
            stripe_geom = stripe_geom.buffer(0.0)
        if isinstance(stripe_geom, ShapelyMultiPolygon):
            stripe_parts = list(stripe_geom.geoms)
        else:
            stripe_parts = [stripe_geom]
        stripe_meshes = []
        for poly in stripe_parts:
            sm = trimesh.creation.extrude_polygon(polygon=poly, height=ground_height + 0.001, engine="manifold")
            sm.visual.vertex_colors = (255, 255, 255, 255)
            stripe_meshes.append(sm)
        meshes_list.append(trimesh.util.concatenate(stripe_meshes))

        # ==================== Group 2 ====================
        # [0] Walkable: empty placeholder
        placeholder = trimesh.creation.box(extents=(0.01, 0.01, 0.001))
        meshes_list.append(placeholder)

        # [1] Non-walkable: terrain on left/right of sidewalk areas only (same Y bands as sidewalks)
        terrain_left_bottom = Polygon(
            [
                [0, 0],
                [walk_left, 0],
                [walk_left, road_bottom],
                [0, road_bottom],
            ]
        )
        terrain_left_top = Polygon(
            [
                [0, road_top],
                [walk_left, road_top],
                [walk_left, terrain_length],
                [0, terrain_length],
            ]
        )
        terrain_right_bottom = Polygon(
            [
                [walk_right, 0],
                [terrain_width, 0],
                [terrain_width, road_bottom],
                [walk_right, road_bottom],
            ]
        )
        terrain_right_top = Polygon(
            [
                [walk_right, road_top],
                [terrain_width, road_top],
                [terrain_width, terrain_length],
                [walk_right, terrain_length],
            ]
        )
        terrain_geom = unary_union([terrain_left_bottom, terrain_left_top, terrain_right_bottom, terrain_right_top])
        if not terrain_geom.is_valid:
            terrain_geom = terrain_geom.buffer(0.0)
        if isinstance(terrain_geom, ShapelyMultiPolygon):
            terrain_parts = list(terrain_geom.geoms)
        else:
            terrain_parts = [terrain_geom]
        terrain_meshes = []
        for poly in terrain_parts:
            tm = trimesh.creation.extrude_polygon(polygon=poly, height=ground_height, engine="manifold")
            terrain_meshes.append(tm)
        meshes_list.append(trimesh.util.concatenate(terrain_meshes))

        # [2] Other: thin placeholder
        thin = trimesh.creation.box(extents=(0.01, 0.01, 0.001))
        meshes_list.append(thin)

        # Combined walkable geometry for sampling / boundary checks
        walkable_geom = unary_union([sidewalk_geom, stripe_geom])
        if not walkable_geom.is_valid:
            walkable_geom = walkable_geom.buffer(0.0)

        origin = np.array([0.0, 0.0, 0.0])

        def sample_pos(walkable=True, num_samples=1, rng=None):
            rng = np.random if rng is None else rng
            if walkable:
                x = np.zeros(num_samples)
                y = np.zeros(num_samples)
                for i in range(num_samples):
                    choice = rng.rand()
                    if choice < 0.35:
                        xi = rng.uniform(walk_left + 0.1, walk_right - 0.1)
                        yi = rng.uniform(0.1, road_bottom - 0.1)
                    elif choice < 0.70:
                        xi = rng.uniform(walk_left + 0.1, walk_right - 0.1)
                        yi = rng.uniform(road_top + 0.1, terrain_length - 0.1)
                    else:
                        xi = rng.uniform(walk_left + 0.1, walk_right - 0.1)
                        yi = crosswalk_center_y + rng.uniform(
                            -crosswalk_total_width / 2.0, crosswalk_total_width / 2.0
                        )
                    x[i] = xi
                    y[i] = yi
                return x, y
            else:
                x = rng.uniform(0, terrain_width, num_samples)
                y = rng.uniform(road_bottom + 0.1, road_top - 0.1, num_samples)
                from shapely.vectorized import contains

                in_walkable = contains(walkable_geom, x, y)
                for _ in range(20):
                    if not np.any(in_walkable):
                        break
                    k = int(np.sum(in_walkable))
                    x[in_walkable] = rng.uniform(0, terrain_width, k)
                    y[in_walkable] = rng.uniform(road_bottom + 0.1, road_top - 0.1, k)
                    in_walkable = contains(walkable_geom, x, y)
                return x, y

        def sample_trajectory(num_points, scenario_type="default", ped_lateral="center", rng=None, **kwargs):
            rng = np.random if rng is None else rng
            if scenario_type in ("intersection", "obstructing", "obstacle_obstruction"):
                ped_direction = kwargs.get("ped_crossing_direction", "bottom_to_top")
                margin = 1.0
                if ped_direction == "top_to_bottom":
                    traj_y = np.linspace(top_sidewalk_polygon.centroid.y, bottom_sidewalk_polygon.centroid.y, num_points)
                    yaw_val = np.deg2rad(-90)
                else:
                    traj_y = np.linspace(bottom_sidewalk_polygon.centroid.y, top_sidewalk_polygon.centroid.y, num_points)
                    yaw_val = np.deg2rad(90)
                traj_x = np.full(num_points, center_x)
                yaw = np.full(num_points, yaw_val)
                return np.stack([traj_x, traj_y, yaw], axis=-1)

            if rng.rand() < 0.5:
                side = rng.choice(["bottom", "top"])
                if side == "bottom":
                    cy = road_bottom / 2.0
                else:
                    cy = (road_top + terrain_length) / 2.0
                traj_x = np.linspace(walk_left + 0.3, walk_right - 0.3, num_points)
                traj_y = np.full(num_points, cy)
                yaw = np.full(num_points, np.deg2rad(0))
            else:
                traj_y = np.linspace(road_bottom - 1.0, road_top + 1.0, num_points)
                traj_x = np.full(num_points, center_x)
                yaw = np.full(num_points, np.deg2rad(90))
            return np.stack([traj_x, traj_y, yaw], axis=-1)

        def sample_start_and_goal(rng=None):
            rng = np.random if rng is None else rng
            start_y = rng.uniform(max(0.1, road_bottom - 1.0), road_bottom)
            goal_y = rng.uniform(road_top, min(terrain_length - 0.1, road_top + 1.0))
            start_x = center_x
            goal_x = center_x
            start_yaw = np.deg2rad(90)
            route_length = float(abs(goal_y - start_y))
            return np.array([start_x, start_y, start_yaw]), np.array([goal_x, goal_y]), route_length

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

        _cl_y_cw = np.linspace(0, terrain_length, 500)
        _cl_xy_cw = np.stack([np.full(500, center_x), _cl_y_cw], axis=1)
        _cl_arc_cw = _cl_y_cw.copy()
        project_to_centerline = _build_centerline_projector(_cl_xy_cw, _cl_arc_cw)

        return (
            meshes_list,
            origin,
            sample_pos,
            sample_trajectory,
            sample_start_and_goal,
            check_outside_boundary,
            project_to_centerline,
            None,
            None,
            None,
            None,
        )


@configclass
class CrosswalkTerrainCfg(BaseTerrainCfg):
    """Configuration for a crosswalk terrain: road with sidewalks and perpendicular crosswalk stripes."""

    class_type = CrosswalkTerrain
