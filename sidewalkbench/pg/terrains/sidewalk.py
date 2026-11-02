from __future__ import annotations

from .base import (
    BaseTerrainCfg,
    Terrain,
    Polygon,
    _build_centerline_projector,
    configclass,
    interpolate,
    np,
    trimesh,
)


class SidewalkTerrain(Terrain):
    """Procedural terrain builder for SideWalkTerrainCfg."""

    def build(self) -> tuple:
        cfg = self.cfg
        generator_cfg = self.generator_cfg
        sidewalk_width = generator_cfg.sidewalk_width
        terrain_width = cfg.size[0]
        terrain_length = cfg.size[1]
        num_control_points = 6
        control_y = np.linspace(0, terrain_length, num_control_points)
        # max_offset = (terrain_width / 2 - sidewalk_width) * 0.3
        max_offset = 0
        control_x = terrain_width / 2 + np.random.uniform(-max_offset, max_offset, num_control_points)
        control_x[0] = terrain_width / 2
        control_x[-1] = terrain_width / 2
        spline = interpolate.CubicSpline(control_y, control_x)
        num_segments = 50
        path_y = np.linspace(0, terrain_length, num_segments + 1)
        path_x = spline(path_y)

        # Clamp path to stay within bounds
        half_width = sidewalk_width / 2
        path_x = np.clip(path_x, half_width + 0.1, terrain_width - half_width - 0.1)
        meshes_list: list[trimesh.Trimesh] = []

        sidewalk_height = generator_cfg.sidewalk_height  # Sidewalk is slightly elevated
        ground_height = generator_cfg.ground_height  # Ground level

        # Build sidewalk polygon vertices (2D for extrusion)
        # Left edge going forward, then right edge going backward
        left_edge = np.stack([path_x - half_width, path_y], axis=-1)
        right_edge = np.stack([path_x + half_width, path_y], axis=-1)

        # Create closed polygon: left edge forward, then right edge backward
        sidewalk_polygon = np.concatenate([left_edge, right_edge[::-1]], axis=0)
        sidewalk_polygon = Polygon(np.array(sidewalk_polygon))
        # Create sidewalk mesh using extrude_polygon
        sidewalk_mesh = trimesh.creation.extrude_polygon(
            polygon=sidewalk_polygon, height=sidewalk_height, engine="manifold"
        )
        meshes_list.append(sidewalk_mesh)

        # Create ground mesh (simple flat rectangle at ground level)
        ground_polygon = [
            [0, 0],
            [terrain_width, 0],
            [terrain_width, terrain_length],
            [0, terrain_length],
        ]
        ground_polygon = Polygon(ground_polygon)
        # Use a very thin extrusion for ground (or create as flat surface)
        ground_mesh = trimesh.creation.extrude_polygon(
            polygon=ground_polygon, height=ground_height, engine="manifold"
        )  # Very thin ground

        meshes_list.append(ground_mesh)

        placeholder_mesh = trimesh.creation.extrude_polygon(polygon=ground_polygon, height=0.001, engine="manifold")
        other_meshes = [placeholder_mesh]

        combined_meshes = trimesh.util.concatenate(other_meshes)
        meshes_list.append(combined_meshes)

        # Origin at center of terrain, fixed at lower right corner
        origin = np.array([0.0, 0.0, 0.0])

        # Precompute dense centerline for projection
        _cl_y = np.linspace(0, terrain_length, 500)
        _cl_x = np.clip(spline(_cl_y), half_width + 0.1, terrain_width - half_width - 0.1)
        _cl_xy = np.stack([_cl_x, _cl_y], axis=1)
        _cl_diffs = np.diff(_cl_xy, axis=0)
        _cl_arc = np.concatenate([[0], np.cumsum(np.linalg.norm(_cl_diffs, axis=1))])
        project_to_centerline = _build_centerline_projector(_cl_xy, _cl_arc)

        def sample_pos(walkable=True, num_samples=1, rng=None) -> tuple[np.ndarray, np.ndarray]:
            """
            Sample a random position on the terrain.

            Args:
                walkable: If True, sample only from walkable (sidewalk) region.
                          If False, sample from entire terrain.

            Returns:
                np.ndarray: Array of shape [2] containing (x, y) coordinates in terrain space.
            """
            rng = np.random if rng is None else rng

            # Sample random y position along the terrain length
            norm_y = rng.uniform(0.1, 0.9, num_samples)
            y = norm_y * terrain_length

            # Use spline to get the center x at this y position
            center_x = spline(y)

            # Clamp center_x to stay within bounds
            center_x = np.clip(center_x, half_width + 0.1, terrain_width - half_width - 0.1)

            # Sample x within the walkable width (with some margin)
            left_x = center_x - half_width
            right_x = center_x + half_width
            if walkable:
                x = rng.uniform(0, sidewalk_width, num_samples) + left_x
            else:
                x = rng.uniform(0, terrain_width - sidewalk_width, num_samples)
                x[x > left_x] += sidewalk_width  # Shift to non-walkable side if in the walkable region
            return x, y

        def sample_trajectory(
            num_points: int, scenario_type: str = "default", ped_lateral: str = "center", rng=None, **kwargs
        ) -> np.ndarray:
            """
            Sample a trajectory of points along the sidewalk path.

            Args:
                num_points: Number of points to sample along the trajectory
                scenario_type: "default", "frontal_approach", "obstructing", or "intersection".
                ped_lateral: "center", "left", or "right" — lateral placement on sidewalk.
            """
            # Lateral offset: center of left quarter, center, or center of right quarter
            if ped_lateral == "left":
                lat_offset = -half_width / 4.0
            elif ped_lateral == "right":
                lat_offset = half_width / 4.0
            else:
                lat_offset = 0.0

            if scenario_type in ("obstructing", "obstacle_obstruction"):
                mid_y = 0.7 * terrain_length
                mid_x = float(np.clip(spline(mid_y), half_width + 0.1, terrain_width - half_width - 0.1)) + lat_offset
                yaw_val = np.deg2rad(270)
                return np.stack(
                    [np.full(num_points, mid_x), np.full(num_points, mid_y), np.full(num_points, yaw_val)], axis=-1
                )

            if scenario_type == "intersection":
                # Human crosses perpendicular to the sidewalk at ~middle of terrain
                cross_y = 0.5 * terrain_length
                center_x = float(np.clip(spline(cross_y), half_width + 0.1, terrain_width - half_width - 0.1))
                ped_crossing_direction = kwargs.get("ped_crossing_direction", "left_to_right")
                margin = half_width + 1.0  # start just outside the sidewalk edge
                if ped_crossing_direction == "right_to_left":
                    start_x = center_x + margin
                    end_x = center_x - margin
                    yaw_val = np.deg2rad(180)
                else:
                    start_x = center_x - margin
                    end_x = center_x + margin
                    yaw_val = np.deg2rad(0)
                traj_x = np.linspace(start_x, end_x, num_points)
                traj_y = np.full(num_points, cross_y)
                yaw = np.full(num_points, yaw_val)
                return np.stack([traj_x, traj_y, yaw], axis=-1)

            num_control_points_ped = 6
            control_x_ped = control_x.copy() + lat_offset
            spline_ped = interpolate.CubicSpline(control_y, control_x_ped)

            if scenario_type == "frontal_approach":
                traj_y = np.linspace(1.0 * terrain_length, 0.3 * terrain_length, num_points)
            else:
                traj_y = np.linspace(0.3 * terrain_length, 1.5 * terrain_length, num_points)

            traj_x = np.clip(spline_ped(traj_y), 0.1, terrain_width - 0.1)
            dx_dy = spline_ped(traj_y, 1)
            if scenario_type == "frontal_approach":
                yaw = np.arctan2(-np.ones_like(dx_dy), -dx_dy)
            else:
                yaw = np.arctan2(np.ones_like(dx_dy), dx_dy)
            return np.stack([traj_x, traj_y, yaw], axis=-1)

        def sample_start_and_goal(rng=None):
            """Sample a start (x, y, yaw) and goal (x, y) on the sidewalk.

            Returns:
                start_x, start_y, start_yaw, goal_x, goal_y
            """
            # Sample two distinct y positions: start in lower half, goal in upper half
            start_y = 0.1  # prevent out of bounds
            goal_y = terrain_length

            # Get sidewalk center x at each y via the spline
            start_cx = float(np.clip(spline(start_y), half_width + 0.1, terrain_width - half_width - 0.1))
            goal_cx = float(np.clip(spline(goal_y), half_width + 0.1, terrain_width - half_width - 0.1))

            # Random lateral offset within the sidewalk
            start_x = start_cx
            goal_x = goal_cx
            # Compute yaw pointing from start toward goal
            start_yaw = np.deg2rad(90)
            # Route length: arc length along spline from start_y to goal_y
            y_dense = np.linspace(start_y, goal_y, 200)
            x_dense = spline(y_dense)
            route_length = float(np.sum(np.sqrt(np.diff(x_dense) ** 2 + np.diff(y_dense) ** 2)))
            return np.array([start_x, start_y, start_yaw]), np.array([goal_x, goal_y]), route_length

        def check_outside_boundary(x, y):
            """Return signed distance to boundary (positive=outside, negative=inside)."""
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            center_x = spline(y)
            center_x = np.clip(center_x, half_width + 0.1, terrain_width - half_width - 0.1)
            return np.abs(x - center_x) - half_width

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
class SideWalkTerrainCfg(BaseTerrainCfg):
    """Configuration for a random uniform height field terrain."""

    class_type = SidewalkTerrain
