from __future__ import annotations

from shapely.geometry import Point

from .base import (
    BaseTerrainCfg,
    Terrain,
    Polygon,
    _build_centerline_projector,
    configclass,
    np,
    trimesh,
)


class CircularTerrain(Terrain):
    """Procedural terrain builder for CircularTerrainCfg."""

    def build(self) -> tuple:
        cfg = self.cfg
        generator_cfg = self.generator_cfg
        sidewalk_width = generator_cfg.sidewalk_width
        terrain_width = cfg.size[0]
        terrain_length = cfg.size[1]
        half_width = sidewalk_width / 2.0

        meshes_list: list[trimesh.Trimesh] = []

        sidewalk_height = generator_cfg.sidewalk_height
        ground_height = generator_cfg.ground_height

        center = np.array([terrain_width / 2.0, terrain_length / 2.0])
        margin = max(half_width + 0.5, 1.0)
        max_radius = min(terrain_width, terrain_length) / 2.0 - margin
        radius = max(2.0, max_radius * 0.75)
        radius = min(radius, max_radius)

        outer_radius = radius + half_width
        inner_radius = max(radius - half_width, 0.05)
        base_point = Point(float(center[0]), float(center[1]))
        outer_disk = base_point.buffer(float(outer_radius), resolution=128)
        inner_disk = base_point.buffer(float(inner_radius), resolution=128)
        circuit_polygon = outer_disk.difference(inner_disk)
        if not circuit_polygon.is_valid:
            circuit_polygon = circuit_polygon.buffer(0.0)

        sidewalk_mesh = trimesh.creation.extrude_polygon(polygon=circuit_polygon, height=sidewalk_height, engine="manifold")
        meshes_list.append(sidewalk_mesh)

        ground_polygon = [
            [0, 0],
            [terrain_width, 0],
            [terrain_width, terrain_length],
            [0, terrain_length],
        ]
        ground_polygon = Polygon(ground_polygon)
        ground_mesh = trimesh.creation.extrude_polygon(polygon=ground_polygon, height=ground_height, engine="manifold")
        meshes_list.append(ground_mesh)

        placeholder_mesh = trimesh.creation.extrude_polygon(
            polygon=ground_polygon, height=0.001, engine="manifold"
        )  # Very thin ground

        other_meshes = [placeholder_mesh]
        combined_meshes = trimesh.util.concatenate(other_meshes)
        meshes_list.append(combined_meshes)

        origin = np.array([0.0, 0.0, 0.0])

        def sample_pos(walkable=True, num_samples=1, rng=None) -> tuple[np.ndarray, np.ndarray]:
            rng = np.random if rng is None else rng
            theta_samples = rng.uniform(0.1, 0.9, num_samples) * (2.0 * np.pi)
            radial_offset = rng.uniform(-half_width, half_width, num_samples) if walkable else np.zeros(num_samples)

            if walkable:
                local_radius = radius + radial_offset
                x = center[0] + local_radius * np.cos(theta_samples)
                y = center[1] + local_radius * np.sin(theta_samples)
                return x, y

            x = rng.uniform(0.0, terrain_width, num_samples)
            y = rng.uniform(0.0, terrain_length, num_samples)
            dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            in_ring = np.abs(dist - radius) <= half_width
            max_resample = 20
            resample_count = 0
            while np.any(in_ring) and resample_count < max_resample:
                k = int(np.sum(in_ring))
                x[in_ring] = rng.uniform(0.0, terrain_width, k)
                y[in_ring] = rng.uniform(0.0, terrain_length, k)
                dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                in_ring = np.abs(dist - radius) <= half_width
                resample_count += 1
            return x, y

        def sample_trajectory(num_points: int, rng=None, **kwargs) -> np.ndarray:
            rng = np.random if rng is None else rng
            start_theta = 0.1 * (2.0 * np.pi)
            end_theta = 0.9 * (2.0 * np.pi)
            traj_theta = np.linspace(start_theta, end_theta, num_points)

            lateral = rng.uniform(-half_width * 0.45, half_width * 0.45)
            traj_radius = radius + lateral

            traj_x = center[0] + traj_radius * np.cos(traj_theta)
            traj_y = center[1] + traj_radius * np.sin(traj_theta)

            tangent_x = -np.sin(traj_theta)
            tangent_y = np.cos(traj_theta)
            yaw = np.arctan2(tangent_y, tangent_x)

            trajectory = np.stack([traj_x, traj_y, yaw], axis=-1)
            return trajectory

        def sample_start_and_goal(rng=None):
            """Sample a start (x, y, yaw) and goal (x, y) on the circular sidewalk.

            Returns:
                start: np.ndarray of shape (3,) — (x, y, yaw)
                goal: np.ndarray of shape (2,) — (x, y)
            """
            rng = np.random if rng is None else rng
            if rng.rand() < 0.5:
                # counterclockwise
                start_theta = rng.uniform(0, 0.1) * (2.0 * np.pi) + np.deg2rad(rng.uniform(-10, 10))
                goal_theta = rng.uniform(0.8, 0.9) * (2.0 * np.pi)
                start_yaw = float(start_theta + np.pi / 2)
            else:
                # clockwise
                start_theta = rng.uniform(0.9, 1.0) * (2.0 * np.pi) + np.deg2rad(rng.uniform(-10, 10))
                goal_theta = rng.uniform(0.1, 0.2) * (2.0 * np.pi)
                start_yaw = float(start_theta - np.pi / 2)

            offset_s = rng.uniform(-half_width * 0.5, half_width * 0.5)
            offset_g = 0

            start_x = center[0] + (radius + offset_s) * np.cos(start_theta)
            start_y = center[1] + (radius + offset_s) * np.sin(start_theta)
            goal_x = center[0] + (radius + offset_g) * np.cos(goal_theta)
            goal_y = center[1] + (radius + offset_g) * np.sin(goal_theta)

            # Route length: arc along the circle (always the longer way around)
            angle_diff = abs(goal_theta - start_theta) % (2 * np.pi)
            if angle_diff < np.pi:
                angle_diff = 2 * np.pi - angle_diff
            route_length = float(radius * angle_diff)

            return np.array([start_x, start_y, start_yaw]), np.array([goal_x, goal_y]), route_length

        def check_outside_boundary(x, y):
            """Return signed distance to boundary (positive=outside, negative=inside)."""
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            return np.abs(dist - radius) - half_width

        # Circular centerline projector
        _cl_n = 500
        _cl_theta = np.linspace(0, 2 * np.pi, _cl_n, endpoint=False)
        _cl_xy_circ = np.stack([center[0] + radius * np.cos(_cl_theta), center[1] + radius * np.sin(_cl_theta)], axis=1)
        _cl_arc_circ = radius * _cl_theta
        project_to_centerline = _build_centerline_projector(_cl_xy_circ, _cl_arc_circ)

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
class CircularTerrainCfg(BaseTerrainCfg):
    """Configuration for a circular terrain."""

    class_type = CircularTerrain
