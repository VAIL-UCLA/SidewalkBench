from __future__ import annotations

from .base import (
    BaseTerrainCfg,
    Terrain,
    Polygon,
    _build_centerline_projector,
    _generate_turn_centerline,
    configclass,
    interpolate,
    np,
    trimesh,
)


def _build_single_turn_terrain(cfg, generator_cfg, turn_left):
    """Shared builder for a single 90° turn sidewalk terrain."""
    sidewalk_width = generator_cfg.sidewalk_width
    terrain_width = cfg.size[0]
    terrain_length = cfg.size[1]
    half_w = sidewalk_width / 2.0
    corner_radius = 4.0

    meshes_list: list[trimesh.Trimesh] = []
    sidewalk_height = generator_cfg.sidewalk_height
    ground_height = generator_cfg.ground_height

    centerline, cl_arc, total_length, turn_start_s, turn_end_s = _generate_turn_centerline(
        corner_radius, turn_left=turn_left, approach_length=10.0, exit_length=10.0
    )

    # Center the path in the terrain
    cl_min = centerline.min(axis=0)
    cl_max = centerline.max(axis=0)
    cl_center = (cl_min + cl_max) / 2
    terrain_center = np.array([terrain_width / 2, terrain_length / 2])
    offset = terrain_center - cl_center
    centerline = centerline + offset

    # Arc-length splines
    spline_x = interpolate.CubicSpline(cl_arc, centerline[:, 0])
    spline_y = interpolate.CubicSpline(cl_arc, centerline[:, 1])

    # Build sidewalk polygon via tangents / normals
    diffs = np.diff(centerline, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    tangents = np.zeros_like(centerline)
    tangents[:-1] = diffs / (seg_lens[:, None] + 1e-8)
    tangents[-1] = tangents[-2]
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

    sidewalk_mesh = trimesh.creation.extrude_polygon(sidewalk_poly, height=sidewalk_height, engine="manifold")

    # Ground mesh
    ground_polygon = Polygon(
        [
            [0, 0],
            [terrain_width, 0],
            [terrain_width, terrain_length],
            [0, terrain_length],
        ]
    )
    ground_mesh = trimesh.creation.extrude_polygon(ground_polygon, height=ground_height, engine="manifold")

    # Placeholder
    placeholder_mesh = trimesh.creation.extrude_polygon(ground_polygon, height=0.001, engine="manifold")

    meshes_list = [sidewalk_mesh, ground_mesh, placeholder_mesh]
    origin = np.array([0.0, 0.0, 0.0])

    def _arc_to_xy(s):
        s = np.clip(np.asarray(s), 0, total_length)
        return np.column_stack([spline_x(s), spline_y(s)])

    def _arc_to_yaw(s):
        s = np.clip(np.asarray(s), 1e-3, total_length - 1e-3)
        dx = spline_x(s, 1)
        dy = spline_y(s, 1)
        return np.arctan2(dy, dx)

    def sample_pos(walkable=True, num_samples=1, rng=None):
        rng = np.random if rng is None else rng
        s = rng.uniform(0.05 * total_length, 0.95 * total_length, num_samples)
        xy = _arc_to_xy(s)
        if walkable:
            lat = rng.uniform(-half_w * 0.8, half_w * 0.8, num_samples)
            yaw = _arc_to_yaw(s)
            normal_dir = np.column_stack([-np.sin(yaw), np.cos(yaw)])
            xy = xy + lat[:, None] * normal_dir
        return xy[:, 0], xy[:, 1]

    def sample_trajectory(num_points, scenario_type="default", ped_lateral="center", rng=None, **kwargs):
        if ped_lateral == "left":
            lat = -half_w / 4.0
        elif ped_lateral == "right":
            lat = half_w / 4.0
        else:
            lat = 0.0
        s_vals = np.linspace(0.05 * total_length, 0.95 * total_length, num_points)
        xy = _arc_to_xy(s_vals)
        yaw = _arc_to_yaw(s_vals)
        normal_dir = np.column_stack([-np.sin(yaw), np.cos(yaw)])
        xy = xy + lat * normal_dir
        return np.column_stack([xy[:, 0], xy[:, 1], yaw])

    def sample_start_and_goal(rng=None):
        s_start = turn_start_s - 3.0
        s_goal = turn_end_s + 3.0
        start_xy = _arc_to_xy(np.array([s_start]))[0]
        goal_xy = _arc_to_xy(np.array([s_goal]))[0]
        start_yaw = float(_arc_to_yaw(np.array([s_start]))[0])
        route_length = float(s_goal - s_start)
        return np.array([start_xy[0], start_xy[1], start_yaw]), np.array([goal_xy[0], goal_xy[1]]), route_length

    def check_outside_boundary(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        s_samples = np.linspace(0, total_length, 200)
        cl_samples = _arc_to_xy(s_samples)
        result = np.zeros(len(x), dtype=np.float64)
        for i in range(len(x)):
            dists = np.sqrt((cl_samples[:, 0] - x[i]) ** 2 + (cl_samples[:, 1] - y[i]) ** 2)
            result[i] = dists.min() - half_w
        return result[0] if scalar else result

    _cl_s = np.linspace(0, total_length, 500)
    _cl_xy = _arc_to_xy(_cl_s)
    project_to_centerline = _build_centerline_projector(_cl_xy, _cl_s)

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


class CurvyLeftTerrain(Terrain):
    """Single 90° left turn sidewalk terrain."""

    def build(self) -> tuple:
        return _build_single_turn_terrain(self.cfg, self.generator_cfg, turn_left=True)


class CurvyRightTerrain(Terrain):
    """Single 90° right turn sidewalk terrain."""

    def build(self) -> tuple:
        return _build_single_turn_terrain(self.cfg, self.generator_cfg, turn_left=False)


@configclass
class CurvyLeftTerrainCfg(BaseTerrainCfg):
    """Configuration for a single 90° left turn sidewalk terrain."""

    class_type = CurvyLeftTerrain


@configclass
class CurvyRightTerrainCfg(BaseTerrainCfg):
    """Configuration for a single 90° right turn sidewalk terrain."""

    class_type = CurvyRightTerrain
