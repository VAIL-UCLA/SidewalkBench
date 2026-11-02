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


class PlazaTerrain(Terrain):
    """Procedural terrain builder for PlazaTerrainCfg."""

    def build(self) -> tuple:
        cfg = self.cfg
        generator_cfg = self.generator_cfg
        """Generate a plaza terrain: a square walkable area in the centre of the terrain.

        Start and goal are placed at diagonally opposite corners of the square.
        Positions and trajectories are sampled randomly within the walkable region.
        """
        sidewalk_width = generator_cfg.sidewalk_width
        terrain_width = cfg.size[0]
        terrain_length = cfg.size[1]

        meshes_list: list[trimesh.Trimesh] = []

        sidewalk_height = generator_cfg.sidewalk_height
        ground_height = generator_cfg.ground_height

        # Plaza square: side length = sidewalk_width, centred in terrain
        plaza_size = terrain_width * 0.8
        cx = terrain_width / 2.0
        cy = terrain_length / 2.0
        half_plaza = plaza_size / 2.0

        p_left = cx - half_plaza
        p_right = cx + half_plaza
        p_bottom = cy - half_plaza
        p_top = cy + half_plaza

        # Sidewalk (plaza square)
        plaza_polygon = Polygon(
            [
                [p_left, p_bottom],
                [p_right, p_bottom],
                [p_right, p_top],
                [p_left, p_top],
            ]
        )
        sidewalk_mesh = trimesh.creation.extrude_polygon(polygon=plaza_polygon, height=sidewalk_height, engine="manifold")
        meshes_list.append(sidewalk_mesh)

        # Ground
        ground_polygon = Polygon(
            [
                [0, 0],
                [terrain_width, 0],
                [terrain_width, terrain_length],
                [0, terrain_length],
            ]
        )
        ground_mesh = trimesh.creation.extrude_polygon(polygon=ground_polygon, height=ground_height, engine="manifold")
        meshes_list.append(ground_mesh)

        # Placeholder / other meshes
        placeholder_mesh = trimesh.creation.extrude_polygon(polygon=ground_polygon, height=0.001, engine="manifold")
        other_meshes = [placeholder_mesh]
        combined_meshes = trimesh.util.concatenate(other_meshes)
        meshes_list.append(combined_meshes)

        origin = np.array([0.0, 0.0, 0.0])

        # Four corners of the plaza (bottom-left, bottom-right, top-right, top-left)
        corners = np.array(
            [
                [p_left, p_bottom],
                [p_right, p_bottom],
                [p_right, p_top],
                [p_left, p_top],
            ]
        )

        def sample_pos(walkable=True, num_samples=1, rng=None) -> tuple[np.ndarray, np.ndarray]:
            """Sample random positions inside (or outside) the plaza square."""
            rng = np.random if rng is None else rng
            if walkable:
                x = rng.uniform(p_left + 0.1, p_right - 0.1, num_samples)
                y = rng.uniform(p_bottom + 0.1, p_top - 0.1, num_samples)
                return x, y
            else:
                x = rng.uniform(0, terrain_width, num_samples)
                y = rng.uniform(0, terrain_length, num_samples)
                in_plaza = (x >= p_left) & (x <= p_right) & (y >= p_bottom) & (y <= p_top)
                max_resample = 20
                resample_count = 0
                while np.any(in_plaza) and resample_count < max_resample:
                    k = int(np.sum(in_plaza))
                    x[in_plaza] = rng.uniform(0, terrain_width, k)
                    y[in_plaza] = rng.uniform(0, terrain_length, k)
                    in_plaza = (x >= p_left) & (x <= p_right) & (y >= p_bottom) & (y <= p_top)
                    resample_count += 1
                return x, y

        def sample_trajectory(num_points: int, rng=None, **kwargs) -> np.ndarray:
            """Sample a random straight-ish trajectory inside the plaza."""
            rng = np.random if rng is None else rng
            # Random start and end within the plaza
            sx = rng.uniform(p_left + 0.1, p_right - 0.1)
            sy = rng.uniform(p_bottom + 0.1, p_top - 0.1)
            ex = rng.uniform(p_left + 0.1, p_right - 0.1)
            ey = rng.uniform(p_bottom + 0.1, p_top - 0.1)

            traj_x = np.linspace(sx, ex, num_points)
            traj_y = np.linspace(sy, ey, num_points)

            dx = np.gradient(traj_x)
            dy = np.gradient(traj_y)
            yaw = np.arctan2(dy, dx)

            trajectory = np.stack([traj_x, traj_y, yaw], axis=-1)
            return trajectory

        def sample_start_and_goal(rng=None):
            """Sample start at bottom-left corner (0,0) of the plaza and goal at the diagonal (top-right).

            Returns:
                start: np.ndarray of shape (3,) — (x, y, yaw)
                goal:  np.ndarray of shape (2,) — (x, y)
            """
            rng = np.random if rng is None else rng
            margin = 1.0  # small inset so the agent isn't exactly on the edge
            start_x = p_left + margin
            start_y = p_bottom + margin
            goal_x = p_right - margin
            goal_y = p_top - margin

            # 45 degrees toward the diagonal goal + random noise
            start_yaw = np.deg2rad(45) + np.deg2rad(rng.uniform(-10, 10))

            route_length = float(np.sqrt((goal_x - start_x) ** 2 + (goal_y - start_y) ** 2))
            return np.array([start_x, start_y, start_yaw]), np.array([goal_x, goal_y]), route_length

        def check_outside_boundary(x, y):
            """Return signed distance to boundary (positive=outside, negative=inside)."""
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            return np.maximum(p_left - x, np.maximum(x - p_right, np.maximum(p_bottom - y, y - p_top)))

        # Diagonal centerline projector (bottom-left to top-right)
        _margin = 1.0
        _cl_t = np.linspace(0, 1, 500)
        _cl_xy_plaza = np.stack(
            [
                p_left + _margin + _cl_t * (p_right - p_left - 2 * _margin),
                p_bottom + _margin + _cl_t * (p_top - p_bottom - 2 * _margin),
            ],
            axis=1,
        )
        _cl_diffs_plaza = np.diff(_cl_xy_plaza, axis=0)
        _cl_arc_plaza = np.concatenate([[0], np.cumsum(np.linalg.norm(_cl_diffs_plaza, axis=1))])
        project_to_centerline = _build_centerline_projector(_cl_xy_plaza, _cl_arc_plaza)

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


# ---------------------------------------------------------------------------
# Block-based procedural sidewalk terrain
# ---------------------------------------------------------------------------


@configclass
class PlazaTerrainCfg(BaseTerrainCfg):
    """Configuration for a plaza terrain (square walkable area)."""

    class_type = PlazaTerrain
