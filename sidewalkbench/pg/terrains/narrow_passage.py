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


class NarrowPassageTerrain(Terrain):
    """Procedural terrain builder for NarrowPassageTerrainCfg."""

    def build(self) -> tuple:
        cfg = self.cfg
        generator_cfg = self.generator_cfg
        """Generate a terrain with a narrow passage (1m width) in the middle of the walkable region.

        The terrain consists of a walkable region with obstacles on both sides. In the middle section,
        the walkable area is constrained to a narrow 1m passage. Position sampling avoids the narrow
        passage region, but trajectory sampling can pass through it.
        """
        sidewalk_width = generator_cfg.sidewalk_width
        terrain_width = cfg.size[0]
        terrain_length = cfg.size[1]
        half_width = sidewalk_width / 2

        meshes_list: list[trimesh.Trimesh] = []

        sidewalk_height = generator_cfg.sidewalk_height
        ground_height = generator_cfg.ground_height

        # Narrow passage parameters
        passage_width = 1.0  # 1m narrow passage
        passage_center_y = terrain_length / 2.0  # Middle of terrain
        passage_length = 5.0  # Length of the narrow section along y
        passage_start_y = passage_center_y - passage_length / 2.0
        passage_end_y = passage_center_y + passage_length / 2.0

        # Center of the walkable path (along x)
        path_center_x = terrain_width / 2.0

        # Narrow passage geometry
        passage_half = passage_width / 2.0
        passage_left = path_center_x - passage_half
        passage_right = path_center_x + passage_half
        lower_left = path_center_x - half_width
        lower_right = path_center_x + half_width

        # Build a single continuous sidewalk polygon that narrows in the middle
        # Left edge going forward (bottom to top), then right edge going backward (top to bottom)
        sidewalk_polygon = Polygon(
            [
                # Bottom-left corner (full width)
                [lower_left, 0],
                # Bottom-right corner (full width)
                [lower_right, 0],
                # Right edge narrows at passage start
                [lower_right, passage_start_y],
                [passage_right, passage_start_y],
                # Right edge of narrow passage
                [passage_right, passage_end_y],
                # Right edge widens after passage
                [lower_right, passage_end_y],
                [lower_right, terrain_length],
                # Top-left corner (full width)
                [lower_left, terrain_length],
                # Left edge widens after passage (going back down)
                [lower_left, passage_end_y],
                [passage_left, passage_end_y],
                # Left edge of narrow passage
                [passage_left, passage_start_y],
                # Left edge narrows at passage start
                [lower_left, passage_start_y],
            ]
        )

        sidewalk_mesh = trimesh.creation.extrude_polygon(
            polygon=sidewalk_polygon, height=sidewalk_height, engine="manifold"
        )
        meshes_list.append(sidewalk_mesh)

        # Create ground mesh (simple flat rectangle at ground level)
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

        # Placeholder for other meshes
        placeholder_mesh = trimesh.creation.extrude_polygon(polygon=ground_polygon, height=0.001, engine="manifold")
        other_meshes = [placeholder_mesh]
        combined_meshes = trimesh.util.concatenate(other_meshes)
        meshes_list.append(combined_meshes)

        origin = np.array([0.0, 0.0, 0.0])

        def sample_pos(walkable=True, num_samples=1, rng=None) -> tuple[np.ndarray, np.ndarray]:
            """
            Sample random positions on the terrain, avoiding the narrow passage region.

            Args:
                walkable: If True, sample only from walkable (sidewalk) region, excluding narrow passage.
                          If False, sample from entire terrain.
                num_samples: Number of samples to generate.

            Returns:
                Tuple of (x, y) arrays of shape (num_samples,).
            """
            rng = np.random if rng is None else rng
            if walkable:
                # Sample from walkable region but avoid the narrow passage
                # Split into lower and upper sections
                lower_length = passage_start_y
                upper_length = terrain_length - passage_end_y
                total_safe_length = lower_length + upper_length

                # Weighted random choice between lower and upper sections
                section_choice = rng.rand(num_samples)
                lower_prob = lower_length / total_safe_length

                y = np.zeros(num_samples)
                # Lower section
                lower_mask = section_choice < lower_prob
                y[lower_mask] = rng.uniform(0.1 * passage_start_y, 0.9 * passage_start_y, int(np.sum(lower_mask)))
                # Upper section
                upper_mask = ~lower_mask
                y[upper_mask] = rng.uniform(
                    passage_end_y + 0.1 * (terrain_length - passage_end_y),
                    passage_end_y + 0.9 * (terrain_length - passage_end_y),
                    int(np.sum(upper_mask)),
                )

                # Sample x within the walkable width
                x = rng.uniform(lower_left + 0.1, lower_right - 0.1, num_samples)
                return x, y
            else:
                # Sample from non-walkable region
                x = rng.uniform(0, terrain_width, num_samples)
                y = rng.uniform(0, terrain_length, num_samples)
                # Resample if in walkable region
                in_walkable = (x >= lower_left) & (x <= lower_right)
                max_resample = 20
                resample_count = 0
                while np.any(in_walkable) and resample_count < max_resample:
                    k = int(np.sum(in_walkable))
                    x[in_walkable] = rng.uniform(0, terrain_width, k)
                    in_walkable = (x >= lower_left) & (x <= lower_right)
                    resample_count += 1
                return x, y

        def sample_trajectory(num_points: int, rng=None, **kwargs) -> np.ndarray:
            """
            Sample a trajectory of points along the walkable path, going through the narrow passage.

            Args:
                num_points: Number of points to sample along the trajectory

            Returns:
                trajectory: An array of shape (num_points, 3) containing (x, y, yaw) at each point.
            """
            rng = np.random if rng is None else rng
            # Sample y coordinates spanning the entire terrain length (through narrow passage)
            traj_y = np.linspace(0.1 * terrain_length, 0.9 * terrain_length, num_points)

            # Compute x position based on y location
            traj_x = np.zeros(num_points)

            for i, y in enumerate(traj_y):
                if passage_start_y <= y <= passage_end_y:
                    # In narrow passage - constrain to passage width with some margin
                    traj_x[i] = path_center_x + rng.uniform(-passage_half * 0.7, passage_half * 0.7)
                else:
                    # Outside narrow passage - can use full sidewalk width
                    traj_x[i] = path_center_x + rng.uniform(-half_width * 0.8, half_width * 0.8)

            # Smooth the trajectory x values
            # Apply a simple moving average for smoother transitions
            window = min(5, num_points // 4)
            if window > 1:
                kernel = np.ones(window) / window
                traj_x_smooth = np.convolve(traj_x, kernel, mode="same")
                # Keep original at boundaries
                traj_x_smooth[: window // 2] = traj_x[: window // 2]
                traj_x_smooth[-window // 2 :] = traj_x[-window // 2 :]
                traj_x = traj_x_smooth

            # Clamp to valid range
            for i, y in enumerate(traj_y):
                if passage_start_y <= y <= passage_end_y:
                    traj_x[i] = np.clip(traj_x[i], passage_left + 0.1, passage_right - 0.1)
                else:
                    traj_x[i] = np.clip(traj_x[i], lower_left + 0.1, lower_right - 0.1)

            # Compute yaw angle (direction of travel)
            dx = np.gradient(traj_x)
            dy = np.gradient(traj_y)
            yaw = np.arctan2(dy, dx)

            trajectory = np.stack([traj_x, traj_y, yaw], axis=-1)
            return trajectory

        def sample_start_and_goal(rng=None):
            """Sample a start (x, y, yaw) and goal (x, y).

            Start is in the lower section (before narrow passage), goal is in the upper section.

            Returns:
                start: np.ndarray of shape (3,) — (x, y, yaw)
                goal: np.ndarray of shape (2,) — (x, y)
            """
            rng = np.random if rng is None else rng
            # Start in lower section
            start_y = rng.uniform(0.0 * terrain_length, 0.1 * terrain_length)
            start_x = path_center_x + rng.uniform(-half_width * 0.5, half_width * 0.5)

            # Goal in upper section
            goal_y = rng.uniform(0.9 * terrain_length, 1.0 * terrain_length)
            goal_x = path_center_x

            # Yaw pointing toward goal (generally upward along y)
            start_yaw = np.deg2rad(90) + np.deg2rad(rng.uniform(-10, 10))

            route_length = float(np.sqrt((goal_x - start_x) ** 2 + (goal_y - start_y) ** 2))
            return np.array([start_x, start_y, start_yaw]), np.array([goal_x, goal_y]), route_length

        def check_outside_boundary(x, y):
            """Return signed distance to boundary (positive=outside, negative=inside)."""
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            in_passage = (y >= passage_start_y) & (y <= passage_end_y)
            # Distance to nearest edge
            dist_passage = np.where(x < path_center_x, passage_left - x, x - passage_right)
            dist_sidewalk = np.where(x < path_center_x, lower_left - x, x - lower_right)
            return np.where(in_passage, dist_passage, dist_sidewalk)

        # Straight centerline projector
        _cl_y_np = np.linspace(0, terrain_length, 500)
        _cl_xy_np = np.stack([np.full(500, path_center_x), _cl_y_np], axis=1)
        _cl_arc_np = _cl_y_np.copy()
        project_to_centerline = _build_centerline_projector(_cl_xy_np, _cl_arc_np)

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
class NarrowPassageTerrainCfg(BaseTerrainCfg):
    """Configuration for a narrow passage terrain."""

    class_type = NarrowPassageTerrain
