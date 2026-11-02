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


class SquareTerrain(Terrain):
    """Procedural terrain builder for SquareTerrainCfg."""

    def build(self) -> tuple:
        cfg = self.cfg
        generator_cfg = self.generator_cfg
        """Generate a terrain with a rounded-square circuit sidewalk.

        The circuit is a square path with smooth rounded corners (quarter-circle arcs)
        instead of sharp 90-degree turns.
        """
        sidewalk_width = generator_cfg.sidewalk_width
        terrain_width = cfg.size[0]
        terrain_length = cfg.size[1]
        half_width = sidewalk_width / 2.0

        meshes_list: list[trimesh.Trimesh] = []

        sidewalk_height = generator_cfg.sidewalk_height
        ground_height = generator_cfg.ground_height

        center = np.array([terrain_width / 2.0, terrain_length / 2.0])
        margin = max(half_width + 0.5, 1.0)
        max_half_side = min(terrain_width, terrain_length) / 2.0 - margin
        half_side = max(2.0, max_half_side * 0.75)
        half_side = min(half_side, max_half_side)

        # Corner radius for smooth turns (larger = smoother)
        corner_radius = half_side * 0.3

        # Build the rounded-square centreline as a Shapely polygon
        # A square with rounded corners is simply a square buffered outward then
        # intersected, but easiest: create a small square and buffer it.
        # Alternatively, build the centreline square and buffer with round join.
        inner_half = half_side - corner_radius
        square_box = Polygon(
            [
                [center[0] - inner_half, center[1] - inner_half],
                [center[0] + inner_half, center[1] - inner_half],
                [center[0] + inner_half, center[1] + inner_half],
                [center[0] - inner_half, center[1] + inner_half],
            ]
        )
        # Buffer outward by corner_radius with round caps → rounded-corner square centreline
        centreline_polygon = square_box.buffer(corner_radius, resolution=64, join_style=1)  # 1 = round

        # Build the sidewalk ring from centreline: outer minus inner
        outer_polygon = square_box.buffer(corner_radius + half_width, resolution=64, join_style=1)
        inner_polygon = square_box.buffer(max(corner_radius - half_width, 0.01), resolution=64, join_style=1)
        circuit_polygon = outer_polygon.difference(inner_polygon)
        if not circuit_polygon.is_valid:
            circuit_polygon = circuit_polygon.buffer(0.0)

        sidewalk_mesh = trimesh.creation.extrude_polygon(polygon=circuit_polygon, height=sidewalk_height, engine="manifold")
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

        placeholder_mesh = trimesh.creation.extrude_polygon(polygon=ground_polygon, height=0.001, engine="manifold")
        other_meshes = [placeholder_mesh]
        combined_meshes = trimesh.util.concatenate(other_meshes)
        meshes_list.append(combined_meshes)

        origin = np.array([0.0, 0.0, 0.0])

        # ---- Helper: sample a point on the centreline at parameter t in [0, 1) ----
        # We parameterise the rounded-square centreline by arc-length.
        # The centreline is the boundary of centreline_polygon.
        raw_ring = np.array(centreline_polygon.exterior.coords)  # Nx2, closed (last == first)

        # Resample the ring uniformly by arc-length so that straight edges also get
        # dense vertices (Shapely only puts 2 vertices on each straight edge).
        _raw_diffs = np.diff(raw_ring, axis=0)
        _raw_seg_len = np.linalg.norm(_raw_diffs, axis=1)
        _raw_cum = np.concatenate([[0.0], np.cumsum(_raw_seg_len)])
        _raw_total = _raw_cum[-1]
        n_resample = 1024  # number of uniformly-spaced vertices (before closing)
        _s_uniform = np.linspace(0.0, _raw_total, n_resample, endpoint=False)
        _idx = np.searchsorted(_raw_cum, _s_uniform, side="right") - 1
        _idx = np.clip(_idx, 0, len(_raw_seg_len) - 1)
        _frac = (_s_uniform - _raw_cum[_idx]) / np.clip(_raw_seg_len[_idx], 1e-9, None)
        centreline_ring = raw_ring[_idx] + _frac[:, None] * _raw_diffs[_idx]
        # Close the ring
        centreline_ring = np.concatenate([centreline_ring, centreline_ring[:1]], axis=0)

        # Rotate the ring so that index 0 starts at the top-middle of the square
        # (center_x, center_y + half_side), which is the midpoint of the top straight edge.
        top_mid = np.array([center[0], center[1] + half_side])
        dists_to_top = np.linalg.norm(centreline_ring[:-1] - top_mid, axis=1)  # exclude closing dup
        start_idx = int(np.argmin(dists_to_top))
        # Roll the ring so start_idx becomes index 0, keep it closed
        centreline_ring = np.concatenate(
            [
                centreline_ring[start_idx:-1],  # from start_idx to second-to-last (drop closing dup)
                centreline_ring[: start_idx + 1],  # wrap around back to start_idx (re-close)
            ],
            axis=0,
        )

        # Compute cumulative arc-length along the centreline
        diffs = np.diff(centreline_ring, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_length = cum_length[-1]

        def _interp_centreline(t_norm):
            """Interpolate position(s) on the centreline.

            Args:
                t_norm: float or 1-D array in [0, 1).

            Returns:
                xy: ndarray of shape (..., 2)
                tangent: ndarray of shape (..., 2) — unit tangent at each point
            """
            t_norm = np.atleast_1d(t_norm).astype(float)
            s = (t_norm % 1.0) * total_length
            # Find segment index for each s
            idx = np.searchsorted(cum_length, s, side="right") - 1
            idx = np.clip(idx, 0, len(seg_lengths) - 1)
            # Local parameter within segment
            frac = (s - cum_length[idx]) / np.clip(seg_lengths[idx], 1e-9, None)
            p0 = centreline_ring[idx]
            p1 = centreline_ring[idx + 1]
            xy = p0 + frac[:, None] * (p1 - p0)
            tangent = diffs[idx]
            tangent = tangent / np.linalg.norm(tangent, axis=-1, keepdims=True)
            return xy, tangent

        def sample_pos(walkable=True, num_samples=1, rng=None) -> tuple[np.ndarray, np.ndarray]:
            rng = np.random if rng is None else rng
            if walkable:
                t = rng.uniform(0.0, 1.0, num_samples)
                pts, _ = _interp_centreline(t)
                # Add lateral offset within sidewalk width
                offsets = rng.uniform(-half_width * 0.9, half_width * 0.9, num_samples)
                # Offset perpendicular to tangent (rotate tangent 90°)
                _, tang = _interp_centreline(t)
                perp = np.stack([-tang[:, 1], tang[:, 0]], axis=-1)
                x = pts[:, 0] + offsets * perp[:, 0]
                y = pts[:, 1] + offsets * perp[:, 1]
                return x, y

            x = rng.uniform(0.0, terrain_width, num_samples)
            y = rng.uniform(0.0, terrain_length, num_samples)
            # Reject points on the sidewalk
            from shapely.vectorized import contains

            in_ring = contains(circuit_polygon, x, y)
            max_resample = 20
            resample_count = 0
            while np.any(in_ring) and resample_count < max_resample:
                k = int(np.sum(in_ring))
                x[in_ring] = rng.uniform(0.0, terrain_width, k)
                y[in_ring] = rng.uniform(0.0, terrain_length, k)
                in_ring = contains(circuit_polygon, x, y)
                resample_count += 1
            return x, y

        def sample_trajectory(num_points: int, rng=None, **kwargs) -> np.ndarray:
            rng = np.random if rng is None else rng
            t_vals = np.linspace(0.05, 0.95, num_points)
            pts, tang = _interp_centreline(t_vals)

            # Small lateral offset for variety
            lateral = rng.uniform(-half_width * 0.45, half_width * 0.45)
            perp = np.stack([-tang[:, 1], tang[:, 0]], axis=-1)
            traj_x = pts[:, 0] + lateral * perp[:, 0]
            traj_y = pts[:, 1] + lateral * perp[:, 1]

            yaw = np.arctan2(tang[:, 1], tang[:, 0])
            trajectory = np.stack([traj_x, traj_y, yaw], axis=-1)
            return trajectory

        def sample_start_and_goal(rng=None):
            """Sample start (x, y, yaw) and goal (x, y) on the square circuit.

            Returns:
                start: np.ndarray of shape (3,) — (x, y, yaw)
                goal:  np.ndarray of shape (2,) — (x, y)
            """
            rng = np.random if rng is None else rng
            cw = rng.rand() < 0.5
            if cw:
                # Forward direction around circuit
                start_t = rng.uniform(0.0, 0.05)
                goal_t = rng.uniform(0.9, 0.95)
            else:
                # Reverse direction
                start_t = rng.uniform(0.95, 1.0)
                goal_t = rng.uniform(0.05, 0.1)
            start_pts, start_tang = _interp_centreline(np.array([start_t]))
            goal_pts, _ = _interp_centreline(np.array([goal_t]))

            offset_s = rng.uniform(-half_width * 0.5, half_width * 0.5)
            perp_s = np.array([-start_tang[0, 1], start_tang[0, 0]])

            start_x = start_pts[0, 0] + offset_s * perp_s[0]
            start_y = start_pts[0, 1] + offset_s * perp_s[1]
            start_yaw = float(np.arctan2(start_tang[0, 1], start_tang[0, 0]))
            start_yaw += np.deg2rad(rng.uniform(-10, 10))

            if not cw:
                start_yaw += np.pi  # Face opposite direction if going counterclockwise

            goal_x = goal_pts[0, 0]
            goal_y = goal_pts[0, 1]

            # Route length: arc along centreline between start_t and goal_t
            if cw:
                route_length = float(abs(goal_t - start_t) * total_length)
            else:
                route_length = float(abs(start_t - goal_t) * total_length)

            return np.array([start_x, start_y, start_yaw]), np.array([goal_x, goal_y]), route_length

        def check_outside_boundary(x, y):
            """Return signed distance to boundary (positive=outside, negative=inside)."""
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            from shapely.geometry import Point as _Pt

            scalar = x.ndim == 0
            x, y = np.atleast_1d(x), np.atleast_1d(y)
            dists = np.array(
                [
                    (
                        circuit_polygon.boundary.distance(_Pt(float(xi), float(yi)))
                        if not circuit_polygon.contains(_Pt(float(xi), float(yi)))
                        else -circuit_polygon.boundary.distance(_Pt(float(xi), float(yi)))
                    )
                    for xi, yi in zip(x, y)
                ]
            )
            return dists[0] if scalar else dists

        # Square centreline projector — reuse centreline_ring and cum_length
        project_to_centerline = _build_centerline_projector(centreline_ring, cum_length)

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
class SquareTerrainCfg(BaseTerrainCfg):
    """Configuration for a rounded-square circuit terrain."""

    class_type = SquareTerrain
