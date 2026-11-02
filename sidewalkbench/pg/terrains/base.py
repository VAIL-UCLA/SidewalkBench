from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import MISSING

import numpy as np
import scipy.interpolate as interpolate  # noqa: F401
import trimesh
from shapely.geometry import Polygon  # noqa: F401

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass
from sidewalkbench.pg.terrains.generator_cfg import CustomTerrainGeneratorCfg  # noqa: F401


class Terrain(ABC):
    """Base class for procedural terrain builders.

    A terrain is instantiated per sub-terrain as ``XTerrain(cfg, generator_cfg)``
    and produces the generator tuple via :meth:`build`::

        (meshes, origin, walkable_sampler, trajectory_sampler, start_goal_sampler,
         boundary_checker, centerline_projector, building_locations,
         car_locations, arc_to_xy, total_length)
    """

    def __init__(self, cfg: SubTerrainBaseCfg, generator_cfg: CustomTerrainGeneratorCfg):
        self.cfg = cfg
        self.generator_cfg = generator_cfg

    @abstractmethod
    def build(self) -> tuple:
        """Build the terrain geometry and samplers."""
        ...

    @property
    def sidewalk_width(self) -> float:
        return self.generator_cfg.sidewalk_width

    @property
    def sidewalk_height(self) -> float:
        return self.generator_cfg.sidewalk_height

    @property
    def ground_height(self) -> float:
        return self.generator_cfg.ground_height


def _build_centerline_projector(cl_xy, cl_arc):
    """Build a projector closure from precomputed centerline points and arc-lengths.

    Args:
        cl_xy: (N, 2) dense centerline points.
        cl_arc: (N,) cumulative arc-length at each point.

    Returns:
        project_to_centerline(x, y, hint_s=None) -> (arc_length, lateral_offset)
    """
    n = len(cl_arc)

    def project_to_centerline(x, y, hint_s=None):
        pt = np.array([x, y])
        if hint_s is not None:
            # Search in a window around hint
            hint_idx = np.searchsorted(cl_arc, hint_s, side="right")
            lo = max(0, hint_idx - 20)
            hi = min(n, hint_idx + 20)
            dists = np.linalg.norm(cl_xy[lo:hi] - pt, axis=1)
            best_local = np.argmin(dists)
            best_idx = lo + best_local
        else:
            dists = np.linalg.norm(cl_xy - pt, axis=1)
            best_idx = np.argmin(dists)
        arc_length = float(cl_arc[best_idx])
        lateral_offset = float(dists[best_local] if hint_s is not None else dists[best_idx])
        return arc_length, lateral_offset

    return project_to_centerline


def _generate_block_centerline(num_blocks, block_length, sidewalk_width, rng=None):
    """Generate a chain of straight/turn blocks, returning centerline points.

    Heading is constrained to (0°, 180°) so the path always advances in +y.

    Returns:
        centerline: (N, 2) dense centerline points
        heading_final: final heading in radians
    """
    if rng is None:
        rng = np.random
    heading = np.pi / 2  # start pointing +y
    pos = np.array([0.0, 0.0])
    all_pts = [pos.copy()]

    samples_per_block = 30  # points per block for smooth curves
    prev_was_turn = False

    for _ in range(num_blocks):
        if prev_was_turn:
            block_type = "straight"
        else:
            block_type = rng.choice(["straight", "turn_left", "turn_right"], p=[0.6, 0.2, 0.2])

        if block_type == "straight":
            end = pos + block_length * np.array([np.cos(heading), np.sin(heading)])
            pts = np.linspace(pos, end, samples_per_block, endpoint=False)[1:]
            pos = end
            prev_was_turn = False
        else:
            # Random turn angle 0°-45°
            if rng.rand() < 0.2:
                angle = rng.uniform(np.deg2rad(30), np.deg2rad(60))
            else:
                angle = np.deg2rad(90)
            if block_type == "turn_left":
                sign = 1
            else:
                sign = -1

            # Check heading constraint: must stay in (0°, 180°)
            new_heading = heading + sign * angle
            if new_heading < 0 or new_heading > np.pi:
                # Fall back to straight
                end = pos + block_length * np.array([np.cos(heading), np.sin(heading)])
                pts = np.linspace(pos, end, samples_per_block, endpoint=False)[1:]
                pos = end
                prev_was_turn = False
            else:
                # Arc: radius chosen so arc length ≈ block_length
                radius = block_length / angle
                # Center of turning circle
                perp = heading + sign * np.pi / 2
                center = pos + radius * np.array([np.cos(perp), np.sin(perp)])
                # Sweep from current angle to new angle
                start_angle = heading - sign * np.pi / 2  # angle from center to pos
                angles = np.linspace(0, sign * angle, samples_per_block, endpoint=False)[1:]
                pts = np.stack(
                    [
                        center[0] + radius * np.cos(start_angle + angles),
                        center[1] + radius * np.sin(start_angle + angles),
                    ],
                    axis=-1,
                )
                pos = center + radius * np.array(
                    [
                        np.cos(start_angle + sign * angle),
                        np.sin(start_angle + sign * angle),
                    ]
                )
                heading = new_heading
                prev_was_turn = True

        all_pts.append(pts)
        all_pts.append(pos.copy().reshape(1, 2))

    centerline = np.concatenate([p.reshape(-1, 2) for p in all_pts], axis=0)
    return centerline, heading


def _generate_turn_centerline(
    corner_radius=4.0, turn_left=True, approach_length=3.0, exit_length=3.0, samples_per_segment=30
):
    """Generate centerline for a single 90° turn with approach and exit straight segments.

    Returns:
        centerline: (N, 2) dense centerline points
        arc: (N,) cumulative arc-length
        total_length: total arc-length
        turn_start_s: arc-length at beginning of arc
        turn_end_s: arc-length at end of arc
    """
    sign = 1 if turn_left else -1
    heading = np.pi / 2
    angle = np.pi / 2
    pos = np.array([0.0, 0.0])
    all_pts = [pos.copy()]

    # Straight approach
    end = pos + approach_length * np.array([np.cos(heading), np.sin(heading)])
    pts = np.linspace(pos, end, samples_per_segment, endpoint=False)[1:]
    pos = end
    all_pts.append(pts)
    all_pts.append(pos.copy().reshape(1, 2))

    # 90° arc
    new_heading = heading + sign * angle
    perp = heading + sign * np.pi / 2
    center = pos + corner_radius * np.array([np.cos(perp), np.sin(perp)])
    start_angle = heading - sign * np.pi / 2
    angles = np.linspace(0, sign * angle, samples_per_segment, endpoint=False)[1:]
    pts = np.stack(
        [
            center[0] + corner_radius * np.cos(start_angle + angles),
            center[1] + corner_radius * np.sin(start_angle + angles),
        ],
        axis=-1,
    )
    pos = center + corner_radius * np.array(
        [
            np.cos(start_angle + sign * angle),
            np.sin(start_angle + sign * angle),
        ]
    )
    heading = new_heading
    all_pts.append(pts)
    all_pts.append(pos.copy().reshape(1, 2))

    # Straight exit
    end = pos + exit_length * np.array([np.cos(heading), np.sin(heading)])
    pts = np.linspace(pos, end, samples_per_segment, endpoint=False)[1:]
    pos = end
    all_pts.append(pts)
    all_pts.append(pos.copy().reshape(1, 2))

    centerline = np.concatenate([p.reshape(-1, 2) for p in all_pts], axis=0)
    diffs = np.diff(centerline, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cl_arc = np.concatenate([[0], np.cumsum(seg_lens)])
    total_length = cl_arc[-1]
    turn_start_s = approach_length
    turn_end_s = approach_length + corner_radius * angle
    return centerline, cl_arc, total_length, turn_start_s, turn_end_s


def _build_terrain(cfg, *args, **kwargs):
    """Legacy ``function`` hook dispatching to the class-based ``class_type`` builder.

    Registered as the default ``function`` so ``SubTerrainBaseCfg`` validation passes
    and any code that still calls ``cfg.function`` keeps working.
    """
    generator_cfg = kwargs.pop("generator_cfg", None)
    if generator_cfg is None and len(args) >= 2:
        generator_cfg = args[1]
    return cfg.class_type(cfg, generator_cfg).build()


@configclass
class BaseTerrainCfg(SubTerrainBaseCfg):
    """Base configuration for all SidewalkBench-PG terrain types."""

    function: Callable = _build_terrain
    """Legacy generation hook; dispatches to :attr:`class_type`."""

    class_type: type[Terrain] = MISSING
    """Terrain builder class instantiated as ``class_type(cfg, generator_cfg)``."""
