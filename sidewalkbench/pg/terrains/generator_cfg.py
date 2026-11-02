# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SidewalkBench-PG terrain generator."""

from __future__ import annotations

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .generator import CustomTerrainGenerator


@configclass
class CustomTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Terrain generator configuration adding sidewalk-specific dimensions."""

    class_type: type = CustomTerrainGenerator
    """The class to use for the terrain generator."""

    sidewalk_height: float = 0.15
    """Height of the sidewalk surface above the ground [m]."""

    ground_height: float = 0.01
    """Height of the ground surface [m]."""

    sidewalk_width: float = 2.0
    """Width of the sidewalk [m]."""
