# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SidewalkBench-PG terrain importer."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .importer import CustomTerrainImporter


@configclass
class CustomTerrainImporterCfg(TerrainImporterCfg):
    """Configuration for the custom terrain importer."""

    class_type: type = CustomTerrainImporter
    """The class to use for the terrain importer."""

    walkable_visual_materials: list[sim_utils.VisualMaterialCfg] | None = [
        sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
    ]
    """Visual materials applied to walkable terrain mesh groups."""

    non_walkable_visual_materials: list[sim_utils.VisualMaterialCfg] | None = [
        sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
    ]
    """Visual materials applied to non-walkable terrain mesh groups."""

    building_enabled: bool = False
    """Whether to spawn buildings on non-walkable terrain. Defaults to False."""

    car_enabled: bool = False
    """Whether to spawn cars on the road. Defaults to False."""
