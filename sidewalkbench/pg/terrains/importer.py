# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from sidewalkbench.pg.utils.math import create_prim_from_mesh

if TYPE_CHECKING:
    from .importer_cfg import CustomTerrainImporterCfg

# import logger
logger = logging.getLogger(__name__)


class CustomTerrainImporter:
    r"""A class to handle terrain meshes and import them into the simulator.

    We assume that a terrain mesh comprises of sub-terrains that are arranged in a grid with
    rows ``num_rows`` and columns ``num_cols``. The terrain origins are the positions of the sub-terrains
    where the robot should be spawned.

    Based on the configuration, the terrain importer handles computing the environment origins from the sub-terrain
    origins. In a typical setup, the number of sub-terrains (:math:`num\_rows \times num\_cols`) is smaller than
    the number of environments (:math:`num\_envs`). In this case, the environment origins are computed by
    sampling the sub-terrain origins.

    If a curriculum is used, it is possible to update the environment origins to terrain origins that correspond
    to a harder difficulty. This is done by calling :func:`update_terrain_levels`. The idea comes from game-based
    curriculum. For example, in a game, the player starts with easy levels and progresses to harder levels.
    """

    terrain_prim_paths: list[str]
    """A list containing the USD prim paths to the imported terrains."""
    #  terrain origins: origin fo the starting point of the robot
    terrain_origins: torch.Tensor | None
    terrain_walkable_samplers: list[Callable] | None
    terrain_trajectory_samplers: list[Callable] | None
    terrain_start_goal_samplers: list[Callable] | None
    terrain_boundary_checkers: list[Callable] | None
    terrain_centerline_projectors: list[Callable] | None
    """The origins of the sub-terrains in the added terrain mesh. Shape is (num_rows, num_cols, 3).

    If terrain origins is not None, the environment origins are computed based on the terrain origins.
    Otherwise, the environment origins are computed based on the grid spacing.
    """

    env_origins: torch.Tensor
    """The origins of the environments. Shape is (num_envs, 3)."""

    def __init__(self, cfg: CustomTerrainImporterCfg):
        """Initialize the terrain importer.

        Args:
            cfg: The configuration for the terrain importer.

        Raises:
            ValueError: If input terrain type is not supported.
            ValueError: If terrain type is 'generator' and no configuration provided for ``terrain_generator``.
            ValueError: If terrain type is 'usd' and no configuration provided for ``usd_path``.
            ValueError: If terrain type is 'usd' or 'plane' and no configuration provided for ``env_spacing``.
        """
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # create buffers for the terrains
        self.terrain_prim_paths = list()
        self.terrain_origins = None
        self.terrain_walkable_samplers = None
        self.terrain_trajectory_samplers = None
        self.terrain_start_goal_samplers = None
        self.terrain_boundary_checkers = None
        self.terrain_centerline_projectors = None
        self.env_origins = None  # assigned later when `configure_env_origins` is called
        # private variables
        self._terrain_flat_patches = dict()

        # auto-import the terrain based on the config
        if self.cfg.terrain_type == "generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain
            terrain_generator = self.cfg.terrain_generator.class_type(
                cfg=self.cfg.terrain_generator, device=self.device
            )
            self.import_meshes("terrain", terrain_generator.terrain_meshes, terrain_generator.cfg)
            # place building instances
            if self.cfg.building_enabled and terrain_generator.terrain_building_locations:
                self._add_building(terrain_generator.terrain_building_locations, terrain_generator.cfg.sidewalk_height)
            # place car instances
            if self.cfg.car_enabled and terrain_generator.terrain_car_locations:
                self._add_car(terrain_generator.terrain_car_locations, terrain_generator.cfg.ground_height)
            if self.cfg.use_terrain_origins:
                # configure the terrain origins based on the terrain generator
                self.configure_env_origins(terrain_generator.terrain_origins)
            else:
                self.configure_env_origins()
            # refer to the flat patches
            self.terrain_walkable_samplers = terrain_generator.terrain_walkable_samplers
            self.terrain_trajectory_samplers = terrain_generator.terrain_trajectory_samplers
            self.terrain_start_goal_samplers = terrain_generator.terrain_start_goal_samplers
            self.terrain_boundary_checkers = terrain_generator.terrain_boundary_checkers
            self.terrain_centerline_projectors = terrain_generator.terrain_centerline_projectors
            self.terrain_centerline_splines = terrain_generator.terrain_centerline_splines
            self._terrain_flat_patches = terrain_generator.flat_patches
        elif self.cfg.terrain_type == "usd":
            # check if config is provided
            if self.cfg.usd_path is None:
                raise ValueError("Input terrain type is 'usd' but no value provided for 'usd_path'.")
            # import the terrain
            self.import_usd("terrain", self.cfg.usd_path)
            # configure the origins in a grid
            self.configure_env_origins()
        elif self.cfg.terrain_type == "plane":
            # load the plane
            self.import_ground_plane("terrain")
            # configure the origins in a grid
            self.configure_env_origins()
        else:
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")

        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    """
    Properties.
    """

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the terrain importer has a debug visualization implemented.

        This always returns True.
        """
        return True

    @property
    def flat_patches(self) -> dict[str, torch.Tensor]:
        """A dictionary containing the sampled valid (flat) patches for the terrain.

        This is only available if the terrain type is 'generator'. For other terrain types, this feature
        is not available and the function returns an empty dictionary.

        Please refer to the :attr:`TerrainGenerator.flat_patches` for more information.
        """
        return self._terrain_flat_patches

    @property
    def terrain_names(self) -> list[str]:
        """A list of names of the imported terrains."""
        return [f"'{path.split('/')[-1]}'" for path in self.terrain_prim_paths]

    """
    Operations - Visibility.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Set the debug visualization of the terrain importer.

        Args:
            debug_vis: Whether to visualize the terrain origins.

        Returns:
            Whether the debug visualization was successfully set. False if the terrain
            importer does not support debug visualization.

        Raises:
            RuntimeError: If terrain origins are not configured.
        """
        # create a marker if necessary
        if debug_vis:
            if not hasattr(self, "origin_visualizer"):
                self.origin_visualizer = VisualizationMarkers(
                    cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/TerrainOrigin")
                )
                if self.terrain_origins is not None:
                    self.origin_visualizer.visualize(self.terrain_origins.reshape(-1, 3))
                elif self.env_origins is not None:
                    self.origin_visualizer.visualize(self.env_origins.reshape(-1, 3))
                else:
                    raise RuntimeError("Terrain origins are not configured.")
            # set visibility
            self.origin_visualizer.set_visibility(True)
        else:
            if hasattr(self, "origin_visualizer"):
                self.origin_visualizer.set_visibility(False)
        # report success
        return True

    """
    Operations - Import.
    """

    def import_ground_plane(self, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)):
        """Add a plane to the terrain importer.

        Args:
            name: The name of the imported terrain. This name is used to create the USD prim
                corresponding to the terrain.
            size: The size of the plane. Defaults to (2.0e6, 2.0e6).

        Raises:
            ValueError: If a terrain with the same name already exists.
        """
        # create prim path for the terrain
        prim_path = self.cfg.prim_path + f"/{name}"
        # check if key exists
        if prim_path in self.terrain_prim_paths:
            raise ValueError(
                f"A terrain with the name '{name}' already exists. Existing terrains: {', '.join(self.terrain_names)}."
            )
        # store the mesh name
        self.terrain_prim_paths.append(prim_path)

        # obtain ground plane color from the configured visual material
        color = (0.0, 0.0, 0.0)
        if self.cfg.visual_material is not None:
            material = self.cfg.visual_material.to_dict()
            # defaults to the `GroundPlaneCfg` color if diffuse color attribute is not found
            if "diffuse_color" in material:
                color = material["diffuse_color"]
            else:
                logger.warning(
                    "Visual material specified for ground plane but no diffuse color found."
                    " Using default color: (0.0, 0.0, 0.0)"
                )

        # get the mesh
        ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=self.cfg.physics_material, size=size, color=color)
        ground_plane_cfg.func(prim_path, ground_plane_cfg)

    def import_meshes(self, name: str, meshes: list[trimesh.Trimesh], terrain_generator_cfg: TerrainGeneratorCfg):
        """Import multiple meshes into the simulator.

        The meshes are imported into the simulator under the prim path ``cfg.prim_path/{key}``. The created path
        contains the meshes as :class:`pxr.UsdGeom` instances along with visual or physics material prims.
        Args:
            name: The name of the imported terrain. This name is used to create the USD prim
                corresponding to the terrain.
            meshes: The list of meshes to import.

        Raises:
            ValueError: If a terrain with the same name already exists.
        """
        all_walkable_meshes = []
        all_non_walkable_meshes = []
        all_other_meshes = []
        border_mesh = meshes[-1]
        prim_path = self.cfg.prim_path + "/border"
        create_prim_from_mesh(
            prim_path, border_mesh, visual_material=self.cfg.visual_material, physics_material=self.cfg.physics_material
        )
        for i, mesh in enumerate(meshes[:-1]):
            if i % 3 == 0:
                all_walkable_meshes.append(mesh)
            elif i % 3 == 1:
                all_non_walkable_meshes.append(mesh)
            else:
                all_other_meshes.append(mesh)
        random.shuffle(all_walkable_meshes)
        all_walkable_meshes = [
            all_walkable_meshes[i :: len(self.cfg.walkable_visual_materials)]
            for i in range(len(self.cfg.walkable_visual_materials))
        ]

        for i, (mesh, visual_materials) in enumerate(zip(all_walkable_meshes, self.cfg.walkable_visual_materials)):
            if len(mesh) == 0:
                continue
            mesh = trimesh.util.concatenate(mesh)
            # add uvs according to sidewalkbench.pg original code
            uvs = []
            for vertex in mesh.vertices:
                uv = [vertex[0], vertex[1]]
                uvs.append(uv)
            uvs = np.array(uvs)
            uvs *= 0.2
            mesh.visual.uvs = uvs
            prim_path = self.cfg.prim_path + f"/walkable_{i}"
            create_prim_from_mesh(
                prim_path, mesh, visual_material=visual_materials, physics_material=self.cfg.physics_material
            )

        all_non_walkable_meshes = [
            all_non_walkable_meshes[i :: len(self.cfg.non_walkable_visual_materials)]
            for i in range(len(self.cfg.non_walkable_visual_materials))
        ]

        for i, (mesh, visual_materials) in enumerate(
            zip(all_non_walkable_meshes, self.cfg.non_walkable_visual_materials)
        ):
            if len(mesh) == 0:
                continue
            mesh = trimesh.util.concatenate(mesh)
            # add uvs according to sidewalkbench.pg original code
            uvs = []
            for vertex in mesh.vertices:
                uv = [vertex[0], vertex[1]]
                uvs.append(uv)
            uvs = np.array(uvs)
            uvs *= 0.2
            mesh.visual.uvs = uvs
            prim_path = self.cfg.prim_path + f"/non_walkable_{i}"
            create_prim_from_mesh(
                prim_path, mesh, visual_material=visual_materials, physics_material=self.cfg.physics_material
            )

        mesh = trimesh.util.concatenate(all_other_meshes)

        prim_path = self.cfg.prim_path + "/other"
        create_prim_from_mesh(prim_path, mesh)

        import omni.usd
        from pxr import Sdf, UsdShade

        stage = omni.usd.get_context().get_stage()
        mesh_prim = stage.GetPrimAtPath(f"{prim_path}/mesh")
        mat_path = f"{prim_path}/matteMaterial"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        # Read vertex colors into diffuseColor so vertex colors stay visible
        primvar_reader = UsdShade.Shader.Define(stage, f"{mat_path}/primvarReader")
        primvar_reader.CreateIdAttr("UsdPrimvarReader_float3")
        primvar_reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            primvar_reader.ConnectableAPI(), "result"
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)

    def _load_building_infos(self):
        import json
        import os

        assets_dir = "assets"
        usd_dir = os.path.join(assets_dir, "usds")
        adj_dir = os.path.join(assets_dir, "adj_parameter_folder")
        building_infos = []
        for fn in sorted(os.listdir(usd_dir)):
            if "building" not in fn.lower() or "non_metric" in fn:
                continue
            uuid_part = fn[:-4].rsplit("_", 1)[-1]
            json_name = f"structures_Building-{uuid_part}.json"
            json_path = os.path.join(adj_dir, json_name)
            if not os.path.exists(json_path):
                continue
            obj_info = json.load(open(json_path))
            usd_path = os.path.join(usd_dir, fn)
            building_infos.append(
                {
                    "usd_path": usd_path,
                    "obj_info": obj_info,
                    "usd_name": fn,
                    "uuid": uuid_part,
                }
            )
        return building_infos

    def _load_car_infos(self):
        import json
        import os

        asset_root = "assets"
        usd_dir = os.path.join(asset_root, "usds")
        adj_dir = os.path.join(asset_root, "adj_parameter_folder")
        # Build UUID → JSON path mapping (JSON may have model-name prefix)
        json_by_uuid = {}
        for jf in os.listdir(adj_dir):
            if not jf.endswith(".json"):
                continue
            stem = jf[4:-5]  # strip 'car_' and '.json'
            uuid_part = stem.split("-", 1)[-1] if "-" in stem else stem
            json_by_uuid[uuid_part] = os.path.join(adj_dir, jf)
        car_infos = []
        allowed_uuids = {
            "e4087350c4d84d8391de09b405c96488",
            "d999506c52cf4313aad5220f4af2da6c",
            "03a6d639648540e3ad45ed38c9b19d1f",
            "389f9ebcaeb84cbfb020a88a25751edc",
            "5416965277554e7682fcc2dc4961a99d",
            "af108d4773df4722a62420f65dd5b8fe",
            "f3c9f419c0924e968deed17196248c98",
            "d893ecd4f4d94f509aa7312306d74c61",
            "c6dc0d2275524bb1a43ddc398d2be3bc",
            "1533810cfbf146ac813da2b03f12befa",
        }
        for fn in sorted(os.listdir(usd_dir)):
            if "non_metric" in fn:
                continue
            uuid_part = fn[4:-4]
            if uuid_part not in allowed_uuids:
                continue
            json_path = json_by_uuid.get(uuid_part)
            if not json_path:
                continue
            obj_info = json.load(open(json_path))
            c_len = obj_info["general"]["length"]
            c_wid = obj_info["general"]["width"]
            c_hei = obj_info["general"]["height"]
            usd_path = os.path.join(usd_dir, fn)
            car_infos.append(
                {
                    "length": c_len,
                    "width": c_wid,
                    "height": c_hei,
                    "usd_path": usd_path,
                    "obj_info": obj_info,
                    "usd_name": fn,
                    "uuid": uuid_part,
                }
            )
        return car_infos

    def _add_building(self, building_locations_list: list[np.ndarray], sidewalk_height: float):
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        stage.DefinePrim("/World/Dataset", "Scope")

        building_infos = self._load_building_infos()
        if not building_infos:
            logger.warning("No building assets found, skipping building spawn.")
            return

        # Create prototypes for each building type
        building_proto_paths = []
        for b_info in building_infos:
            uuid = b_info["uuid"]
            proto_path = f"/World/Dataset/Building_{uuid}"
            building_cfg = sim_utils.UsdFileCfg(
                usd_path=b_info["usd_path"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=min(b_info["obj_info"].get("mass", 1000), 1000)),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            )
            proto_prim = building_cfg.func(proto_path, building_cfg, translation=(0.0, 0.0, 0.0))
            proto_prim.SetInstanceable(True)
            for child in proto_prim.GetChildren():
                child.SetInstanceable(True)
            proto_prim.SetActive(False)
            building_proto_paths.append(proto_path)

        all_locs = np.concatenate(building_locations_list, axis=0)
        instance_counter = 0
        for loc in all_locs:
            x, y, z, yaw, b_idx = float(loc[0]), float(loc[1]), float(loc[2]), float(loc[3]), int(loc[4])
            if b_idx < 0 or b_idx >= len(building_proto_paths):
                continue
            proto_path = building_proto_paths[b_idx]
            inst_path = f"/World/BuildingInstance_{instance_counter:04d}"
            stage.DefinePrim(inst_path, "Xform")
            inst_prim = stage.GetPrimAtPath(inst_path)
            inst_prim.GetReferences().AddReference(Sdf.Reference("", proto_path))
            inst_xform = UsdGeom.Xformable(inst_prim)
            inst_xform.GetTranslateOp().Set(Gf.Vec3f(x, y, z))
            inst_xform.AddRotateZOp().Set(float(np.degrees(yaw)))
            inst_prim.SetActive(True)
            instance_counter += 1

    def _add_car(self, car_locations_list: list[np.ndarray], ground_height: float):
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        stage.DefinePrim("/World/Dataset", "Scope")

        car_infos = self._load_car_infos()
        if not car_infos:
            logger.warning("No car assets found, skipping car spawn.")
            return

        car_proto_paths = []
        for c_info in car_infos:
            uuid = c_info["uuid"]
            proto_path = f"/World/Dataset/Car_{uuid}"
            car_cfg = sim_utils.UsdFileCfg(
                usd_path=c_info["usd_path"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=min(c_info["obj_info"].get("MASS", 1000), 1000)),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            )
            proto_prim = car_cfg.func(proto_path, car_cfg, translation=(0.0, 0.0, 0.0))
            proto_prim.SetInstanceable(True)
            for child in proto_prim.GetChildren():
                child.SetInstanceable(True)
            proto_prim.SetActive(False)
            car_proto_paths.append(proto_path)

        all_locs = np.concatenate(car_locations_list, axis=0)
        instance_counter = 0
        for loc in all_locs:
            x, y, z, yaw, c_idx = float(loc[0]), float(loc[1]), float(loc[2]), float(loc[3]), int(loc[4])
            if c_idx < 0 or c_idx >= len(car_proto_paths):
                continue
            proto_path = car_proto_paths[c_idx]
            inst_path = f"/World/CarInstance_{instance_counter:04d}"
            stage.DefinePrim(inst_path, "Xform")
            inst_prim = stage.GetPrimAtPath(inst_path)
            inst_prim.GetReferences().AddReference(Sdf.Reference("", proto_path))
            inst_xform = UsdGeom.Xformable(inst_prim)
            inst_xform.GetTranslateOp().Set(Gf.Vec3f(x, y, z))
            inst_xform.AddRotateZOp().Set(
                float(np.degrees(yaw + np.pi / 2))
            )  # !!!! TODO: fix this later, this is sooooooooooo hardcoded
            inst_prim.SetActive(True)
            instance_counter += 1

    def import_usd(self, name: str, usd_path: str):
        """Import a mesh from a USD file.

        This function imports a USD file into the simulator as a terrain. It parses the USD file and
        stores the mesh under the prim path ``cfg.prim_path/{key}``. If multiple meshes are present in
        the USD file, only the first mesh is imported.

        The function doe not apply any material properties to the mesh. The material properties should
        be defined in the USD file.

        Args:
            name: The name of the imported terrain. This name is used to create the USD prim
                corresponding to the terrain.
            usd_path: The path to the USD file.

        Raises:
            ValueError: If a terrain with the same name already exists.
        """
        # create prim path for the terrain
        prim_path = self.cfg.prim_path + f"/{name}"
        # check if key exists
        if prim_path in self.terrain_prim_paths:
            raise ValueError(
                f"A terrain with the name '{name}' already exists. Existing terrains: {', '.join(self.terrain_names)}."
            )
        # store the mesh name
        self.terrain_prim_paths.append(prim_path)

        # add the prim path
        cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
        cfg.func(prim_path, cfg)

    """
    Operations - Origins.
    """

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
        """Configure the origins of the environments based on the added terrain.

        Args:
            origins: The origins of the sub-terrains. Shape is (num_rows, num_cols, 3).
        """
        # decide whether to compute origins in a grid or based on curriculum
        if origins is not None:
            # convert to numpy
            if isinstance(origins, np.ndarray):
                origins = torch.from_numpy(origins)
            # store the origins
            self.terrain_origins = origins.to(self.device, dtype=torch.float)
            # compute environment origins
            self.env_origins = self._compute_env_origins_curriculum(self.cfg.num_envs, self.terrain_origins)
        else:
            self.terrain_origins = None
            # check if env spacing is valid
            if self.cfg.env_spacing is None:
                raise ValueError("Environment spacing must be specified for configuring grid-like origins.")
            # compute environment origins
            self.env_origins = self._compute_env_origins_grid(self.cfg.num_envs, self.cfg.env_spacing)

    def update_env_origins(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor):
        """Update the environment origins based on the terrain levels."""
        # check if grid-like spawning
        if self.terrain_origins is None:
            return
        # update terrain level for the envs
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # robots that solve the last level are sent to a random one
        # the minimum level is zero
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
        # update the env origins
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    """
    Internal helpers.
    """

    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        """Compute the origins of the environments defined by the sub-terrains origins."""
        # extract number of rows and cols
        num_rows, num_cols = origins.shape[:2]
        # maximum initial level possible for the terrains
        if self.cfg.max_init_terrain_level is None:
            max_init_level = num_rows - 1
        else:
            max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)
        # store maximum terrain level possible
        self.max_terrain_level = num_rows
        # define all terrain levels and types available
        terrain_idxs = torch.arange(num_envs, device=self.device).long()
        terrain_idxs = terrain_idxs % (num_rows * num_cols)

        self.terrain_levels = terrain_idxs // num_cols
        self.terrain_types = terrain_idxs % num_cols
        # create tensor based on number of environments
        env_origins = torch.zeros(num_envs, 3, device=self.device)
        env_origins[:] = origins[self.terrain_levels, self.terrain_types]
        return env_origins

    def _compute_env_origins_grid(self, num_envs: int, env_spacing: float) -> torch.Tensor:
        """Compute the origins of the environments in a grid based on configured spacing."""
        # create tensor based on number of environments
        env_origins = torch.zeros(num_envs, 3, device=self.device)
        # create a grid of origins
        num_rows = np.ceil(num_envs / int(np.sqrt(num_envs)))
        num_cols = np.ceil(num_envs / num_rows)
        ii, jj = torch.meshgrid(
            torch.arange(num_rows, device=self.device), torch.arange(num_cols, device=self.device), indexing="ij"
        )
        env_origins[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing
        env_origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing
        env_origins[:, 2] = 0.0
        return env_origins

    """
    Deprecated.
    """

    @property
    def warp_meshes(self):
        """A dictionary containing the terrain's names and their warp meshes.

        .. deprecated:: v2.1.0
            The `warp_meshes` attribute is deprecated. It is no longer stored inside the class.
        """
        logger.warning(
            "The `warp_meshes` attribute is deprecated. It is no longer stored inside the `TerrainImporter` class."
            " Returning an empty dictionary."
        )
        return {}

    @property
    def meshes(self) -> dict[str, trimesh.Trimesh]:
        """A dictionary containing the terrain's names and their tri-meshes.

        .. deprecated:: v2.1.0
            The `meshes` attribute is deprecated. It is no longer stored inside the class.
        """
        logger.warning(
            "The `meshes` attribute is deprecated. It is no longer stored inside the `TerrainImporter` class."
            " Returning an empty dictionary."
        )
        return {}
