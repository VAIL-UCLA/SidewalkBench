from __future__ import annotations

import os
from dataclasses import MISSING
from glob import glob

import numpy as np

import isaaclab.sim as sim_utils
import isaaclab_tasks.core.velocity.mdp as loc_mdp
import sidewalkbench.pg.mdp.commands as custom_cmd
import sidewalkbench.pg.mdp.rewards as custom_rew
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

use_fixed_texture = os.environ.get("USE_FIXED_TEXTURE") == "1"


# Random path
material_path = "./assets/materials"

if not use_fixed_texture:
    walkable_material_path_list = sorted(glob(f"{material_path}/material_sidewalk_mdl/*.mdl"))
    walkable_material_path_list = [np.random.choice(walkable_material_path_list)]
    non_walkable_material_path_list = sorted(glob(f"{material_path}/material_terrain_mdl/*.mdl")) + sorted(
        glob(f"{material_path}/material_road_mdl/*.mdl")
    )
    non_walkable_material_path_list = [np.random.choice(non_walkable_material_path_list)]
else:
    walkable_material_path_list = [f"{material_path}/Ground/Rough_Gravel.mdl"]
    non_walkable_material_path_list = [
        f"{material_path}/material_terrain_mdl/Grass005_2K_PNG.mdl",
    ]
walkable_visual_materials = []
for mdl_path in walkable_material_path_list:
    walkable_visual_materials.append(
        sim_utils.MdlFileCfg(mdl_path=mdl_path, project_uvw=True, texture_scale=(1.0, 1.0))
    )
non_walkable_visual_materials = []
for mdl_path in non_walkable_material_path_list:
    non_walkable_visual_materials.append(
        sim_utils.MdlFileCfg(mdl_path=mdl_path, project_uvw=True, texture_scale=(1.0, 1.0))
    )

fixed_light_intensity = np.random.uniform(880.0, 900.0)

back_ground_hdr = sorted(glob(f"{material_path}/material_background_hdr/*.hdr"))


from sidewalkbench.pg.terrains.generator_cfg import CustomTerrainGeneratorCfg
from sidewalkbench.pg.terrains.importer_cfg import CustomTerrainImporterCfg
from sidewalkbench.pg.terrains import (
    BlockSidewalkTerrainCfg,
    CircularTerrainCfg,
    CrosswalkTerrainCfg,
    CurvyLeftTerrainCfg,
    CurvyRightTerrainCfg,
    CurvyTerrainCfg,
    IntersectionTerrainCfg,
    NarrowPassageTerrainCfg,
    PlazaTerrainCfg,
    SideWalkTerrainCfg,
    SquareTerrainCfg,
)

GO2_FILTER_TERRAIN_CFG = CustomTerrainGeneratorCfg(
    seed=0,
    size=(30.0, 20.0),
    border_width=10.0,
    border_height=0.0,
    num_rows=1,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    sidewalk_height=0.15 + 0.01,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "sidewalk": SideWalkTerrainCfg(),
        "plaza": PlazaTerrainCfg(),
        "square": SquareTerrainCfg(),
        "intersection": IntersectionTerrainCfg(),
        "curvy": CurvyTerrainCfg(),
        "circular": CircularTerrainCfg(),
        "narrow_passage": NarrowPassageTerrainCfg(),
        "block_sidewalk": BlockSidewalkTerrainCfg(),
        "crosswalk": CrosswalkTerrainCfg(),
        "curvy_left": CurvyLeftTerrainCfg(),
        "curvy_right": CurvyRightTerrainCfg(),
    },
)
# ------------------------------------------------
# Scene Config
from scipy.spatial.transform import Rotation as R

q_base = R.from_quat([0.5, -0.5, 0.5, -0.5])


roll_var = -20.0  # 10 degree roll around local Z-axis
pitch_var = 0.0  # Looking downwards (positive rotation around local X-axis tilts it down)
yaw_var = 0.0  # Side-to-side heading

# 3. Create the local offset rotation
q_offset = R.from_euler("xyz", [roll_var, pitch_var, yaw_var], degrees=True)

# 4. Combine them (Intrinsic/Local multiplication order)
q_final = (q_base * q_offset).as_quat()


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration"""

    env_spacing: float = 21.0
    # procedural generation config
    pg_config: dict = MISSING
    # robot
    robot: ArticulationCfg = MISSING

    terrain = CustomTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GO2_FILTER_TERRAIN_CFG,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
            texture_scale=(4.0, 4.0),
        ),
        walkable_visual_materials=walkable_visual_materials,
        non_walkable_visual_materials=non_walkable_visual_materials,
        debug_vis=False,
        building_enabled=True,
        car_enabled=True,
    )


    # cameras
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.005,
        height=288,
        width=512,
        data_types=["rgb", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[790.0, 0.0, 960.0, 0.0, 790.0, 540.0, 0.0, 0.0, 1.0],
            width=1920,
            height=1080,
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.51, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        update_latest_camera_pose=True,
    )

    camera2 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/third_cam",
        update_period=0.005,
        height=1080,
        width=1920,
        data_types=["rgb", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[790.0, 0.0, 960.0, 0.0, 790.0, 540.0, 0.0, 0.0, 1.0],
            width=1920,
            height=1080,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-3, 0.0, 2.0), rot=(q_final[0], q_final[1], q_final[2], q_final[3]), convention="ros"
        ),
        update_latest_camera_pose=True,
    )

    # Optional free goal camera (pose set explicitly from the goal). Populated by
    # the environment only when a goal image is needed.
    goal_camera: CameraCfg | None = None

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=np.random.uniform(500.0, 1000.0) if not use_fixed_texture else fixed_light_intensity,
            texture_file=(
                np.random.choice(back_ground_hdr)
                if not use_fixed_texture
                else f"{material_path}/kloofendal_48d_partly_cloudy_puresky_4k.hdr"
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 1.0, 1.0),
            angle=20,
            intensity=fixed_light_intensity,
        ),
    )


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    eye: tuple[float, float, float] = (0, -4, 2)

    lookat: tuple[float, float, float] = (0, 0, 0)

    cam_prim_path: str = "/OmniverseKit_Persp"

    resolution: tuple[int, int] = (512, 288)

    origin_type: str = "asset_root"

    env_index: int = 0

    asset_name: str | None = "robot"


# ------------------------------------------------
# Specific settings for training
# Note:
# 1. define action cfg for each type of robots
# 2. if you want to change the reward, command or any other settings, change it in custom_env
# ------------------------------------------------
@configclass
class CommandsCfg:
    """Command specifications for the MDP (none active)."""


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class SensorCfg(ObsGroup):
        rgb = ObsTerm(func=custom_cmd.get_image, params={"sensor_cfg": SceneEntityCfg("camera"), "type": "RGB"})

    # observation groups
    sensor: SensorCfg = SensorCfg(history_length=5, flatten_history_dim=False)


@configclass
class EventCfg:
    """Configuration for events."""

    # Establish the episode/retry identity before sampling either event below.
    reset_randomization = EventTerm(func=custom_cmd.begin_scene_reset, mode="reset")

    reset_base = EventTerm(
        func=custom_cmd.reset_start_goal,
        mode="reset",
    )

    reset_obj = EventTerm(
        func=custom_cmd.reset_obj,
        mode="reset",
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=loc_mdp.time_out, time_out=True)
    out_of_region = DoneTerm(
        func=custom_rew.out_of_region,
        time_out=False,
        params={"buffer": 1.0},
    )
    collision_static = DoneTerm(func=custom_rew.custom_collision_static, time_out=False)
    collision_dynamic = DoneTerm(func=custom_rew.custom_collision_dynamic, time_out=False)
    arrive = DoneTerm(func=custom_rew.arrive, time_out=False)
    freeze = DoneTerm(
        func=custom_rew.freeze,
        time_out=False,
        params={"freeze_duration": 5.0, "move_threshold": 0.1},
    )
    pedestrian_group_intrusion = DoneTerm(func=custom_rew.pedestrian_group_intrusion, time_out=False)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
