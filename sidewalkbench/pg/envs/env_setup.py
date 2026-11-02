# Evaluation script for pretrained models

# launch omniverse
import argparse
import os
import sys
from glob import glob

import gymnasium as gym
import numpy as np
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to SidewalkBench-PG Environments!")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--save_images", action="store_true", default=False, help="Whether to save images during the simulation."
)
parser.add_argument(
    "--online_viewer",
    action="store_true",
    default=False,
    help="Show live composite third-person + first-person view via OpenCV window.",
)
parser.add_argument("--free_view", action="store_true", default=False, help="Enable free view mode for the camera.")
parser.add_argument(
    "--config",
    type=str,
    default="configs/unit_test/straight/FlowPilot.yaml",
    help="Path to environment config file.",
)
parser.add_argument("--exp_name", type=str, default="test", help="Experiment name.")
parser.add_argument(
    "--kinematics_only",
    action="store_true",
    default=False,
    help="Whether to use kinematics-only version of COCO (no rigid body dynamics).",
)
parser.add_argument("--robot_type", type=str, default="coco", help="Type of robot to use in the simulation.")
parser.add_argument("--obstacle_id", type=int, default=-1, help="ID of the obstacle to use in the simulation.")
parser.add_argument(
    "--save_video", action="store_true", default=False, help="Save the composite visualization as an mp4."
)
parser.add_argument("--save_mcap", action="store_true", default=False, help="Record a Foxglove MCAP per episode.")
parser.add_argument("--max_resets", type=int, default=10, help="Number of episodes to evaluate per environment.")
parser.add_argument(
    "--use_fixed_texture", action="store_true", default=False, help="Use fixed terrain textures and the benchmark lighting preset."
)
parser.add_argument(
    "--num_obstacles",
    type=int,
    default=None,
    help="Number of static obstacles (default: config value, or 36).",
)
parser.add_argument(
    "--sidewalk_width",
    type=float,
    default=None,
    help="Sidewalk width in meters (default: config value, or 4.0).",
)
parser.add_argument(
    "--use_behavior",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Use behavior pedestrians (default: enabled for long horizon, disabled for short horizon).",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument(
    "--texture_seed",
    type=int,
    help="Seed for terrain texture sampling (default: config value, or 21).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
import yaml

with open(args_cli.config) as f:
    env_config: dict = yaml.safe_load(f)

try:
    args_cli.sidewalk_width = float(
        args_cli.sidewalk_width
        if args_cli.sidewalk_width is not None
        else env_config.get("sidewalk_width", 4.0)
    )
except (TypeError, ValueError):
    parser.error("Sidewalk width must be a finite positive number of meters.")
if not 0.0 < args_cli.sidewalk_width < float("inf"):
    parser.error("Sidewalk width must be a finite positive number of meters.")
env_config["sidewalk_width"] = args_cli.sidewalk_width

if "terrain_segment" not in env_config or env_config["terrain_segment"] is None:
    env_config["terrain_segment"] = "default"
exp_folder = os.path.join(
    "experiments",
    args_cli.exp_name,
    f"{env_config['model_type']}_{env_config['terrain_segment']}_{env_config['scenario_type']}_{env_config['num_peds_per_terrain']}_{args_cli.seed}",
)
exp_folder = exp_folder + f"_{env_config['ped_lateral']}" if "ped_lateral" in env_config else exp_folder
exp_folder = (
    exp_folder + f"_{env_config['ped_crossing_direction']}" if "ped_crossing_direction" in env_config else exp_folder
)
# Non-default embodiments get their own result folders.
exp_folder = exp_folder + f"_{args_cli.robot_type}" if args_cli.robot_type != "coco" else exp_folder
os.makedirs(exp_folder, exist_ok=True)
if os.path.exists(f"{exp_folder}/results.pkl"):
    sys.exit(0)  # skip if results already exist
# launch omniverse app
# Enable the Isaac Sim cloner extension required by SidewalkScenePG without editing the app .kit files.
# RTX budgets its streaming memory as (free VRAM + streaming usage - gpuSafetyInMiB). Once a model's ONNX Runtime arena
# (FlowPilot: ~3.4 GB) and texture streaming leave the geometry streamer a grant under ~320 MiB, it silently unloads
# every scene mesh and all cameras see only the sky dome for the rest of the run. A negative reserve keeps the geometry
# grant intact (it only needs a few hundred MiB) while texture streaming still adapts to the free VRAM.
args_cli.kit_args = (
    f"--enable isaacsim.core.cloner --/rtx-transient/memorybudgetmanager/gpuSafetyInMiB=-1024 {args_cli.kit_args or ''}"
).strip()
app_launcher = AppLauncher(args_cli, enable_cameras=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

if args_cli.texture_seed is None:
    args_cli.texture_seed = env_config.get("texture_seed", 21)
np.random.seed(args_cli.texture_seed)
if args_cli.use_fixed_texture:
    os.environ["USE_FIXED_TEXTURE"] = "1"
else:
    os.environ.pop("USE_FIXED_TEXTURE", None)

import isaaclab_tasks.core.velocity.mdp as loc_mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm

import sidewalkbench.pg.mdp.rewards as custom_rew
from sidewalkbench.pg.primitives.navigation.random_env_cfg import (
    CameraCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    SceneCfg,
    TerminationsCfg,
    ViewerCfg,
)
from sidewalkbench.pg.primitives.robot.coco import (
    COCO_KINEMATICS_CFG,
    GO2_KINEMATICS_CFG,
    M20_KINEMATICS_CFG,
    COCOWaypointKinematicsActionsCfg,
)
from sidewalkbench.pg.primitives.robot.registry import get_robot_configs, get_robot_profile

robot_profile = get_robot_profile(args_cli.robot_type)
robot_asset_cfg, robot_actions_cfg = get_robot_configs(args_cli.robot_type)


@configclass
class EmptyRewardsCfg:
    pass


## setup robot (the same geometry is used by GS scenes)
robot_cfg = robot_profile.as_dict()

if args_cli.num_obstacles is not None and args_cli.num_obstacles >= 0:
    env_config["num_static_obstacles"] = args_cli.num_obstacles
else:
    env_config.setdefault("num_static_obstacles", 36)
args_cli.num_obstacles = env_config["num_static_obstacles"]

# setup scenario
from sidewalkbench.pg.utils.scene_random import SCENE_RNG_VERSION

pg_config = dict(
    seed=args_cli.seed,
    scene_rng_version=SCENE_RNG_VERSION,
    scenario_generation_method="limited async procedural generation",
    num_walkable_objects_per_terrain=env_config["num_static_obstacles"],
    num_non_walkable_objects_per_terrain=0,
    num_peds_per_terrain=env_config["num_peds_per_terrain"],
    simplified=False,
    use_rigid_dynamics=True,
    training=False,
    sidewalk_width=args_cli.sidewalk_width,
    scenario_type=env_config.get("scenario_type", "default"),
    terrain_segment=env_config.get("terrain_segment", "straight"),
    ped_lateral=env_config.get("ped_lateral", "center"),
    ped_crossing_direction=env_config.get("ped_crossing_direction", "left_to_right"),
    robot_bbox=robot_cfg["bbox"],
    robot_height_offset=robot_cfg["height_offset"],
    obstacle_id=args_cli.obstacle_id,
    use_behavior=bool(args_cli.use_behavior),
    terrain_type=env_config["terrain_type"],
    # Behavior efficiency: full ORCA + animation updates are restricted to
    # pedestrians inside the robot front-camera view (within behavior_radius);
    # the rest use goal steering. Pedestrians within behavior_proximity_radius
    # stay fully active regardless of the camera orientation.
    behavior_fov_cull=env_config.get("behavior_fov_cull", True),
    behavior_fov_margin=env_config.get("behavior_fov_margin", 0.15),
    behavior_radius=env_config.get("behavior_radius", 25.0),
    behavior_proximity_radius=env_config.get("behavior_proximity_radius", 5.0),
    # Pedestrian/ORCA update rate [Hz], decoupled from the policy step rate.
    behavior_update_hz=env_config.get("behavior_update_hz", 5.0),
    # Long-horizon route segmentation: length [m] of each goal segment and the
    # arc-length margin [m] at which the goal advances to the next segment.
    goal_segment_length=env_config.get("goal_segment_length", 30.0),
    goal_advance_margin=env_config.get("goal_advance_margin", 1.0),
    # Sample all texture pools on each env-0 reset unless fixed textures are requested.
    use_fixed_texture=args_cli.use_fixed_texture,
    terrain_texture_seed=args_cli.texture_seed,
)

## setup model
from sidewalkbench.models.controller_config import controller_from_config
from sidewalkbench.pg.inference_scripts.get_model import get_model

controller = controller_from_config(env_config, env_config["model_type"])
model = get_model(env_config["model_type"], controller)
if env_config["model_type"] == "CityWalker":
    model.configure_reset_history(env_config.get(
        "citywalker_reset_history_speed_mps", model._info.get("reset_history_speed_mps", 1.0)))
# Goal inputs are opt-in for each scenario.
pg_config["use_goal"] = env_config.get("use_goal", False)
model_input_shape = model._info["image_resolution"]
model_input_length = model._info["context_size"]
if env_config["model_type"] == "InternVLA_N1":
    model.set_instruction("Follow the sidewalk.")


def vis_requested(args) -> bool:
    """Whether any visualization output flag was requested on the CLI."""
    return bool(args.save_images or args.online_viewer or args.save_video or args.save_mcap)


@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    recovery: bool = False
    seed = args_cli.seed
    scene: SceneCfg = SceneCfg(pg_config=pg_config, num_envs=args_cli.num_envs)

    viewer = ViewerCfg()

    observations: ObservationsCfg = ObservationsCfg()
    if args_cli.kinematics_only:
        actions: COCOWaypointKinematicsActionsCfg = COCOWaypointKinematicsActionsCfg()
    else:
        actions = robot_actions_cfg

    rewards: EmptyRewardsCfg = EmptyRewardsCfg()

    terminations: TerminationsCfg = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.scene.pg_config["seed"] = self.seed
        self.scene.pg_config["long_horizon"] = self.recovery
        # Recovery is the long-horizon mode; explicit CLI choices take priority.
        self.scene.pg_config["use_behavior"] = (
            self.recovery if args_cli.use_behavior is None else args_cli.use_behavior
        )

        # The third-person camera is only used by the visualization utilities.
        # Drop it entirely for plain evaluation so it is never spawned or rendered.
        if not vis_requested(args_cli) and os.environ.get("SIDEWALKBENCH_KEEP_CAMERA2", "0") != "1":
            self.scene.camera2 = None

        # general settings
        self.sim.dt = 0.005
        if getattr(model, "control_dt", None) is not None:
            self.decimation = round(model.control_dt / self.sim.dt)
            self.sim.render_interval = self.decimation
        elif env_config["model_type"] == "Keyboard":
            self.decimation = 10
            self.sim.render_interval = 10
        elif env_config["model_type"] in ("NoMaD", "ViNT"):
            # NavDP goal-free models — all trained at 4 Hz observation rate.
            self.decimation = 50
            self.sim.render_interval = 50
        else:
            self.decimation = 40
            self.sim.render_interval = 40
        # Short-horizon episodes are capped at 60 s; long-horizon (recovery) episodes
        # override this to 3600 s below.
        self.episode_length_s = 60
        # simulation settings

        self.num_rerenders_on_reset = 5

        self.scene.camera.height = 288
        self.scene.camera.width = 512
        if args_cli.free_view:

            @configclass
            class FreeViewerCfg:
                """Configuration of the scene viewport camera."""

                eye: tuple[float, float, float] = (-20, -20, 7.2)
                lookat: tuple[float, float, float] = (20, 20, -5)
                cam_prim_path: str = "/OmniverseKit_Persp"
                resolution: tuple[int, int] = (1920, 1080)
                origin_type: str = "world"
                env_index: int = 0
                asset_name: str | None = None

            self.viewer = FreeViewerCfg()
        if args_cli.kinematics_only:
            if args_cli.robot_type == "coco":
                self.scene.robot = COCO_KINEMATICS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            elif args_cli.robot_type == "go2":
                self.scene.robot = GO2_KINEMATICS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            elif args_cli.robot_type == "m20":
                self.scene.robot = M20_KINEMATICS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        else:
            self.scene.robot = robot_asset_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.init_state.pos = (
            0,
            0,
            robot_cfg["height_offset"] + self.scene.terrain.terrain_generator.sidewalk_height,
        )
        self.scene.terrain.terrain_generator.sidewalk_width = pg_config["sidewalk_width"]
        # One terrain column per environment (rows hold the long-horizon rounds).
        self.scene.terrain.terrain_generator.num_cols = args_cli.num_envs
        self.scene.terrain.terrain_generator.sub_terrains = {
            env_config["terrain_type"]: self.scene.terrain.terrain_generator.sub_terrains[env_config["terrain_type"]]
        }  # only use one terrain type
        if env_config["terrain_type"] == "crosswalk":
            road_material_path = sorted(glob("./assets/materials/material_road_mdl/*.mdl"))
            self.scene.terrain.non_walkable_visual_materials = [
                sim_utils.MdlFileCfg(
                    mdl_path=np.random.choice(road_material_path), project_uvw=True, texture_scale=(1.0, 1.0)
                )
            ] + self.scene.terrain.non_walkable_visual_materials
        elif env_config["terrain_type"] == "intersection":
            self.scene.terrain.terrain_generator.size = (100.0, 100.0)
            road_material_path = sorted(glob("./assets/materials/material_road_mdl/*.mdl"))
            terrain_segment = env_config.get("terrain_segment", "straight")
            self.scene.terrain.terrain_generator.sub_terrains["intersection"].terrain_segment = terrain_segment
            self.scene.terrain.terrain_generator.sub_terrains["intersection"].scenario_type = pg_config["scenario_type"]
            self.scene.terrain.terrain_generator.sub_terrains["intersection"].use_ramp = terrain_segment == "crosswalk"
            if args_cli.use_fixed_texture:
                self.scene.terrain.non_walkable_visual_materials = [
                    sim_utils.MdlFileCfg(
                        mdl_path="assets/materials/material_road_mdl/Asphalt015_2K_PNG.mdl",
                        project_uvw=True,
                        texture_scale=(1.0, 1.0),
                    )
                ] + self.scene.terrain.non_walkable_visual_materials
            else:
                self.scene.terrain.non_walkable_visual_materials = [
                    sim_utils.MdlFileCfg(
                        mdl_path=np.random.choice(road_material_path), project_uvw=True, texture_scale=(1.0, 1.0)
                    )
                ] + self.scene.terrain.non_walkable_visual_materials
        if env_config["terrain_type"] == "block_sidewalk":
            self.scene.terrain.terrain_generator.size = (2000.0, 2000.0)
            self.scene.terrain.non_walkable_visual_materials = [
                sim_utils.MdlFileCfg(
                    mdl_path="assets/materials/material_terrain_mdl/Grass005_2K_PNG.mdl",
                    project_uvw=True,
                    texture_scale=(1.0, 1.0),
                ),
                sim_utils.MdlFileCfg(
                    mdl_path="assets/materials/material_road_mdl/Asphalt015_2K_PNG.mdl",
                    project_uvw=True,
                    texture_scale=(1.0, 1.0),
                ),
            ]

        elif env_config["terrain_type"] in ["square", "circular"]:
            self.scene.terrain.terrain_generator.size = (30.0, 30.0)
        # modify env
        self.observations.sensor.history_length = 21  # hardcoded
        # self.scene.terrain.terrain_generator.sidewalk_height = 0.15
        # del self.terminations.arrive
        # del self.terminations.out_of_region
        # del self.terminations.time_out
        # del self.rewards.arrive_reward
        # del self.rewards.out_of_region_penalty
        # del self.rewards.final_progress_reward
        self.scene.camera.prim_path = f"{{ENV_REGEX_NS}}/Robot/{robot_profile.camera_link}/front_cam"
        self.scene.camera.offset = CameraCfg.OffsetCfg(
            pos=robot_profile.camera_offset, rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
        )
        # The third-person camera follows the same body as the front camera.
        if self.scene.camera2 is not None:
            self.scene.camera2.prim_path = self.scene.camera.prim_path.replace("front_cam", "third_cam")

        self.scene.camera.spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=robot_cfg["camera_intrinsics"].flatten().tolist(),
            width=512,
            height=288,
        )
        # Optional free goal camera. Its world pose is set explicitly from the
        # current goal, so it has an identity offset and does not track a body.
        if vis_requested(args_cli) or (pg_config.get("use_goal", False) and model.goal_type == "image"):
            self.scene.goal_camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/GoalCam",
                update_period=0.005,
                height=288,
                width=512,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                    intrinsic_matrix=robot_cfg["camera_intrinsics"].flatten().tolist(),
                    width=512,
                    height=288,
                ),
                offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
                update_latest_camera_pose=False,
            )
        if self.recovery:
            self.terminations.time_out = DoneTerm(func=loc_mdp.time_out, time_out=True)
            self.terminations.arrive = DoneTerm(func=custom_rew.arrive, time_out=False)
            self.episode_length_s = 3600
            # Long-horizon row grid: one terrain row per success round. The route
            # advances to the next row after every env finishes the current row.
            self.scene.terrain.terrain_generator.num_rows = args_cli.max_resets
            for attr in ("collision_static", "collision_dynamic", "out_of_region", "freeze", "pedestrian_group_intrusion"):
                if hasattr(self.terminations, attr):
                    delattr(self.terminations, attr)


"""register env"""
gym.register(
    id="SidewalkBench-PG-Base",
    entry_point="sidewalkbench.pg.envs.abstract_env:AbstractEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BaseEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
)
