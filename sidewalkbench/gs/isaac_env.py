"""sidewalkbench.gs Isaac integration: parallel GS route evaluation, PG-format output.

A ManagerBasedRLEnv where each env is a copy of one captured scene (the scene
collision mesh + one shared gsplat for observations); the scene's annotated
routes are evaluated IN ORDER, each exactly once (with N envs the next route in
order goes to whichever env is free), and route outcomes / metrics come from the
independent sbgs core.

Output (same format as the PG entry point random_env_pretrain.py), under
``experiments/<exp_name>/<model_type>_<scene>_gs_0_<seed>/``:
  results.pkl              EvalTracker schema (27 keys, joblib) -> sidewalkbench/gs/tools/collect_results.py
  evaluation_metrics.csv   coco-gs per-route CSV (extra)
  routes.jsonl             one line per route: outcome, failures (long horizon), video index, route type (extra)
  visualize_reset<k>.mp4   --save_video: PG composite per route k (every env)
  log_images/trial_<k>_<outcome>/   --save_images (observation/rgb/demo jpgs)
Like PG, a run is skipped when results.pkl already exists (``--overwrite`` to rerun).

Behavioral reference: coco-gs @73c1d32 (static_scene_nav mesh-route eval) for the
checks/metrics/spawn; the PG env (random_env_pretrain.py) for the per-model
model pipeline.
  * Model pipeline (PG): every per-model detail comes from sidewalkbench/gs/model_spec.py: the PG release
    zoo (sidewalkbench/models, run as PG release runs them): policy cadence (action held in between), the ``context_size`` most
    recent CONSECUTIVE frames prepared by PG's own ``prepare_model_input`` (resize to ``image_resolution``
    except for the wrappers that resize themselves; a model card's ``input_crop`` first), ``camera`` intrinsics, ``controller``, per-route model reset.
  * Checks (defaults aligned with PG where the rules coincide), run once per policy period at its end (PG's env
    step is one policy period; so are its tracker's samples of position, power and pedestrians): timeout 60 s short horizon /
    3600 s long horizon (PG episode_length_s); collision = summed contact force on non-wheel robot
    bodies >= 1 N (touching the scene mesh), or any robot contact with a cone (wheels included); freeze = XY displacement < 0.1 m over 5 s. Success and
    off-sidewalk follow the research spec (centerline projection within 0.1 m of
    the goal projection; robot within 1.0 m (= PG out_of_region buffer, --off_sidewalk_margin_m)
    of the nearest cell centre of the route's walkable region on the semantic grid,
    flood-filled from the route centerline -- sampler.SemanticBounds -- and within 2.5 m of the
    route centerline, sampler.OFF_ROUTE_M).
  * Metrics: route_completion = as PG: clip((final progress - start) / (goal - start), 0, 1), 1 on success.
  * Modes: each route is ``short_horizon``, ``long_horizon`` or ``cocoa`` (generated on a cocoa
    capture's trajectory, sidewalkbench/gs/annotate/import_cocoa.py; evaluated like short horizon); ``--mode`` picks which run.
    Long horizon follows PG's recovery eval (random_env_pretrain.py --recovery_eval): a
    collision / off-sidewalk / freeze is counted and the robot is teleported 1 m further
    along the centerline (facing along it; recurrent state + freeze window reset) instead
    of ending the route, which ends only at the goal or the timeout; results.pkl is PG's
    recovery summary (failures per 100 m, average velocity).
  * Pedestrians (sidewalkbench/gs/humans.py; per route, both modes; --no_humans turns them all off), each a
    baked avatar of its identity (gs_assets/pedestrians/<identity>/) at its spawn point on the
    route centerline, rendered into the policy's camera and the videos:
      * static  -- standing, facing its annotated yaw;
      * dynamic -- walking the route centerline back and forth (first toward the start), steered
        by RVO2 around the scene's walkability grid (gs_assets/<scene>/walkable.npz), the cones,
        the static pedestrians, each other and the robot.
    The robot touching one (its footprint rectangle vs the human's circle) is a collision; the
    robot's closest approach to any of them is logged per route.

Imports from the PG release include isaaclab, the shared robot registry, the
models (sidewalkbench.models via sidewalkbench.gs.model_spec) and PG's
visualization utilities (sidewalkbench.pg.utils.vis, for byte-identical video/image layout). The scene,
gsplat obs, checks and metrics are sbgs-owned.

Usage (or sidewalkbench/gs/run_gs_eval.sh <model_type> <scene> [args]):
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. python sidewalkbench/gs/isaac_env.py \
      --model_type FlowPilot --scene beverly_hills_crosswalk --viz none \
      [--num_envs 1 --seed 42 --exp_name gs --save_video --save_images]
  Diagnostics (env vars):
    SBGS_TRAJ_CSV=/path/traj.csv    per-step kinematics for all envs (pose, rpy, speed,
                                    wheel speed, command, contact) -- body speed should
                                    equal wheel speed * radius if the robot is rolling
    SBGS_BLACK_OBS=1                feed the policy black images: a blind baseline that
                                    must NOT complete routes if vision is what steers
"""

import argparse
import json
import math
import os
import sys
import traceback
import types
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from isaaclab.app import AppLauncher

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)  # the sidewalkbench package

parser = argparse.ArgumentParser()
from sidewalkbench.gs.config import add_variant_arguments, output_directory, resolve_arguments, variant_suffix  # noqa: E402

add_variant_arguments(parser)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments (copies of the scene).")
parser.add_argument("--max_steps", type=int, default=100000, help="Safety cap on env steps.")
parser.add_argument("--save_video", action="store_true", default=False,
                    help="Write visualize_reset<k>.mp4 (PG composite) for every route k (all envs).")
parser.add_argument("--save_images", action="store_true", default=False,
                    help="Write log_images/trial_<k>_<outcome>/ jpgs for every route k (all envs).")
parser.add_argument("--routes", type=str, default="",
                    help="Comma-separated route ids to run, in this order (default: every route of the scene in annotation order).")
parser.add_argument("--overwrite", action="store_true", default=False,
                    help="Rerun even if results.pkl already exists (PG skips such runs).")
parser.add_argument("--off_sidewalk_margin_m", type=float, default=1.0,
                    help="Off-sidewalk = robot farther than this from the route's walkable semantic-grid cells "
                         "(default 1.0 m = PG out_of_region buffer).")
parser.add_argument("--no_semantic_region", action="store_true", default=False,
                    help="No off-sidewalk check (the robot is always in bounds).")
parser.add_argument("--route_timeout_s", type=float, default=None,
                    help="Route timeout (default as PG: 60 s short horizon / cocoa, 3600 s long horizon).")
parser.add_argument("--collision_force_threshold_n", type=float, default=1.0,
                    help="Collision threshold for bodies other than support wheels/feet (default: 1 N).")
parser.add_argument("--no_humans", action="store_true", default=False,
                    help="run without the routes' pedestrians (static and dynamic).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
try:
    SETTINGS = resolve_arguments(args_cli)
except (ValueError, OSError) as exc:
    parser.error(str(exc))

SCENE_DIR = os.path.join(REPO_ROOT, "gs_assets", args_cli.scene)
ANNOTATION = os.path.join(SCENE_DIR, "annotation.json")
COLLISION_MESH = os.path.join(SCENE_DIR, "mesh.ply")
GSPLAT_PLY = os.path.join(SCENE_DIR, "gs.ply")
SEMANTIC_GRID = os.path.join(SCENE_DIR, "semantic_grid_0p1m.npz")

# model card (no Isaac deps): everything model-specific comes from here
from sidewalkbench.gs.model_spec import get_zoo_model, load_model_spec, runnable_reason  # noqa: E402

SPEC = load_model_spec(args_cli.model_type)
_why = runnable_reason(SPEC)
if _why:
    sys.exit(f"[sbgs] cannot run {args_cli.model_type}: {_why}")

# PG-style experiment folder: experiments/<exp_name>/<model>_<terrain_segment>_<scenario>_<num_peds>_<seed>
# with terrain_segment = scene, scenario_type = "gs", num_peds = 0.
env_config = {
    **SETTINGS,
    "seed": int(args_cli.seed),
    "model_type": args_cli.model_type,
    "terrain_type": args_cli.scene,
    "terrain_segment": args_cli.scene,
    "scenario_type": {"short_horizon": "gs", "long_horizon": "gs_long_horizon", "cocoa": "gs_cocoa"}[args_cli.mode],
    "ped_lateral": "center",
    "ped_crossing_direction": "left_to_right",
    "num_peds_per_terrain": 0,
    "num_static_obstacles": 0,
}
pg_config = {
    **SETTINGS,
    "scenario_generation_method": "gaussian splat scene",
    "num_walkable_objects_per_terrain": 0,
    "num_non_walkable_objects_per_terrain": 0,
    "num_peds_per_terrain": 0,
    "scenario_type": env_config["scenario_type"],
    "terrain_segment": args_cli.scene,
    "terrain_type": args_cli.scene,
    "scene": args_cli.scene,
}
EXP_FOLDER = str(output_directory(SETTINGS, REPO_ROOT))
os.makedirs(EXP_FOLDER, exist_ok=True)
if os.path.exists(os.path.join(EXP_FOLDER, "results.pkl")) and not args_cli.overwrite:
    print(f"[sbgs] results.pkl already exists in {EXP_FOLDER}; skipping (use --overwrite to rerun).", flush=True)
    sys.exit(0)

# --- launch Kit BEFORE isaaclab-dependent imports ---
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
import gymnasium as gym  # noqa: E402
import carb  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

np.random.seed(args_cli.seed)
torch.manual_seed(args_cli.seed)

# shared robot primitive + shared model loader
from sidewalkbench.pg.primitives.robot.registry import get_robot_configs, get_robot_profile  # noqa: E402
from sidewalkbench.models.controller_config import controller_from_config  # noqa: E402
from sidewalkbench.pg.inference_scripts.get_model import get_model  # noqa: E402
from sidewalkbench.pg.utils.eval import prepare_model_input  # noqa: E402

# sbgs core (independent)
from sidewalkbench.gs.annotation import CONE_ROUTE_RADIUS_M, load_scene_annotation  # noqa: E402
from sidewalkbench.gs.sampler import OFF_ROUTE_M, RouteQueue, RouteRuntime  # noqa: E402
from sidewalkbench.gs.collision_mesh import load_collision_mesh  # noqa: E402
from sidewalkbench.gs.checks import RouteEvaluator, CheckConfig, StepInput  # noqa: E402
from sidewalkbench.gs import checks as sbgs_checks  # noqa: E402
from sidewalkbench.gs.metrics import RouteMetricAccumulator, RouteResult, MetricsReport  # noqa: E402
from sidewalkbench.gs import obs as sbgs_obs  # noqa: E402
from sidewalkbench.gs.isaac_helpers import ConeSpawnerCfg, author_collision_mesh  # noqa: E402
from sidewalkbench.gs.pg_results import PGEpisodeAccumulator, PGResults  # noqa: E402
from sidewalkbench.gs.humans import HumanCrowd, merge_gaussians, touches_robot  # noqa: E402
from sidewalkbench.gs.cones import cone_gaussians  # noqa: E402
from sidewalkbench.gs.route_types import classify as route_type  # noqa: E402
from sidewalkbench.gs.goals import GSGoalManager  # noqa: E402

ROBOT_PROFILE = get_robot_profile(args_cli.robot_type)
ROBOT_CFG, ROBOT_ACTIONS = get_robot_configs(args_cli.robot_type)
ROBOT_CFG.spawn.activate_contact_sensors = True
ROBOT_USD = ROBOT_CFG.spawn.usd_path
if "://" not in ROBOT_USD and not os.path.isabs(ROBOT_USD):
    ROBOT_USD = os.path.join(REPO_ROOT, ROBOT_USD)
ROBOT_CFG.spawn.usd_path = ROBOT_USD
CAM_W, CAM_H = 512, 288
CAM_OFFSET_POS = ROBOT_PROFILE.camera_offset
# per-model camera intrinsics at the 512x288 render, from the model card (default = PG's 790 K scaled)
_K = np.array([[SPEC.camera["fx"], 0.0, SPEC.camera["cx"]], [0.0, SPEC.camera["fy"], SPEC.camera["cy"]], [0.0, 0.0, 1.0]])
if args_cli.robot_type != "coco":
    _K = ROBOT_PROFILE.as_dict()["camera_intrinsics"]

SPAWN_CLEARANCE_M = 0.02   # placed on the raycast surface + this
CONE_PARK_DEPTH_M = 1000.0  # a cone the current route does not use is parked this far below its settled pose
RECOVERY_AHEAD_M = 1.0     # long horizon: teleport this far along the centerline after a failure (PG look_ahead)
RECOVERABLE = (sbgs_checks.COLLISION, sbgs_checks.OFF_SIDEWALK, sbgs_checks.FREEZE)
HUMAN_COLLISION = "human_collision"  # long horizon: a collision with reason human_contact (PG dynamic collision)

# ------------------------------------------------------------------ mesh spawner
_COLLISION_MESH = None  # the scene's mesh.ply, parsed in main(); the spawner authors it once per env clone


def _spawn_collision_mesh(prim_path, cfg, translation=None, orientation=None):
    return author_collision_mesh(prim_path, _COLLISION_MESH)


@configclass
class CollisionMeshSpawnerCfg(sim_utils.SpawnerCfg):
    func = _spawn_collision_mesh


# ------------------------------------------------------------------ scene
@configclass
class GSSceneCfg(InteractiveSceneCfg):
    # NO ground plane: the scene collision mesh IS the ground. An Isaac
    # terrain_type="plane" is a solid half-space at z=0, but this scene's ground is
    # at z~-1.95 (below 0), so a z=0 plane buries the whole scene and pushes the
    # robot up out of it. env_origins come from InteractiveSceneCfg env_spacing.
    robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene_collision = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene", spawn=CollisionMeshSpawnerCfg()
    )
    # No Isaac RTX camera: observations are gsplat renders from the robot root state
    # (obs.get_gsplat_image) with the same intrinsics / mount as the reference's
    # front camera (CAM_OFFSET_POS, level "ros" optical frame), so an RTX sensor
    # would only render the invisible collision mesh at a per-env GPU cost.
    # all robot bodies, like the reference's "robot_contact"; non-wheel bodies are
    # summed for the collision check
    contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=0, update_period=0.0)


@configclass
class ObservationsCfg:
    @configclass
    class SensorCfg(ObsGroup):
        rgb = ObsTerm(func=sbgs_obs.get_gsplat_image)

    # history_length=1: the model-cadence frame history is kept by the eval loop.
    sensor: SensorCfg = SensorCfg(history_length=1, flatten_history_dim=False)


@configclass
class EventCfg:
    pass


@configclass
class GSEnvCfg(ManagerBasedRLEnvCfg):
    seed = args_cli.seed
    scene: GSSceneCfg = GSSceneCfg(num_envs=args_cli.num_envs, env_spacing=200.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions = ROBOT_ACTIONS
    terminations = None  # routes end by the sbgs checks
    events: EventCfg = EventCfg()
    rewards = None
    commands = None
    curriculum = None

    def __post_init__(self):
        self.sim.dt = 0.005
        self.decimation = 10  # 20 Hz env, as PG and the reference (0.01 x 5)
        self.episode_length_s = 100000.0  # route timeouts handled by sbgs checks
        self.scene.robot.init_state.pos = (0.0, 0.0, ROBOT_PROFILE.height_offset)


# ------------------------------------------------------------------ helpers
def _yaw_quat(yaw: float, device) -> torch.Tensor:
    """Pure-yaw root quaternion in the (x, y, z, w) order this Isaac Lab build uses.

    ``write_root_state_to_sim`` hands ``root_state[:, 3:7]`` straight to
    ``write_root_link_pose_to_sim_index`` (contract: x, y, z, w) and ``root_quat_w``
    is a ``wp.quatf`` (also x, y, z, w). Writing the wxyz form ``[cos, 0, 0, sin]``
    is read as x=cos, w=sin -> a ~90 deg ROLL: the robot spawns on its side, rests on
    two wheels, barrel-rolls when driven and never translates. This matches the
    reference's ``--quat_format xyzw`` default (``[0, 0, sin, cos]``).
    """
    return torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], device=device)


def _rpy_deg(qx, qy, qz, qw):
    """Roll/pitch/yaw in degrees from an (x, y, z, w) quaternion (see _yaw_quat)."""
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = max(-1.0, min(1.0, 2 * (qw * qy - qz * qx)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _reset_model_env(model, i: int) -> None:
    """Reset ONLY env i's recurrent state (a model_scripts model's per-env reset is done here directly, so it
    cannot touch the other envs' state). Shared model_scripts code is left untouched.
    """
    buf = getattr(model, "feature_buffer", None)
    if buf is not None and buf.shape[0] > i:
        buf[i] = 0.0


def _joint_power_w(robot) -> torch.Tensor | None:
    """sum_j |tau_j * qdot_j| per env (W), as EvalTracker's energy metric; None if
    the articulation exposes no applied torque (PG then counts 0 energy)."""
    try:
        torque = torch.as_tensor(robot.data.applied_torque)
        jvel = torch.as_tensor(robot.data.joint_vel)
        return torch.abs(torque * jvel).sum(dim=-1)
    except (AttributeError, NotImplementedError, TypeError, RuntimeError):
        return None


def _save_recovery(rec: dict, dt: float) -> None:
    """PG's RecoveryTracker.summarize: the long-horizon results.pkl (failures per 100 m, average velocity)."""
    import joblib

    res = {"successes": int(rec["successes"]), "static_collisions": rec[sbgs_checks.COLLISION],
           "dynamic_collisions": rec[HUMAN_COLLISION],
           "oob_count": rec[sbgs_checks.OFF_SIDEWALK], "freeze_count": rec[sbgs_checks.FREEZE],
           "total_steps": rec["total_steps"], "total_route_length": float(rec["total_route_length"])}
    per = 100.0 / res["total_route_length"] if res["total_route_length"] > 0 else 0.0
    vel = res["total_route_length"] / (res["total_steps"] * dt) if res["total_steps"] else 0.0
    print(f"[sbgs] long horizon: successes={res['successes']} route_length={res['total_route_length']:.1f} m "
          f"avg_velocity={vel:.2f} m/s | per 100 m: collisions={res['static_collisions'] * per:.2f} "
          f"human_collisions={res['dynamic_collisions'] * per:.2f} "
          f"off_sidewalk={res['oob_count'] * per:.2f} freezes={res['freeze_count'] * per:.2f}", flush=True)
    joblib.dump(res, os.path.join(EXP_FOLDER, "results.pkl"))


# ------------------------------------------------------------------ eval loop
@dataclass
class EnvRoute:
    """One env's current route and its bookkeeping (made by ``place``)."""

    k: int                          # the route's index in the run (= its video visualize_reset<k>.mp4)
    runtime: RouteRuntime
    evaluator: RouteEvaluator
    metrics: RouteMetricAccumulator
    start_step: int                 # the route clock starts after the warm-up, on the first command
    world_min: np.ndarray           # BEV-world extent for the recorder (route centerline bbox + 3 m)
    world_size: np.ndarray
    fails: dict                     # long horizon: failures per type on this route
    pg: PGEpisodeAccumulator = field(default_factory=PGEpisodeAccumulator)
    human_min: float = math.inf     # closest robot-human centre distance


def main():
    global _COLLISION_MESH
    ann = load_scene_annotation(ANNOTATION, scene_name=args_cli.scene)
    cmesh = load_collision_mesh(COLLISION_MESH)
    _COLLISION_MESH = cmesh.mesh  # the spawner authors this mesh
    routes = [r for r in ann.routes if r.mode == args_cli.mode]  # the scene's routes of this mode, in order
    if not routes:
        print(f"[sbgs] {args_cli.scene} has no {args_cli.mode} routes; nothing to run", flush=True)
        return
    if args_cli.routes:
        by_id = {r.route_id: r for r in routes}
        wanted = [r.strip() for r in args_cli.routes.split(",") if r.strip()]
        unknown = [r for r in wanted if r not in by_id]
        if unknown:
            sys.exit(f"[sbgs] --routes: unknown route id(s) {unknown}; scene has {sorted(by_id)}")
        routes = [by_id[r] for r in wanted]
        if any(r.mode != args_cli.mode for r in routes):
            sys.exit(f"[sbgs] --routes: {[r.route_id for r in routes if r.mode != args_cli.mode]} are not {args_cli.mode}")
        print(f"[sbgs] --routes: running {wanted}", flush=True)
    queue = RouteQueue(routes)
    NUM_ROUTES = len(queue)
    _no_region = [r.route_id for r in routes if r.region_xy is None]
    print(f"[sbgs] off-sidewalk: {len(routes) - len(_no_region)}/{len(routes)} route(s) against their annotated region"
          + (f"; WARNING {_no_region} have none (semantic grid + {OFF_ROUTE_M} m from the centerline)" if _no_region else ""),
          flush=True)
    sbgs_obs.configure_gsplat(GSPLAT_PLY, CAM_W, CAM_H, _K, offset_pos=CAM_OFFSET_POS,
                             depth_mesh=cmesh.mesh,
                             camera_link=ROBOT_PROFILE.camera_link if args_cli.robot_type == "g1" else None)
    route_cones = {r.route_id: ann.route_cones(r) for r in routes}  # the cones each route uses (within 3 m of it)

    zoo = SPEC.source == "pg_release"  # PG release wrapper (sidewalkbench/models): inference_vw(obs, robot), reset(env)
    controller = controller_from_config(SETTINGS, args_cli.model_type)
    if not zoo and (controller is not None or args_cli.use_goal):
        raise ValueError("Controller overrides and annotated goals require a shared BaseModel zoo wrapper.")
    model = get_zoo_model(args_cli.model_type, controller=controller) if zoo else get_model(args_cli.model_type)
    goal_type = getattr(model, "goal_type", None)
    if args_cli.use_goal and goal_type not in ("point", "image"):
        raise ValueError(f"{args_cli.model_type} does not accept point or image goals; omit --use_goal.")
    model_input_shape = tuple(SPEC.image_resolution)
    model_input_length = int(SPEC.context_size)
    hist_len = model_input_length  # PG: the context_size most recent consecutive frames
    use_pp = not zoo and SPEC.controller == "pure_pursuit" and hasattr(model, "inference_vw_pp")
    model_crop = SPEC.input_crop

    check_cfg = CheckConfig(
        off_sidewalk_margin_m=args_cli.off_sidewalk_margin_m,
        collision_force_threshold_n=args_cli.collision_force_threshold_n,
        route_timeout_s=args_cli.route_timeout_s if args_cli.route_timeout_s is not None
        else 3600.0 if args_cli.mode == "long_horizon" else 60.0,
    )
    long_horizon = args_cli.mode == "long_horizon"
    resolved = dict(SETTINGS, camera=ROBOT_PROFILE.camera_link, camera_offset=CAM_OFFSET_POS,
                    camera_intrinsics=_K.tolist(), goal_type=goal_type,
                    variant=variant_suffix(SETTINGS),
                    resolved_controller=asdict(model.controller_cfg) if zoo else SPEC.controller)
    env_config["resolved_controller"] = resolved["resolved_controller"]
    env_config["variant"] = resolved["variant"]
    pg_config["robot_type"] = args_cli.robot_type
    with open(os.path.join(EXP_FOLDER, "run_config.json"), "w") as f:
        json.dump(resolved, f, indent=2)

    env_cfg = GSEnvCfg()
    # cones: kinematic, at the settled pose from settle_cones.sh, each with a contact
    # sensor filtered against every robot body (PhysX wants one filter path per body), so any
    # robot-cone contact, wheels included, is a collision
    robot_usd = Usd.Stage.Open(ROBOT_USD)
    if robot_usd is None:
        raise FileNotFoundError(f"Could not load robot USD: {ROBOT_USD}")
    robot_root = robot_usd.GetDefaultPrim()
    robot_bodies = [str(p.GetPath().MakeRelativePath(robot_root.GetPath()))
                    for p in Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies())
                    if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    cone_spawn = ConeSpawnerCfg(rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                collision_props=sim_utils.CollisionPropertiesCfg(), activate_contact_sensors=True)
    for k, pose in enumerate(ann.cones):
        setattr(env_cfg.scene, f"cone_{k}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Cone_{k}", spawn=cone_spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pose[:3]), rot=tuple(pose[3:]))))
        setattr(env_cfg.scene, f"cone_contact_{k}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Cone_{k}", filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{b}" for b in robot_bodies],
            history_length=0, update_period=0.0))
    if len(ann.cones):
        print(f"[sbgs] {len(ann.cones)} cone(s) at settled poses; each route uses those within {CONE_ROUTE_RADIUS_M:g} m of its centerline "
              f"({', '.join(f'{k}: {int(v.sum())}' for k, v in route_cones.items())}): rendered, a collider "
              "(contact = collision) and a pedestrian obstacle; the others are parked out of the scene", flush=True)
    env = gym.make("SBGS-Eval", cfg=env_cfg, render_mode="rgb_array")
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    step_dt = env.unwrapped.step_dt
    robot = env.unwrapped.scene["robot"]
    origins = env.unwrapped.scene.env_origins  # (N, 3)

    crowd = None  # the routes' pedestrians, per env
    n_static, n_dynamic = (sum(len(getattr(r, f"{k}_pedestrians")) for r in routes) for k in ("static", "dynamic"))
    if not args_cli.no_humans and n_static + n_dynamic:
        walkable = os.path.join(SCENE_DIR, "walkable.npz") if n_dynamic else None
        if walkable and not os.path.isfile(walkable):
            sys.exit(f"[sbgs] dynamic pedestrians need {walkable}: run sidewalkbench/gs/annotate/bev_walkable.py "
                     f"{args_cli.scene} (or pass --no_humans)")
        crowd = HumanCrowd(walkable, num_envs, step_dt, cmesh.surface_z, ann.cones, routes, device=str(device),
                           robot_radius=ROBOT_PROFILE.pedestrian_radius)
        print(f"[sbgs] humans: {n_static} static + {n_dynamic} dynamic pedestrian(s) over {len(routes)} route(s) "
              "(RVO2 on walkable.npz + cones + robot), contact = collision", flush=True)
    # per env, the Gaussians added to the splat for its route: its cones (CONE_GAUSS, one set per cone) + pedestrians
    if len(ann.cones):
        g = cone_gaussians(ann.cones, device)
        n_g = len(g[0]) // len(ann.cones)
        CONE_GAUSS = [{k: t[c * n_g:(c + 1) * n_g] for k, t in zip(("means", "quats", "scales", "opacities", "colors"), g)}
                      for c in range(len(ann.cones))]
    env_cones = [None] * num_envs  # per env: its route's cones' Gaussians (merged), set in place()
    extras = types.SimpleNamespace(gaussians=lambda i: merge_gaussians(
        [env_cones[i]] + [crowd.gaussians(i) if crowd is not None else None]))
    sbgs_obs.set_extras(extras)

    env_hz = 1.0 / step_dt
    model_hz = float(args_cli.model_hz) if args_cli.model_hz else float(SPEC.input_fps)
    every_n = max(1, int(round(env_hz / model_hz)))
    _fed = prepare_model_input(torch.zeros(1, model_input_length, 3, CAM_H, CAM_W), model_input_length,
                               model_input_shape, args_cli.model_type).shape[-2:]  # what PG's prep hands the wrapper
    print(
        f"[sbgs] {args_cli.model_type}: env {env_hz:.1f} Hz; policy every {every_n} step(s) = {env_hz / every_n:.2f} Hz; "
        f"input = last {model_input_length} consecutive frames ({every_n / env_hz * 1000:.0f} ms apart), "
        f"crop={model_crop}, frames given to the wrapper {_fed[0]}x{_fed[1]}, "
        f"controller={model.controller_cfg.type if zoo else SPEC.controller}, "
        f"robot={args_cli.robot_type}, goal={goal_type if args_cli.use_goal else 'disabled'}, "
        f"recurrent={SPEC.recurrent}, fx={_K[0, 0]:.1f}",
        flush=True,
    )
    print(f"[sbgs] output -> {EXP_FOLDER}", flush=True)

    contact = env.unwrapped.scene["contact"]
    body_names = list(contact.body_names)
    collision_mask = ROBOT_PROFILE.collision_body_mask(body_names)
    body_mask = torch.tensor(collision_mask, device=device, dtype=torch.float32)
    print(f"[sbgs] collision force = sum over {[n for n, keep in zip(body_names, collision_mask) if keep]}", flush=True)
    cone_sensors = [env.unwrapped.scene[f"cone_contact_{k}"] for k in range(len(ann.cones))]

    carb.settings.get_settings().set("/app/runLoops/main/rateLimitEnabled", False)

    obs_dict, _ = env.reset(seed=args_cli.seed)

    recorders = None  # one per env; outputs are named by route index
    if args_cli.save_video or args_cli.save_images:
        from sidewalkbench.gs.mesh_render import robot_mesh as load_robot_mesh
        from sidewalkbench.gs.pg_vis import PGRecorder

        robot_mesh = load_robot_mesh(ROBOT_USD, robot.body_names, device)
        recorders = [PGRecorder(EXP_FOLDER, save_video=args_cli.save_video, save_images=args_cli.save_images,
                                obs_intrinsics=_K, robot_mesh=robot_mesh, robot_cfg=ROBOT_PROFILE.as_dict())
                     for _ in range(num_envs)]

    goals = None
    if args_cli.use_goal or recorders is not None:
        goals = GSGoalManager(num_envs, mode=args_cli.mode, ground_z=cmesh.surface_z,
                              camera_height=ROBOT_PROFILE.camera_height, intrinsics=_K,
                              segment_length=args_cli.goal_segment_length,
                              advance_margin=args_cli.goal_advance_margin,
                              capture_images=recorders is not None or goal_type == "image", device=device)

    # one line per finished route (routes.jsonl): outcome, failures (long horizon), video index, type -- tools/make_report.py
    crosswalk_lines = [np.asarray(c["xy"], dtype=float) for c in
                       json.load(open(os.path.join(SCENE_DIR, "centerlines.json")))["centerlines"] if c.get("crosswalk")]
    routes_log = open(os.path.join(EXP_FOLDER, "routes.jsonl"), "w")
    cur: list[EnvRoute] = [None] * num_envs  # per env: its current (or last) route
    warmup_until = [0] * num_envs  # per env: its route's warm-up runs until this step (its first command)
    active = [False] * num_envs
    hist_reset = [True] * num_envs
    start_frame = [None] * num_envs  # per env: the frame rendered right after its route's placement (PG's reset render)
    start_frame_pending = []         # envs placed since the last render; refresh_obs() takes their start_frame
    history = None  # (N, hist_len, 3, H, W) at model cadence
    held_action = torch.zeros(num_envs, 2, device=device)
    held_traj = None

    def warmup_end(p):
        """PG's episode start: a route placed at step p (pedestrians paused) gets one zero-action policy period
        (PG's warm-up env.step) before its first command, standing still, paused, unchecked and unclocked.
        The model runs for all envs on one shared cadence (every ``every_n`` steps), so the warm-up ends on
        the first policy step at least one period after p; routes end on a period's end, so p is a policy step
        and the warm-up is exactly one period (the rounding is a safeguard)."""
        return -(-(p + every_n) // every_n) * every_n

    report = MetricsReport(scene=ann.scene)
    pg_results = PGResults(num_envs, NUM_ROUTES)
    recovery = dict.fromkeys(RECOVERABLE + (HUMAN_COLLISION, "successes", "total_steps", "total_route_length"), 0)

    def teleport(i, x, y, yaw):
        """Env i's robot at rest at scene (x, y) on the collision surface (+ clearance), facing ``yaw``."""
        rs = robot.data.default_root_state[i].clone()
        rs[0], rs[1] = float(x) + float(origins[i, 0]), float(y) + float(origins[i, 1])
        rs[2] = float(cmesh.spawn_z(x, y, base_link_height_m=ROBOT_PROFILE.height_offset)) + SPAWN_CLEARANCE_M + origins[i, 2]
        rs[3:7] = _yaw_quat(yaw, device)
        rs[7:13] = 0.0
        robot.write_root_state_to_sim(rs.unsqueeze(0), env_ids=torch.tensor([i], device=device))

    def place(i, gstep):
        k = len(queue) - queue.remaining  # this route's index in the run
        if recorders is not None:
            recorders[i].start_episode(k)
        route = queue.next()
        teleport(i, *route.start_xy, route.start_yaw_rad)
        eid = torch.tensor([i], device=device)
        # canonical Isaac reset = root pose + default joint state, so a re-placed env
        # does not inherit the previous route's wheel spin / steer / suspension state.
        jp = robot.data.default_joint_pos[i].clone().unsqueeze(0)
        jv = robot.data.default_joint_vel[i].clone().unsqueeze(0)
        robot.write_joint_state_to_sim(jp, jv, env_ids=eid)
        robot.reset(eid)
        env.unwrapped.action_manager.reset(eid)
        on = route_cones[route.route_id]
        for c in range(len(ann.cones)):  # the route's cones at their settled pose, the others parked out of the scene
            pose = torch.tensor(ann.cones[c], dtype=torch.float32, device=device)
            pose[:3] += origins[i] + torch.tensor([0.0, 0.0, 0.0 if on[c] else -CONE_PARK_DEPTH_M], device=device)
            env.unwrapped.scene[f"cone_{c}"].write_root_link_pose_to_sim_index(root_pose=pose[None], env_ids=eid)
        env_cones[i] = merge_gaussians([CONE_GAUSS[c] for c in np.flatnonzero(on)]) if len(ann.cones) else None
        if crowd is not None:
            crowd.reset(i, route, on)
        runtime = RouteRuntime.from_route(
            route, None if args_cli.no_semantic_region else SEMANTIC_GRID,
            off_sidewalk_margin_m=check_cfg.off_sidewalk_margin_m,
            success_tolerance_m=check_cfg.success_tolerance_m,
        )
        if goals is not None:
            goals.reset(i, runtime)
        cl = route.centerline_xy
        warmup_until[i] = warmup_end(gstep)  # zero action, paused pedestrians, no checks until then
        cur[i] = EnvRoute(k=k, runtime=runtime, evaluator=RouteEvaluator(runtime, check_cfg),
                          metrics=RouteMetricAccumulator(), start_step=warmup_until[i],
                          world_min=cl.min(axis=0) - 3.0, world_size=(cl.max(axis=0) - cl.min(axis=0)) + 6.0,
                          fails=dict.fromkeys(RECOVERABLE + (HUMAN_COLLISION,), 0))
        held_action[i] = 0.0
        hist_reset[i] = True   # frames + recurrent state are reset when the warm-up ends
        start_frame_pending.append(i)
        active[i] = True

    def recover(i, x, y):
        """PG long-horizon recovery: put the robot 1 m further along the centerline (facing along it)."""
        rt = cur[i].runtime
        (nx, ny), yaw = rt.centerline.point_at(min(rt.robot_progress_m(x, y) + RECOVERY_AHEAD_M, rt.goal_progress_m))
        teleport(i, nx, ny, yaw)
        if args_cli.robot_type != "coco":
            eid = torch.tensor([i], device=device)
            robot.write_joint_state_to_sim(robot.data.default_joint_pos[i:i + 1],
                                           robot.data.default_joint_vel[i:i + 1], env_ids=eid)
            robot.reset(eid)
            env.unwrapped.action_manager.reset(eid)
        cur[i].pg.moved_to((nx, ny))  # the teleport is not driven distance
        cur[i].evaluator.reset_freeze()
        if zoo:
            model.reset(i)  # PG release: model.reset(env_id) before every recovery teleport
        elif SPEC.recurrent:
            _reset_model_env(model, i)

    def finish_route(i, oc, x, y, t, gstep, route_index):
        """Record env i's finished route: the CSV report, PG's results, routes.jsonl, the long-horizon totals."""
        r, rt = cur[i], cur[i].runtime
        met = r.metrics.finalize(outcome=oc.status, start_progress_m=rt.start_progress_m, goal_progress_m=rt.goal_progress_m,
                                 path_length_m=r.pg.path_length)
        report.add(RouteResult(scene=ann.scene, route_index=route_index, route_id=rt.route.route_id,
                               route_source="polygon_annotation", outcome=oc.status, reason=oc.reason, metrics=met))
        pg_results.add_episode(
            i,
            success=oc.status == sbgs_checks.SUCCESS,
            collision_static=oc.status == sbgs_checks.COLLISION,
            oob=oc.status == sbgs_checks.OFF_SIDEWALK,
            timeout=oc.status == sbgs_checks.TIMEOUT,
            freeze=oc.status == sbgs_checks.FREEZE,
            path_length=r.pg.path_length,
            directed_progress_m=rt.robot_progress_m(x, y) - rt.start_progress_m,
            route_length_m=rt.goal_progress_m - rt.start_progress_m,
            episode_time_s=t,
            energy_j=r.pg.energy_j,
        )
        if long_horizon:  # PG RecoveryTracker: steps until success, length of successes
            for kind in r.fails:
                recovery[kind] += r.fails[kind]
            recovery["successes"] += oc.status == sbgs_checks.SUCCESS
            # PG RecoveryTracker counts every step (policy period) of the route except the one that reached the
            # goal; kept in env steps (x step_dt in _save_recovery / make_tables --dt)
            recovery["total_steps"] += (gstep + 1 - r.start_step) - every_n * (oc.status == sbgs_checks.SUCCESS)
            recovery["total_route_length"] += (rt.goal_progress_m - rt.start_progress_m) * (oc.status == sbgs_checks.SUCCESS)
        print(f"[sbgs] step {gstep} env {i} route {rt.route.route_id} -> {oc.status} "
              f"(t={t:.1f}s completion={met['route_completion']:.2f}"
              + (f" failures={dict(r.fails)}" if long_horizon else "")
              + (f" closest_human={r.human_min:.2f}m" if crowd is not None else "") + ")", flush=True)
        routes_log.write(json.dumps({
            "scene": ann.scene, "route_id": rt.route.route_id, "mode": args_cli.mode, "video_index": r.k,
            "type": route_type({"centerline_xy": rt.route.centerline_xy}, crosswalk_lines)[0],
            "outcome": oc.status, "reason": oc.reason, "completion": met["route_completion"], "spl": round(met["spl"], 4),
            "path_length_m": round(r.pg.path_length, 2),
            "time_s": round(t, 2), "avg_speed_mps": met["average_speed_mps"],
            "length_m": round(rt.goal_progress_m - rt.start_progress_m, 2),
            "cones": int(route_cones[rt.route.route_id].sum()),
            "failures": dict(r.fails) if long_horizon else None,
            "closest_human_m": round(r.human_min, 2) if math.isfinite(r.human_min) else None}) + "\n")
        routes_log.flush()
        if recorders is not None:
            recorders[i].episode_end(oc.status)

    def refresh_obs():
        # The obs in obs_dict was rendered by the last env.step/env.reset, i.e. from the pose
        # BEFORE place() teleported the robot. Re-render from the new pose so neither the
        # policy's frame history nor the video's first frame shows the previous route.
        obs = env.unwrapped.observation_manager.compute(update_history=True)
        for i in start_frame_pending:  # the routes placed since the last render: their start frame (PG's reset render)
            start_frame[i] = obs["sensor"][i, -1].to(device).clone()
        start_frame_pending.clear()
        return obs

    for i in range(num_envs):
        if queue.remaining:
            place(i, 0)
    obs_dict = refresh_obs()

    routes_done = 0
    _traj_path = os.environ.get("SBGS_TRAJ_CSV")
    _traj = open(_traj_path, "w") if _traj_path else None
    if _traj:
        _traj.write("gstep,env,route_id,t,x,y,z,roll_deg,pitch_deg,yaw_deg,speed,wheel_speed,action_v,action_w,contact_f\n")
    _wj_ids = robot.find_joints(ROBOT_PROFILE.wheel_joint_pattern)[0] if ROBOT_PROFILE.wheel_joint_pattern else []
    _blind = bool(os.environ.get("SBGS_BLACK_OBS"))

    for gstep in range(args_cli.max_steps):
        frame = obs_dict["sensor"][:, -1].to(device)  # (N, 3, H, W): newest render

        # ---- policy at model cadence, action held in between ----
        if gstep % every_n == 0:
            camera_positions = sbgs_obs.camera_poses(env.unwrapped)[0] if goals is not None else None
            if history is None:
                history = frame.unsqueeze(1).repeat(1, hist_len, 1, 1, 1)
            else:
                history = torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)
            for i in range(num_envs):
                if hist_reset[i] and active[i] and gstep >= warmup_until[i]:  # (an env never given a route stays idle)
                    # Route start = PG episode start: PG's reset render fills its whole frame history and
                    # the warm-up step appends one frame, so the first input is [R, .., R, F0] (R: the
                    # render right after placement, F0: now) with zero recurrent state. Nothing from the
                    # warm-up-time calls or the previous route survives.
                    history[i] = torch.cat([start_frame[i].unsqueeze(0).repeat(hist_len - 1, 1, 1, 1),
                                            frame[i].unsqueeze(0)])
                    start_frame[i] = None
                    if zoo:
                        model.reset(i)  # PG release: model.reset(env_id) at every episode start
                    elif SPEC.recurrent:
                        _reset_model_env(model, i)
                    if goals is not None:
                        xy = (robot.data.root_pos_w[i, :2] - origins[i, :2]).cpu().numpy()
                        goals.set_camera_height(i, float(camera_positions[i, 2]) - cmesh.surface_z(*xy))
                    hist_reset[i] = False
            with torch.no_grad():
                if model_crop is not None:  # a model card's crop; no PG release model has one
                    y0, y1, x0, x1 = model_crop
                    history_in = history[:, :, :, y0:y1, x0:x1]
                else:
                    history_in = history
                # PG's own input preparation: the context_size most recent frames, resized to image_resolution
                # except for the models whose wrappers resize themselves
                om = prepare_model_input(history_in, model_input_length, model_input_shape, args_cli.model_type)
                if _blind:
                    om = torch.zeros_like(om)
                goal_kwargs = {}
                if goals is not None:
                    local_xy = robot.data.root_pos_w[:, :2] - origins[:, :2]
                    goals.update(local_xy.cpu().numpy())
                    goals.render_pending(sbgs_obs.get_renderer(), extras=lambda i: env_cones[i])
                    if args_cli.use_goal:
                        if goal_type == "point":
                            goal_kwargs["goal_xy"] = goals.goal_xy_local(local_xy, robot.data.root_quat_w)
                        else:
                            goal_kwargs["goal_image"] = goals.images
                if zoo:
                    out = model.inference_vw(om, robot=robot, **goal_kwargs)
                else:
                    out = model.inference_vw_pp(om, robot) if use_pp else model.inference_vw(om)
                act, held_traj = out if isinstance(out, tuple) else (out, None)
            held_action = act.to(device).clone()
        action = held_action.clone()
        for i in range(num_envs):
            if (not active[i]) or gstep < warmup_until[i] or hist_reset[i]:
                # idle, warming up, or the route-start reset + fresh inference has not run yet
                # (the held command was inferred from warm-up frames; PG's first command is always fresh)
                action[i] = 0.0

        # PG records the obs the policy saw + its action BEFORE stepping
        for i in (range(num_envs) if recorders is not None else ()):
            if active[i]:
                recorders[i].frame(
                    renderer=sbgs_obs.get_renderer(), env_origin=origins[i], pos_w=robot.data.root_pos_w[i],
                    quat_xyzw=robot.data.root_quat_w[i], body_pos_w=robot.data.body_link_pos_w[i],
                    body_quat_xyzw=robot.data.body_link_quat_w[i],
                    obs_rgb=frame[i], action=action[i:i + 1],
                    traj=None if held_traj is None else held_traj[i:i + 1],
                    world_min=cur[i].world_min, world_size=cur[i].world_size,
                    human=extras.gaussians(i),
                    goal_world=None if goals is None else goals.xy[i],
                    goal_yaw=None if goals is None else goals.yaw[i],
                    goal_image=None if goals is None or goals.images is None else goals.images[i],
                )

        if crowd is not None:  # pedestrians react to the robot's current state, then the step renders them
            # a warming-up env's pedestrians are paused (PG pause_pedestrians): their first tick is its first command
            crowd.step((robot.data.root_pos_w[:, :2] - origins[:, :2]).cpu().numpy(),
                       robot.data.root_vel_w[:, :2].cpu().numpy(),
                       paused=[active[i] and gstep < warmup_until[i] for i in range(num_envs)])
        obs_dict, _, _, _, _ = env.step(action)

        pos = robot.data.root_pos_w[:, :2] - origins[:, :2]
        speed = torch.linalg.norm(robot.data.root_vel_w[:, :2], dim=1)
        body_f = torch.linalg.norm(contact.data.net_forces_w, dim=-1)  # (N, B)
        forces = (body_f * body_mask).sum(dim=1)                        # (N,) non-support bodies
        cone_touch = torch.zeros(num_envs, dtype=torch.bool, device=device)  # any robot body on any cone
        for sensor in cone_sensors:
            cone_touch |= torch.linalg.norm(torch.as_tensor(sensor.data.force_matrix_w), dim=-1).flatten(1).amax(1) > 0
        power = _joint_power_w(robot)

        placed = False
        for i in range(num_envs):
            if active[i] and gstep == warmup_until[i] - 1:  # the warm-up's last step: path measured from here
                cur[i].pg.reset((float(pos[i, 0]), float(pos[i, 1])))  # (PG tracker.init_pos after its warm-up)
            if not active[i] or gstep < warmup_until[i]:
                continue
            r = cur[i]
            x, y = float(pos[i, 0]), float(pos[i, 1])
            roll_deg, pitch_deg, yaw_deg = _rpy_deg(*robot.data.root_quat_w[i].tolist())  # xyzw
            if gstep == warmup_until[i]:
                print(
                    f"[sbgs] env{i} {r.runtime.route.route_id} first command at step {gstep}: "
                    f"z={float(robot.data.root_pos_w[i, 2]):.2f} "
                    f"(spawn_z={cmesh.spawn_z(*r.runtime.route.start_xy, base_link_height_m=ROBOT_PROFILE.height_offset):.2f}) "
                    f"rpy_deg=({roll_deg:.0f},{pitch_deg:.0f},{yaw_deg:.0f}) "
                    f"start_yaw_deg={math.degrees(r.runtime.route.start_yaw_rad):.0f}",
                    flush=True,
                )
            f = float(forces[i])
            if _traj:  # every step (diagnostic)
                z = float(robot.data.root_pos_w[i, 2])
                wheel_speed = (float(robot.data.joint_vel[i, _wj_ids].mean()) * ROBOT_PROFILE.wheel_radius
                               * ROBOT_PROFILE.wheel_direction) if _wj_ids else float("nan")
                _traj.write(
                    f"{gstep},{i},{r.runtime.route.route_id},{(gstep - r.start_step) * step_dt:.3f},{x:.3f},{y:.3f},{z:.3f},"
                    f"{roll_deg:.2f},{pitch_deg:.2f},{yaw_deg:.1f},"
                    f"{float(speed[i]):.3f},{wheel_speed:.3f},{float(action[i, 0]):.3f},{float(action[i, 1]):.3f},{f:.2f}\n"
                )
            # PG's env step is one policy period: it checks the route (and its tracker samples position,
            # power, pedestrians) once per period, at the period's end. Routes start on a policy step, so
            # a period ends right before the next one and a recovery or a new route starts on a policy step.
            steps_done = gstep + 1 - r.start_step  # steps since the route's first command
            if steps_done % every_n:
                continue
            t = (steps_done - every_n) * step_dt  # PG's episode time: (periods - 1) x period
            human_touch = False
            people = crowd.xy(i) if crowd is not None else []
            if people:
                yaw = math.radians(yaw_deg)
                r.human_min = min(r.human_min, *(math.hypot(hx - x, hy - y) for hx, hy in people))
                human_touch = any(touches_robot(p, (x, y), yaw, ROBOT_PROFILE.bbox[:2]) for p in people)
            r.metrics.update(r.runtime.robot_progress_m(x, y), float(speed[i]))
            r.pg.step((x, y), float(power[i]) if power is not None else 0.0, every_n * step_dt)
            oc = r.evaluator.step(StepInput(x=x, y=y, sim_time_s=t, collision_force_n=f,
                                            cone_contact=bool(cone_touch[i]), human_contact=human_touch))
            if oc is None:
                continue
            if long_horizon and oc.status in RECOVERABLE:
                kind = HUMAN_COLLISION if oc.reason == sbgs_checks.HUMAN_CONTACT else oc.status
                r.fails[kind] += 1
                print(f"[sbgs] step {gstep} env {i} route {r.runtime.route.route_id} {oc.status} ({oc.reason}) "
                      f"#{r.fails[kind]} at t={t:.1f}s -> recover 1 m ahead", flush=True)
                recover(i, x, y)
                placed = True  # re-render from the new pose
                continue
            finish_route(i, oc, x, y, t, gstep, routes_done)
            routes_done += 1
            if queue.remaining:
                # step `gstep` has already run: the warm-up starts at step gstep+1, as the first
                # placement's starts at step 0
                place(i, gstep + 1)
                placed = True
            else:
                active[i] = False
        if placed:
            obs_dict = refresh_obs()
        if routes_done >= NUM_ROUTES:
            break

    unfinished = [f"env{i}:{cur[i].runtime.route.route_id}" for i in range(num_envs) if active[i]]
    if unfinished:
        print(f"[sbgs] WARNING: --max_steps reached with routes still in flight (not scored): {unfinished}", flush=True)
    if _traj:
        _traj.close()
        print(f"[sbgs] trajectory log -> {_traj_path}", flush=True)
    for rec in recorders or ():
        rec.close()
    csv_path = os.path.join(EXP_FOLDER, "evaluation_metrics.csv")
    report.write_csv(csv_path)
    print("[sbgs] " + report.summary_line(requested=NUM_ROUTES), flush=True)
    print(f"[sbgs] metrics -> {csv_path}", flush=True)
    if long_horizon:
        _save_recovery(recovery, step_dt)
    else:
        pg_results.save(os.path.join(EXP_FOLDER, "results.pkl"), env_config, pg_config)
    env.close()


gym.register(
    id="SBGS-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": GSEnvCfg},
)

if __name__ == "__main__":
    # Kit's fast shutdown (atexit) exits the process with code 0 even after an uncaught
    # exception, so batch drivers would count a crash as success; exit 1 explicitly.
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if code:
            print(e, file=sys.stderr, flush=True)
        simulation_app.close(exit_code=code)
        raise
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        simulation_app.close(exit_code=1)
        os._exit(1)
    simulation_app.close()
