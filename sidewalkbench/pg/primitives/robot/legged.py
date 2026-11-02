"""Legged embodiments driven by the benchmark's ``(v, w)`` action.

Go2 and G1 track ``(v, w)`` with a learned flat-ground velocity-tracking policy at 50 Hz.
M20 is wheel-legged: its legs hold a fixed stance while it skid-steers on its wheels.
"""

from __future__ import annotations

import os
from dataclasses import MISSING

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab_tasks.contrib.navigation.mdp.pre_trained_policy_action import PreTrainedPolicyAction
from isaaclab_tasks.contrib.navigation.mdp.pre_trained_policy_action_cfg import PreTrainedPolicyActionCfg
from sidewalkbench.pg.primitives.robot.coco import set_wheel_velocity_vectorized

M20_WHEEL_RADIUS = 0.09
# Physical track 0.453 m times a wheel-separation multiplier: lateral skid resistance makes
# skid steering reach only about 40% of the kinematic yaw rate (0.97 rad/s in place with this factor).
M20_TRACK_WIDTH = 2.5 * 0.453152
M20_MAX_ACCEL = 1.0  # m/s^2; the tucked stance pitches over under step speed commands
M20_CFG = ArticulationCfg(
    # Absolute path: Kit cannot resolve the relative payloads inside M20.usd from a relative reference.
    spawn=sim_utils.UsdFileCfg(usd_path=os.path.abspath("assets/robots/M20_usd/M20.usd")),
    # Low Z-fold stance (thighs swept back, calves swept forward) that puts the wheels 0.09 m inboard of the hips:
    # the wheelbase (0.44 m) must stay shorter than the track (0.45 m) for skid steering to overcome the wheels'
    # lateral friction, while a base this long keeps a stalled drive (e.g. against a curb) from flipping the robot.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            "f[lr]_hipy_joint": -1.0,
            "f[lr]_knee_joint": 1.484,
            "h[lr]_hipy_joint": -0.484,
            "h[lr]_knee_joint": 1.484,
        },
    ),
    actuators={
        # Stiff enough that wheel skid forces do not rock the held stance.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip[xy]_joint", ".*_knee_joint"],
            effort_limit_sim=200.0,
            stiffness=2000.0,
            damping=50.0,
        ),
        # Stiff velocity drive: skid steering has to break the wheels' lateral grip. The torque cap keeps the
        # four stalled wheels (48 N m) below the moment that would tip the 34 kg robot over its rear axle.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"], effort_limit_sim=12.0, stiffness=0.0, damping=100.0
        ),
    },
)


@configclass
class LocomotionObsCfg(ObsGroup):
    """Inputs of the IsaacLab flat-ground locomotion policies, in training order."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    # Replaced by the action term with the (vx, vy, wz) command.
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    # Replaced by the action term with the policy's previous output.
    actions = ObsTerm(func=mdp.last_action)


class VWLocomotionPolicyAction(PreTrainedPolicyAction):
    """Tracks a ``(v, w)`` action with a learned ``(vx, vy, wz)`` velocity-tracking policy."""

    cfg: VWLocomotionPolicyActionCfg

    def __init__(self, cfg: VWLocomotionPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        # The base class feeds this buffer to the policy as its (vx, vy, wz) command.
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        return 2

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:, 0] = actions[:, 0].clamp(self.cfg.min_lin_vel, self.cfg.max_lin_vel)
        self._raw_actions[:, 2] = actions[:, 1].clamp(-self.cfg.max_ang_vel, self.cfg.max_ang_vel)

    def apply_actions(self):
        # The policy output must not carry autograd state into the simulation buffers.
        with torch.inference_mode():
            super().apply_actions()

    def reset(self, env_ids=None) -> None:
        """Clear only the re-placed robots' locomotion command and previous action."""
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self.low_level_actions[ids] = 0.0
        self._low_level_action_term.reset(env_ids)
        self._low_level_action_term.process_actions(self.low_level_actions)
        self._low_level_obs_manager.reset(env_ids)


@configclass
class VWLocomotionPolicyActionCfg(PreTrainedPolicyActionCfg):
    class_type: type[ActionTerm] = VWLocomotionPolicyAction
    low_level_observations: ObsGroup = LocomotionObsCfg()
    # Command range [m/s], [rad/s] the policy was trained on.
    min_lin_vel: float = -1.0
    max_lin_vel: float = 1.0
    max_ang_vel: float = 1.0
    debug_vis: bool = False


# Policies trained with IsaacLab's flat velocity tasks plus stance regularizers on PhysX (see README "Embodiments").
@configclass
class GO2ActionsCfg:
    locomotion = VWLocomotionPolicyActionCfg(
        asset_name="robot",
        policy_path="assets/robots/Go2/policy_flat.pt",
        low_level_actions=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
        ),
    )


@configclass
class G1ActionsCfg:
    locomotion = VWLocomotionPolicyActionCfg(
        asset_name="robot",
        policy_path="assets/robots/G1/policy_flat.pt",
        low_level_actions=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        ),
        min_lin_vel=0.0,  # trained on forward commands only
    )


class M20SkidSteerAction(ActionTerm):
    """Converts a ``(v, w)`` action into left/right wheel velocity targets."""

    cfg: M20SkidSteerActionCfg

    def __init__(self, cfg: M20SkidSteerActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._speed = torch.zeros(self.num_envs, device=self.device)
        self._left_wheel_ids, _ = self.robot.find_joints("[fh]l_wheel_joint")
        self._right_wheel_ids, _ = self.robot.find_joints("[fh]r_wheel_joint")

    def reset(self, env_ids=None) -> None:
        self._speed[slice(None) if env_ids is None else env_ids] = 0.0

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        max_dv = M20_MAX_ACCEL * self._env.physics_dt
        self._speed += (self._raw_actions[:, 0] - self._speed).clamp(-max_dv, max_dv)
        w = self._raw_actions[:, 1]
        # The wheel joints spin about -y, so forward rolling is a negative joint velocity.
        left = -(self._speed - 0.5 * M20_TRACK_WIDTH * w) / M20_WHEEL_RADIUS
        right = -(self._speed + 0.5 * M20_TRACK_WIDTH * w) / M20_WHEEL_RADIUS
        set_wheel_velocity_vectorized(self.robot, self._left_wheel_ids, left)
        set_wheel_velocity_vectorized(self.robot, self._right_wheel_ids, right)


@configclass
class M20SkidSteerActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = M20SkidSteerAction
    asset_name: str = MISSING


@configclass
class M20ActionsCfg:
    wheels = M20SkidSteerActionCfg(asset_name="robot")
