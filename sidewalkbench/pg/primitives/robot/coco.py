from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

# Robot instance
WHEEL_RADIUS = 0.11
WHEEL_BASE = 0.456
TRACK_WIDTH = 0.478687
MAX_STEER_RAD = 0.35
COCO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/robots/coco_one/coco_one_cocosim.usd",
        # usd_path=f"assets/robots/coco_one_cylwheel.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1e5,
            max_angular_velocity=1e5,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.27668),
        joint_pos={
            ".*wheel_joint*": 0.0,
            "base_to_front_axle_joint": 0.0,
        },
        joint_vel={
            ".*wheel_joint": 0.0,
            "base_to_front_axle_joint": 0.0,
        },
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel_joint"],
            effort_limit_sim=2.75,
            stiffness={
                ".*_wheel_joint": 0.0,
            },
            damping={".*_wheel_joint": 5.0},
        ),
        "axle": ImplicitActuatorCfg(
            joint_names_expr=["base_to_front_axle_joint"],
            # This value was measured from data by looking at the derivative of the steering angle sensor.
            velocity_limit_sim=1.75,
            # Stiffness was manually tuned in simulator
            stiffness=500.0,
            damping=0.0,
        ),
        "shock": ImplicitActuatorCfg(
            joint_names_expr=[".*shock_joint"],
            stiffness=50000.0,
            damping=25000.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

COCO_KINEMATICS_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/robots/coco_one/coco_one_rigid.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.27668),
    ),
)


GO2_KINEMATICS_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/robots/Go2/go2_rigid.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.44891),
    ),
)

M20_KINEMATICS_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/robots/M20_usd/M20_rigid.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5900668874382973),
    ),
)


def ackermann_wheel_speeds_vectorized(
    forward_speed: torch.Tensor,
    steering_angle: torch.Tensor,
    wheel_base: float,
    track_width: float,
    wheel_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert forward speed and steering to left/right wheel angular rates (Ackermann)."""

    R = wheel_base / torch.tan(steering_angle + 1e-8)
    R_left = R - track_width * 0.5
    R_right = R + track_width * 0.5
    v_left = forward_speed * (R_left / R)
    v_right = forward_speed * (R_right / R)
    w_left = v_left / wheel_radius
    w_right = v_right / wheel_radius
    return w_left, w_right


def set_wheel_velocity_vectorized(robot, joint_ids: Sequence[int], wheel_speeds: torch.Tensor) -> None:
    _, D = robot.num_instances, len(joint_ids)
    targets = wheel_speeds.repeat(D, 1).T
    robot.set_joint_velocity_target(targets, joint_ids=joint_ids)


def set_steering_vectorized(robot, joint_ids: Sequence[int], steering_angles: torch.Tensor) -> None:
    clipped_angles = torch.clamp(steering_angles, -MAX_STEER_RAD, MAX_STEER_RAD)
    _, D = robot.num_instances, len(joint_ids)
    targets = clipped_angles.repeat(D, 1).T
    robot.set_joint_position_target(targets, joint_ids=joint_ids)


def filter_joint_ids(joint_ids: Sequence[int], joint_names: Sequence[str], keyword: str) -> list[int]:
    return [jid for jid, name in zip(joint_ids, joint_names) if keyword in name]


class ClassicalCarWaypointAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: ClassicalCarWaypointActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ClassicalCarWaypointActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.vw = cfg.vw
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._axle_joint_ids, _ = self.robot.find_joints("base_to_front_axle_joint")
        # Joint ids are static; resolve them once instead of every substep.
        wheel_joint_ids, wheel_joint_names = self.robot.find_joints(".*wheel_joint")
        self._left_wheel_ids = filter_joint_ids(wheel_joint_ids, wheel_joint_names, "left")
        self._right_wheel_ids = filter_joint_ids(wheel_joint_ids, wheel_joint_names, "right")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # actions = actions / 10
        if self.vw:
            v, w = actions[:, 0], actions[:, 1]
            eps = 1e-6  # Small value to avoid division by zero
            turning = torch.abs(w) > eps
            R = torch.where(turning, v / torch.where(turning, w, torch.ones_like(w)), torch.full_like(v, 1e6))
            steering_angle = torch.atan(WHEEL_BASE / R)
            self._raw_actions[:, 0] = v
            self._raw_actions[:, 1] = steering_angle
        else:
            self._raw_actions = actions

    def apply_actions(self) -> None:
        left_wheel_speeds, right_wheel_speeds = ackermann_wheel_speeds_vectorized(
            self.raw_actions[:, 0],
            self.raw_actions[:, 1],
            WHEEL_BASE,
            TRACK_WIDTH,
            WHEEL_RADIUS,
        )

        set_wheel_velocity_vectorized(self.robot, self._left_wheel_ids, left_wheel_speeds)
        set_wheel_velocity_vectorized(self.robot, self._right_wheel_ids, right_wheel_speeds)
        set_steering_vectorized(self.robot, self._axle_joint_ids, self.raw_actions[:, 1])


@configclass
class ClassicalCarWaypointActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = ClassicalCarWaypointAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    action_dt: int = MISSING
    vw: bool = True


@configclass
class COCOWaypointActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = ClassicalCarWaypointActionCfg(asset_name="robot", action_dt=4)


class ClassicalCarWaypointKinematicsAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: ClassicalCarWaypointKinematicsActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ClassicalCarWaypointKinematicsActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: RigidObject = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # actions = actions / 10
        v, w = actions[:, 0], actions[:, 1]
        self._raw_actions[:, 0] = v
        self._raw_actions[:, 1] = w

    def apply_actions(self):
        """
        Given your _raw_actions (v, w) in robot frame, transform to world frame and apply.
        Robot frame: x-forward, z-up. World frame: x-forward, z-up.
        """
        v = self._raw_actions[:, 0]  # linear velocity (forward)
        w = self._raw_actions[:, 1]  # angular velocity (yaw rate)

        # Extract yaw from quaternion (x, y, z, w)
        quat = self.robot.data.root_quat_w  # (num_envs, 4)
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)  # (num_envs,)

        # Transform local forward velocity to world frame
        vx = v * torch.cos(yaw)
        vy = v * torch.sin(yaw)

        # Build world-frame velocity: [vx, vy, vz, wx, wy, wz]
        root_velocity = torch.zeros(self.num_envs, 6, device=self.device)
        root_velocity[:, 0] = vx
        root_velocity[:, 1] = vy
        root_velocity[:, 5] = w  # angular velocity around z-axis

        self.robot.write_root_velocity_to_sim(root_velocity)


@configclass
class ClassicalCarWaypointKinematicsActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = ClassicalCarWaypointKinematicsAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    action_dt: int = MISSING


@configclass
class COCOWaypointKinematicsActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = ClassicalCarWaypointKinematicsActionCfg(asset_name="robot", action_dt=4)
