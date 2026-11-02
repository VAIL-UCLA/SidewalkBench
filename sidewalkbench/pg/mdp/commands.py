from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera
from sidewalkbench.pg.utils.math import _yaw_to_quat


def begin_scene_reset(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Select the random streams before any reset event samples the scene."""
    env.scene.begin_reset(env_ids)


def reset_obj(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    env.scene.resample(env_ids)


def get_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    type: str = "RGB",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Wrapper to choose between rgb and rgbd processing based on params."""
    if type == "RGBD":
        return rgbd_processed(
            env,
            sensor_cfg=sensor_cfg,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=normalize,
        )
    if type == "LIDAR_1D":
        return lidar_1d_processed(
            env,
            sensor_cfg=sensor_cfg,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=normalize,
        )
    return rgb_processed(
        env,
        sensor_cfg=sensor_cfg,
        convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
        normalize=normalize,
    )


def lidar_1d_processed(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    depth = sensor.data.output["distance_to_camera"]

    # depth image conversion
    if convert_perspective_to_orthogonal:
        depth = math_utils.orthogonalize_perspective_depth(depth, sensor.data.intrinsic_matrices)

    depth[depth == float("inf")] = 5
    depth = depth.clamp(0.0, 5.0) / 5.0
    depth = depth[:, : depth.shape[1] // 2, ::16, 0].min(dim=1)[0]

    return depth.clone()


def rgbd_processed(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output["rgb"]
    depth = sensor.data.output["distance_to_camera"]

    # depth image conversion
    if convert_perspective_to_orthogonal:
        depth = math_utils.orthogonalize_perspective_depth(depth, sensor.data.intrinsic_matrices)

    # rgb/depth image normalization
    if normalize:
        images = images.float() / 255.0
        images = images.permute(0, 3, 1, 2)
        B, C, H, W = images.shape
        depth[depth == float("inf")] = 0
        depth = depth.reshape(B, 1, H, W).clamp(0.0, 20.0) / 20.0

    return torch.cat([images, depth], dim=1).clone()


def rgb_processed(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output["rgb"]
    images = images.float() / 255.0
    if env.scene.clone_dynamic_trajectories is not None:
        human_image, human_mask, human_depth = env.scene.render_human(return_depth=True)
        depth = sensor.data.output["distance_to_camera"]
        human_mask = torch.logical_and(human_mask, human_depth < depth)
        images = human_image * human_mask + images * (
            ~human_mask
        )  # use rasterized image for foreground and original image for background

    # depth = sensor.data.output['distance_to_camera']
    # rgb/depth image normalization
    images = images.permute(0, 3, 1, 2)
    B, C, H, W = images.shape
    # depth[depth == float("inf")] = 0
    # depth  =  depth.reshape(B, 1, H, W).clamp(0., 20.) / 20.
    return images


def reset_start_goal(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # get default root pose and velocity (reading default_root_state is deprecated)
    default_pose = asset.data.default_root_pose[env_ids].clone()
    positions = default_pose[:, 0:3]
    orientations = default_pose[:, 3:7]
    velocities = asset.data.default_root_vel[env_ids].clone()

    # set into the physics simulation
    for i, env_id in enumerate(env_ids):
        row, col = env.scene.terrain.terrain_levels[env_id], env.scene.terrain.terrain_types[env_id]
        sampler = env.scene.terrain.terrain_start_goal_samplers[row][col]
        result = sampler(rng=env.scene.randomization.numpy(int(row), int(col), "start_goal"))
        if len(result) == 4:
            start_pos, goal_pos, route_length, projector = result
            env.scene.terrain.terrain_centerline_projectors[row][col] = projector
        else:
            start_pos, goal_pos, route_length = result
            projector = env.scene.terrain.terrain_centerline_projectors[row][col]
        start_pos = torch.tensor(start_pos, dtype=positions.dtype, device=positions.device)
        goal_pos = torch.tensor(goal_pos, dtype=positions.dtype, device=positions.device)
        start_pos[:2] += env.scene.env_origins[env_id][:2]
        goal_pos[:2] += env.scene.env_origins[env_id][:2]
        positions[i, 0:2] = start_pos[:2]
        positions[i, 2] = env.scene.terrain.cfg.terrain_generator.sidewalk_height + env.scene.robot_height_offset
        env.scene.env_goals[env_id] = goal_pos[:2]
        env.scene.env_goal_yaws[env_id] = goal_pos[2] if goal_pos.numel() >= 3 else float("nan")
        env.scene.route_lengths[env_id] = route_length
        # Set arc-length goal for arrive check
        local_start = start_pos[:2] - env.scene.env_origins[env_id][:2]
        local_goal = goal_pos[:2] - env.scene.env_origins[env_id][:2]
        s_start, _ = projector(local_start[0].item(), local_start[1].item())
        s_goal, _ = projector(local_goal[0].item(), local_goal[1].item())
        env.scene.last_arc_length[env_id] = s_start
        env.scene.arc_length_goal[env_id] = s_goal
        env.scene.arc_length_start[env_id] = s_start
        orientations[i] = _yaw_to_quat(start_pos[2])

    velocities = torch.zeros_like(velocities)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    # Clear physical joint state as well as body velocity. Otherwise spinning
    # wheels from the previous episode can move the robot after the teleport.
    if hasattr(asset, "write_joint_state_to_sim"):
        joint_pos = asset.data.default_joint_pos[env_ids].clone()
        joint_vel = asset.data.default_joint_vel[env_ids].clone()
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        asset.set_joint_position_target(joint_pos, env_ids=env_ids)
        asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
