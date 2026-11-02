import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from sidewalkbench.pg.utils.math import _quat_to_yaw, check_collision_bbox
from sidewalkbench.pg.utils.scenario_rules import pedestrian_group_intrusion


def custom_collision_static(env: ManagerBasedRLEnv) -> torch.Tensor:
    collision = torch.zeros(env.num_envs, dtype=torch.bool).to(env.device)
    if env.scene.clone_static_states is None:
        return collision

    num_rows, num_cols = env.scene.terrain.terrain_origins.shape[:2]

    static_states = env.scene.clone_static_states
    robot_pos = env.scene["robot"].data.root_pos_w
    robot_yaw = _quat_to_yaw(env.scene["robot"].data.root_quat_w)
    robot_bbox = torch.tensor(env.scene.robot_bbox).to(env.device).unsqueeze(0).expand_as(robot_pos)
    robot_states = torch.cat([robot_pos, robot_bbox, robot_yaw.unsqueeze(-1)], dim=-1)
    robot_terrain_idxs = env.scene.terrain.terrain_levels * num_cols + env.scene.terrain.terrain_types

    for i, static_states_i in static_states.items():
        robot_mask = robot_terrain_idxs == i
        if robot_mask.sum() == 0:
            continue
        result = check_collision_bbox(robot_states[robot_mask], static_states_i)
        collision[robot_mask] = result

    return collision


def custom_collision_dynamic(env: ManagerBasedRLEnv) -> torch.Tensor:
    collision = torch.zeros(env.num_envs, dtype=torch.bool).to(env.device)
    if env.scene.clone_dynamic_states is None:
        return collision
    num_rows, num_cols = env.scene.terrain.terrain_origins.shape[:2]
    robot_pos = env.scene["robot"].data.root_pos_w
    robot_yaw = _quat_to_yaw(env.scene["robot"].data.root_quat_w)
    robot_bbox = torch.tensor(env.scene.robot_bbox).to(env.device).unsqueeze(0).expand_as(robot_pos)
    robot_states = torch.cat([robot_pos, robot_bbox, robot_yaw.unsqueeze(-1)], dim=-1)
    robot_terrain_idxs = env.scene.terrain.terrain_levels * num_cols + env.scene.terrain.terrain_types

    for i, dynamic_states_i in env.scene.clone_dynamic_states.items():
        robot_mask = robot_terrain_idxs == i
        if robot_mask.sum() == 0:
            continue
        dynamic_pos, dynamic_yaw = dynamic_states_i[:, :3], dynamic_states_i[:, 3]
        dynamic_bbox = torch.tensor([0.3, 0.3, 1.7]).to(env.device)
        dynamic_states = torch.cat(
            [dynamic_pos, dynamic_bbox.unsqueeze(0).expand_as(dynamic_pos), dynamic_yaw.unsqueeze(-1)], dim=-1
        )
        result = check_collision_bbox(robot_states[robot_mask], dynamic_states)
        collision[robot_mask] = result

    return collision


def out_of_region(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), buffer: float = 0.0
) -> torch.Tensor:
    """Terminate when the asset is outside the walkable region (with optional buffer).

    Args:
        buffer: Extra distance (meters) beyond the boundary before triggering. Default 0.
    """
    robot_pos = env.scene["robot"].data.root_pos_w[..., :2] - env.scene.env_origins[..., :2]
    out_of_region = torch.zeros(env.num_envs, dtype=torch.bool).to(env.device)
    for i in range(env.num_envs):
        row, col = env.scene.terrain.terrain_levels[i], env.scene.terrain.terrain_types[i]
        dist = env.scene.terrain.terrain_boundary_checkers[row][col](robot_pos[i][0].item(), robot_pos[i][1].item())
        out_of_region[i] = float(dist) > buffer
    return out_of_region


def arrive(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold=0.1) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    arrive = torch.zeros(env.num_envs, dtype=torch.bool).to(env.device)
    for i in range(env.num_envs):
        row, col = env.scene.terrain.terrain_levels[i], env.scene.terrain.terrain_types[i]
        projector = env.scene.terrain.terrain_centerline_projectors[row][col]
        local_pos = asset.data.root_pos_w[i, :2] - env.scene.env_origins[i, :2]
        s, lat = projector(local_pos[0].item(), local_pos[1].item())
        env.scene.last_arc_length[i] = s
        arrive[i] = s >= env.scene.arc_length_goal[i].item() - threshold
        if env.scene.terrain_segment == "crosswalk":
            arrive[i] = arrive[i] & (lat <= 1.0)
    return arrive


def freeze(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    freeze_duration: float = 5.0,
    move_threshold: float = 0.1,
) -> torch.Tensor:
    """Terminate when the robot hasn't moved more than move_threshold over freeze_duration seconds."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_pos = asset.data.root_pos_w[:, :2]
    dt = env.step_dt
    buf_len = max(1, int(freeze_duration / dt))

    # Lazy-init ring buffer and per-env step counter
    if not hasattr(env, "_freeze_pos_buf"):
        env._freeze_pos_buf = cur_pos.clone().unsqueeze(1).repeat(1, buf_len, 1)
        env._freeze_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # Detect resets via episode_length_buf
    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        env._freeze_pos_buf[just_reset] = cur_pos[just_reset].unsqueeze(1).expand(-1, buf_len, -1)
        env._freeze_step[just_reset] = 0

    # Store current position
    env._freeze_pos_buf[torch.arange(env.num_envs), env._freeze_step % buf_len] = cur_pos
    env._freeze_step += 1

    # Only check after buffer is full
    not_ready = env._freeze_step < buf_len
    oldest = env._freeze_pos_buf[torch.arange(env.num_envs), env._freeze_step % buf_len]
    dist = torch.linalg.norm(cur_pos - oldest, dim=-1)
    return (dist < move_threshold) & ~not_ready
