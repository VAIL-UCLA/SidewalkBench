"""Regulated pure pursuit: ego-frame path -> forward speed and Ackermann steering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

WHEEL_BASE = 0.456
MAX_STEER_RAD = 0.35


@dataclass
class PurePursuitConfig:
    lookahead_base: float = 1.0
    lookahead_gain: float = 0.1
    max_lookahead: float = 2.0
    wheelbase: float = WHEEL_BASE
    max_steering_angle: float = MAX_STEER_RAD


def _circle_segment_intersection(
    p1: torch.Tensor,
    p2: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """Circle at origin, radius ``r``; segment ``p1``–``p2``."""
    x1, y1 = p1[..., 0], p1[..., 1]
    x2, y2 = p2[..., 0], p2[..., 1]
    dx = x2 - x1
    dy = y2 - y1
    dr2 = dx * dx + dy * dy
    d_cross = x1 * y2 - x2 * y1

    d1 = x1 * x1 + y1 * y1
    d2 = x2 * x2 + y2 * y2
    dd = d2 - d1
    sign_dd = torch.where(dd >= 0, torch.ones_like(dd), -torch.ones_like(dd))
    inner = r * r * dr2 - d_cross * d_cross
    sqrt_term = torch.sqrt(inner)
    px = (d_cross * dy + sign_dd * dx * sqrt_term) / dr2
    py = (-d_cross * dx + sign_dd * dy * sqrt_term) / dr2
    return torch.stack([px, py], dim=-1)


def _find_target_point(path: torch.Tensor, lookahead_distance: float) -> torch.Tensor:
    """``path``: (H, 2) in robot frame; robot at origin. Returns (2,) float tensor on ``path`` device."""
    if path.shape[0] == 0:
        return torch.zeros(2, dtype=path.dtype, device=path.device)

    dists = torch.linalg.vector_norm(path, dim=-1)
    ge = dists >= lookahead_distance
    if not torch.any(ge):
        return path[-1].clone()

    first = int(torch.nonzero(ge, as_tuple=False)[0].item())
    if first == 0:
        return path[0].clone()

    p_prev = path[first - 1]
    p_goal = path[first]
    r = torch.tensor(lookahead_distance, dtype=path.dtype, device=path.device)
    return _circle_segment_intersection(p_prev, p_goal, r)


def _steering_from_target(
    target: torch.Tensor,
    position: torch.Tensor,
    heading: torch.Tensor,
    cfg: PurePursuitConfig,
) -> torch.Tensor:
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    target_angle = torch.atan2(dy, dx)
    angle_diff = target_angle - heading
    dist_to_target = torch.hypot(dx, dy)
    raw_steering_angle = torch.atan2(2.0 * cfg.wheelbase * torch.sin(angle_diff), dist_to_target + 1e-8)
    steering_angle = torch.clamp(raw_steering_angle, -cfg.max_steering_angle, cfg.max_steering_angle)
    return steering_angle


def _velocity_at_index(
    path_xy: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Predicted speed at waypoint ``n`` per env: (num_envs,)."""
    return path_xy[:, n, 2]


def _batch_ego_speed(robot: Any) -> torch.Tensor:
    """Planar speed ``‖v_xy‖`` per environment from root world linear velocity."""
    v_xy = robot.data.root_vel_w[..., :2]
    return torch.linalg.vector_norm(v_xy, dim=-1)


class PurePursuit:
    """Ego-frame regulated pure pursuit.

    Forward speed follows waypoint ``waypoint_index``; lookahead uses **current** planar ego speed from ``robot``.
    """

    def __init__(
        self,
        action_dt: float,
        *,
        config: PurePursuitConfig | None = None,
        waypoint_index: int = 0,
    ):
        self.action_dt = action_dt
        self.config = config if config is not None else PurePursuitConfig()
        self.waypoint_index = waypoint_index
        self._ego_log_target_point: torch.Tensor | None = None

    def ego_log_target_point(self) -> torch.Tensor | None:
        return self._ego_log_target_point

    def step(
        self,
        ego_waypoints: torch.Tensor,
        *,
        robot: Any = None,
        axle_joint_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del axle_joint_ids
        if robot is None:
            raise TypeError("PurePursuit.step requires ``robot`` for current ego speed (lookahead distance).")

        path_xy = ego_waypoints[..., 0:2]  # only the x, y points
        num_envs = path_xy.shape[0]
        device = path_xy.device
        dtype = path_xy.dtype

        v_cmd = _velocity_at_index(ego_waypoints, self.waypoint_index)

        ego_speed = _batch_ego_speed(robot).to(device=device, dtype=dtype)

        position = torch.zeros(num_envs, 2, device=device, dtype=dtype)
        heading = torch.zeros(num_envs, 1, device=device, dtype=dtype)

        steer_out = torch.empty(num_envs, device=device, dtype=dtype)
        target_ego = torch.empty(num_envs, 2, device=device, dtype=dtype)
        cfg = self.config

        for e in range(num_envs):
            speed_e = float(ego_speed[e].item())
            ld = min(cfg.lookahead_base + cfg.lookahead_gain * speed_e, cfg.max_lookahead)
            target = _find_target_point(path_xy[e], ld)
            target_ego[e] = target
            steer_out[e] = _steering_from_target(
                target,
                position[e],
                heading[e],
                cfg,
            )[0]

        self._ego_log_target_point = target_ego
        v_w = v_cmd / cfg.wheelbase * torch.tan(steer_out)
        return v_cmd, v_w
