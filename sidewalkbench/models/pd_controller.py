"""Classic PD waypoint controller shared by all navigation models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class PDControllerCfg:
    """Configuration for :class:`PDController`.

    Attributes:
        waypoint_mode: ``lookahead`` uses one time-limited waypoint for both
            commands. ``half_horizon`` uses the waypoint nearest half of the
            prediction horizon for speed, and the final waypoint for heading.
        lookahead_s: Target time and maximum lookahead [s] in lookahead mode.
        dt: Optional time divisor [s], capped by ``lookahead_s`` in lookahead
            mode. Half-horizon mode defaults to half the final prediction time.
        speed_waypoint_index: Optional speed waypoint override. BaseModel
            otherwise selects the waypoint nearest the mode's target time.
        heading_waypoint_index: Optional heading index; half-horizon mode
            defaults to the final waypoint. In lookahead mode it
            must select the same waypoint as ``speed_waypoint_index``.
        max_v: Maximum forward speed [m/s].
        max_w: Maximum angular velocity [rad/s].
        smoothing: Apply acceleration limits to the linear velocity.
        clamp: Apply upper speed and angular limits. Linear speed is always
            nonnegative, including when this option is disabled.
    """

    waypoint_mode: str = "half_horizon"
    lookahead_s: float = 1.0
    dt: float | None = None
    speed_waypoint_index: int | None = None
    heading_waypoint_index: int | None = None
    max_v: float = 1.5
    max_w: float = 0.65
    smoothing: bool = True
    clamp: bool = True


class PDController:
    """Convert predicted waypoints to ``(v, w)`` commands.

    BaseModel selects speed and heading waypoints from the configured mode.
    Standalone calls without indices use the middle array index for both.
    Linear commands
    are always nonnegative; waypoints behind the robot cannot request reverse.
    """

    def __init__(
        self,
        *,
        speed_waypoint_index: int | None = None,
        heading_waypoint_index: int | None = None,
        max_v: float = 1.5,
        max_w: float = 0.65,
        smoothing: bool = True,
        clamp: bool = True,
    ) -> None:
        self.speed_waypoint_index = speed_waypoint_index
        self.heading_waypoint_index = heading_waypoint_index
        self.max_v = max_v
        self.max_w = max_w
        self.smoothing = smoothing
        self.clamp = clamp
        self.last_v: torch.Tensor | None = None
        self._has_last: torch.Tensor | None = None

    def reset(self, env_idx: int | Sequence[int] | None = None) -> None:
        """Clear the per-env velocity smoothing state.

        Args:
            env_idx: Environment indices to clear; ``None`` clears all.
        """
        if self.last_v is None:
            return
        if env_idx is None:
            self.last_v = None
            self._has_last = None
        else:
            self._has_last[env_idx] = False

    def __call__(
        self,
        waypoints: torch.Tensor,
        dt: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``(v, w)`` from ego-frame waypoints ``(B, N, >=2)``.

        Args:
            waypoints: Ego-frame waypoint positions, shape ``[B, N, >=2]``.
            dt: Controller time divisor [s], resolved by BaseModel.

        Returns:
            Linear and angular velocity, each shape ``[B]``.
        """
        eps = 1e-6
        count = waypoints.shape[1]
        idx = (count - 1) // 2 if self.speed_waypoint_index is None else self.speed_waypoint_index
        heading_idx = idx if self.heading_waypoint_index is None else self.heading_waypoint_index
        if not (-count <= idx < count and -count <= heading_idx < count):
            raise ValueError("PD waypoint indices are outside the predicted trajectory.")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("PD dt must be finite and positive.")
        dx = waypoints[:, idx, 0]
        dxr = waypoints[:, heading_idx, 0]
        dyr = waypoints[:, heading_idx, 1]

        v = (dx / dt).clamp_min(0.0)
        w = torch.atan2(dyr, dxr.abs().clamp(min=eps)) * dxr.sign() / dt

        near_zero = dx.abs() < eps
        v = torch.where(near_zero, torch.zeros_like(v), v)
        w = torch.where(near_zero, dyr.sign() * (math.pi / 20.0), w)

        if self.smoothing:
            if self.last_v is None or self.last_v.shape[0] != v.shape[0]:
                self.last_v = v.clone()
                self._has_last = torch.zeros_like(v, dtype=torch.bool)
            v = torch.where(
                self._has_last,
                v.clamp(self.last_v - 0.5, self.last_v + 0.4),
                v,
            ).clamp_min(0.0)
        if self.clamp:
            v = v.clamp(0.0, self.max_v)
            w = w.clamp(-self.max_w, self.max_w)
        if self.smoothing:
            # The historical controller smooths from the command actually sent.
            self.last_v = v.clone()
            self._has_last[:] = True
        return v, w
