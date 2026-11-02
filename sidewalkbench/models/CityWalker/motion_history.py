"""Measured position history in the robot's current planar coordinate frame."""

from collections.abc import Sequence

import numpy as np


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


class MotionHistory:
    """Keep one oldest-first position buffer per robot in a batched articulation.

    Each sample corresponds to a policy observation. Missing startup history is
    padded with the first measured position by default. An optional reset speed
    seeds only missing poses with forward motion in the reset robot frame.
    New measured poses replace these seeds on subsequent distinct policy ticks.
    Quaternions follow IsaacLab 3's
    XYZW convention, matching SidewalkBench's robot-frame goal conversion.
    """

    def __init__(self, context_size: int = 5, *, reset_speed_mps: float = 0.0,
                 sample_dt_s: float = 0.2):
        if not isinstance(context_size, int) or isinstance(context_size, bool) or context_size < 1:
            raise ValueError("CityWalker context size must be a positive integer.")
        reset_speed_mps, sample_dt_s = float(reset_speed_mps), float(sample_dt_s)
        if not np.isfinite(reset_speed_mps) or reset_speed_mps < 0:
            raise ValueError("CityWalker reset history speed must be finite and nonnegative.")
        if not np.isfinite(sample_dt_s) or sample_dt_s <= 0:
            raise ValueError("CityWalker history sample time must be finite and positive.")
        self.context_size = context_size
        self.reset_speed_mps = reset_speed_mps
        self.sample_dt_s = sample_dt_s
        self.reset()

    def reset(self, env_idx: int | Sequence[int] | None = None):
        if env_idx is None:
            self._robot = None
            self._positions = self._initialized = self._ticks = None
        elif self._positions is not None:
            if hasattr(env_idx, "detach"):
                env_idx = env_idx.detach().cpu().numpy()
            self._positions[env_idx] = 0
            self._initialized[env_idx] = False
            self._ticks[env_idx] = -1

    def sample(self, robot, tick: int | None = None) -> np.ndarray:
        """Return ``(B, context, 2)`` metres in ``[forward, left]`` coordinates."""
        positions = _numpy(robot.data.root_pos_w)
        quaternions = _numpy(robot.data.root_quat_w)
        if positions.ndim != 2 or positions.shape[1] < 2:
            raise ValueError("CityWalker robot positions must have shape (B, >=2).")
        batch = len(positions)
        if quaternions.shape != (batch, 4):
            raise ValueError("CityWalker robot quaternions must have shape (B, 4), in XYZW order.")
        positions = positions[:, :2]
        if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
            raise ValueError("CityWalker requires finite measured robot poses.")
        if (self._robot is not robot or self._positions is None
                or self._positions.shape[0] != batch):
            self._robot = robot
            self._positions = np.zeros((batch, self.context_size, 2), np.float32)
            self._initialized = np.zeros(batch, dtype=bool)
            self._ticks = np.full(batch, -1, dtype=np.int64)

        # Seed in world coordinates along each robot's own reset heading. Once
        # the buffer fills, every row is measured, including for a stopped robot.
        x, y, z, w = quaternions.T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        offsets = (np.arange(1 - self.context_size, 1, dtype=np.float32)
                   * self.reset_speed_mps * self.sample_dt_s)
        fresh = ~self._initialized
        advance = self._initialized & (True if tick is None else self._ticks != tick)
        self._positions[fresh, :, 0] = positions[fresh, 0, None] + offsets * c[fresh]
        self._positions[fresh, :, 1] = positions[fresh, 1, None] + offsets * s[fresh]
        self._positions[advance, :-1] = self._positions[advance, 1:]
        self._positions[advance, -1] = positions[advance]
        self._initialized[:] = True
        self._ticks[:] = -1 if tick is None else tick

        # World XY -> the current robot frame; the newest point is the origin.
        delta = self._positions - positions[:, None]
        history = np.empty_like(delta)
        history[..., 0] = c * delta[..., 0] + s * delta[..., 1]
        history[..., 1] = -s * delta[..., 0] + c * delta[..., 1]
        return history
