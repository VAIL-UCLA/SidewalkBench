# Copyright (c) 2022-2025, The SidewalkBench-PG Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Goal conditioning helpers for evaluation.

Exposes the current goal to models that declare
:attr:`sidewalkbench.models.base_model.BaseModel.goal_type`:

- ``"point"`` models consume a local goal point ``[x_fwd, y_left]`` [m].
- ``"image"`` models consume an egocentric goal image ``(B, 3, H, W)`` in
  ``[0, 1]``.

Goal images are captured with a **dedicated free goal camera**. Capture is folded
into the environment loop: :meth:`GoalManager.pose_current` sets the camera to the
current goal *before* ``env.step`` (whose internal physics step propagates the pose
to the renderer) and :meth:`GoalManager.read_current` reads it back *after* the
step. This avoids teleporting the main robot, issues no extra physics step, and
produces pedestrian-free images (pedestrians are not USD prims). The goal image
therefore lags the goal point by one iteration when the goal changes. Two modes:

- ``"short"``: one image at the scenario end goal.
- ``"long"``: the route is split into ``segment_length`` [m] segments; one image
  is captured per segment goal and swapped in as the robot advances (advance when
  the robot is within ``advance_margin`` [m] arc-length of the segment goal).

Capture never calls ``scene.write_data_to_sim`` / the env step loop, so ORCA,
pedestrians, commands, events, and terminations are not ticked during capture.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from sidewalkbench.pg.utils.math import _quat_to_yaw

# Arc-length step [m] used to estimate the route tangent at a goal.
_TANGENT_EPS = 0.5

# Distance [m] ahead of the goal camera used to build its look-at target.
_GOAL_LOOK_AHEAD = 1.0


def advance_segment(
    s_target: float, s_goal: float, s_robot: float, segment_length: float, advance_margin: float
) -> float:
    """Return the route-segment target arc-length after one update.

    The target is advanced by ``segment_length`` (clamped to ``s_goal``) once the
    robot's arc-length ``s_robot`` reaches ``s_target - advance_margin``; it is
    left unchanged at or beyond the final goal.
    """
    if s_target >= s_goal - 1e-6:
        return s_target
    if s_robot >= s_target - advance_margin:
        return min(s_target + segment_length, s_goal)
    return s_target


class GoalManager:
    """Track the current goal point/yaw and (optionally) its captured image.

    Args:
        env: The gymnasium environment (unwrapped access is used).
        mode: ``"short"`` (scenario end goal) or ``"long"`` (route segments).
        segment_length: Route segment length [m] (long mode).
        advance_margin: Arc-length margin [m] at which the goal advances (long mode).
        capture_images: Whether to capture goal images with the free goal camera.
    """

    def __init__(
        self,
        env,
        mode: str = "short",
        segment_length: float = 30.0,
        advance_margin: float = 1.0,
        capture_images: bool = True,
    ):
        self.env = env
        self.scene = env.unwrapped.scene
        self.mode = mode
        self.segment_length = float(segment_length)
        self.advance_margin = float(advance_margin)
        self.capture_images = bool(capture_images)

        self.robot = self.scene["robot"]
        # Front camera is used only for the goal-eye elevation; images are captured
        # with the dedicated free goal camera.
        self.camera = self.scene.sensors["camera"]
        self.goal_camera = self.scene.sensors.get("goal_camera")

        num_envs = env.unwrapped.num_envs
        device = env.unwrapped.device
        self._goal_world = self.scene.env_goals.clone()
        self._goal_yaw = self.scene.env_goal_yaws.clone()
        self._goal_images: torch.Tensor | None = None
        # Per-env list of segments: {"s": float, "world": (2,) tensor, "yaw": float}
        self._segments: dict[int, list[dict]] = {}
        self._idx: dict[int, int] = {i: 0 for i in range(num_envs)}
        self._running = False
        # Cameras that must be re-posed before the next env step, and cameras whose
        # pose was set this iteration so their render is read after the next step.
        self._dirty = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._pending_read = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """(Re)build the goal segment(s) for the given envs and mark them for re-posing.

        Called after an episode reset (statics/peds re-sampled). Never called on a
        recovery teleport. Goal images are (re)captured lazily: the camera is posed
        on the next :meth:`pose_current` and read on the following :meth:`read_current`.
        """
        env_ids_t = self._as_env_ids(env_ids)
        if env_ids_t.numel() == 0:
            return
        if self.capture_images and self._goal_images is None:
            cam = self.goal_camera or self.camera
            self._goal_images = torch.zeros(
                self.env.unwrapped.num_envs, 3, cam.cfg.height, cam.cfg.width,
                device=self.env.unwrapped.device,
            )
        for env_id in env_ids_t.tolist():
            self._segments[env_id] = self._build_segments(env_id)
            self._idx[env_id] = 0
            self._set_current(env_id)

    def update(self, env_ids: Sequence[int] | None = None) -> None:
        """Advance the current goal along the route once the robot is close enough (long mode)."""
        if self.mode != "long":
            return
        env_ids_t = self._as_env_ids(env_ids)
        if env_ids_t.numel() == 0:
            return
        s_robot = self._robot_arc_length(env_ids_t)
        for k, env_id in enumerate(env_ids_t.tolist()):
            segs = self._segments.get(env_id)
            if not segs:
                continue
            idx = self._idx[env_id]
            while idx < len(segs) - 1 and float(s_robot[k]) >= segs[idx]["s"] - self.advance_margin:
                idx += 1
            if idx != self._idx[env_id]:
                self._idx[env_id] = idx
                self._set_current(env_id)

    # ------------------------------------------------------------------
    # Segment/goal construction
    # ------------------------------------------------------------------

    def _build_segments(self, env_id: int) -> list[dict]:
        if self.mode == "long":
            s_start = float(self.scene.arc_length_start[env_id])
            s_goal = float(self.scene.arc_length_goal[env_id])
            arcs = []
            s = s_start + self.segment_length
            while s < s_goal - 1e-6:
                arcs.append(s)
                s += self.segment_length
            arcs.append(s_goal)
        else:
            arcs = [float(self.scene.arc_length_goal[env_id])]

        segments = []
        for s in arcs:
            world, yaw = self._pose_at_arc(env_id, s)
            segments.append({"s": s, "world": world, "yaw": yaw})
        return segments

    def _pose_at_arc(self, env_id: int, s: float) -> tuple[torch.Tensor, float]:
        """World goal position/yaw at route arc-length ``s`` (fallbacks for short/point)."""
        num_rows, num_cols = self.scene.terrain.terrain_origins.shape[:2]
        row = int(self.scene.terrain.terrain_levels[env_id]) % num_rows
        col = int(self.scene.terrain.terrain_types[env_id]) % num_cols
        origin = self.scene.env_origins[env_id, :2]
        splines = self.scene.terrain.terrain_centerline_splines[row][col]
        if self.mode == "long" and splines is not None:
            arc_to_xy, total_len = splines
            xy = np.asarray(arc_to_xy(np.array([s])), dtype=np.float64)[0]
            s0, s1 = max(0.0, s - _TANGENT_EPS), min(float(total_len), s + _TANGENT_EPS)
            p0 = np.asarray(arc_to_xy(np.array([s0])), dtype=np.float64)[0]
            p1 = np.asarray(arc_to_xy(np.array([s1])), dtype=np.float64)[0]
            delta = p1 - p0
            world = torch.as_tensor(xy, device=origin.device, dtype=torch.float32) + origin
            return world, float(np.arctan2(delta[1], delta[0]))
        # Short mode (or missing spline): use the scenario goal and its provided yaw.
        world = self.scene.env_goals[env_id].clone()
        yaw = float(self.scene.env_goal_yaws[env_id].item())
        if np.isnan(yaw):
            delta = world - self.robot.data.root_pos_w[env_id, :2]
            yaw = float(torch.atan2(delta[1], delta[0]))
        return world, yaw

    def _set_current(self, env_id: int) -> None:
        seg = self._segments[env_id][self._idx[env_id]]
        self._goal_world[env_id] = seg["world"]
        self._goal_yaw[env_id] = seg["yaw"]
        self._dirty[env_id] = True

    # ------------------------------------------------------------------
    # Capture (dedicated free goal camera, folded into the env step)
    # ------------------------------------------------------------------

    def pose_current(self, env_ids: Sequence[int] | None = None) -> None:
        """Pose the goal camera to each env's current goal (call before ``env.step``).

        The env's own physics step then propagates the pose to the renderer; the
        image is read back by :meth:`read_current` after that step. Goal images
        therefore lag the goal point by one iteration when the goal changes.
        """
        cam = self.goal_camera
        if cam is None or not self.capture_images:
            return
        target = self._as_env_ids(env_ids)
        ids = target[self._dirty[target]]
        if ids.numel() == 0:
            return
        eye_z = self.camera.data.pos_w[ids, 2]
        eye = torch.stack([self._goal_world[ids, 0], self._goal_world[ids, 1], eye_z], dim=-1)
        yaw = self._goal_yaw[ids]
        fwd = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
        cam.set_world_poses_from_view(eye, eye + _GOAL_LOOK_AHEAD * fwd, env_ids=ids)
        self._pending_read[ids] = True
        self._dirty[ids] = False

    def read_current(self, env_ids: Sequence[int] | None = None) -> None:
        """Read the goal camera image for each env posed this iteration (after ``env.step``)."""
        if self.goal_camera is None or not self.capture_images or self._goal_images is None:
            return
        target = self._as_env_ids(env_ids)
        ids = target[self._pending_read[target]]
        if ids.numel() == 0:
            return
        rgb = self.goal_camera.data.output["rgb"][ids]  # (N, H, W, 3)
        rgb = rgb.to(device=self.env.unwrapped.device, dtype=torch.float32) / 255.0
        self._goal_images[ids] = rgb.permute(0, 3, 1, 2).contiguous()
        self._pending_read[ids] = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _as_env_ids(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        device = self.env.unwrapped.device
        if env_ids is None:
            return torch.arange(self.env.unwrapped.num_envs, device=device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=device, dtype=torch.long).reshape(-1)
        return torch.tensor(list(env_ids), device=device, dtype=torch.long)

    def _robot_arc_length(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Robot centerline arc-length [m] per env, recomputed from the current pose."""
        num_rows, num_cols = self.scene.terrain.terrain_origins.shape[:2]
        s = self.scene.last_arc_length[env_ids].clone()
        for k, env_id in enumerate(env_ids.tolist()):
            row = int(self.scene.terrain.terrain_levels[env_id]) % num_rows
            col = int(self.scene.terrain.terrain_types[env_id]) % num_cols
            projector = self.scene.terrain.terrain_centerline_projectors[row][col]
            if projector is None:
                continue
            local = self.robot.data.root_pos_w[env_id, :2] - self.scene.env_origins[env_id, :2]
            s[k] = projector(local[0].item(), local[1].item())[0]
        return s

    # ------------------------------------------------------------------
    # Model/vis-facing accessors
    # ------------------------------------------------------------------

    def goal_xy_local(self) -> np.ndarray:
        """Current goals in the robot frame ``[x_fwd, y_left]`` [m], shape ``(B, 2)``."""
        pos = self.robot.data.root_pos_w[:, :2]
        yaw = _quat_to_yaw(self.robot.data.root_quat_w)
        delta = self._goal_world - pos
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        x_fwd = delta[:, 0] * cos_yaw + delta[:, 1] * sin_yaw
        y_left = -delta[:, 0] * sin_yaw + delta[:, 1] * cos_yaw
        return torch.stack([x_fwd, y_left], dim=-1).cpu().numpy().astype(np.float32)

    def goal_world(self) -> np.ndarray:
        """Current goal world positions ``(B, 2)`` [m]."""
        return self._goal_world.cpu().numpy()

    def goal_yaw(self) -> np.ndarray:
        """Current goal forward headings ``(B,)`` [rad] (NaN if unknown)."""
        return self._goal_yaw.cpu().numpy()

    def goal_image(self) -> torch.Tensor | None:
        """Current goal images ``(B, 3, H, W)`` in ``[0, 1]``, or ``None`` if not captured."""
        return self._goal_images
