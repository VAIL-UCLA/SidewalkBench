#!/usr/bin/env python3
"""Cosmos3-Official inference wrapper (Cosmos3-Nano ``wam``/``av``, fast 1-frame export, goal free).

    nav = Cosmos3Navigator(device="cuda")
    traj, scores = nav.inference_trajectory(obs)        # (B, 1, 40, 3) [x_fwd, y_left, speed], (B, 1)
    vw, best = nav.inference_vw(obs)                    # (B, 2) [v, w], (B, 40, 3)
    vw, best = nav.inference_vw(obs, pose=pose)         # pose (B, 3) world [x, y, yaw]: reuse plans
    nav.reset()

obs: (B, T, 3, H, W) float32 in [0,1]; the last frame is used (pinhole, see model_info.yaml).
One plan is 20 UniPC steps of denoiser.onnx (~0.8 s, ~1.7 s with mirror_ensemble). With ``pose``
a plan is kept for ``replan_interval_s`` and re-expressed in the current robot frame; without it
every call plans. Keep each .onnx next to its .onnx.data.
"""

import json
import os

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageOps
from sidewalkbench.models.base_model import BaseModel, ControllerCfg

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Camera frame (x right, y down, z forward) -> body frame (x forward, y left, z up).
BODY_R_CAMERA = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
CAMERA_HEIGHT = 0.456


def rot6d_to_matrix(r6):
    """First two rotation columns (..., 6) -> rotation matrices (..., 3, 3)."""
    a, b = r6[..., :3], r6[..., 3:6]
    x = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (x * b).sum(-1, keepdims=True) * x
    y = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def actions_to_ego(actions, scale):
    """Per-step camera poses (N, 9) ``[t_xyz | rot6d]`` in model metres -> body poses
    (N+1, 3) ``[x_fwd, y_left, yaw]`` in robot metres; row 0 is the start pose."""
    rel = np.broadcast_to(np.eye(4), (len(actions), 4, 4)).copy()
    rel[:, :3, :3] = rot6d_to_matrix(np.asarray(actions[:, 3:9], dtype=np.float64))
    rel[:, :3, 3] = np.asarray(actions[:, :3], dtype=np.float64) / scale
    cam = [np.eye(4)]
    for r in rel:
        cam.append(cam[-1] @ r)
    body_t_camera = np.eye(4)
    body_t_camera[:3, :3] = BODY_R_CAMERA
    body_t_camera[2, 3] = CAMERA_HEIGHT
    body = body_t_camera @ np.stack(cam) @ np.linalg.inv(body_t_camera)
    yaw = np.unwrap(np.arctan2(body[:, 1, 0], body[:, 0, 0]))
    return np.column_stack([body[:, 0, 3], body[:, 1, 3], yaw])


class Cosmos3Navigator(BaseModel):
    """Base Cosmos3-Nano world-action model as a goal-less sidewalk policy."""

    context_size = 1
    multimodal = False
    goal_type = None

    def __init__(self, device="cuda", step_dt=0.2, controller: ControllerCfg | None = None):
        """step_dt: seconds between calls (used to advance a kept plan)."""
        super().__init__(device=device, controller=controller,
                         info_path=os.path.join(MODEL_DIR, 'model_info.yaml'))
        self.step_dt = float(step_dt)
        if not np.isfinite(self.step_dt) or self.step_dt <= 0:
            raise ValueError('Cosmos3 step_dt must be finite and positive.')
        self._env = None
        self._batch_size = None
        with open(os.path.join(MODEL_DIR, "meta.json")) as f:
            self.meta = json.load(f)
        with open(os.path.join(MODEL_DIR, "io_spec.json")) as f:
            spec = json.load(f)
        sched = self.meta["schedules"][str(self._info["denoising_steps"])]
        self.model_t, self.coefs = sched["model_t"], sched["unipc"]
        self.latent_shape = self.meta["layout"]["latent_shape"][-4:]  # C, T, H, W
        self.cond = np.asarray(self.meta["layout"]["condition_frames"], dtype=bool)
        self.canvas_wh = tuple(spec["encoder"]["canvas_wh"])
        self.chunk = self.meta["chunk"]
        self.plan_fps = float(self.meta["fps"])
        self.raw_dim = self.meta["arch"]["raw_action_dim"]
        self.action_dim = self.meta["arch"]["action_dim"]
        self.scale = float(self._info.get("robot_scale") or self.meta["robot_scale"])
        self.replan_interval = float(self._info["replan_interval_s"])
        self.mirror_ensemble = bool(self._info.get("mirror_ensemble", True))
        self.seed = 0
        self.num_waypoints = int(self._info["num_waypoints"])

        self._encoder = self._load_session(os.path.join(MODEL_DIR, spec['encoder']['file']))
        self._denoiser = self._load_session(os.path.join(MODEL_DIR, spec['denoiser']['file']))
        self._plans = {}

    def _load_session(self, path):
        if torch.device(self.device).type == 'cpu':
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            return ort.InferenceSession(path, sess_options=options, providers=['CPUExecutionProvider'])
        return self._make_session(path)

    def bind_env(self, env):
        self._env = env.unwrapped
        self.reset()

    def _reset_state(self, env_idx=None):
        if env_idx is None:
            self._plans = {}
            self._batch_size = None
        else:
            if hasattr(env_idx, 'detach'):
                env_idx = env_idx.detach().cpu().numpy()
            for index in np.asarray(env_idx).reshape(-1):
                self._plans.pop(int(index), None)

    def _measured_pose(self, robot=None):
        if robot is None and self._env is not None:
            robot = self._env.scene['robot']
        if robot is None:
            return None
        def numpy(value):
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float64)
        position = numpy(robot.data.root_pos_w)
        q = numpy(robot.data.root_quat_w)  # XYZW, as in the benchmark goal transform.
        x, y, z, w = q.T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.column_stack([position[:, :2], yaw])

    def _canvas(self, frame):
        """(3, H, W) float in [0,1] -> uint8 canvas (1, 3, 1, 480, 832): centre crop + Lanczos."""
        rgb = (np.clip(frame, 0.0, 1.0) * 255.0).round().astype(np.uint8).transpose(1, 2, 0)
        im = ImageOps.fit(Image.fromarray(rgb), self.canvas_wh, method=Image.Resampling.LANCZOS)
        return np.ascontiguousarray(np.asarray(im).transpose(2, 0, 1)[None, :, None])

    def _sample(self, cond_latent):
        """UniPC (order 2, bh2, x0-prediction) over the joint [vision | action] state.

        Returns:
            Raw av actions (chunk, 9): per-step camera pose ``[t_xyz | rot6d]``.
        """
        C, T, H, W = self.latent_shape
        rng = np.random.default_rng(self.seed)
        xv = rng.standard_normal((1, C, T, H, W)).astype(np.float32)
        xa = np.zeros((1, self.chunk, self.action_dim), dtype=np.float32)
        xa[..., : self.raw_dim] = rng.standard_normal((1, self.chunk, self.raw_dim))
        if self.meta.get("action_pad_noise") == "gaussian":
            xa[..., self.raw_dim :] = rng.standard_normal((1, self.chunk, self.action_dim - self.raw_dim))
        xv[:, :, self.cond] = cond_latent[:, :, : int(self.cond.sum())]

        x = (xv, xa)
        outs, last = [], None
        for k, d in enumerate(self.coefs):
            t = np.array([self.model_t[k]], dtype=np.float32)
            v = self._denoiser.run(None, {"x_vision": x[0].astype(np.float32), "x_action": x[1].astype(np.float32), "t": t})
            x0 = tuple(xi - d["sigma"] * vi for xi, vi in zip(x, v))
            c = d["corrector"]
            if c is not None:
                m0 = outs[-1]
                xt = tuple(c["x_coef"] * ls + c["m0_coef"] * m for ls, m in zip(last, m0))
                if c["order"] == 2:
                    xt = tuple(a + c["d1s_coef"] * (m2 - m) / c["rk"] for a, m2, m in zip(xt, outs[-2], m0))
                x = tuple(a + c["d1t_coef"] * (b - m) for a, b, m in zip(xt, x0, m0))
            outs.append(x0)
            last = x
            p = d["predictor"]
            xt = tuple(p["x_coef"] * a + p["m0_coef"] * b for a, b in zip(x, x0))
            if p["order"] == 2:
                xt = tuple(a + p["d1s_coef"] * (m2 - b) / p["rk"] for a, m2, b in zip(xt, outs[-2], x0))
            x = xt
        return x[1][0, :, : self.raw_dim]

    def _plan_poses(self, frame):
        """One frame (3, H, W) -> body poses (chunk + 1, 3). With ``mirror_ensemble`` the
        plan on the mirrored frame, mirrored back, is averaged in; this cancels steering
        that does not depend on the image."""
        latent = self._encoder.run(None, {"frames": self._canvas(frame)})[0]
        poses = actions_to_ego(self._sample(latent), self.scale)
        if self.mirror_ensemble:
            latent = self._encoder.run(None, {"frames": self._canvas(np.ascontiguousarray(frame[:, :, ::-1]))})[0]
            mirrored = actions_to_ego(self._sample(latent), self.scale) * np.array([1.0, -1.0, -1.0])
            poses = (poses + mirrored) / 2
        return poses

    def _offset(self, plan):
        """Plan steps elapsed since the plan was made."""
        return int(round(plan["age"] * self.step_dt * self.plan_fps))

    def _needs_plan(self, plan, pose):
        if plan is None or pose is None or plan["origin"] is None:
            return True
        if plan["age"] * self.step_dt >= self.replan_interval - 1e-6:
            return True
        return self._offset(plan) + self.num_waypoints > self.chunk

    def _window(self, plan, pose):
        """The next ``num_waypoints`` of ``plan`` in the current body frame -> (N, 3) x, y, speed."""
        offset = self._offset(plan)
        xy = plan["poses"][offset + 1 : offset + 1 + self.num_waypoints, :2]
        speed = plan["speed"][offset : offset + self.num_waypoints]
        if pose is not None:
            origin = plan["origin"]
            c, s = np.cos(origin[2]), np.sin(origin[2])
            world = xy @ np.array([[c, s], [-s, c]]) + origin[:2]
            c, s = np.cos(pose[2]), np.sin(pose[2])
            xy = (world - pose[:2]) @ np.array([[c, -s], [s, c]])
        return np.column_stack([xy, speed]).astype(np.float32)

    def inference_trajectory(self, obs, pose=None):
        """obs: (B, T, 3, H, W) float32 in [0,1]. pose: optional (B, 3) world [x, y, yaw].

        Returns trajectory (B, 1, 40, 3) [x_fwd, y_left (m), speed (m/s)] and scores (B, 1).
        """
        obs_np = obs.detach().cpu().numpy() if hasattr(obs, 'detach') else np.asarray(obs, dtype=np.float32)
        if obs_np.ndim != 5 or obs_np.shape[2] != 3 or obs_np.shape[1] < 1:
            raise ValueError('Cosmos3 expects (B, T, 3, H, W) RGB observations.')
        B = obs_np.shape[0]
        if self._batch_size != B:
            self.reset()
            self._batch_size = B
        if pose is None:
            pose = self._measured_pose()
        if pose is not None:
            if hasattr(pose, 'detach'):
                pose = pose.detach().cpu().numpy()
            pose = np.asarray(pose, dtype=np.float64)
            if pose.shape != (B, 3) or not np.isfinite(pose).all():
                raise ValueError('Cosmos3 pose must be finite world [x, y, yaw], shape (B, 3).')
        trajectory = np.zeros((B, 1, self.num_waypoints, 3), dtype=np.float32)
        for b in range(B):
            p = None if pose is None else np.asarray(pose[b], dtype=np.float64)
            plan = self._plans.get(b)
            if self._needs_plan(plan, p):
                poses = self._plan_poses(obs_np[b, -1])
                speed = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1) * self.plan_fps
                plan = self._plans[b] = {"poses": poses, "speed": speed, "origin": None if p is None else p.copy(), "age": 0}
            trajectory[b, 0] = self._window(plan, p)
            plan["age"] += 1
        if not np.isfinite(trajectory).all():
            raise RuntimeError('Cosmos3 returned non-finite predictions.')
        return trajectory, np.ones((B, 1), dtype=np.float32)

    def inference_vw(self, obs, robot=None, pose=None):
        """Use measured world poses for plan reuse and the shared 1 s PD controller."""
        if pose is None:
            pose = self._measured_pose(robot)
        return super().inference_vw(obs, robot=robot, pose=pose)
