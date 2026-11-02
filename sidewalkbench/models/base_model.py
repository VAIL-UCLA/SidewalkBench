"""Shared base class for SidewalkBench navigation model wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
import yaml

from .pd_controller import PDController, PDControllerCfg
from .pure_pursuit import PurePursuitConfig, PurePursuit

if TYPE_CHECKING:
    import onnxruntime as ort

_DEFAULT_T_MID = 1.0


@dataclass
class ControllerCfg:
    """Selects the low-level controller that converts waypoints to ``(v, w)``.

    Attributes:
        type: ``"pd"`` (default) or ``"pp"``.
        pd: Parameters for the PD controller.
        pp: Parameters for the pure-pursuit controller.
        pp_action_dt: Replanning interval [s] passed to the pure-pursuit controller.
    """

    type: str = "pd"
    pd: PDControllerCfg | None = None
    pp: PurePursuitConfig = field(default_factory=PurePursuitConfig)
    pp_action_dt: float = 3.0


class BaseModel:
    """Common base for navigation model wrappers.

    Subclasses implement :meth:`inference_trajectory`; the shared
    :meth:`inference_vw` converts the predicted trajectory to ``(v, w)`` using the
    configured controller (PD by default, pure pursuit optionally).

    Class Attributes:
        context_size: Number of observation frames the model consumes.
        multimodal: Whether ``inference_trajectory`` returns multiple trajectory modes.
        goal_type: Goal conditioning consumed by this model: ``None`` (ignores
            goals), ``"point"`` (accepts a ``(B, 2)`` local goal ``[x_fwd, y_left]``
            via ``goal_xy``), or ``"image"`` (accepts an egocentric goal image
            ``(B, 3, H, W)`` in ``[0, 1]`` via ``goal_image``).
    """

    context_size: int = 21
    multimodal: bool = False
    goal_type: str | None = None

    def __init__(
        self,
        *,
        device: str = "cuda",
        controller: ControllerCfg | None = None,
        info_path: str | None = None,
        t_mid: float | None = None,
    ) -> None:
        """Initialize shared controller, model metadata, and horizon timing.

        Args:
            device: Torch device string for converted waypoints.
            controller: Controller selection and parameters (defaults to PD).
            info_path: Path to ``model_info.yaml`` describing the prediction horizon.
            t_mid: Optional initial controller time [s]. Inference resolves the
                time from the returned trajectory or ``controller.pd.dt``.
        """
        self.device = device
        self._info: dict = self._load_info(info_path) if info_path else {}
        requested = controller if controller is not None else ControllerCfg()
        self.controller_cfg = replace(
            requested,
            pd=requested.pd if requested.pd is not None else PDControllerCfg(**self._info.get("pd_controller", {})),
        )
        if self.controller_cfg.pd.waypoint_mode not in {"lookahead", "half_horizon"}:
            raise ValueError("Unknown PD waypoint_mode; use lookahead or half_horizon.")

        self._controller = PDController(
            speed_waypoint_index=self.controller_cfg.pd.speed_waypoint_index,
            heading_waypoint_index=self.controller_cfg.pd.heading_waypoint_index,
            max_v=self.controller_cfg.pd.max_v,
            max_w=self.controller_cfg.pd.max_w,
            smoothing=self.controller_cfg.pd.smoothing,
            clamp=self.controller_cfg.pd.clamp,
        )
        self._pp = PurePursuit(
            self.controller_cfg.pp_action_dt,
            config=self.controller_cfg.pp,
            waypoint_index=self._mid_index(),
        )
        self.t_mid = t_mid if t_mid is not None else self._pd_time_divisor(self._num_waypoints() or 0)

    # --- Model metadata / horizon -------------------------------------------------

    @staticmethod
    def _load_info(path: str) -> dict:
        """Load a ``model_info.yaml`` file."""
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def _num_waypoints(self) -> int | None:
        """Predicted trajectory length, from ``num_waypoints`` or ``len_traj_pred``."""
        for key in ("num_waypoints", "len_traj_pred"):
            if self._info.get(key):
                return int(self._info[key])
        return None

    def _fps(self) -> float | None:
        """Output waypoint rate [Hz], from ``output_fps`` or ``model_fps``."""
        for key in ("output_fps", "model_fps"):
            if self._info.get(key) is not None:
                fps = float(self._info[key])
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError(f"{key} must be finite and positive.")
                return fps
        return None

    def _mid_index(self) -> int:
        """Middle array index used by the optional pure-pursuit controller."""
        num_waypoints = self._num_waypoints()
        return (num_waypoints - 1) // 2 if num_waypoints else 0

    def _waypoint_times_seconds(self, num_waypoints: int) -> np.ndarray:
        """Resolve the returned trajectory's timestamps without a GPU round trip.

        Explicit timestamps take priority over the output rate. Untimed spatial
        paths use a declared benchmark horizon (2 s by default), rather than
        claiming that the evaluation frequency is a prediction frequency.
        """
        if num_waypoints < 1:
            raise ValueError("A trajectory must contain at least one waypoint.")
        fps = self._fps()
        timestamps = self._info.get("output_timestamps_s")
        ticks = self._info.get("output_timestamps_5hz")
        includes_origin = bool(self._info.get("trajectory_includes_origin", False))
        if timestamps is not None or ticks is not None:
            times = np.asarray(timestamps if timestamps is not None else ticks, dtype=np.float64)
            if times.ndim != 1 or len(times) < num_waypoints:
                raise ValueError("Prediction timestamps do not cover the returned trajectory.")
            times = times[:num_waypoints]
            if timestamps is None:
                times = times / 5.0
        elif fps:
            start = 0 if includes_origin else 1
            times = np.arange(start, num_waypoints + start, dtype=np.float64) / fps
        else:
            horizon = float(self._info.get("prediction_horizon_s", 2 * _DEFAULT_T_MID))
            if includes_origin and num_waypoints == 1:
                times = np.zeros(1, dtype=np.float64)
            else:
                start = 0 if includes_origin else 1
                denominator = num_waypoints - 1 if start == 0 else num_waypoints
                times = np.arange(start, num_waypoints + start, dtype=np.float64) * horizon / denominator
        origin_only = includes_origin and num_waypoints == 1 and times[0] == 0
        if (not np.isfinite(times).all() or times[0] < 0 or (times[-1] <= 0 and not origin_only)
                or np.any(np.diff(times) <= 0)):
            raise ValueError("Prediction timestamps must be finite and strictly increasing.")
        return times

    def waypoint_times(self, num_waypoints: int) -> torch.Tensor:
        """Waypoint timestamps [s], including a declared origin when present."""
        return torch.as_tensor(self._waypoint_times_seconds(num_waypoints), dtype=torch.float32, device=self.device)

    def _t_mid_for(self, num_waypoints: int) -> float:
        """Controller divisor [s] for the configured waypoint mode."""
        if self.controller_cfg.pd.waypoint_mode == "half_horizon":
            horizon = (float(self._waypoint_times_seconds(num_waypoints)[-1]) if num_waypoints
                       else float(self._info.get("prediction_horizon_s", 2 * _DEFAULT_T_MID)))
            # A singleton origin is a stop plan, with a positive fallback divisor.
            return horizon / 2 if horizon > 0 else _DEFAULT_T_MID
        limit = self._pd_lookahead_limit()
        if not num_waypoints:
            return limit
        horizon = float(self._waypoint_times_seconds(num_waypoints)[-1])
        # A singleton origin represents a stop plan and still needs a positive divisor.
        return min(horizon, limit) if horizon > 0 else limit

    def _pd_time_divisor(self, num_waypoints: int) -> float:
        """Configured divisor, or the mode's target time [s]."""
        override = self.controller_cfg.pd.dt
        if override is not None:
            value = float(override)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("PD dt must be finite and positive.")
            return (value if self.controller_cfg.pd.waypoint_mode == "half_horizon"
                    else min(value, self._pd_lookahead_limit()))
        return self._t_mid_for(num_waypoints)

    def _pd_lookahead_limit(self) -> float:
        """Validate the maximum controller lookahead [s]."""
        limit = float(self.controller_cfg.pd.lookahead_s)
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("PD lookahead_s must be finite and positive.")
        return limit

    def _pd_speed_index(self, num_waypoints: int) -> int:
        """Speed waypoint nearest the mode's target time; earlier on ties."""
        times = self._waypoint_times_seconds(num_waypoints)
        if self.controller_cfg.pd.waypoint_mode == "half_horizon":
            override = self.controller_cfg.pd.speed_waypoint_index
            if override is not None:
                if not -num_waypoints <= override < num_waypoints:
                    raise ValueError("PD waypoint index is outside the predicted trajectory.")
                return int(override) % num_waypoints
            distance = np.abs(times - times[-1] / 2)
            return int(np.flatnonzero(np.isclose(distance, distance.min(), rtol=1e-12, atol=1e-12))[0])
        limit = self._pd_lookahead_limit()
        override = self.controller_cfg.pd.speed_waypoint_index
        if override is not None:
            index = int(override)
            if not -num_waypoints <= index < num_waypoints:
                raise ValueError("PD waypoint index is outside the predicted trajectory.")
            index %= num_waypoints
            if times[index] > limit + 1e-12:
                raise ValueError("PD waypoint override exceeds the lookahead cap.")
            return index
        eligible = np.flatnonzero(times <= limit + 1e-12)
        if not len(eligible):
            raise ValueError("No predicted waypoint is within the PD lookahead cap.")
        target = min(float(times[-1]), limit)
        distance = np.abs(times[eligible] - target)
        # Float rounding must not choose the later point in an exact time tie.
        nearest = np.flatnonzero(np.isclose(distance, distance.min(), rtol=1e-12, atol=1e-12))[0]
        return int(eligible[nearest])

    def _pd_control(self, waypoints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the configured speed and heading targets for this trajectory."""
        count = waypoints.shape[1]
        index = self._pd_speed_index(count)
        heading_override = self.controller_cfg.pd.heading_waypoint_index
        if self.controller_cfg.pd.waypoint_mode == "half_horizon":
            heading_index = count - 1 if heading_override is None else int(heading_override)
            if not -count <= heading_index < count:
                raise ValueError("PD heading index is outside the predicted trajectory.")
            heading_index %= count
        else:
            heading_index = index
        if self.controller_cfg.pd.waypoint_mode == "lookahead" and heading_override is not None:
            if not -count <= heading_override < count or heading_override % count != index:
                raise ValueError("Linear and angular velocity must use the same PD waypoint.")
        self._controller.speed_waypoint_index = index
        self._controller.heading_waypoint_index = heading_index
        self.t_mid = self._pd_time_divisor(count)
        return self._controller(waypoints[..., :2], dt=self.t_mid)

    def _derive_t_mid(self) -> float:
        """Controller target time [s] derived from the model metadata."""
        return self._t_mid_for(self._num_waypoints() or 0)

    # --- Shared helpers -----------------------------------------------------------

    @staticmethod
    def _make_session(onnx_path: str, *, disable_graph_opt: bool = False) -> ort.InferenceSession:
        """Create an ONNX Runtime session, preferring the CUDA execution provider.

        Args:
            onnx_path: Path to the ONNX model.
            disable_graph_opt: Disable graph optimization (needed by some models).

        Returns:
            The created inference session.
        """
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        providers = [
            ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
            "CPUExecutionProvider",
        ]
        if disable_graph_opt:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            return ort.InferenceSession(onnx_path, sess_options=options, providers=providers)
        return ort.InferenceSession(onnx_path, providers=providers)

    @staticmethod
    def _obs_to_numpy(obs: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert an observation batch to a numpy array."""
        if isinstance(obs, torch.Tensor):
            return obs.cpu().numpy()
        return np.asarray(obs, dtype=np.float32)

    def _to_torch(self, array: np.ndarray) -> torch.Tensor:
        """Move a numpy array to the model device as float32."""
        return torch.from_numpy(array).float().to(self.device)

    @staticmethod
    def _select_best(
        trajectory: np.ndarray, scores: np.ndarray | None = None
    ) -> np.ndarray:
        """Select the best trajectory mode. Override for multimodal models."""
        return trajectory[:, 0]

    def _waypoint_dt(self, num_waypoints: int) -> torch.Tensor:
        """Per-waypoint timestep [s] between adjacent waypoints.

        Args:
            num_waypoints: Number of predicted waypoints.

        Returns:
            Length-``num_waypoints`` tensor of local steps; element ``k`` is the
            time between waypoints ``k-1`` and ``k`` (element 0 is the time to the
            first waypoint). For an explicit origin at time zero, element 0
            uses the first positive interval to keep its zero speed finite.
        """
        times = self.waypoint_times(num_waypoints)
        steps = torch.empty_like(times)
        steps[0] = times[0]
        steps[1:] = times[1:] - times[:-1]
        if num_waypoints > 1:
            steps[0] = torch.where(times[0] > 0, times[0], steps[1])
        else:
            steps[0] = torch.where(times[0] > 0, times[0], torch.ones_like(times[0]))
        return steps

    def _waypoints_with_speed(self, best_traj: np.ndarray) -> torch.Tensor:
        """Return ego-frame waypoints with a speed column.

        The speed is the trajectory derivative: adjacent-waypoint displacement
        divided by the local per-step timestep [m/s]. When the trajectory already
        carries a speed column, it is returned unchanged.
        """
        waypoints = self._to_torch(best_traj)
        if waypoints.shape[-1] >= 3:
            return waypoints
        origin = torch.zeros_like(waypoints[:, :1])
        prev = torch.cat([origin, waypoints[:, :-1]], dim=1)
        delta = torch.linalg.vector_norm(waypoints[..., :2] - prev[..., :2], dim=-1)
        speed = delta / self._waypoint_dt(waypoints.shape[1])
        return torch.cat([waypoints, speed.unsqueeze(-1)], dim=-1)

    # --- Interface -----------------------------------------------------------------

    def inference_trajectory(self, obs: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Run the model and return ``(trajectory, scores)``. Must be implemented."""
        raise NotImplementedError

    def inference_vw(
        self,
        obs: Any,
        robot: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Run the model and convert the best trajectory to ``(v, w)``.

        Args:
            obs: Model observation batch.
            robot: IsaacLab articulation handle (required by pure pursuit).
            **kwargs: Model-specific inference arguments.

        Returns:
            ``(vw, best_traj)`` where ``vw`` is a ``[B, 2]`` tensor and
            ``best_traj`` is the selected numpy trajectory.
        """
        trajectory, scores = self.inference_trajectory(obs, **kwargs)
        best_traj = self._select_best(trajectory, scores)
        if self.controller_cfg.type == "pp":
            waypoints = self._waypoints_with_speed(best_traj)
            self._pp.waypoint_index = (waypoints.shape[1] - 1) // 2
            v, w = self._pp.step(waypoints, robot=robot)
        else:
            v, w = self._pd_control(self._to_torch(best_traj))
        return torch.stack([v, w], dim=1), best_traj

    def reset(self, env_idx: int | Sequence[int] | None = None) -> None:
        """Reset per-episode controller and model state.

        Called by the environment when an episode ends. Resets only ``env_idx``
        when provided; otherwise resets all environments.

        Args:
            env_idx: Environment indices to reset (``None`` = all).
        """
        self._controller.reset(env_idx)
        self._reset_state(env_idx)

    def _reset_state(self, env_idx: int | Sequence[int] | None = None) -> None:
        """Hook for subclasses to reset model-specific state."""
        pass
