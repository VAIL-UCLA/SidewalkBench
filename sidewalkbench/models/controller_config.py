"""Build the same controller from PG or GS YAML settings."""

from collections.abc import Mapping
from dataclasses import fields
import math
from pathlib import Path

import yaml

from .base_model import ControllerCfg
from .pd_controller import PDControllerCfg
from .pure_pursuit import PurePursuitConfig


def _params(value, cls, key):
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {key} setting(s): {', '.join(sorted(unknown))}")
    return dict(value)


def controller_from_config(config: Mapping, model_type: str) -> ControllerCfg | None:
    """None preserves the wrapper's defaults, including model timing metadata."""
    if not any(key in config for key in ("controller", "pd", "pp", "pp_action_dt")):
        return None
    kind = config.get("controller") or "pd"
    if kind not in ("pd", "pp"):
        raise ValueError("controller must be 'pd' or 'pp'")
    info_path = Path(__file__).parent / model_type / "model_info.yaml"
    info = yaml.safe_load(info_path.read_text()) if info_path.is_file() else {}
    pd = PDControllerCfg(**{**info.get("pd_controller", {}), **_params(config.get("pd", {}), PDControllerCfg, "pd")})
    pp = PurePursuitConfig(**_params(config.get("pp", {}), PurePursuitConfig, "pp"))
    if pd.waypoint_mode not in ("half_horizon", "lookahead"):
        raise ValueError("pd.waypoint_mode must be 'half_horizon' or 'lookahead'")
    for key in ("lookahead_s", "max_v", "max_w"):
        value = getattr(pd, key)
        if not math.isfinite(value) or value < 0 or (key == "lookahead_s" and value == 0):
            raise ValueError(f"pd.{key} must be finite and {'positive' if key == 'lookahead_s' else 'nonnegative'}")
    if pd.dt is not None and (not math.isfinite(pd.dt) or pd.dt <= 0):
        raise ValueError("pd.dt must be finite and positive")
    for key in ("lookahead_base", "max_lookahead", "wheelbase", "max_steering_angle"):
        value = getattr(pp, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"pp.{key} must be finite and positive")
    if not math.isfinite(pp.lookahead_gain) or pp.lookahead_gain < 0:
        raise ValueError("pp.lookahead_gain must be finite and nonnegative")
    if pp.max_lookahead < pp.lookahead_base:
        raise ValueError("pp.max_lookahead must be >= pp.lookahead_base")
    action_dt = float(config.get("pp_action_dt", 3.0))
    if not math.isfinite(action_dt) or action_dt <= 0:
        raise ValueError("pp_action_dt must be finite and positive")
    return ControllerCfg(type=kind, pd=pd, pp=pp, pp_action_dt=action_dt)
