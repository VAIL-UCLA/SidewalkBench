"""GS variant options and output paths, usable without launching Isaac Sim."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

from sidewalkbench.pg.primitives.robot.registry import ROBOT_PROFILES


DEFAULTS = dict(model_type="FlowPilot", scene="beverly_hills_crosswalk", exp_name="gs", seed=42,
                mode="short_horizon", robot_type="coco", use_goal=False,
                goal_segment_length=30.0, goal_advance_margin=1.0, model_hz=None)


def add_variant_arguments(parser):
    parser.add_argument("--config", help="Optional PG-style YAML: model_type, robot_type, use_goal, controller, pd, pp.")
    for key in ("model_type", "scene", "exp_name"):
        parser.add_argument(f"--{key}", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", choices=("short_horizon", "long_horizon", "cocoa"), default=None)
    parser.add_argument("--model_hz", type=float, default=None, help="Policy Hz; default: model's input rate.")
    parser.add_argument("--robot_type", choices=tuple(ROBOT_PROFILES), default=None)
    parser.add_argument("--use_goal", action=argparse.BooleanOptionalAction, default=None,
                        help="Feed the annotated route goal to point/image-goal models (default: disabled).")
    parser.add_argument("--goal_segment_length", type=float, default=None, help="Long-horizon goal spacing (default: 30 m).")
    parser.add_argument("--goal_advance_margin", type=float, default=None, help="Goal advancement margin (default: 1 m).")
    parser.add_argument("--controller", choices=("pd", "pp"), default=None)
    parser.add_argument("--pd_waypoint_mode", choices=("half_horizon", "lookahead"), default=None)
    parser.add_argument("--pd_lookahead_s", type=float, default=None)
    parser.add_argument("--max_v", type=float, default=None, help="PD maximum forward velocity (m/s).")
    parser.add_argument("--max_w", type=float, default=None, help="PD maximum angular velocity (rad/s).")


def resolve_arguments(args) -> dict:
    """CLI overrides YAML; omitted options retain the existing GS defaults."""
    config = yaml.safe_load(Path(args.config).expanduser().read_text()) if args.config else {}
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("--config must contain a YAML mapping")
    settings = {key: getattr(args, key) if getattr(args, key) is not None else config.get(key, default)
                for key, default in DEFAULTS.items()}
    if settings["robot_type"] not in ROBOT_PROFILES:
        raise ValueError(f"Unknown robot_type: {settings['robot_type']!r}")
    if settings["mode"] not in ("short_horizon", "long_horizon", "cocoa"):
        raise ValueError(f"Unknown mode: {settings['mode']!r}")
    if not isinstance(settings["use_goal"], bool):
        raise ValueError("use_goal must be true or false")
    for key in ("goal_segment_length", "goal_advance_margin", "model_hz"):
        if settings[key] is None and key == "model_hz":
            continue
        value = float(settings[key])
        if not math.isfinite(value) or value < 0 or (key != "goal_advance_margin" and value == 0):
            raise ValueError(f"{key} must be finite and {'nonnegative' if key == 'goal_advance_margin' else 'positive'}")
        settings[key] = value
    if settings["goal_advance_margin"] >= settings["goal_segment_length"]:
        raise ValueError("goal_advance_margin must be less than goal_segment_length")
    for key in ("controller", "pd", "pp", "pp_action_dt"):
        if key in config:
            settings[key] = config[key]
    if args.controller is not None:
        settings["controller"] = args.controller
    for arg, key in (("pd_waypoint_mode", "waypoint_mode"), ("pd_lookahead_s", "lookahead_s"),
                     ("max_v", "max_v"), ("max_w", "max_w")):
        if getattr(args, arg) is not None:
            if not isinstance(settings.get("pd", {}), dict):
                raise ValueError("pd must be a mapping")
            settings["pd"] = {**settings.get("pd", {}), key: getattr(args, arg)}
    for key in DEFAULTS:
        setattr(args, key, settings[key])
    return settings


def variant_suffix(settings: dict) -> str:
    """Readable selection plus a stable digest of explicit variant parameters."""
    changed = {k: v for k, v in settings.items()
               if k in ("controller", "pd", "pp", "pp_action_dt") or
               (k in ("robot_type", "use_goal", "goal_segment_length", "goal_advance_margin", "model_hz")
                and v != DEFAULTS[k])}
    if not changed:
        return ""
    digest = hashlib.sha256(json.dumps(changed, sort_keys=True, allow_nan=False).encode()).hexdigest()[:10]
    return f"_{settings['robot_type']}_{'goal' if settings['use_goal'] else 'goalless'}_{settings.get('controller', 'pd')}_{digest}"


def output_directory(settings: dict, root: str | Path) -> Path:
    tag = {"short_horizon": "gs", "long_horizon": "gs_long_horizon", "cocoa": "gs_cocoa"}[settings["mode"]]
    name = f"{settings['model_type']}_{settings['scene']}_{tag}_0_{settings['seed']}" + variant_suffix(settings)
    return Path(root) / "experiments" / settings["exp_name"] / name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print a GS run's output directory without starting the simulator.")
    add_variant_arguments(parser)
    parser.add_argument("--print_field", choices=("output_dir", "exp_name", "mode"), default="output_dir")
    args, _ = parser.parse_known_args()
    settings = resolve_arguments(args)
    print(output_directory(settings, Path(__file__).resolve().parents[2])
          if args.print_field == "output_dir" else settings[args.print_field])
