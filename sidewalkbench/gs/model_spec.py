"""Per-model facts for the GS eval, read from the model's ``model_info.yaml``.

Two model sources; a name in both resolves to the PG release one:
  * PG release zoo (``sidewalkbench/models/<model>/``, vendored from origin/release/pg; weights via
    ``python sidewalkbench/models/download_zoo.py``): FlowPilot, NoMaD, ViNT, MIMIC, ... Run as PG
    release runs them: ``inference_vw(obs, robot=robot)``, ``reset(env)`` at every route start /
    recovery, the policy at 1 / control_dt, else 4 Hz for NoMaD / ViNT, else 5 Hz (env_setup.py).
  * ``model_scripts/models/<model>/`` (the older zoo; not part of the PG release, absent here).

The eval code has no model-specific branches: everything that differs between
models is a card field, with zoo-wide defaults that equal PG's else-branches
(random_env_pretrain.py) so a card that says nothing runs exactly as PG runs it.

  image_resolution  [H, W] model input                                (required)
  context_size      frames fed to the model                           (required)
  input_fps         policy cadence in Hz; action held in between      (default 5.0 = PG decimation 40)
  camera            {fx, fy, cx, cy} at the 512x288 render            (default = PG's 790-intrinsics
                                                                       scaled: 210.667 / 256 / 144)
  input_crop        [y0, y1, x0, x1] on the 512x288 frame, pre-resize (default none)
  controller        "vw" | "pure_pursuit" (needs inference_vw_pp)     (default vw)
  recurrent         per-env recurrent state reset between routes      (default false)
  video_fps         PG save_video fps, 20 or 80                       (default 20)
  goal_type         metadata; PG wrappers declare their point/image interface on BaseModel.
                    GS can feed annotated goals when use_goal is enabled.

``python sidewalkbench/gs/model_spec.py --list`` shows every card and why a
model would or would not run; ``--runnable`` prints the runnable names.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass, field

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "model_scripts", "models")
ZOO_DIR = os.path.join(REPO_ROOT, "sidewalkbench", "models")
# PG release's get_model (sidewalkbench/pg/inference_scripts/get_model.py): name -> (module, class)
ZOO_CLASSES = {
    "CityWalker": ("CityWalker.inference", "CityWalkerNavigator"),
    "FlowPilot": ("FlowPilot.inference", "FlowPilotNavigator"),
    "Cosmos3": ("Cosmos3.inference", "Cosmos3Navigator"),
    "InternVLA_N1": ("InternVLA_N1.inference", "DualVLNNavigator"),
    "MBRA": ("MBRA.inference", "MBRAPGNavigator"),
    "MIMIC": ("MIMIC.inference", "MIMICNavigator"),
    "NoMaD": ("NoMaD.inference", "NoMaDGLNavigator"),
    "S2E": ("S2E.inference", "S2ENavigator"),
    "ViNT": ("ViNT.inference", "VintGLOfficialNavigator"),
}
ZOO_HZ = {"NoMaD": 4.0, "ViNT": 4.0}  # PG release decimation 50; others 40 (5 Hz) unless control_dt
ZOO_INSTRUCTION = {"InternVLA_N1": "Follow the sidewalk."}  # PG release env_setup.py
GET_MODEL_PY = os.path.join(REPO_ROOT, "sidewalkbench", "pg", "inference_scripts", "get_model.py")

RENDER_W, RENDER_H = 512, 288
DEFAULT_CAMERA = {"fx": 790.0 * RENDER_W / 1920, "fy": 790.0 * RENDER_W / 1920, "cx": RENDER_W / 2, "cy": RENDER_H / 2}
DEFAULT_INPUT_FPS = 5.0   # PG: decimation 40 at dt 0.005 for models without a special case
GOAL_LESS = ("goal_less", "goal_free")
WEIGHT_GLOBS = ("*.onnx", "*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.bin")
NON_MODELS = ("Keyboard",)


@dataclass
class ModelSpec:
    model_type: str
    model_dir: str
    image_resolution: tuple[int, int]
    context_size: int
    input_fps: float = DEFAULT_INPUT_FPS
    camera: dict = field(default_factory=lambda: dict(DEFAULT_CAMERA))
    input_crop: tuple[int, int, int, int] | None = None
    controller: str = "vw"
    recurrent: bool = False
    video_fps: float = 20.0
    goal_type: str = "unknown"
    registered: bool = False
    weights: list[str] = field(default_factory=list)
    source: str = "model_scripts"  # "pg_release" (sidewalkbench/models) | "model_scripts"


def registered_model_types() -> set[str]:
    """Model names get_model() knows (parsed from its if/elif chain)."""
    with open(GET_MODEL_PY, encoding="utf-8") as fh:
        return set(re.findall(r'model_type\s*==\s*["\']([^"\']+)["\']', fh.read()))


def get_zoo_model(model_type: str, controller=None, *, reset_history_speed_mps=None):
    """A PG release zoo model, built with the same controller settings as PG."""
    import importlib

    module, cls = ZOO_CLASSES[model_type]
    model = getattr(importlib.import_module(f"sidewalkbench.models.{module}"), cls)(controller=controller)
    if model_type == "CityWalker":  # as PG release env_setup.py (default reset_history_speed_mps)
        model.configure_reset_history(model._info.get("reset_history_speed_mps", 1.0)
                                      if reset_history_speed_mps is None else reset_history_speed_mps)
    if model_type in ZOO_INSTRUCTION:
        model.set_instruction(ZOO_INSTRUCTION[model_type])
    return model


def _zoo_hz(model_type: str) -> float:
    """PG release's policy rate: 1 / control_dt when the wrapper declares one, else ZOO_HZ / 5 Hz."""
    import ast

    module = os.path.join(ZOO_DIR, ZOO_CLASSES[model_type][0].replace(".", os.sep) + ".py")
    for node in ast.walk(ast.parse(open(module, encoding="utf-8").read())):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "control_dt" for t in node.targets):
            return 1.0 / float(ast.literal_eval(node.value))
    return ZOO_HZ.get(model_type, DEFAULT_INPUT_FPS)


def _load_zoo_spec(model_type: str) -> ModelSpec:
    model_dir = os.path.join(ZOO_DIR, model_type)
    with open(os.path.join(model_dir, "model_info.yaml"), encoding="utf-8") as fh:
        info = yaml.safe_load(fh) or {}
    files = info.get("checkpoint_files") or {}
    return ModelSpec(
        model_type=model_type,
        model_dir=model_dir,
        image_resolution=(int(info["image_resolution"][0]), int(info["image_resolution"][1])),
        context_size=int(info["context_size"]),
        input_fps=_zoo_hz(model_type),
        goal_type=str(info.get("goal_type", "unknown")).replace("-", "_"),
        registered=True,
        weights=sorted(f for f in files if os.path.isfile(os.path.join(model_dir, f))) if len(files) and all(
            os.path.isfile(os.path.join(model_dir, f)) for f in files) else [],
        source="pg_release",
    )


def load_model_spec(model_type: str) -> ModelSpec:
    if model_type in ZOO_CLASSES and os.path.isfile(os.path.join(ZOO_DIR, model_type, "model_info.yaml")):
        return _load_zoo_spec(model_type)
    model_dir = os.path.join(MODELS_DIR, model_type)
    card = os.path.join(model_dir, "model_info.yaml")
    if not os.path.isfile(card):
        raise FileNotFoundError(f"no model card: {card}")
    with open(card, encoding="utf-8") as fh:
        info = yaml.safe_load(fh) or {}
    for key in ("image_resolution", "context_size"):
        if key not in info:
            raise ValueError(f"{card}: missing required field {key!r}")
    cam = dict(DEFAULT_CAMERA)
    if isinstance(info.get("camera"), dict):
        cam.update({k: float(info["camera"][k]) for k in ("fx", "fy", "cx", "cy") if k in info["camera"]})
    crop = info.get("input_crop")
    controller = str(info.get("controller", "vw"))
    if controller not in ("vw", "pure_pursuit"):
        raise ValueError(f"{card}: controller must be 'vw' or 'pure_pursuit', got {controller!r}")
    weights = sorted(os.path.basename(p) for g in WEIGHT_GLOBS for p in glob.glob(os.path.join(model_dir, g)))
    return ModelSpec(
        model_type=model_type,
        model_dir=model_dir,
        image_resolution=(int(info["image_resolution"][0]), int(info["image_resolution"][1])),
        context_size=int(info["context_size"]),
        input_fps=float(info.get("input_fps", DEFAULT_INPUT_FPS)),
        camera=cam,
        input_crop=tuple(int(v) for v in crop) if crop else None,
        controller=controller,
        recurrent=bool(info.get("recurrent", False)),
        video_fps=float(info.get("video_fps", 20.0)),
        goal_type=str(info.get("goal_type", "unknown")),
        registered=model_type in registered_model_types(),
        weights=weights,
    )


def runnable_reason(spec: ModelSpec) -> str | None:
    """None if the model can run in the GS eval, else why not."""
    if spec.model_type in NON_MODELS:
        return "interactive, not a model"
    if not spec.registered:
        return "not registered in sidewalkbench/pg/inference_scripts/get_model.py"
    # PG release runs every zoo model goal-less by default (use_goal: false): point-goal wrappers sample a
    # random goal, image-goal ones a noise image, InternVLA_N1 gets ZOO_INSTRUCTION.
    if spec.source != "pg_release" and not any(
            g.lower().startswith(spec.goal_type.lower()) or spec.goal_type.lower().startswith(g) for g in GOAL_LESS):
        return f"goal_type={spec.goal_type!r} (only goal-less models run here)"
    if not spec.weights:
        return ("missing checkpoint files (python sidewalkbench/models/download_zoo.py --models "
                f"{spec.model_type})" if spec.source == "pg_release" else "no weights file in the model directory")
    return None


def list_models() -> list[tuple[str, ModelSpec | None, str | None]]:
    rows = []
    for d in sorted(set(ZOO_CLASSES) | set(os.listdir(MODELS_DIR) if os.path.isdir(MODELS_DIR) else ())):
        if d.startswith(".") or not (d in ZOO_CLASSES or os.path.isdir(os.path.join(MODELS_DIR, d))):
            continue
        try:
            spec = load_model_spec(d)
        except (FileNotFoundError, ValueError) as exc:
            rows.append((d, None, str(exc)))
            continue
        rows.append((d, spec, runnable_reason(spec)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="table of all cards and runnability")
    ap.add_argument("--runnable", action="store_true", help="print runnable model names (space-separated)")
    args = ap.parse_args()
    rows = list_models()
    if args.runnable:
        print(" ".join(name for name, spec, why in rows if spec is not None and why is None))
        return
    print(f"{'model':26} {'run?':5} {'goal':10} {'res':9} {'ctx':3} {'fps':5} {'ctrl':12} {'crop':4} {'rec':3} note")
    for name, spec, why in rows:
        if spec is None:
            print(f"{name:26} {'no':5} {'-':10} {'-':9} {'-':3} {'-':5} {'-':12} {'-':4} {'-':3} {why}")
            continue
        res = f"{spec.image_resolution[0]}x{spec.image_resolution[1]}"
        print(f"{name:26} {'yes' if why is None else 'no':5} {spec.goal_type[:10]:10} {res:9} {spec.context_size:<3} "
              f"{spec.input_fps:<5g} {spec.controller:12} {'yes' if spec.input_crop else '-':4} "
              f"{'yes' if spec.recurrent else '-':3} {why or ''}")


if __name__ == "__main__":
    main()
