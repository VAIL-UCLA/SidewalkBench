# InternVLA-N1 (DualVLN) navigator

InternVLA-N1 (DualVLN) is a language-instruction navigation model: an S2 VLM
(Qwen2.5-VL) emits a pixel goal or a discrete action, and an S1 diffusion policy
(NextDiT, async) turns the S2 latent + RGB into a continuous trajectory.

This directory contains the SidewalkBench wrapper (`inference.py`), the model
spec sheet (`model_info.yaml`), and the checkpoint downloader
(`download_checkpoints.sh`). The upstream InternNav code is vendored as a git
submodule at `third_party/InternNav`.

The wrapper restores SidewalkBench's earlier always-S1 pipeline. Every control
step runs S2 on the current RGB frame, extracts the complete output's latent,
and calls S1 diffusion for a continuous trajectory. This applies to coordinates,
`STOP`, arrow tokens, and unrecognized text. S1 receives two copies of the current
RGB frame and zero depth.

## Folder layout

```
third_party/InternNav/                 ← git submodule (InternRobotics/InternNav)
└── internnav/...                       ← upstream model + agent code

sidewalkbench/models/InternVLA_N1/
├── README.md                 ← this file
├── model_info.yaml           ← spec sheet (resolution, context, pipeline params)
├── inference.py              ← DualVLNNavigator (BaseModel interface)
├── download_checkpoints.sh   ← fetches the two checkpoints into ckpts/
└── ckpts/                                ← checkpoints live with the model
    ├── InternVLA-N1-DualVLN/                       ← HF model dir (~8 GB)
    └── depth_anything_v2_metric_hypersim_vits.pth  ← depth backbone (~95 MB)
```

The wrapper passes the absolute `ckpts/` path to InternNav at runtime, so runs
are independent of the current working directory.

## Installation

1. Fetch the submodule (and its own submodules `LongCLIP` and `diffusion-policy`):

   ```bash
   git submodule update --init --recursive third_party/InternNav
   ```

   Set `INTERNNAV_ROOT=/path/to/InternNav` to use a checkout elsewhere.

2. Dependencies. InternVLA-N1 targets `transformers==4.51.0` / `diffusers==0.33.1`.
   These must stay on transformers 4.x because **MotionLCM's human-animation text
   encoder uses `sentence-transformers`, whose 6.x release forces transformers 5**.
   The project pins the consistent set in `pyproject.toml`:

   | Package | Pin |
   |---|---|
   | `transformers` | `==4.51.0` |
   | `diffusers` | `==0.33.1` |
   | `sentence-transformers` | `>=4.1,<6` (MotionLCM requirement) |
   | `accelerate` | present (needed for `from_pretrained(device_map=...)`) |

   Apply with `uv sync`. `flash-attn` is **not required** (the wrapper loads the
   model with `attn_implementation="sdpa"`).

3. Checkpoints (downloaded into `sidewalkbench/models/InternVLA_N1/ckpts/`):

   ```bash
   bash sidewalkbench/models/InternVLA_N1/download_checkpoints.sh
   # optional: bash .../download_checkpoints.sh /custom/target
   ```

   Optional env vars: `HF_TOKEN`, `HF_ENDPOINT` (e.g. `https://hf-mirror.com`),
   `NO_HF_TRANSFER=1`, `PYTHON=/path/python`.

## Usage in the benchmark

`configs/long_horizon/InternVLA_N1.yaml` (and the `unit_test` /
`pedestrian_reactive` variants) select the model via `model_type: InternVLA_N1`.
`env_setup.py` sets the fixed instruction, "Follow the sidewalk." The wrapper
consumes the latest RGB frame and uses zero depth. `bind_env` remains available
for compatibility with the evaluation scripts.

## Standalone API

```python
from sidewalkbench.models.InternVLA_N1.inference import DualVLNNavigator

nav = DualVLNNavigator(device="cuda:0")           # defaults to this model's ckpts/InternVLA-N1-DualVLN
nav.set_instruction("Follow the sidewalk.")

traj, scores = nav.inference_trajectory(obs)       # (B, 1, N, 2), (B, 1)
vw, traj = nav.inference_vw(obs)                   # (B, 2), (B, N, 2)
nav.reset()                                        # between episodes
```

`obs` is the zoo-shaped `(B, context, 3, H, W)` float in `[0, 1]`; only the last
frame is consumed. The trajectory is in the robot frame (`+x` forward, `+y` left,
meters) and is converted to `(v, w)` by the shared PD controller.

## Behaviour notes

- **Always S1.** `generate_latents` and `generate_traj` run after every S2 output.
  `traj_to_actions(..., use_discrate_action=False)` decodes continuous waypoints.
  A `STOP` token therefore still produces an S1 trajectory, matching the earlier
  benchmark implementation.
- **Current frame.** S2 uses one frame at 384x384. S1 uses that same image twice
  at 224x224, with a zero depth tensor of shape `(1, 2, 224, 224, 1)`.
- **Shared PD.** The decoder prepends the origin to 32 future points. Since the
  model has no published waypoint times, metadata declares a nominal 2 s
  benchmark horizon. Both velocities track index 16 at the nominal 1 s time,
  with `T=1` s. These times are a benchmark convention for the untimed S1 path.
- **Batch and reset.** Each environment runs the same S2 → latent → S1 sequence.
  `reset(env_idx)` clears only the selected environments' PD smoothing state.

## Gotchas

- **CWD-independent** — `internvla_n1_arch.py`'s `MODEL_PATH_TO` is redirected to the
  absolute `<model_dir>/ckpts` at import time, so the DepthAnything backbone resolves
  regardless of where the process is launched.
- `INTERNNAV_ROOT` (or the `repo_root=` kwarg) overrides the vendored checkout.
- `setuptools<81` is only needed when installing InternNav as a package
  (`pkg_resources` via LongCLIP); importing the model classes from the submodule
  does not require it.
