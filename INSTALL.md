
## Installation

1. Clone the repository with its submodules (IsaacLab 3.0.0-EA, MotionLCM, and InternNav):
   ```shell
   git clone --recurse-submodules --shallow-submodules https://github.com/VAIL-UCLA/SidewalkBench.git
   ```
   Run the remaining steps from `SidewalkBench/`.

2. Install dependencies (creates/syncs the `uv` environment):
   ```shell
   uv sync
   ```
   This compiles nvdiffrast, gsplat and RVO2 from source, and needs a CUDA toolkit of the same major version as
   PyTorch (CUDA 12; 12.8 recommended), found via `CUDA_HOME`, `nvcc` on `PATH`, or `/usr/local/cuda`.

3. Download the MotionLCM assets into `third_party/MotionLCM/`. Only the text-to-motion
   generation assets are required (`git-lfs`, `gdown`, and `unzip` must be installed):

   ```shell
   cd third_party/MotionLCM
   uv run bash prepare/prepare_t5.sh                  # deps/sentence-t5-large
   uv run bash prepare/download_pretrained_models.sh  # experiments_t2m/motionlcm_humanml/motionlcm_humanml.ckpt
   uv run bash prepare/prepare_tiny_humanml3d.sh      # datasets/humanml3d (Mean.npy, Std.npy)
   cd -
   ```


4. Download the asset bundle into `assets/` from the project root:

   ```shell
   uv run hf download UCLA-VAIL/SidewalkBench-Assets --repo-type dataset --local-dir assets
   ```

   For the real-world scanned scenes, download the assets into
   `gs_assets/`:

   ```shell
   uv run hf download UCLA-VAIL/SidewalkBench-GS-Assets --repo-type dataset --local-dir gs_assets
   ```

5. Convert the bundled texture paths to absolute paths for this machine. The RTX
   MDL material conversion does not resolve texture paths relative to the USD layer,
   so this step is required after downloading the bundle:

   ```shell
   uv run python -m sidewalkbench.tools.absolutize_asset_paths
   ```

6. Download the model checkpoints before evaluation using the
   [public checkpoint download instructions](#download-public-checkpoints) below.

The SMPL body model (`SMPL_NEUTRAL.pkl`, `smpl_uv.obj`) is license-restricted and
is not part of the bundle. Download it from the [official SMPL website](https://smpl.is.tue.mpg.de/) and
place it under `assets/smplx/`.

The SMPLitex texture (`smplx/SMPLitex-textures/*.png`) is likewise not
included because it comes from another public dataset. Download it from [the
official SMPLitex project page](https://dancasas.github.io/projects/SMPLitex/) and place it under `assets/smplx/SMPLitex-textures/`.

## Models
The benchmark models are CityWalker-PG-Official, ViNT, NoMaD, MBRA,
MIMIC, S2E, InternVLA-N1, FlowPilot-GL-Official, and Cosmos3-Official. The OpenPilot checkpoint is currently not publically available.

### Download public checkpoints

From the repository root, download the checkpoints for all nine current
benchmark models:

```shell
uv run python -m sidewalkbench.models.download_zoo
```

This downloads ViNT, NoMaD, MBRA, CityWalker, InternVLA-DualVLN (`InternVLA_N1`),
Cosmos3, MIMIC, S2E, and FlowPilot. 

To download a subset, pass the model names explicitly:

```shell
uv run python -m sidewalkbench.models.download_zoo --models ViNT NoMaD FlowPilot
```

The ONNX models come from the [public navigation model zoo](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public).
InternVLA uses the [DualVLN checkpoint](https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN).

