# Workflow guide

Run commands from the repository root after installation and configuration as
shown in the [README](../README.md).

## 1. Check the environment and inputs

```bash
export CF_MATCH_CONFIG="$PWD/configs/local.json"
python pipeline.py fetch-weights
python pipeline.py doctor --data
```

The environment check reports files, schemas, packages, FFmpeg and CUDA
availability. Full rendering requires a working GPU driver and a compatible CUDA
compiler for gsplat's first kernel build. On Ubuntu, system utilities include:

```bash
sudo apt-get install ffmpeg fonts-dejavu-core build-essential libgl1 libglib2.0-0
```

`fetch-weights` downloads the configured matchers and tracker, verifies their
hashes, and reuses verified files. Once dependencies and weights are available,
processing can run offline.

## 2. Prepare and initialize

```bash
python pipeline.py --dry-run prepare
python pipeline.py prepare
python pipeline.py bootstrap
```

`prepare` decodes the native camera frames, fits the camera clock, estimates
image-to-mocap delay and records pose validity. `bootstrap` matches keyframes to
known scene viewpoints, fits scene and camera-axis corrections, rechecks timing
and applies the corrected initialization. Inspect its preview sheets under
`workspace/output/captures/bootstrap/`.

When the scene registration and camera mount are independently calibrated, use
`bootstrap --use-seed` to initialize directly from that calibration. The normal
bootstrap estimates a mount rotation while retaining the configured lever arm.
Recalibrate the full camera-to-body transform after changing a rigid body's pivot.

## 3. Generate and review room features

```bash
python pipeline.py features
python pipeline.py review --run run_20260925T122400 --open
```

Review each configured run. A candidate has the same ID and color in the real
and rendered views. Check its point, connecting line and magnified crops across
several frames. Accept with **A** or reject with **R**. Favor distinct corners on
fixed room structure, spread across the image and visible from several positions.

```bash
python pipeline.py align --track
```

Alignment uses the confirmed points, propagates observations with CoTracker3,
compares global registrations, extends supported tracks, searches every observed
frame independently and renders the corrected poses. The global search uses up
to 120 distributed observed frames; per-frame search uses each frame's available
observations.

For automatic processing after geometric filtering:

```bash
python pipeline.py align --track --accept-geometric
```

This option records batch admission in the annotations and preserves previous
rejections. Inspect its visual output carefully around repeated patterns and
objects that may have moved.

## 4. Inspect geometry

```bash
python pipeline.py verify
python pipeline.py video
python pipeline.py video --only-fitted
```

`verify` measures residual shifts of gradient patches at fitted poses. Inspect
these alongside feature reprojection errors and Real / Sim / Overlay videos.
The per-frame CSV and NPZ identify `fitted`, `interpolated` and `global` poses.

`video` includes all accepted frames. `video --only-fitted` includes independently
corrected frames. `render` regenerates images from saved poses. Add `--run RUN`
to these commands to process one recording.

## 5. Apply or fit appearance

Apply the configured frozen response:

```bash
python pipeline.py appearance-export
```

Fit a shared response from the aligned recordings and export it:

```bash
python pipeline.py appearance-fit
python pipeline.py appearance-export \
  --model workspace/output/captures/appearance_v3/selection.json
```

Fitting stages run in this order: `exposure`, `tone`, `lighting`, `run-gain`,
`v2-check`, `camera`, `v3-check`. Resume a stage with
`appearance-fit --stage STAGE`. A reserved-check result freezes that fitting
stage; select a fresh workspace before developing a revised model.

### Temporal split requirements

The supplied fitting protocol uses informative multi-run recordings with varied
viewpoints and known exposure/gain. It assigns 10-second blocks to training,
development and checking, with one-second guards. Refinement stages reserve
windows around `40 + 50*n` and `50 + 50*n` seconds, each with a 0.65-second guard.
Each configured run needs usable frames in the required partitions.

For short clips, change the protocol explicitly and reserve suitable frames
before fitting. Evaluate camera poses separately: the appearance split measures
image agreement conditional on the fitted poses. Transfer to another scene or
camera requires its own validation.

### Apply the response in Python

```python
import json
import sys
from pathlib import Path

repo = Path('/path/to/crazyflie-mocap-visual-matching')
sys.path.insert(0, str(repo / 'scripts'))
from fit_appearance import apply

model = json.loads((repo / 'configs/appearance_selection.json').read_text())['selected']
# rgb: uint8 RGB [240, 320, 3]
# R_GC: 3 x 3 camera-to-scene rotation
fitted = apply(
    rgb, model, run=None, rotation=R_GC,
    relative_exposure=(exposure_ms * digital_gain) / (8.33 * 1.5),
)
```

The result is a grayscale float image on [0, 1]. `run=None` uses unit residual
run gain. The response requires the simulated image, camera orientation and
exposure/gain settings; paired real images are needed during fitting and scoring.

## 6. Resume and manage outputs

All results stay under the configured `workspace_dir`. Keep a separate workspace
for each change to input recordings, scene, reviewed features or model settings.
The CLI fingerprints code, configuration and calibration in
`workspace/pipeline_inputs.json` and detects incompatible cached results.
Stage files allow completed work to be reused under the same configuration.

The CLI uses the active Python interpreter for child processes. For an advanced
per-run stage, configure its paths through the wrapper:

```bash
python scripts/run_stage.py run_20260925T122400 \
  per_frame_registration --help
```

## 7. Troubleshooting

| Symptom | Action |
|---|---|
| CUDA unavailable | Check the driver and GPU visibility with `nvidia-smi`. |
| Missing MINIMA checkpoint | Run `fetch-weights`, or set `cross_matcher` to `stock` explicitly. |
| Missing scene transforms | Supply `dataparser_transforms.json` and `world_frame.json` from the checkpoint export. |
| Too few timing pairs | Use a segment with camera translation, visible corners and broad track coverage. |
| Bootstrap has poor overlap | Check atlas viewpoints, lens calibration, seed registration and camera extrinsics. |
| Too few confirmed features | Review more distinct fixed room points across the trajectory. |
| GPU memory exhausted | Reduce competing GPU use; the default rendering workload was used with 16 GB VRAM. |
| No reserved appearance samples | Check capture duration, pose validity and the temporal partitions. |
| Good feature residuals but poor image agreement | Inspect repeated-edge matches, feature coverage, changed objects and lighting. |

## 8. Validate and build documentation

```bash
python pipeline.py test
python pipeline.py smoke
python tools/verify_package.py
python tools/build_master_guide.py
```

Tests cover time and coordinate conventions, image metrics, saved-output parity,
configuration paths and image/video export. The smoke examples provide a quick
installation check. Add `--match` or `--render` to exercise the image matcher or
CUDA renderer with configured weights and scene assets.

The PDF builder uses the packaged images and LaTeX source. Install pdfLaTeX with
Latin Modern fonts, for example `texlive-latex-extra` and `lmodern` on Ubuntu.
