# Crazyflie mocap and visual matching

Align a reconstructed room to Crazyflie footage using motion capture and shared
visual features. Refine each camera pose, render the 3D Gaussian splatting scene
with the camera's lens, and fit the simulated appearance to the recordings.

The pipeline produces corrected poses, grayscale simulation images, feature and
image error measurements, and Real / Sim / Overlay videos. Recorded images keep
their native **320 × 240** pixel geometry.

**Start with the [illustrated process guide](docs/master_pipeline_guide.pdf).**
For implementation details, see [the algorithms](docs/METHODS.md),
[input and output formats](docs/FILES.md), and [the workflow guide](docs/WORKFLOW.md).

## Pipeline

```text
Recordings + mocap + camera calibration + 3DGS scene
                         ↓
Synchronize clocks and interpolate camera poses
                         ↓
Initialize the camera-to-scene registration
                         ↓
Match fixed room features and review correspondences
                         ↓
Fit the global transform, then correct each observed frame
                         ↓
Render with the Crazyflie lens model
                         ↓
Fit exposure, tone, spatial and view-dependent brightness
                         ↓
Export poses, corrected images, metrics and comparison videos
```

Each observed frame receives its own pose search. The output labels unsupported
frames as interpolated or global-only. Appearance fitting follows pose alignment.

## Installation

Use Linux with Python 3.11. Full scene rendering requires an NVIDIA GPU, a working
CUDA driver and a compatible CUDA toolkit/compiler. Install FFmpeg and DejaVu
fonts for video and document output.

```bash
conda env create -f environment.yml
conda activate crazyflie-alignment
python -m pip install -r requirements-torch-cu128.txt
python -m pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` includes the CPU dependencies and pinned renderer/matcher
versions. gsplat compiles CUDA kernels on the first render.

For CPU geometry, image correction and the included examples:

```bash
python -m pip install -r requirements.txt
python pipeline.py test
python pipeline.py smoke
```

The smoke command reproduces three saved example images and writes a comparison
sheet to `workspace/smoke/`.

## Configure inputs

Copy `configs/project.json` to `configs/local.json`, then set:

| Setting | Input |
|---|---|
| `data_dir`, `runs` | Recording directories containing camera frames, timestamps and drone mocap |
| `scene_dir`, `checkpoint` | Nerfstudio Splatfacto checkpoint and its scene transforms |
| `camera`, `lens` | Intrinsics, lens model and camera-to-body transform |
| `seed_registration`, `reference_atlas` | Initial room registration and known camera viewpoints |
| `exposure` | Exposure time and digital gain for each recording |
| `workspace_dir` | Output subdirectory, such as `workspace` |

Relative paths resolve from the repository root. Absolute input paths are also
supported. The included calibration and model profiles apply to the supplied
Crazyflie/arena examples; calibrate these inputs for another camera or scene.
See [file formats](docs/FILES.md) for the complete schema and directory layout.

```bash
export CF_MATCH_CONFIG="$PWD/configs/local.json"
python pipeline.py fetch-weights
python pipeline.py doctor --data
```

The environment check verifies input files, recording schemas and GPU readiness.
Model downloads are checked against recorded SHA-256 hashes.

## Run the pipeline

```bash
python pipeline.py prepare
python pipeline.py bootstrap
python pipeline.py features
python pipeline.py review --run run_20260925T122400 --open
```

Review each configured recording. The viewer shows the same feature ID and color
on both images, with a connecting line and magnified views. Press **A** to accept
or **R** to reject a candidate.

```bash
python pipeline.py align --track
python pipeline.py verify
python pipeline.py video
python pipeline.py appearance-export
```

`align` tracks accepted features, corrects poses and renders the scene.
`appearance-export` applies the configured frozen camera response and writes
corrected PNGs, image metrics and a comparison video.

To fit a new appearance response from the aligned recordings:

```bash
python pipeline.py appearance-fit
python pipeline.py appearance-export \
  --model workspace/output/captures/appearance_v3/selection.json
```

Use `--run RUN` for one recording on the per-run commands. `video --only-fitted`
exports independently fitted frames. `--dry-run` before a command shows its
execution plan. The [workflow guide](docs/WORKFLOW.md) covers review, temporal
splits, resuming and troubleshooting.

## Find the outputs

Results are stored under `workspace/output/captures/` by default.

| Result | Location |
|---|---|
| Raw images and timestamps | `frames/RUN/real_raw/`, `frames/RUN/raw_timing.npz` |
| Timing and pose validity | `sync/`, `prepared/RUN/timing.json` |
| Initial scene/camera registration | `bootstrap/` |
| Reviewed features | `RUN/annotations.json` |
| Global registration | `RUN/fixed_features/registration/` |
| Per-frame poses and fit status | `RUN/fixed_features/per_frame/RUN/poses_per_frame.csv` and `.npz` |
| Aligned renders | `RUN/fixed_features/per_frame/RUN/render/RUN/` |
| Alignment video | `RUN/alignment_all_accepted.mp4` |
| Appearance-corrected images, scores and video | `appearance_export/RUN/` |
| Stage logs | `journal.jsonl`, `RUN/logs/` |

Join records by run and frame `index`; the original `frame_id` and camera time are
also retained. A frame's fit status tells you whether its pose was independently
corrected or filled from nearby constraints.

## Documentation and development

- [Illustrated process guide](docs/master_pipeline_guide.pdf): the method, equations and real examples.
- [Methods](docs/METHODS.md): loss functions, search schedules and metric definitions.
- [File reference](docs/FILES.md): configuration, input schemas and output arrays.
- [Workflow](docs/WORKFLOW.md): commands, review, fitting and troubleshooting.
- [Validation](verification/README.md): numerical and integration checks.

Rebuild the process guide with `python tools/build_master_guide.py`. This requires
pdfLaTeX, Latin Modern fonts and the Python dependencies in `requirements.txt`.
Create a source archive with `python tools/package_source.py` after adding the
source files to Git.
