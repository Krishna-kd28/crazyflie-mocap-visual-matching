# File paths, schemas and coordinate conventions

All relative **configuration paths** resolve from the repository
root. All generated output paths resolve under `workspace_dir`. Raw data,
calibration inputs and reconstruction files are read-only pipeline inputs.
An absolute input path is allowed; runtime output outside this package is not.

## Configuration

`configs/project.json` is the portable template. Set `--config path.json` before
the command, or export `CF_MATCH_CONFIG=/absolute/path/to/config.json`.

| Key | Meaning / required file |
|---|---|
| `workspace_dir` | New output subdirectory of this repo; default `workspace` |
| `data_dir` | Directory containing the configured run subdirectories |
| `runs` | Explicit list of run directory names; no automatic dataset discovery |
| `exposure` | `RUN: {exposure_ms, digital_gain}`; values come from recording notes |
| `scene_dir` | Nerfstudio scene containing `dataparser_transforms.json` and `world_frame.json` |
| `checkpoint` | Corresponding `nerfstudio_models/step-*.ckpt` file |
| `camera` | Camera intrinsics, archived distortion and assumed camera-to-body extrinsic |
| `lens` | Active empirical angle-model JSON used for projection/rendering |
| `seed_registration` | Initial mocap-to-metric-scene similarity |
| `reference_atlas` | NPZ with camera poses in that metric scene; bootstrap rerenders them |
| `nominal_meters_per_scene_unit` | Original scene metric conversion; 0.85 for this reconstruction |
| `minima_weights` | MINIMA SuperPoint-LightGlue checkpoint |
| `cotracker_weights` | CoTracker3 `scaled_offline.pth` checkpoint |
| `cross_matcher` | `minima` (default) or explicitly `stock`; no silent fallback |
| `appearance_model` | Frozen selected appearance model JSON or selection JSON |
| `baseline_appearance` | Reference grayscale response used in appearance comparison |
| `exclude_floor` | Whether the geometric feature filter excludes the known floor |
| `floor_plane_abc` | Floor plane `z = a*x + b*y + c`, in metric scene coordinates |
| `exclude_central_radius_m` | XY radius around scene origin excluded from fixed-room alignment |

The included recording profiles are:

| Run | Exposure (ms) | Digital gain |
|---|---:|---:|
| `run_20260925T122400` | 8.33 | 1.5 |
| `run_20260925T123004` | 8.33 | 0.75 |
| `run_20260925T123534` | 5.55 | 1.5 |
| `run_20260925T125125` | 11.09 | 1.5 |

These are run-level notes, not measured per-frame automatic exposure telemetry.

## Required recording layout

```text
data_dir/
  RUN/
    manifest.json
    camera.parquet
    camera_frames.bin
    mocap.parquet
```

`manifest.json` must contain `streams.camera.rows` and `streams.mocap.rows`.
The ingestion check verifies row counts, increasing host timestamps, payload
offsets, image dimensions and successful decoding.

| Camera Parquet column | Type / meaning |
|---|---|
| `seq` | Integer row sequence |
| `host_ns` | Integer camera-PC Unix receipt timestamp, nanoseconds |
| `frame_id` | Capture frame identifier; not necessarily the zero-based row index |
| `deck_ms` | Camera deck timestamp, milliseconds |
| `width`, `height` | 320 and 240 |
| `pixel_format` | 1: encoded JPEG; 0 with `depth=1`: raw grayscale |
| `depth` | Bytes per raw pixel for the supported raw format |
| `blob_offset`, `size` | Byte range of the payload in `camera_frames.bin` |

| Mocap Parquet column | Type / meaning |
|---|---|
| `seq` | Integer row sequence |
| `host_ns` | Integer mocap-PC receipt timestamp, nanoseconds |
| `x`, `y`, `z` | Body position in metres, in the recorded mocap frame |
| `qx`, `qy`, `qz`, `qw` | Unit quaternion, XYZW order; body-to-mocap rotation |

Export the tracked drone body's position and orientation into this schema.
Additional timestamp columns may remain in the tables; the adapter uses the
fields above. Converting another recording format requires an adapter that
preserves clock units, frame identity and quaternion conventions.

Separate PCs' Unix clocks are not guaranteed synchronized. The finite lag search
does not solve an arbitrary clock offset or an unknown body-to-camera mount.

## Reconstruction inputs

The checkpoint loader reads Nerfstudio Splatfacto Gaussian parameters
(`means`, `scales`, `quats`, `opacities`, `features_dc`, `features_rest`, inside
the supported pipeline state dictionary). Use a trusted checkpoint: PyTorch loads this file as a serialized model state.

`dataparser_transforms.json` contains a rigid 3 x 4 `transform` and positive
`scale`; `world_frame.json` contains the 4 x 4 `world_transform`. These matrices
and the checkpoint must belong to the same scene export.

`configs/reference_atlas.npz` contains `T_GC` (64 x 4 x 4 camera-to-scene poses)
and `index` (stable view identifiers). `reference_atlas.json` records their
selection. Bootstrap renders the atlas at these poses using the configured
scene. For a new scene, provide useful viewpoints in its coordinate frame.
Use `bootstrap --use-seed` when the initial registration and camera mount are
already independently calibrated.

## Input camera / registration JSON

`camera.json` contains `K`, `width`, `height`,
`distortion_opencv_k1_k2_p1_p2_k3`, `camera_to_body_rotation_assumed` ($R_{BC}$)
and `camera_origin_in_body_m` ($p_{BC}$). The archived polynomial remains for
provenance; `lens_model.json` controls active geometry. Its `type` is `angle`
and `k` stores the coefficients in increasing powers of squared ray angle.

`seed_registration.json` stores:

```text
R_scene_metric_from_mocap = R_GM
meters_per_scene_unit = nominal_meters_per_scene_unit / lambda
mocap_origin_m = -R_GM.T @ t_G / lambda
```

The seed initializes the registration search. Its transform must use the same
scene coordinate frame as the reference atlas and reconstruction.

## Intermediate and final arrays

Let `O = workspace/output/captures`.

| File | Important contents |
|---|---|
| `O/frames/RUN/raw_timing.npz` | `host_ns`, `frame_id`, `time_s`; original camera receipt time |
| `O/prepared/RUN/poses.npz` | `camera_to_mocap_assumed` (N x 4 x 4), `accepted`, `host_ns`, `pose_query_host_ns`, `frame_id`, body position/quaternion |
| `O/frames/RUN/poses.npz` | Accepted-row `index`, `frame_id`, pose-query `time_s`, `T_MC`, `T_GC`, `T_CS`, `K`, archived distortion |
| `O/frames/RUN/index.json` | Accepted zero-based row indices, times and source identity |
| `O/RUN/annotations.json` | Feature ID, `status`, `point_G`, real/render clicks, quality checks and decision provenance |
| `O/RUN/fixed_features/observations*.npz` | Feature/run/index, raw `pixel`, `pixel_pinhole`, `sigma_px`, `seed_index`, tracking provenance |
| `O/RUN/fixed_features/registration/registration.json` | Selected global renderer-compatible transform |
| `O/RUN/fixed_features/registration/search.json` | Search bounds, candidates, loss, refinement and elapsed time |
| `O/RUN/fixed_features/per_frame/RUN/poses_per_frame.npz` | Index/time/frame ID, `T_GC`, `T_SC`, `T_CS`, six-value correction, `mode`, `n_features`, residuals |
| `O/RUN/fixed_features/per_frame/RUN/poses_per_frame.csv` | Readable per-frame fields and before/after residuals |
| `O/appearance_v3/selection.json` | Selected response, prior baseline, parent model and protocol identity |
| `O/appearance_export/RUN/manifest.json` | Applied model/pose hashes and playback convention |
| `O/appearance_export/RUN/frame_metrics.csv` | Before/after whole-image component metrics, index, camera time |
| `O/appearance_export/RUN/video_frame_index.csv` | Encoded-frame to source-image/time mapping |

Join by **run plus zero-based `index`**, not by filename order or row count alone.
`frame_id` is retained as a second identity. Pose-query time includes clock/lag
correction; appearance playback uses recorded camera time. Intervals rejected
by pose synchronization are not independently fitted; playback holds the prior
usable image through those gaps.

## Repository layout

```text
pipeline.py                 Command-line entry point
configs/                    Camera, scene, exposure and model profiles
scripts/                    Synchronization, alignment, rendering and appearance
examples/minimal_pair/      Recorded image pairs and expected corrected images
docs/                      Process PDF, equations, file reference and workflow
tests/                      Geometry, image and export checks
tools/                      Documentation and source-archive builders
weights/                    Downloaded matcher/tracker checkpoints
workspace/                  Generated images, poses, metrics, logs and caches
```

`configs/local.json` stores machine-specific paths. Workspaces, local config,
recordings, scene assets and downloaded weights are Git-ignored. The source
archive is built from Git-tracked files.
