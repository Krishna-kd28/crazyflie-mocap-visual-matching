# Validation

Run the numerical and export checks with:

```bash
python pipeline.py test
python pipeline.py smoke
```

Validation recorded on 7 October 2026:

| Check | Result / record |
|---|---|
| Numerical, path and export integration | 14 passed; [test output](tests.txt) |
| Saved appearance outputs | Three examples reproduced pixel for pixel; [smoke record](smoke_results.json) |
| SuperPoint + MINIMA LightGlue | CPU inference completed on the first example; [smoke record](smoke_results.json) |
| Lens inverse and image coverage | Sensor corners and overscan white-image test passed |
| Camera geometry | Known-point triangulation and transform conventions passed |
| Timing | Clock fit with arrival outliers and outage rejection passed |
| Pose objective | Tensor and NumPy implementations agree |
| Image/video export | Corrected pixels and irregular-time frame mapping passed |
| Core numerical functions | 22 source comparisons passed; [algorithm audit](algorithm_parity.json) |
| Scene/checkpoint transform | Agrees with the saved camera matrices; [transform audit](scene_transform_check.json) |
| Commands outside the repository directory | 11 plans resolved; [CLI checks](cli_plans.json) |
| Per-run stage entry points | Nine import/help checks passed; [stage checks](stage_help_checks.json) |
| Source archive relocation | [Extraction and relocation check](relocation.json) |
| File integrity and documentation links | [Package check](package_checks.json) |

The three example pairs are reproducibility fixtures. Their image scores verify
consistent behavior and do not estimate accuracy on new recordings.

The CUDA driver was unavailable during this validation, so checks cover CPU
matching, geometry, appearance, export and package relocation. Full rendering and
alignment require a working CUDA environment; check it with
`python pipeline.py doctor --data`. Dependency versions were inspected in the
available environment; a fresh installation still needs its own environment check.
