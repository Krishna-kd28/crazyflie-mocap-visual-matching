"""Portable paths for the copied research algorithms.

All relative paths are relative to this repository, never the caller's cwd.
CF_MATCH_CONFIG selects an alternate configuration, including in subprocesses.
Only the workspace is writable pipeline state; recordings and scene are inputs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get('CF_MATCH_CONFIG', CODE_ROOT / 'configs/project.json')).expanduser().resolve()
CONFIG = json.loads(CONFIG_PATH.read_text())


def resolve(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (CODE_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def path(key):
    if not CONFIG.get(key):
        raise ValueError(f'Missing {key!r} in {CONFIG_PATH}')
    return resolve(CONFIG[key])


WORKSPACE = path('workspace_dir')
if not WORKSPACE.is_relative_to(CODE_ROOT) or WORKSPACE == CODE_ROOT:
    raise ValueError('workspace_dir must be a subdirectory of this standalone repository')
os.environ.setdefault('TORCH_HOME', str(WORKSPACE / 'cache/torch'))
os.environ.setdefault('MPLCONFIGDIR', str(WORKSPACE / 'cache/matplotlib'))
for _key in ('data_dir', 'scene_dir'):
    _input = path(_key)
    if _input == WORKSPACE or _input.is_relative_to(WORKSPACE) or WORKSPACE.is_relative_to(_input):
        raise ValueError(f'{_key} and workspace_dir must be disjoint directories')

DATA = path('data_dir')
OUT = WORKSPACE / 'output/captures'
RUNS = list(CONFIG['runs'])
if not RUNS or len(set(RUNS)) != len(RUNS) or any(Path(r).name != r or r in ('.', '..') for r in RUNS):
    raise ValueError('runs must contain unique directory names')
if CONFIG.get('cross_matcher', 'minima') not in ('minima', 'stock'):
    raise ValueError('cross_matcher must be minima or stock')
EXPOSURE = CONFIG['exposure']
for _run in RUNS:
    _settings = EXPOSURE[_run]
    if _settings['exposure_ms'] <= 0 or _settings['digital_gain'] <= 0:
        raise ValueError(f'Positive exposure_ms and digital_gain are required for {_run}')


def read_camera():
    camera = json.loads(path('camera').read_text())
    if (camera.get('width'), camera.get('height')) != (320, 240):
        raise ValueError('This pipeline is calibrated for native 320x240 Crazyflie images')
    return camera


def load_cotracker(device='cuda'):
    """Load the pinned installed implementation and an explicit checkpoint."""
    from cotracker.predictor import CoTrackerPredictor
    checkpoint = path('cotracker_weights')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'{checkpoint}: run python pipeline.py fetch-weights')
    return CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, window_len=60).to(device).eval()


def relative_or_absolute(value):
    value = Path(value).resolve()
    return str(value.relative_to(WORKSPACE)) if value.is_relative_to(WORKSPACE) else str(value)
