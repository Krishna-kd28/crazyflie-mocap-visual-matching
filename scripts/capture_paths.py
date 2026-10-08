"""Configured recording paths, exposure metadata, and output directories."""
import datetime
import json
from pathlib import Path
import shutil
import project_config as PC
import fixed_features_common as C

ROOT, DATA, OUT, RUNS, EXPOSURE = PC.WORKSPACE, PC.DATA, PC.OUT, PC.RUNS, PC.EXPOSURE


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def event(stage, **details):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'journal.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(timespec='seconds'), stage=stage, **details)) + '\n')


def seed_path(run):
    return OUT / run / 'seed_registration.json'


def snapshot_seeds():
    OUT.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        target = seed_path(run)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(PC.path('seed_registration'), target)
    paths = [PC.path(k) for k in ('seed_registration', 'lens', 'camera')]
    record = OUT / 'protected_originals.json'
    if not record.exists():
        write(record, {str(p): C.sha256(p) for p in paths})


def configure(run):
    if run not in RUNS:
        raise ValueError(run)
    C.DATA, C.POSES_DIR, C.FRAMES = DATA, OUT / 'prepared', OUT / 'frames'
    C.FF_OUT = OUT / run / 'fixed_features'
    C.FF_OUT.mkdir(parents=True, exist_ok=True)
    C.ANNOTATIONS, C.START_REGISTRATION = OUT / run / 'annotations.json', seed_path(run)
    C.RUNS, C.SHORT = [run], {run: run[-6:]}
    return C
