"""Synchronize Crazyflie mocap, align 3DGS views, and fit camera appearance."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='JSON configuration (default configs/project.json)')
    p.add_argument('--dry-run', action='store_true', help='Print stage commands without executing them')
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('doctor', help='Check configured files, environment and input schemas')
    d.add_argument('--data', action='store_true', help='Also check recording Parquet schemas')
    sub.add_parser('fetch-weights', help='Download MINIMA and CoTracker checkpoints into weights/')
    sub.add_parser('prepare', help='Decode native frames, fit clocks and estimate mocap lag')
    b = sub.add_parser('bootstrap', help='Estimate scene/mount correction using the known-pose atlas')
    b.add_argument('--use-seed', action='store_true', help='Use the supplied registration and mount without estimating a correction')
    b.add_argument('--atlas', type=int, default=64)
    b.add_argument('--queries', type=int, default=24)
    for command in ('features', 'align', 'render', 'video', 'verify'):
        s = sub.add_parser(command)
        s.add_argument('--run', help='One configured run; default all')
        if command == 'align':
            s.add_argument('--track', action='store_true', help='Run CoTracker3 before fitting')
            s.add_argument('--accept-geometric', action='store_true', help='Explicit batch admission; recorded as not individually reviewed')
        if command == 'video':
            s.add_argument('--only-fitted', action='store_true')
    r = sub.add_parser('review', help='Separate real/render feature-review interface')
    r.add_argument('--run', required=True)
    r.add_argument('--port', type=int, default=8777)
    r.add_argument('--open', action='store_true')
    a = sub.add_parser('appearance-fit', help='Refit exposure, spatial and camera-direction response')
    a.add_argument('--stage', choices=['all', 'exposure', 'tone', 'lighting', 'run-gain', 'v2-check', 'camera', 'v3-check'], default='all')
    a = sub.add_parser('appearance-export', help='Apply a frozen response; export corrected PNGs, metrics and video')
    a.add_argument('--run')
    a.add_argument('--model', type=Path, help='selection.json or model JSON; default packaged selected model')
    a.add_argument('--out', type=Path, help='Destination below configured workspace')
    a.add_argument('--no-video', action='store_true')
    s = sub.add_parser('smoke', help='Check the three packaged image pairs against saved outputs')
    s.add_argument('--render', action='store_true', help='Also rerender the first pair on CUDA')
    s.add_argument('--match', action='store_true', help='Also exercise MINIMA/LightGlue on the first pair')
    sub.add_parser('test', help='Run numerical/configuration integration tests on CPU')
    return p


def doctor(PC, data=False):
    import importlib.metadata as metadata
    import importlib.util
    required = ['numpy', 'scipy', 'cv2', 'pyarrow', 'PIL', 'matplotlib', 'torch', 'gsplat', 'lightglue', 'cotracker']
    missing = [x for x in required if importlib.util.find_spec(x) is None]
    files = {k: {'path': str(PC.path(k)), 'exists': PC.path(k).exists()} for k in
             ['data_dir', 'scene_dir', 'checkpoint', 'camera', 'lens', 'seed_registration', 'reference_atlas', 'minima_weights', 'cotracker_weights', 'appearance_model']}
    result = dict(config=str(PC.CONFIG_PATH), workspace=str(PC.WORKSPACE), runs=PC.RUNS,
                  files=files, missing_modules=missing, ffmpeg=shutil.which('ffmpeg'), python=sys.version.split()[0])
    result['versions'] = {}
    for name in ['numpy', 'scipy', 'opencv-python', 'torch', 'gsplat', 'lightglue', 'cotracker']:
        try: result['versions'][name] = metadata.version(name)
        except metadata.PackageNotFoundError: pass
    if 'torch' not in missing:
        import torch
        result['cuda_available'] = torch.cuda.is_available()
    if data:
        import pyarrow.parquet as pq
        camera = {'seq','host_ns','frame_id','deck_ms','width','height','pixel_format','depth','size','blob_offset'}
        mocap = {'seq','host_ns','x','y','z','qx','qy','qz','qw'}
        result['recordings'] = {}
        for run in PC.RUNS:
            checks = {}
            for name, columns in [('camera', camera), ('mocap', mocap)]:
                file = PC.DATA / run / f'{name}.parquet'
                if not file.exists(): checks[name] = {'missing_file': str(file)}; continue
                p = pq.ParquetFile(file)
                checks[name] = dict(rows=p.metadata.num_rows, missing_columns=sorted(columns-set(p.schema.names)))
            checks['payload_exists'] = (PC.DATA/run/'camera_frames.bin').is_file()
            checks['manifest_exists'] = (PC.DATA/run/'manifest.json').is_file()
            result['recordings'][run] = checks
    result['ready_for_full_pipeline'] = (not missing and all(x['exists'] for x in files.values())
                                        and bool(result['ffmpeg']) and result.get('cuda_available', False))
    if data:
        result['ready_for_full_pipeline'] &= all(v['payload_exists'] and v['manifest_exists'] and
            all(not v[k].get('missing_file') and not v[k].get('missing_columns') for k in ('camera','mocap'))
            for v in result['recordings'].values())
    print(json.dumps(result, indent=2))
    return result


def guard_workspace(PC):
    """Reject cached outputs under changed code/configuration/calibration inputs."""
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    snapshot = dict(config=PC.CONFIG, code={p.name:sha(p) for p in sorted((PC.CODE_ROOT/'scripts').glob('*.py'))},
                    calibration={k:sha(PC.path(k)) for k in ('camera','lens','seed_registration','reference_atlas')})
    record = PC.WORKSPACE/'pipeline_inputs.json'
    if record.exists() and json.loads(record.read_text()) != snapshot:
        raise RuntimeError('Code, configuration or calibration changed. Select a fresh workspace_dir; do not reuse cached stage markers.')
    PC.WORKSPACE.mkdir(parents=True, exist_ok=True)
    if not record.exists(): record.write_text(json.dumps(snapshot,indent=2)+'\n')


def main():
    args = parser().parse_args()
    if args.config:
        os.environ['CF_MATCH_CONFIG'] = str(args.config.expanduser().resolve())
    import project_config as PC
    run = getattr(args, 'run', None)
    if run is not None and run not in PC.RUNS:
        raise SystemExit(f'Run {run!r} is not configured. Choices: {PC.RUNS}')
    runs = [run] if run else PC.RUNS
    if not args.dry_run and args.command in ('prepare','bootstrap','features','align','render','verify','appearance-fit'):
        guard_workspace(PC)
    env = dict(os.environ, CF_MATCH_CONFIG=str(PC.CONFIG_PATH), PYTHONDONTWRITEBYTECODE='1')
    env.setdefault('MPLCONFIGDIR', str(PC.WORKSPACE/'cache/matplotlib'))
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')

    def execute(name, *rest):
        cmd = [sys.executable, str(PC.CODE_ROOT/'scripts'/f'{name}.py'), *map(str, rest)]
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            PC.WORKSPACE.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, env=env, cwd=PC.CODE_ROOT, check=True)

    def wrapped(r, name, *rest):
        execute('run_stage', r, name, *rest)

    if args.command == 'doctor':
        result = doctor(PC, args.data)
        if not result['ready_for_full_pipeline']:
            raise SystemExit(1)
    elif args.command == 'fetch-weights':
        execute('fetch_weights')
    elif args.command == 'prepare':
        execute('prepare_recordings', 'all')
    elif args.command == 'bootstrap':
        if args.use_seed:
            if not args.dry_run:
                for r in runs:
                    target = PC.OUT/r/'bootstrap_registration.json'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(PC.path('seed_registration'), target)
                print('Copied the seed registration; the configured mount is assumed unchanged.')
            else: print('Copy seed registration into each run; retain prepared camera poses.')
        else:
            execute('bootstrap_registration', 'localize', '--atlas', args.atlas, '--queries', args.queries)
            execute('bootstrap_registration', 'fit')
            initial = PC.OUT/'bootstrap/initial_hand_eye.json'
            if not args.dry_run:
                if initial.exists():
                    raise SystemExit('Bootstrap timing already started. Resume explicit stages or use a fresh workspace.')
                shutil.copy2(PC.OUT/'bootstrap/hand_eye.json', initial)
            execute('prepare_recordings', 'sync', '--timing-camera-correction', initial)
            execute('bootstrap_registration', 'fit')
            execute('bootstrap_registration', 'apply')
    elif args.command in ('features', 'align'):
        options = ['--run', run] if run else []
        if args.command == 'align':
            if args.track: options += ['--track']
            if args.accept_geometric: options += ['--accept-geometric']
        execute('run_alignment', 'features' if args.command == 'features' else 'fit', *options)
    elif args.command == 'review':
        wrapped(run, 'auto_feature_viewer', '--run', run, '--port', args.port, *(['--open'] if args.open else []))
    elif args.command == 'render':
        for r in runs:
            wrapped(r, 'per_frame_registration', '--run', r, '--render-only')
    elif args.command == 'video':
        for r in runs:
            suffix = 'matched_only' if args.only_fitted else 'all_accepted'
            wrapped(r, 'build_per_frame_comparison_video', '--run', r, '--panels', 'real,per_frame,blend',
                    '--plain-labels', '--out', PC.OUT/r/f'alignment_{suffix}.mp4',
                    *(['--only-fitted'] if args.only_fitted else []))
    elif args.command == 'verify':
        for r in runs:
            wrapped(r, 'verify_fixed_feature_registration', '--annotations', PC.OUT/r/'annotations.json', '--extra-annotations')
    elif args.command == 'appearance-fit':
        stages = [
            ('exposure', 'fit_exposure', []),
            ('tone', 'fit_appearance', ['fit']),
            ('lighting', 'fit_appearance', ['lighting']),
            ('run-gain', 'fit_appearance', ['run-gain']),
            ('v2-check', 'fit_appearance', ['evaluate']),
            ('camera', 'fit_camera_response', ['fit']),
            ('v3-check', 'fit_camera_response', ['evaluate'])]
        for key, name, options in stages:
            if args.stage in ('all', key): execute(name, *options)
    elif args.command == 'appearance-export':
        options = []
        for key in ('run','model','out'):
            if getattr(args,key): options += ['--'+key, str(getattr(args,key))]
        if args.no_video: options += ['--no-video']
        execute('export_appearance', *options)
    elif args.command == 'smoke':
        execute('smoke_pipeline', *(['--render'] if args.render else []), *(['--match'] if args.match else []))
    elif args.command == 'test':
        cmd = [sys.executable, '-m', 'unittest', 'discover', '-s', str(PC.CODE_ROOT/'tests'), '-v']
        print(shlex.join(cmd), flush=True)
        if not args.dry_run: subprocess.run(cmd, env=env, cwd=PC.CODE_ROOT, check=True)
