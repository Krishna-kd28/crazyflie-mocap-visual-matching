#!/usr/bin/env python3
"""Render every accepted frame of a run under a registration, lens model applied.

Writes a directory that scripts/render_trajectory_video.py accepts as
--gsplat-dir: <out>/<run>/frame_XXXXXX.png in raw pixel geometry (the render is
made on the padded pinhole canvas and warped with the supplied distortion),
plus render_metadata.json, render_index.csv and render_poses.npz.
Run in the CUDA environment.
"""
from __future__ import annotations

import project_config as PC

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_features_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default='run_20260730T183828')
    ap.add_argument('--registration', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--pinhole', action='store_true', help='undistorted 320x240 renders instead')
    args = ap.parse_args()
    from PIL import Image
    from prepare_fixed_feature_frames import Renderer
    reg = C.load_registration(args.registration)
    folder, index, arrays = C.frame_store(args.run)
    idx = np.array(index['accepted_indices'])
    T_GC = C.apply_registration(arrays['T_MC'], reg['R'], reg['t'], reg['lam'])
    T_SC, T_CS = C.checkpoint_views(T_GC)
    renderer = Renderer(distorted=not args.pinhole)
    print('render method:', renderer.method, flush=True)
    dest = args.out / args.run
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for j, i in enumerate(idx):
        path = dest / f'frame_{int(i):06d}.png'
        Image.fromarray(renderer.rgb(T_CS[j])).save(path)
        records.append(dict(run=args.run, index=int(i), frame_id=int(index['frame_id'][j]), png_path=str(path)))
        if j % 250 == 0:
            print(f'{j + 1}/{len(idx)}', flush=True)
    with (args.out / 'render_index.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    np.savez_compressed(args.out / 'render_poses.npz', camera_to_checkpoint=T_SC, checkpoint_to_camera=T_CS, index=idx)
    C.write_json(args.out / 'render_metadata.json', dict(
        status='complete', run=args.run, pose_sha256=C.sha256(index['pose_source']),
        registration=dict(status='provisional_landmark_fit', label=reg.get('label', ''), path=str(args.registration)),
        camera_model='pinhole',
        distortion='none' if args.pinhole else renderer.method,
        scene_edited=False, checkpoint=str(C.CHECKPOINT),
        selection=[dict(run=args.run, index=int(i), frame_id=int(fid)) for i, fid in zip(idx, index['frame_id'])]))
    print(f'{len(records)} renders -> {args.out}')


if __name__ == '__main__':
    main()
