#!/usr/bin/env python3
"""Comparison videos from the per-frame dataset.

Default: four panels, real | global-fit render | per-frame render | blend.
--panels real,per_frame,blend gives a single row of three. --only-fitted keeps
only frames that see a feature (played at --constant-rate frames/s); otherwise
playback keeps the recorded frame timing. Each frame is labelled with its mode
(fitted / interpolated / global), feature count, correction and residuals.
"""
from __future__ import annotations

import project_config as PC

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_features_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default='run_20260730T183828')
    ap.add_argument('--per-frame', type=Path, default=C.FF_OUT / 'per_frame')
    ap.add_argument('--global-renders', type=Path, default=C.ROOT / 'output/gsplat_video_183828/fixed_features_v2')
    ap.add_argument('--out', type=Path, default=C.ROOT / 'output/videos/183828_per_frame/per_frame_comparison.mp4')
    ap.add_argument('--keep-frames', action='store_true')
    ap.add_argument('--panels', default='real,global,per_frame,blend', help='comma list of real, global, per_frame, blend')
    ap.add_argument('--only-fitted', action='store_true', help='keep only frames with feature observations')
    ap.add_argument('--constant-rate', type=float, default=10.0, help='frames/s for --only-fitted playback')
    ap.add_argument('--plain-labels', action='store_true', help='titles only; time and frame on the real panel, no numbers')
    args = ap.parse_args()
    run = args.run
    d = args.per_frame / run
    with np.load(d / 'poses_per_frame.npz', allow_pickle=False) as z:
        idx, times, mode, n_feat = z['index'], z['time_s'], z['mode'], z['n_features']
        corr = z['correction']
    rows = {int(r['index']): r for r in __import__('csv').DictReader((d / 'poses_per_frame.csv').open())}
    frame_dir = args.out.parent / ('.' + args.out.stem + '_frames')
    frame_dir.mkdir(parents=True, exist_ok=True)
    S = 2
    W, H = C.WIDTH * S, C.HEIGHT * S
    durations = np.r_[np.diff(times), np.median(np.diff(times))]
    panels = [x.strip() for x in args.panels.split(',')]
    keep = [j for j in range(len(idx)) if (not args.only_fitted) or mode[j] == 'fitted']
    for j in keep:
        i = idx[j]
        real = cv2.cvtColor(cv2.imread(str(C.real_raw_path(run, int(i))), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        g = cv2.imread(str(args.global_renders / run / f'frame_{int(i):06d}.png')) if 'global' in panels else None
        p = cv2.imread(str(d / 'render' / run / f'frame_{int(i):06d}.png'))
        if p is None or ('global' in panels and g is None):
            raise FileNotFoundError(f'missing render for frame {i}')
        blend = cv2.addWeighted(real, 0.5, p, 0.5, 0)
        r = rows[int(i)]
        source = dict(real=(real, f'real  {times[j]:.2f} s  frame {int(i)}'), global_=(g, 'global fit render'),
                      per_frame=(p, f'sim at per-frame pose: {mode[j]} ({n_feat[j]} feat, {np.degrees(np.linalg.norm(corr[j, :3])):.2f} deg, {100 * np.linalg.norm(corr[j, 3:6]):.1f} cm)'
                                 + (f'  residual {r["median_before_px"]} -> {r["median_after_px"]} px' if r['median_after_px'] else '')),
                      blend=(blend, 'real over sim (50% blend)'))
        source['global'] = source.pop('global_')
        if args.plain_labels:
            source['per_frame'] = (p, 'sim (3DGS) at per-frame pose')
            source['global'] = (g, 'sim (3DGS) at global pose')
            source['blend'] = (blend, 'real over sim')
        tiles = [cv2.resize(source[k][0], (W, H), interpolation=cv2.INTER_LINEAR) for k in panels]
        labels = [source[k][1] for k in panels]
        for t, lab in zip(tiles, labels):
            cv2.rectangle(t, (0, 0), (W, 22), (0, 0, 0), -1)
            cv2.putText(t, lab, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        colour = {'fitted': (60, 200, 60), 'interpolated': (40, 170, 230), 'global': (60, 60, 220)}[str(mode[j])]
        if 'per_frame' in panels and len(panels) == 4 and not args.plain_labels:
            cv2.rectangle(tiles[panels.index('per_frame')], (W - 26, 4), (W - 6, 18), colour, -1)
        frame = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]) if len(tiles) == 4 else np.hstack(tiles)
        cv2.imwrite(str(frame_dir / f'{j:06d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    ticks = np.rint(np.r_[times, times[-1] + durations[-1]] * 1000).astype(np.int64)
    enc = np.diff(ticks) * .001
    seq_durations = [1.0 / args.constant_rate] * len(keep) if args.only_fitted else [float(enc[j]) for j in keep]
    concat = frame_dir / 'frames.ffconcat'
    with concat.open('w') as f:
        f.write('ffconcat version 1.0\n')
        for j, dur in zip(keep, seq_durations):
            f.write(f"file '{j:06d}.jpg'\noption framerate 1000\nduration {max(dur, 0.001):.3f}\n")
        f.write(f"file '{keep[-1]:06d}.jpg'\noption framerate 1000\n")
    if args.out.exists():
        args.out.unlink()
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-f', 'concat', '-safe', '0', '-i', str(concat), '-an',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-fps_mode', 'vfr',
                    '-enc_time_base', '1:1000', '-bf', '0', '-video_track_timescale', '1000000', '-movflags', '+faststart', str(args.out)], check=True)
    if not args.keep_frames:
        shutil.rmtree(frame_dir)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
