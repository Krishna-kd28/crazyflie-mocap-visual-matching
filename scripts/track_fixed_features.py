#!/usr/bin/env python3
"""Propagate each manual real-image click to neighbouring frames by tracking.

Every confirmed feature's real clicks are tracked forwards and backwards
through consecutive accepted frames. Default tracker: CoTracker3 (offline,
--tracker cotracker3), run on the window of frames around the click, which in
scripts/benchmark_trackers.py reached a median error of 0.9 px against the
user's own later clicks (Lucas-Kanade: 2.2 px, with drift on a tenth of the
tracks); its visibility estimate and a forward-backward re-track cut the
track. --tracker lk keeps the pyramidal Lucas-Kanade tracker with its
forward-backward and appearance checks. Tracks are cut at recorded time gaps
and tracked observations get pixel uncertainties that grow with the distance
from the click. Output: output/fixed_features/observations.npz.
"""
from __future__ import annotations

import project_config as PC

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_features_common as C  # noqa: E402

LK = dict(winSize=(15, 15), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def load_gray(run, i):
    """Raw recorded frame; clicks and tracks live in raw pixel coordinates."""
    im = cv2.imread(str(C.real_raw_path(run, i)), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(C.real_raw_path(run, i))
    return im


def patch_ncc(a, b, xy_a, xy_b, r=7):
    ha, wa = a.shape

    def crop(im, xy):
        x, y = int(round(xy[0])), int(round(xy[1]))
        if not (r <= x < wa - r and r <= y < ha - r):
            return None
        return im[y - r:y + r + 1, x - r:x + r + 1].astype(np.float32)
    pa, pb = crop(a, xy_a), crop(b, xy_b)
    if pa is None or pb is None or pa.std() < 1 or pb.std() < 1:
        return -1.0
    return float(np.mean((pa - pa.mean()) * (pb - pb.mean())) / (pa.std() * pb.std()))


def track_seed(run, index, arrays, j0, xy0, max_steps, fb_tol, ncc_min, sigma0):
    """Track one click forward and backward; return list of (j, xy, steps, fb_error, ncc)."""
    idx = index['accepted_indices']
    times = np.asarray(index['time_s'])
    seed = load_gray(run, idx[j0])
    out = []
    for direction in (1, -1):
        prev_j, prev_xy, prev_im = j0, np.asarray(xy0, dtype=np.float32), seed
        for step in range(1, max_steps + 1):
            j = prev_j + direction
            if j < 0 or j >= len(idx):
                break
            if abs(times[j] - times[prev_j]) > 0.25:      # recorded gap: stop
                break
            im = load_gray(run, idx[j])
            p0 = prev_xy.reshape(1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_im, im, p0, None, **LK)
            if st[0, 0] != 1:
                break
            p0b, stb, _ = cv2.calcOpticalFlowPyrLK(im, prev_im, p1, None, **LK)
            fb = float(np.linalg.norm(p0b[0, 0] - p0[0, 0]))
            if stb[0, 0] != 1 or fb > fb_tol:
                break
            xy = p1[0, 0]
            if not (4 <= xy[0] <= C.WIDTH - 5 and 4 <= xy[1] <= C.HEIGHT - 5):
                break
            ncc = patch_ncc(seed, im, xy0, xy)
            if ncc < ncc_min:
                break
            out.append((j, xy.astype(float), step, fb, ncc))
            prev_j, prev_xy, prev_im = j, xy, im
    return out


class CoTracker:
    """CoTracker3 offline: one window of frames around the click, tracked in both directions."""

    def __init__(self, device='cuda'):
        import torch
        self.torch = torch
        self.model = PC.load_cotracker(device)
        self.device = device

    def run(self, frames_gray, queries):
        """queries: list of (t, x, y). Returns tracks [Q, N, 2] and visibility [Q, N] (forward from t)."""
        torch = self.torch
        video = torch.from_numpy(np.stack([np.repeat(f[..., None], 3, axis=2) for f in frames_gray]))
        video = video.permute(0, 3, 1, 2)[None].float().to(self.device)
        q = torch.tensor([[[float(t), float(x), float(y)] for t, x, y in queries]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            tracks, vis = self.model(video, queries=q)
        return tracks[0].permute(1, 0, 2).cpu().numpy(), vis[0].permute(1, 0).cpu().numpy()


def track_seed_cotracker(tracker, run, index, j0, xy0, max_steps, fb_tol, sigma0):
    """Each direction is one clip that starts at the click (the backward side is the reversed
    frame order), tracked forward by CoTracker3; a second pass from the far end of the track
    back to the click is the consistency check. The track is cut at recorded time gaps and at
    the first frame that is invisible, off-image or inconsistent."""
    idx = index['accepted_indices']
    times = np.asarray(index['time_s'])
    out = []
    for direction in (1, -1):
        js = [j0]
        for step in range(1, max_steps + 1):
            j = j0 + direction * step
            if j < 0 or j >= len(idx) or abs(times[j] - times[js[-1]]) > 0.25:      # recorded gap: stop
                break
            js.append(j)
        if len(js) < 2:
            continue
        frames = [load_gray(run, idx[j]) for j in js]
        fwd, vis = tracker.run(frames, [(0, xy0[0], xy0[1])])
        fwd, vis = fwd[0], vis[0]
        inside = (fwd[:, 0] >= 4) & (fwd[:, 0] <= C.WIDTH - 5) & (fwd[:, 1] >= 4) & (fwd[:, 1] <= C.HEIGHT - 5)
        ok = (vis > 0.5) & inside
        t_end = 0
        while t_end + 1 < len(js) and ok[t_end + 1]:
            t_end += 1
        if t_end == 0:
            continue
        clip = frames[:t_end + 1][::-1]                        # far end first
        back = tracker.run(clip, [(0, fwd[t_end, 0], fwd[t_end, 1])])[0][0][::-1]
        for t in range(1, t_end + 1):
            fb = float(np.linalg.norm(fwd[t] - back[t]))
            if fb > fb_tol:
                break
            out.append((js[t], fwd[t].astype(float), t, fb, 1.0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotations', type=Path, default=C.ANNOTATIONS)
    ap.add_argument('--extra-annotations', type=Path, nargs='*', default=[C.ROOT / 'research/fixed_features/auto_features.json'],
                    help='further annotation files whose confirmed features are merged in (default: the automatic feature maker output)')
    ap.add_argument('--out', type=Path, default=C.FF_OUT / 'observations.npz')
    ap.add_argument('--tracker', choices=['cotracker3', 'lk'], default='cotracker3')
    ap.add_argument('--max-steps', type=int, default=None, help='frames tracked in each direction (default 60 cotracker3, 30 lk)')
    ap.add_argument('--fb-tol', type=float, default=None, help='forward-backward tolerance (px; default 1.5 cotracker3, 0.6 lk)')
    ap.add_argument('--ncc-min', type=float, default=0.6, help='appearance similarity to the seed patch')
    ap.add_argument('--sigma-growth', type=float, default=0.05, help='allowance growth per tracked frame (px); CoTracker3 error stays near 1 px, so a small margin')
    args = ap.parse_args()
    if args.max_steps is None:
        args.max_steps = 60 if args.tracker == 'cotracker3' else 30
    if args.fb_tol is None:
        args.fb_tol = 1.5 if args.tracker == 'cotracker3' else 0.6
    tracker = None
    if args.tracker == 'cotracker3':
        try:
            tracker = CoTracker()
        except Exception as error:  # noqa: BLE001
            print(f'CoTracker3 unavailable ({error.__class__.__name__}: {str(error)[:80]}); using Lucas-Kanade', flush=True)
            args.tracker, args.max_steps, args.fb_tol = 'lk', 30, 0.6
    ann = json.loads(args.annotations.read_text())
    if ann.get('coords') != 'raw':
        sys.exit('Annotations are not in raw coordinates: run scripts/migrate_annotations_to_raw.py first')
    have = {f['id'] for f in ann['features']}
    for path in args.extra_annotations or []:
        if Path(path).exists():
            for f in json.loads(Path(path).read_text()).get('features', []):
                if f.get('status') == 'confirmed' and f['id'] not in have:
                    ann['features'].append(f); have.add(f['id'])
    stores = {run: C.frame_store(run) for run in C.RUNS}
    positions = {run: {int(i): j for j, i in enumerate(stores[run][1]['accepted_indices'])} for run in C.RUNS}
    rows = []       # (feature, run, index, u, v, sigma, kind, seed_index, steps)
    summary = {}
    features = [f for f in ann['features'] if f.get('status') == 'confirmed']
    for f in features:
        manual, tracked = 0, 0
        seen = set()
        automatic = f.get('source') == 'auto_features'
        for c in sorted(f.get('real_clicks', []), key=lambda c: (c['run'], c['index'])):
            run, i = c['run'], int(c['index'])
            if c.get('method') == 'auto_cotracker3':
                # propagated by scripts/auto_features.py track; shares its matched frame's weight,
                # allowance recomputed here from its distance so --sigma-growth applies to it too
                steps_c = int(c.get('steps', 0))
                rows.append((f['id'], run, i, c['xy'][0], c['xy'][1], float(np.hypot(2.5, args.sigma_growth * steps_c)), 'tracked',
                             int(c.get('seed_index', i)), steps_c))
                tracked += 1
                continue
            rows.append((f['id'], run, i, c['xy'][0], c['xy'][1], float(c.get('sigma_px', 2.0)), 'matched' if automatic else 'manual', i, 0))
            seen.add((run, i))
            manual += 1
        if automatic:
            summary[f['id']] = dict(manual=manual, tracked=tracked)
            print(f'{f["id"]}: {manual} matched, {tracked} propagated by the feature maker', flush=True)
            continue
        # Track every click; a frame reached from several clicks keeps the nearest one.
        candidates = {}
        for c in sorted(f.get('real_clicks', []), key=lambda c: (c['run'], c['index'])):
            run, i = c['run'], int(c['index'])
            folder, index, arrays = stores[run]
            j0 = positions[run][i]
            steps = args.max_steps
            tracked_points = (track_seed_cotracker(tracker, run, index, j0, c['xy'], steps, args.fb_tol, c.get('sigma_px', 2.0))
                              if tracker is not None else
                              track_seed(run, index, arrays, j0, c['xy'], steps, args.fb_tol, args.ncc_min, c.get('sigma_px', 2.0)))
            for j, xy, steps, fb, ncc in tracked_points:
                key = (run, index['accepted_indices'][j])
                if key in seen:
                    continue
                sigma = float(np.hypot(c.get('sigma_px', 2.0), args.sigma_growth * steps))
                if key not in candidates or steps < candidates[key][-1]:
                    candidates[key] = (f['id'], run, key[1], float(xy[0]), float(xy[1]), sigma, 'tracked', i, steps)
        for key in sorted(candidates):
            rows.append(candidates[key])
            tracked += 1
        summary[f['id']] = dict(manual=manual, tracked=tracked)
        print(f'{f["id"]}: {manual} manual, {tracked} tracked', flush=True)
    if not rows:
        sys.exit('No confirmed real clicks found')
    feature_ids = sorted({r[0] for r in rows})
    pixel = np.array([[r[3], r[4]] for r in rows], dtype=np.float64)
    pixel_pinhole = C.raw_to_pinhole(pixel)
    keep = np.isfinite(pixel_pinhole).all(axis=1)
    if not keep.all():
        print(f'{int((~keep).sum())} observations beyond the valid lens radius dropped', flush=True)
        rows = [r for r, k in zip(rows, keep) if k]
        pixel, pixel_pinhole = pixel[keep], pixel_pinhole[keep]
    arr = dict(
        feature=np.array([r[0] for r in rows]), run=np.array([r[1] for r in rows]),
        index=np.array([r[2] for r in rows], dtype=np.int64),
        pixel=pixel, pixel_pinhole=pixel_pinhole, coords=np.array('raw'),
        sigma_px=np.array([r[5] for r in rows], dtype=np.float64),
        kind=np.array([r[6] for r in rows]), seed_index=np.array([r[7] for r in rows], dtype=np.int64),
        steps=np.array([r[8] for r in rows], dtype=np.int64), feature_ids=np.array(feature_ids))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arr)
    C.write_json(args.out.with_suffix('.json'), dict(
        annotations=str(args.annotations), annotations_sha256=C.sha256(args.annotations),
        parameters=dict(tracker=args.tracker, max_steps=args.max_steps, fb_tol_px=args.fb_tol, ncc_min=args.ncc_min,
                        sigma_growth_px_per_step=args.sigma_growth, lk=dict(winSize=LK['winSize'], maxLevel=LK['maxLevel'])),
        per_feature=summary, total=len(rows), manual=int(np.sum(arr['kind'] == 'manual')),
        tracked=int(np.sum(arr['kind'] == 'tracked'))))
    print(f'{len(rows)} observations ({int(np.sum(arr["kind"] == "manual"))} manual) -> {args.out}')


if __name__ == '__main__':
    main()
