#!/usr/bin/env python3
"""Extend feature observations to every frame in which a feature is visible.

For each confirmed feature with a scene point and each accepted frame of a run,
the point is projected with the fitted global registration. If it lands inside
the image, the feature is searched in the real image by normalized
cross-correlation of gradient magnitude within a window around the prediction.
Templates are the real-image patches of the user's own manual clicks of that
feature, taken from the two most similar camera poses; a match is kept only
when both templates agree within 1.5 px and correlate well. Manual and tracked
observations take precedence. Output: observations_extended.npz.
"""
from __future__ import annotations

import project_config as PC

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_features_common as C  # noqa: E402
from fixed_feature_picker import Store  # noqa: E402
from gradient_features import gradient_magnitude  # noqa: E402

P = 10   # template half size (21 x 21)


def match(template, image_gm, u, v, search):
    h, w = image_gm.shape
    x, y = int(round(u)), int(round(v))
    if not (P + search <= x < w - P - search and P + search <= y < h - P - search):
        return None
    window = image_gm[y - P - search:y + P + search + 1, x - P - search:x + P + search + 1]
    if template.std() < 1e-3 or window.std() < 1e-3:
        return None
    sc = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    py, px = np.unravel_index(np.argmax(sc), sc.shape)
    peak = float(sc[py, px])
    masked = sc.copy()
    masked[max(0, py - 3):py + 4, max(0, px - 3):px + 4] = -1
    distinct = peak - float(masked.max())
    du, dv = px - search, py - search
    if 0 < px < sc.shape[1] - 1:
        l, c, r = sc[py, px - 1], sc[py, px], sc[py, px + 1]
        d = l - 2 * c + r
        du += float(np.clip(0.5 * (l - r) / d, -0.5, 0.5)) if abs(d) > 1e-9 else 0.0
    if 0 < py < sc.shape[0] - 1:
        l, c, r = sc[py - 1, px], sc[py, px], sc[py + 1, px]
        d = l - 2 * c + r
        dv += float(np.clip(0.5 * (l - r) / d, -0.5, 0.5)) if abs(d) > 1e-9 else 0.0
    return x + du, y + dv, peak, distinct


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', action='append', help='default: all runs with manual clicks')
    ap.add_argument('--annotations', type=Path, default=C.ANNOTATIONS)
    ap.add_argument('--extra-annotations', type=Path, nargs='*', default=[C.ROOT / 'research/fixed_features/auto_features.json'],
                    help='further annotation files whose confirmed features are merged in (default: the automatic feature maker output)')
    ap.add_argument('--observations', type=Path, default=C.FF_OUT / 'observations.npz')
    ap.add_argument('--fit', type=Path, default=C.FF_OUT / 'registration')
    ap.add_argument('--out', type=Path, default=C.FF_OUT / 'observations_extended.npz')
    ap.add_argument('--search', type=int, default=14, help='half window around the prediction (px)')
    ap.add_argument('--min-peak', type=float, default=0.6)
    ap.add_argument('--min-distinct', type=float, default=0.08)
    ap.add_argument('--agree-px', type=float, default=1.5)
    ap.add_argument('--sigma-px', type=float, default=2.5)
    args = ap.parse_args()
    store = Store(args.annotations, gpu=False, extra=args.extra_annotations)
    points = store.feature_points()
    reg = C.load_registration(args.fit / 'registration.json')
    with np.load(args.observations, allow_pickle=False) as d:
        obs = {k: d[k].copy() for k in d.files if k not in ('feature_ids', 'coords')}
    if 'pixel_pinhole' not in obs:
        sys.exit('observations.npz has no raw/pinhole pixels: rerun scripts/track_fixed_features.py')
    existing = {(str(f), str(r), int(i)) for f, r, i in zip(obs['feature'], obs['run'], obs['index'])}
    runs = args.run or sorted({str(r) for r in obs['run']})
    new_rows = []
    summary = {}
    gm_cache = {}

    def gm(run, i):
        key = (run, int(i))
        if key not in gm_cache:
            im = cv2.imread(str(C.real_raw_path(run, int(i))), cv2.IMREAD_GRAYSCALE)
            gm_cache[key] = gradient_magnitude(im)
            if len(gm_cache) > 4000:
                gm_cache.pop(next(iter(gm_cache)))
        return gm_cache[key]
    for run in runs:
        folder, index, arrays = C.frame_store(run)
        idx = np.array(index['accepted_indices'])
        T_GC = C.apply_registration(arrays['T_MC'], reg['R'], reg['t'], reg['lam'])
        pos = {int(i): j for j, i in enumerate(idx)}
        per_feature = {}
        for f in store.annotations['features']:
            p = points.get(f['id'])
            if f.get('status') != 'confirmed' or p is None or p['X'] is None or p['method'] != 'triangulation':
                continue
            manual = [c for c in f.get('real_clicks', []) if c['run'] == run and int(c['index']) in pos]
            if not manual:
                continue
            X = np.asarray(p['X'])
            # template bank: patch + camera pose of every manual click
            bank = []
            for c in manual:
                j = pos[int(c['index'])]
                g = gm(run, c['index'])
                x, y = int(round(c['xy'][0])), int(round(c['xy'][1]))
                if not (P <= x < C.WIDTH - P and P <= y < C.HEIGHT - P):
                    continue
                bank.append(dict(index=int(c['index']), patch=g[y - P:y + P + 1, x - P:x + P + 1].copy(),
                                 p=T_GC[j, :3, 3], R=T_GC[j, :3, :3]))
            if not bank:
                continue
            bank_positions = np.array([b['p'] for b in bank])
            bank_rotations = np.array([b['R'] for b in bank])
            uv, z = C.project(T_GC, np.repeat(X[None], len(T_GC), 0))
            uv = C.pinhole_to_raw(uv)          # search in the raw image around the raw prediction
            added = 0
            for j, i in enumerate(idx):
                key = (f['id'], run, int(i))
                if key in existing or z[j] <= 0.3:
                    continue
                u, v = uv[j]
                if not (P + args.search <= u < C.WIDTH - P - args.search and P + args.search <= v < C.HEIGHT - P - args.search):
                    continue
                # two templates from the most similar camera poses
                # tr(R_b^T R_j) gives the same geodesic angle as the rotation
                # vector norm, evaluated for the whole template bank at once.
                cosine = (np.einsum('bij,ij->b', bank_rotations, T_GC[j, :3, :3]) - 1) / 2
                dist = np.linalg.norm(bank_positions - T_GC[j, :3, 3], axis=1) + np.arccos(np.clip(cosine, -1, 1))
                order = np.argsort(dist)[:2]
                g = gm(run, i)
                results = [match(bank[k]['patch'], g, u, v, args.search) for k in order]
                results = [r for r in results if r is not None and r[2] >= args.min_peak and r[3] >= args.min_distinct]
                if len(order) >= 2:
                    if len(results) < 2 or np.hypot(results[0][0] - results[1][0], results[0][1] - results[1][1]) > args.agree_px:
                        continue
                    mu = (results[0][0] + results[1][0]) / 2
                    mv = (results[0][1] + results[1][1]) / 2
                    peak = min(results[0][2], results[1][2])
                else:
                    if not results or results[0][2] < 0.75:
                        continue
                    mu, mv, peak = results[0][0], results[0][1], results[0][2]
                pin = C.raw_to_pinhole(np.array([[mu, mv]]))[0]
                if not np.isfinite(pin).all():
                    continue
                new_rows.append((f['id'], run, int(i), float(mu), float(mv), args.sigma_px, 'matched',
                                 int(bank[order[0]]['index']), 0, float(peak), float(pin[0]), float(pin[1])))
                existing.add(key)
                added += 1
            per_feature[f['id']] = added
            if len(per_feature) % 25 == 0:
                print(f'{run}: {len(per_feature)} feature banks processed, {sum(per_feature.values())} new matches', flush=True)
        summary[run] = per_feature
        print(f'{run}: ' + ', '.join(f'{k} +{v}' for k, v in per_feature.items()), flush=True)
    n_old = len(obs['pixel'])
    merged = dict(
        feature=np.concatenate([obs['feature'], np.array([r[0] for r in new_rows], dtype=obs['feature'].dtype)]) if new_rows else obs['feature'],
        run=np.concatenate([obs['run'], np.array([r[1] for r in new_rows], dtype=obs['run'].dtype)]) if new_rows else obs['run'],
        index=np.concatenate([obs['index'], np.array([r[2] for r in new_rows], dtype=np.int64)]) if new_rows else obs['index'],
        pixel=np.vstack([obs['pixel'], np.array([[r[3], r[4]] for r in new_rows], dtype=float)]) if new_rows else obs['pixel'],
        pixel_pinhole=np.vstack([obs['pixel_pinhole'], np.array([[r[10], r[11]] for r in new_rows], dtype=float)]) if new_rows else obs['pixel_pinhole'],
        sigma_px=np.concatenate([obs['sigma_px'], np.array([r[5] for r in new_rows], dtype=float)]) if new_rows else obs['sigma_px'],
        kind=np.concatenate([obs['kind'].astype('<U8'), np.array([r[6] for r in new_rows], dtype='<U8')]) if new_rows else obs['kind'],
        seed_index=np.concatenate([obs['seed_index'], np.array([r[7] for r in new_rows], dtype=np.int64)]) if new_rows else obs['seed_index'],
        steps=np.concatenate([obs['steps'], np.array([r[8] for r in new_rows], dtype=np.int64)]) if new_rows else obs['steps'],
    )
    merged['feature_ids'] = np.array(sorted(set(map(str, merged['feature']))))
    merged['coords'] = np.array('raw')
    np.savez_compressed(args.out, **merged)
    C.write_json(args.out.with_suffix('.json'), dict(
        source_observations=str(args.observations), registration=str(args.fit / 'registration.json'),
        parameters=dict(search_px=args.search, min_peak=args.min_peak, min_distinct=args.min_distinct,
                        agree_px=args.agree_px, sigma_px=args.sigma_px, patch=2 * P + 1),
        added=len(new_rows), total=len(merged['pixel']), previous=n_old, per_run=summary))
    print(f'{n_old} -> {len(merged["pixel"])} observations (+{len(new_rows)} matched) -> {args.out}')


if __name__ == '__main__':
    main()
