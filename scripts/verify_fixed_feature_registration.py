#!/usr/bin/env python3
"""Check a fitted registration directly in rendered images.

For every manual real-image observation, the scene is rendered at the frame's
corrected pose under the start and the fitted registration, warped with the
supplied lens model into raw pixel geometry. A 21x21 patch of the raw real
image around the click is located in that render by normalized
cross-correlation of gradient magnitude within +/-12 px of the click, giving
the image-space displacement between real and rendered feature. This uses the
rendered pixels, not the render clicks, so it is an independent check of the
reprojection metric. A contact sheet of crops and a JSON summary are written.

Optionally (--full-metrics) every accepted frame of every run is rendered under
the fitted registration and compared with the raw real image by grayscale MAE,
SSIM and gradient correlation, before and after, over the pixels inside the
lens model's valid radius.

Run in the CUDA environment.
"""
from __future__ import annotations

import project_config as PC

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_features_common as C  # noqa: E402
from fixed_feature_picker import Store  # noqa: E402
from gradient_features import gradient_magnitude  # noqa: E402

PATCH, SEARCH = 10, 12


def render_rgb(renderer, T_CS):
    return renderer.rgb(T_CS[0])


def locate(real_gm, rend_gm, u, v):
    """Displacement (du, dv, ncc) of the real patch at (u,v) inside the render around (u,v)."""
    h, w = real_gm.shape
    x, y = int(round(u)), int(round(v))
    if not (PATCH + SEARCH <= x < w - PATCH - SEARCH and PATCH + SEARCH <= y < h - PATCH - SEARCH):
        return None
    template = real_gm[y - PATCH:y + PATCH + 1, x - PATCH:x + PATCH + 1]
    window = rend_gm[y - PATCH - SEARCH:y + PATCH + SEARCH + 1, x - PATCH - SEARCH:x + PATCH + SEARCH + 1]
    if template.std() < 1e-3 or window.std() < 1e-3:
        return None
    score = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    py, px = np.unravel_index(np.argmax(score), score.shape)
    du, dv = px - SEARCH, py - SEARCH
    # sub-pixel parabola
    if 0 < px < score.shape[1] - 1:
        l, c, r = score[py, px - 1], score[py, px], score[py, px + 1]
        d = l - 2 * c + r
        du += float(np.clip(0.5 * (l - r) / d, -0.5, 0.5)) if abs(d) > 1e-9 else 0
    if 0 < py < score.shape[0] - 1:
        l, c, r = score[py - 1, px], score[py, px], score[py + 1, px]
        d = l - 2 * c + r
        dv += float(np.clip(0.5 * (l - r) / d, -0.5, 0.5)) if abs(d) > 1e-9 else 0
    return float(du), float(dv), float(score[py, px])


def gray_metrics(real, rend_rgb):
    valid = C.distortion_canvas()['valid']
    real = real.astype(np.float32) / 255
    g = cv2.cvtColor(rend_rgb.astype(np.float32) / 255, cv2.COLOR_RGB2GRAY)
    err = (g - real)[valid]
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT_101)[5:-5, 5:-5]
    a, b = blur(real), blur(g)
    va, vb, cov = blur(real ** 2) - a * a, blur(g ** 2) - b * b, blur(real * g) - a * b
    ssim = ((2 * a * b + .01 ** 2) * (2 * cov + .03 ** 2)) / ((a * a + b * b + .01 ** 2) * (va + vb + .03 ** 2))
    v5 = valid[5:-5, 5:-5]
    gr, gg = gradient_magnitude(real * 255), gradient_magnitude(g * 255)
    gcorr = float(np.corrcoef(gr[valid].ravel(), gg[valid].ravel())[0, 1])
    return dict(mae=float(np.mean(np.abs(err))), ssim=float(ssim[v5].mean()), gradient_corr=gcorr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fit', type=Path, default=C.FF_OUT / 'registration')
    ap.add_argument('--start', type=Path, default=C.START_REGISTRATION)
    ap.add_argument('--annotations', type=Path, default=C.ANNOTATIONS)
    ap.add_argument('--extra-annotations', type=Path, nargs='*', default=[C.ROOT / 'research/fixed_features/auto_features.json'],
                    help='further annotation files whose confirmed features are merged in (default: the automatic feature maker output)')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--full-metrics', action='store_true')
    ap.add_argument('--full-stride', type=int, default=1, help='frame stride for full-image metrics')
    args = ap.parse_args()
    out = args.out or args.fit / 'verify'
    out.mkdir(parents=True, exist_ok=True)
    import torch
    from prepare_fixed_feature_frames import Renderer
    renderer = Renderer(distorted=True)
    start = C.load_registration(args.start)
    final = C.load_registration(args.fit / 'registration.json')
    store = Store(args.annotations, gpu=False, extra=args.extra_annotations)
    points = store.feature_points()
    rows, tiles = [], []
    t0 = time.time()
    todo = []
    for f in store.annotations['features']:
        p = points.get(f['id'])
        if f.get('status') != 'confirmed' or p is None or p['X'] is None:
            continue
        for c in f.get('real_clicks', []):
            todo.append((f, np.asarray(p['X']), c))
    hand = [t for t in todo if not str(t[2].get('method', '')).startswith('auto_')]
    if hand:
        todo = hand                     # the rendered-patch check is about the user's own clicks
    else:                               # automatic features only: an evenly spaced sample of the matched positions
        todo = [t for t in todo if t[2].get('method') == 'auto_lightglue']
        todo = todo[::max(1, len(todo) // 200)][:200]
        print(f'no hand clicks: checking {len(todo)} automatic matched positions', flush=True)
    with torch.no_grad():
        for f, X, c in todo:
            if True:
                run, i = c['run'], int(c['index'])
                r = store.runs[run]
                j = r['index']['position'][i]
                T_MC = r['arrays']['T_MC'][j:j + 1]
                real = cv2.imread(str(C.real_raw_path(run, i)), cv2.IMREAD_GRAYSCALE)
                real_gm = gradient_magnitude(real)
                click_pin = store.to_pinhole(c['xy'])
                if not np.isfinite(click_pin).all():
                    continue
                record = dict(feature=f['id'], run=C.SHORT[run], index=i, u=c['xy'][0], v=c['xy'][1])
                crops = [real]
                for tag, reg in (('start', start), ('final', final)):
                    T_GC = C.apply_registration(T_MC, reg['R'], reg['t'], reg['lam'])
                    _, T_CS = C.checkpoint_views(T_GC)
                    rgb = render_rgb(renderer, T_CS)
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    uv, z = C.project(T_GC, X[None])
                    record[f'{tag}_reproj_px'] = float(np.hypot(uv[0, 0] - click_pin[0], uv[0, 1] - click_pin[1]))
                    loc = locate(real_gm, gradient_magnitude(gray), c['xy'][0], c['xy'][1])
                    if loc is None:
                        record[f'{tag}_ncc_shift_px'] = None
                        record[f'{tag}_ncc'] = None
                    else:
                        record[f'{tag}_ncc_shift_px'] = float(np.hypot(loc[0], loc[1]))
                        record[f'{tag}_ncc'] = loc[2]
                    pred_raw = C.pinhole_to_raw(uv)[0]
                    record[f'{tag}_pred'] = [float(pred_raw[0]), float(pred_raw[1])]
                    crops.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                rows.append(record)
                if c.get('method', 'manual') == 'manual':
                    tiles.append((record, crops))
    print(f'{len(rows)} observations checked in {time.time() - t0:.0f} s', flush=True)

    def stats(key):
        vals = np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)
        if not len(vals):
            return None
        return dict(n=int(len(vals)), rms=float(np.sqrt(np.mean(vals ** 2))), median=float(np.median(vals)),
                    p90=float(np.percentile(vals, 90)), within_3px=float(np.mean(vals <= 3)))
    summary = {tag: dict(reprojection=stats(f'{tag}_reproj_px'), ncc_shift=stats(f'{tag}_ncc_shift_px'),
                         mean_ncc=(lambda v: float(np.mean(v)) if v else None)([r[f'{tag}_ncc'] for r in rows if r.get(f'{tag}_ncc') is not None]))
               for tag in ('start', 'final')}
    summary['per_feature'] = {}
    for fid in sorted({r['feature'] for r in rows}):
        sel = [r for r in rows if r['feature'] == fid]
        summary['per_feature'][fid] = {tag: dict(
            reproj_rms=float(np.sqrt(np.mean([r[f'{tag}_reproj_px'] ** 2 for r in sel]))),
            ncc_shift_median=(lambda v: float(np.median(v)) if v else None)([r[f'{tag}_ncc_shift_px'] for r in sel if r.get(f'{tag}_ncc_shift_px') is not None]))
            for tag in ('start', 'final')}
    with (out / 'observation_checks.csv').open('w', newline='') as fh:
        keys = ['feature', 'run', 'index', 'u', 'v', 'start_reproj_px', 'final_reproj_px', 'start_ncc_shift_px',
                'final_ncc_shift_px', 'start_ncc', 'final_ncc']
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    # Contact sheet: real | start render | final render crops with the click and predictions.
    def crop(im, cx, cy, marks, size=24, zoom=5):
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        pad = cv2.copyMakeBorder(im, size, size, size, size, cv2.BORDER_CONSTANT)
        x, y = int(round(cx)), int(round(cy))
        c = pad[y:y + 2 * size + 1, x:x + 2 * size + 1]
        c = cv2.resize(c, ((2 * size + 1) * zoom, (2 * size + 1) * zoom), interpolation=cv2.INTER_NEAREST)
        for (mx, my), col in marks:
            px = int(round((mx - x + size + 0.5) * zoom)), int(round((my - y + size + 0.5) * zoom))
            cv2.drawMarker(c, px, col, cv2.MARKER_CROSS, 22, 1, cv2.LINE_AA)
        return c
    sheet_rows = []
    for record, crops in tiles[:60]:
        u, v = record['u'], record['v']
        yellow, red, green = (0, 255, 255), (0, 0, 255), (0, 220, 0)
        a = crop(crops[0], u, v, [((u, v), yellow)])
        b = crop(crops[1], u, v, [((u, v), yellow), (record['start_pred'], red)])
        c = crop(crops[2], u, v, [((u, v), yellow), (record['final_pred'], green)])
        tile = np.hstack([a, b, c])
        label = f"{record['feature']} {record['run']}:{record['index']}  start {record['start_reproj_px']:.1f} px, final {record['final_reproj_px']:.1f} px"
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 16), (0, 0, 0), -1)
        cv2.putText(tile, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
        sheet_rows.append(tile)
    if sheet_rows:
        while len(sheet_rows) % 2:
            sheet_rows.append(np.zeros_like(sheet_rows[0]))
        sheet = np.vstack([np.hstack(sheet_rows[k:k + 2]) for k in range(0, len(sheet_rows), 2)])
        cv2.imwrite(str(out / 'feature_checks.png'), sheet)
        C.write_json(out / 'feature_checks_legend.json', dict(
            columns=['raw real (yellow = click)', 'start render, lens model applied (red = predicted feature)',
                     'final render, lens model applied (green = predicted feature)'],
            crop_half_size_px=24, zoom=5))

    if args.full_metrics:
        full = {}
        with torch.no_grad():
            for run in C.RUNS:
                r = store.runs[run]
                idx = r['index']['accepted_indices'][::args.full_stride]
                T_MC = r['arrays']['T_MC'][::args.full_stride]
                acc = {tag: [] for tag in ('start', 'final')}
                for tag, reg in (('start', start), ('final', final)):
                    T_GC = C.apply_registration(T_MC, reg['R'], reg['t'], reg['lam'])
                    _, T_CS = C.checkpoint_views(T_GC)
                    for k, i in enumerate(idx):
                        real = cv2.imread(str(C.real_raw_path(run, i)), cv2.IMREAD_GRAYSCALE)
                        if tag == 'start' and C.render_distorted_path(run, i).exists():
                            rgb = cv2.cvtColor(cv2.imread(str(C.render_distorted_path(run, i))), cv2.COLOR_BGR2RGB)
                        else:
                            rgb = render_rgb(renderer, T_CS[k:k + 1])
                        acc[tag].append(gray_metrics(real, rgb))
                full[C.SHORT[run]] = {tag: {m: float(np.mean([x[m] for x in acc[tag]])) for m in ('mae', 'ssim', 'gradient_corr')}
                                      for tag in ('start', 'final')}
                full[C.SHORT[run]]['frames'] = len(idx)
                print(f'{run}: {json.dumps(full[C.SHORT[run]])}', flush=True)
        summary['full_image'] = full
    C.write_json(out / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('start', 'final')}, indent=1))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
