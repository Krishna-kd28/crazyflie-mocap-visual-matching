#!/usr/bin/env python3
"""Brute-force refinement of the mocap-to-scene registration on fixed features.

Starting from the current registration (p_G = R0 p_M + t0), a correction
(omega, delta, s) is searched on nested grids:

    R = exp([omega]_x) R0,      p_G = e^s R_c (p_G0 - c) + c + delta,

where p_G0 is the camera centre under the starting registration, c is the
mean camera centre over the observation frames, R_c = exp([omega]_x), and s is
the log scale. Each candidate moves every camera of every run together; the
score is the mean Huber loss of the sigma-normalised reprojection error of the
fixed scene points at the annotated (and tracked) real-image observations. The
grid is refined around the best candidates (coarse-to-fine, top-K retained),
then polished with Huber least squares. Held-out checks: whole frames, whole
runs and whole features.
"""
from __future__ import annotations

import project_config as PC

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_features_common as C  # noqa: E402
from fixed_feature_picker import Store  # noqa: E402

BEHIND = 50.0     # loss for a point behind the camera
try:
    import torch as TORCH
    DEVICE = 'cuda' if TORCH.cuda.is_available() else 'cpu'
except ImportError:  # pragma: no cover
    TORCH, DEVICE = None, None


# ----------------------------------------------------------------------------
# Data assembly
# ----------------------------------------------------------------------------
def load_observations(args, store):
    """Return dict with per-observation arrays and per-feature scene points."""
    points = store.feature_points()
    features = {}
    for f in store.annotations['features']:
        p = points.get(f['id'])
        if f.get('status') != 'confirmed' or p is None or p['X'] is None:
            continue
        if p['method'] == 'triangulation' and p['residual_px'] and max(p['residual_px']) > args.max_render_residual:
            print(f'warning: {f["id"]} render clicks disagree ({p["residual_px"]} px); skipped', flush=True)
            continue
        weak = p['method'] != 'triangulation' or p.get('parallax_deg', 0) < args.min_parallax_deg
        if weak and not args.allow_weak_points:
            print(f'warning: {f["id"]} has no well-triangulated scene point ({p["method"]}, parallax '
                  f'{p.get("parallax_deg")} deg); skipped. Add a render click from a distant frame, or pass --allow-weak-points', flush=True)
            continue
        features[f['id']] = np.asarray(p['X'], dtype=float)
    if not features:
        sys.exit('No confirmed features with scene points')
    if args.observations.exists() and not args.manual_only:
        with np.load(args.observations, allow_pickle=False) as d:
            obs = {k: d[k].copy() for k in d.files if k not in ('feature_ids', 'coords')}
        if 'pixel_pinhole' in obs:          # raw-domain observations: fit in pinhole coordinates
            obs['pixel_raw'] = obs['pixel']
            obs['pixel'] = obs.pop('pixel_pinhole')
        keep = np.array([f in features for f in obs['feature']])
        obs = {k: v[keep] for k, v in obs.items()}
    else:
        rows = []
        for f in store.annotations['features']:
            if f['id'] not in features:
                continue
            for c in f.get('real_clicks', []):
                pin = store.to_pinhole(c['xy'])
                if not np.isfinite(pin).all():
                    continue
                rows.append((f['id'], c['run'], int(c['index']), float(pin[0]), float(pin[1]), float(c.get('sigma_px', 2.0))))
        obs = dict(feature=np.array([r[0] for r in rows]), run=np.array([r[1] for r in rows]),
                   index=np.array([r[2] for r in rows]), pixel=np.array([[r[3], r[4]] for r in rows], dtype=float),
                   sigma_px=np.array([r[5] for r in rows]), kind=np.array(['manual'] * len(rows)),
                   seed_index=np.array([r[2] for r in rows]), steps=np.zeros(len(rows), dtype=int))
    n = len(obs['pixel'])
    if n == 0:
        sys.exit('No observations')
    fids = sorted(features)
    obs['feature_index'] = np.array([fids.index(f) for f in obs['feature']])
    obs['X'] = np.array([features[f] for f in fids])
    obs['feature_ids'] = np.array(fids)
    # Camera poses (mocap) for every observation.
    T_MC = np.zeros((n, 4, 4))
    for run in C.RUNS:
        sel = obs['run'] == run
        if not sel.any():
            continue
        r = store.runs[run]
        pos = r['index']['position']
        T_MC[sel] = r['arrays']['T_MC'][[pos[int(i)] for i in obs['index'][sel]]]
    obs['T_MC'] = T_MC
    return obs


class Model:
    """Vectorised reprojection under corrections of a starting registration."""

    def __init__(self, obs, start, mask=None, prior_sigma=None):
        self.obs = obs
        # Gaussian prior on the correction (rotation rad x3, translation m x3, log scale);
        # infinite sigma disables it. Expressed per unit seed weight, see loss().
        self.prior_sigma = np.full(7, np.inf) if prior_sigma is None else np.asarray(prior_sigma, dtype=float)
        self.R0, self.t0 = start['R'], start['t']
        self.start_lam = start['lam']
        m = np.ones(len(obs['pixel']), bool) if mask is None else mask
        self.mask = m
        T_GC0 = C.apply_registration(obs['T_MC'], self.R0, self.t0, start['lam'])
        self.c = T_GC0[m, :3, 3].mean(axis=0)
        self.R_GC0 = T_GC0[:, :3, :3]
        self.a = obs['X'][obs['feature_index']] - self.c                       # X - c
        self.b = np.einsum('nji,nj->ni', self.R_GC0, T_GC0[:, :3, 3] - self.c)  # R_GC0^T (p0 - c)
        self.u = obs['pixel']
        self.sigma = obs['sigma_px']
        # Each manual click and the observations tracked from it share unit weight,
        # so long tracks cannot outvote independent clicks.
        seeds = np.array([f'{f}|{r}|{s}' for f, r, s in zip(obs['feature'], obs['run'], obs['seed_index'])])
        _, inverse, counts = np.unique(seeds, return_inverse=True, return_counts=True)
        self.weight = 1.0 / counts[inverse]

    def project(self, params):
        """params[B,7] = omega(3), delta(3), s -> pixels[B,N,2], depth[B,N]."""
        params = np.atleast_2d(params)
        Rc = Rotation.from_rotvec(params[:, :3]).as_matrix()          # B,3,3
        lam = np.exp(params[:, 6])
        a_shift = self.a[None] - params[:, None, 3:6]                     # B,N,3
        tmp = np.einsum('bji,bnj->bni', Rc, a_shift)                      # R_c^T (a - delta)
        q = np.einsum('nji,bnj->bni', self.R_GC0, tmp) - lam[:, None, None] * self.b[None]
        h = q @ C.K.T
        z = h[..., 2]
        safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
        return h[..., :2] / safe[..., None], q[..., 2]

    def residual_px(self, params, mask=None):
        pix, z = self.project(params)
        e = np.linalg.norm(pix - self.u[None], axis=-1)
        e = np.where(z > 0.05, e, np.inf)
        return e[0] if mask is None else e[0][mask]

    def loss(self, params, mask=None, chunk=400):
        params = np.atleast_2d(params)
        m = self.mask if mask is None else mask
        w = self.weight[m] / self.weight[m].sum()
        prior = 0.5 * np.sum((params / self.prior_sigma) ** 2, axis=1) / self.weight[m].sum()
        if TORCH is not None:
            return self._loss_torch(params, m, w) + prior
        out = np.empty(len(params))
        for k in range(0, len(params), chunk):
            pix, z = self.project(params[k:k + chunk])
            e = np.linalg.norm(pix[:, m] - self.u[None, m], axis=-1) / self.sigma[None, m]
            rho = np.where(e <= 1, 0.5 * e ** 2, e - 0.5)
            rho = np.where(z[:, m] > 0.05, rho, BEHIND)
            out[k:k + chunk] = rho @ w
        return out + prior

    def _loss_torch(self, params, m, w, chunk=4096):
        torch, dev = TORCH, DEVICE
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
        R_GC0, a, b, u, sigma, wt = f(self.R_GC0[m]), f(self.a[m]), f(self.b[m]), f(self.u[m]), f(self.sigma[m]), f(w)
        Kt = f(C.K)
        out = np.empty(len(params))
        for k in range(0, len(params), chunk):
            p = f(params[k:k + chunk])
            Rc = f(Rotation.from_rotvec(params[k:k + chunk, :3]).as_matrix())
            lam = torch.exp(p[:, 6])
            tmp = torch.einsum('bji,bnj->bni', Rc, a[None] - p[:, None, 3:6])
            q = torch.einsum('nji,bnj->bni', R_GC0, tmp) - lam[:, None, None] * b[None]
            h = q @ Kt.T
            z = h[..., 2]
            safe = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
            pix = h[..., :2] / safe[..., None]
            e = torch.linalg.norm(pix - u[None], dim=-1) / sigma[None]
            rho = torch.where(e <= 1, 0.5 * e ** 2, e - 0.5)
            rho = torch.where(z > 0.05, rho, torch.full_like(rho, BEHIND))
            out[k:k + chunk] = (rho @ wt).cpu().numpy()
        return out

    def registration(self, params):
        """Similarity (R, t, lam) for a correction vector."""
        omega, delta, s = params[:3], params[3:6], params[6]
        Rc = Rotation.from_rotvec(omega).as_matrix()
        lam = float(np.exp(s))
        R = Rc @ self.R0
        t = lam * Rc @ (self.t0 - self.c) + self.c + delta
        return R, t, self.start_lam * lam


def summarise(e, sigma=None):
    e = np.asarray(e, dtype=float)
    finite = np.isfinite(e)
    if not finite.any():
        return dict(n=int(len(e)), rms_px=None, median_px=None, p90_px=None, within_3px=0.0, behind=int((~finite).sum()))
    ef = e[finite]
    return dict(n=int(len(e)), rms_px=float(np.sqrt(np.mean(ef ** 2))), median_px=float(np.median(ef)),
                p90_px=float(np.percentile(ef, 90)), max_px=float(ef.max()),
                within_3px=float(np.mean(ef <= 3)), behind=int((~finite).sum()))


# ----------------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------------
def grid(centre, half, steps, scale_values):
    axes = [np.linspace(centre[i] - half[i], centre[i] + half[i], steps) for i in range(6)]
    axes.append(np.asarray(scale_values) + centre[6])
    mesh = np.meshgrid(*axes, indexing='ij')
    return np.stack([m.ravel() for m in mesh], axis=1)


def diverse_top(params, losses, k, step):
    order = np.argsort(losses)
    picked = []
    for i in order:
        if all(np.max(np.abs(params[i] - params[j]) / step) > 1.5 for j in picked):
            picked.append(i)
        if len(picked) >= k:
            break
    return [int(i) for i in picked]


def search(model, args, mask=None, log=print):
    rot0, tr0 = np.radians(args.rot_deg), args.trans_m
    schedule = [  # (rotation half-width rad, translation half-width m, log-scale offsets, steps, top-K)
        (rot0, tr0, [0.0], args.coarse_steps, 6),
        (rot0 / 4, tr0 / 3, [-0.03, 0, 0.03], 7, 4),
        (rot0 / 11, tr0 / 9, [-0.012, 0, 0.012], 7, 3),
        (rot0 / 30, tr0 / 25, [-0.005, 0, 0.005], 7, 2),
        (rot0 / 90, tr0 / 70, [-0.002, 0, 0.002], 7, 2),
        (rot0 / 250, tr0 / 200, [-0.0008, 0, 0.0008], 7, 1),
    ]
    if args.fix_scale:
        schedule = [(r, t, [0.0], s, k) for r, t, _, s, k in schedule]
    centres = [np.zeros(7)]
    history = []
    best = (np.inf, np.zeros(7))
    for level, (hr, ht, sv, steps, k) in enumerate(schedule):
        cand = np.concatenate([grid(c, [hr] * 3 + [ht] * 3, steps, sv) for c in centres])
        cand = np.unique(np.round(cand, 10), axis=0)
        t0 = time.time()
        losses = model.loss(cand, mask)
        step = np.array([2 * hr / (steps - 1)] * 3 + [2 * ht / (steps - 1)] * 3 + [max(sv) - min(sv) + 1e-9])
        top = diverse_top(cand, losses, k, step)
        i = int(np.argmin(losses))
        if losses[i] < best[0]:
            best = (float(losses[i]), cand[i].copy())
        e = model.residual_px(best[1], mask if mask is not None else model.mask)
        history.append(dict(level=level, candidates=int(len(cand)), rotation_half_width_deg=float(np.degrees(hr)),
                            translation_half_width_m=float(ht), scale_offsets=list(map(float, sv)), steps=steps,
                            top_k=k, best_loss=best[0], best_rms_px=summarise(e)['rms_px'],
                            best_median_px=summarise(e)['median_px'], seconds=round(time.time() - t0, 1),
                            best_params=best[1].tolist()))
        log(f'  level {level}: {len(cand):>8d} candidates, best loss {best[0]:.4f}, '
            f'rms {history[-1]["best_rms_px"]:.2f} px, median {history[-1]["best_median_px"]:.2f} px ({time.time() - t0:.0f} s)')
        centres = [cand[j] for j in top]
    # Local continuous minimisation of the same objective from the best grid point.
    hr, ht = schedule[-1][0], schedule[-1][1]
    span = np.r_[np.full(3, 4 * hr), np.full(3, 4 * ht), 0.004 if not args.fix_scale else 1e-12]
    bounds = list(zip(best[1] - span, best[1] + span))
    res = minimize(lambda p: model.loss(p[None], mask)[0], best[1], method='Powell', bounds=bounds,
                   options=dict(xtol=1e-7, ftol=1e-9, maxfev=4000))
    polished = res.x
    if args.fix_scale:
        polished[6] = 0.0
    polished_loss = float(model.loss(polished[None], mask)[0])
    final = polished if polished_loss <= best[0] else best[1]
    log(f'  polish: loss {best[0]:.5f} -> {polished_loss:.5f} ({"accepted" if polished_loss <= best[0] else "grid kept"})')
    return dict(params=final, grid_params=best[1], grid_loss=best[0], polished_params=polished,
                polished_loss=polished_loss, history=history)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--annotations', type=Path, default=C.ANNOTATIONS)
    ap.add_argument('--extra-annotations', type=Path, nargs='*', default=[C.ROOT / 'research/fixed_features/auto_features.json'],
                    help='further annotation files whose confirmed features are merged in (default: the automatic feature maker output)')
    ap.add_argument('--observations', type=Path, default=C.FF_OUT / 'observations.npz')
    ap.add_argument('--start', type=Path, default=C.START_REGISTRATION)
    ap.add_argument('--out', type=Path, default=C.FF_OUT / 'registration')
    ap.add_argument('--manual-only', action='store_true', help='ignore tracked observations')
    ap.add_argument('--fix-scale', action='store_true', help='search rotation and translation only')
    ap.add_argument('--rot-deg', type=float, default=10.0, help='coarse rotation half-width per axis')
    ap.add_argument('--trans-m', type=float, default=0.6, help='coarse translation half-width per axis')
    ap.add_argument('--coarse-steps', type=int, default=9)
    ap.add_argument('--max-render-residual', type=float, default=4.0)
    ap.add_argument('--min-parallax-deg', type=float, default=5.0, help='render clicks must span at least this ray angle')
    ap.add_argument('--allow-weak-points', action='store_true', help='also use single-click or weak-parallax scene points')
    ap.add_argument('--no-validation', action='store_true')
    ap.add_argument('--prior-rot-deg', type=float, default=3.0, help='prior sigma per rotation axis (0 = none)')
    ap.add_argument('--prior-trans-m', type=float, default=0.2, help='prior sigma per translation axis (0 = none)')
    ap.add_argument('--prior-scale', type=float, default=0.03, help='prior sigma on log scale (0 = none)')
    ap.add_argument('--label', default='Brute-force fixed-feature fit')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    store = Store(args.annotations, gpu=False, extra=args.extra_annotations)
    start = C.load_registration(args.start)
    obs = load_observations(args, store)
    n = len(obs['pixel'])
    print(f'{n} observations of {len(obs["feature_ids"])} features over runs '
          f'{sorted(set(obs["run"]))} ({int(np.sum(obs["kind"] == "manual"))} manual); '
          f'evaluation on {DEVICE or "numpy"}', flush=True)
    inf = np.inf
    prior = np.r_[np.full(3, np.radians(args.prior_rot_deg) if args.prior_rot_deg > 0 else inf),
                  np.full(3, args.prior_trans_m if args.prior_trans_m > 0 else inf),
                  args.prior_scale if args.prior_scale > 0 else inf]
    model = Model(obs, start, prior_sigma=prior)
    e0 = model.residual_px(np.zeros(7))
    print('start:', json.dumps(summarise(e0)), flush=True)
    print('full search on all observations', flush=True)
    full = search(model, args)
    e1 = model.residual_px(full['params'])
    print('final:', json.dumps(summarise(e1)), flush=True)
    R, t, lam = model.registration(full['params'])
    manual = obs['kind'] == 'manual'

    def groups(e):
        out = dict(all=summarise(e), manual=summarise(e[manual]), tracked=summarise(e[~manual]),
                   per_run={C.SHORT[r]: summarise(e[obs['run'] == r]) for r in C.RUNS if (obs['run'] == r).any()},
                   per_feature={f: summarise(e[obs['feature'] == f]) for f in obs['feature_ids']})
        return out
    metrics = dict(start=groups(e0), final=groups(e1), observations=n, features=list(obs['feature_ids']))

    validation = {}
    if not args.no_validation:
        # (a) whole held-out seed frames: every third manual seed frame (with its tracks) per run.
        seeds = sorted({(r, int(s)) for r, s in zip(obs['run'], obs['seed_index'])})
        held = {sd for k, sd in enumerate(seeds) if k % 3 == 2}
        test = np.array([(r, int(s)) in held for r, s in zip(obs['run'], obs['seed_index'])])
        if test.any() and (~test).any():
            print(f'validation (frames): fit on {int((~test).sum())}, check {int(test.sum())} observations', flush=True)
            fit = search(model, args, mask=~test)
            validation['held_out_frames'] = dict(
                held_out_seed_frames=[f'{C.SHORT[r]}:{s}' for r, s in sorted(held)],
                start=summarise(model.residual_px(np.zeros(7), test)),
                fitted=summarise(model.residual_px(fit['params'], test)),
                fitted_on_training=summarise(model.residual_px(fit['params'], ~test)),
                params=fit['params'].tolist())
        # (b) leave-one-run-out.
        runs_present = [r for r in C.RUNS if (obs['run'] == r).any()]
        if len(runs_present) >= 2:
            validation['leave_one_run_out'] = {}
            for r in runs_present:
                test = obs['run'] == r
                print(f'validation (runs): hold out {C.SHORT[r]}', flush=True)
                fit = search(model, args, mask=~test)
                validation['leave_one_run_out'][C.SHORT[r]] = dict(
                    start=summarise(model.residual_px(np.zeros(7), test)),
                    fitted=summarise(model.residual_px(fit['params'], test)),
                    fitted_manual_only=summarise(model.residual_px(fit['params'], test & manual)),
                    params=fit['params'].tolist())
        # (c) leave-one-feature-out (scene-point consistency).
        if len(obs['feature_ids']) >= 3:
            validation['leave_one_feature_out'] = {}
            lofo_ids = [f for f in obs['feature_ids'] if not str(f).startswith('auto_')]
            if len(lofo_ids) < len(obs['feature_ids']):
                print(f'validation (features): {len(obs["feature_ids"]) - len(lofo_ids)} automatic features are not held out one by one '
                      f'(scripts/validate_auto_features.py checks them as a set)', flush=True)
            for f in lofo_ids:
                test = obs['feature'] == f
                print(f'validation (features): hold out {f}', flush=True)
                fit = search(model, args, mask=~test)
                validation['leave_one_feature_out'][f] = dict(
                    start=summarise(model.residual_px(np.zeros(7), test)),
                    fitted=summarise(model.residual_px(fit['params'], test)))
    metrics['validation'] = validation

    # Outputs.
    provenance = dict(
        rotation_provenance='Brute-force coarse-to-fine grid over rotation, translation and scale corrections '
                            'of the starting registration, scored on fixed-feature reprojection; Huber polish',
        translation_provenance='same search', scale_provenance='fixed at source scale' if args.fix_scale else 'searched jointly',
        fit_path=str((args.out / 'search.json').resolve()),
        start_registration=str(args.start), annotations_sha256=C.sha256(args.annotations),
        observations=n, features=list(obs['feature_ids']),
        caveat='Scene points come from render clicks/depth of the reconstruction; observations from manual clicks '
               'and short optical-flow tracks. Timing, intrinsics and the camera mount are unchanged.')
    reg = C.registration_json(R, t, lam, args.label, provenance)
    C.write_json(args.out / 'registration.json', reg)
    correction = dict(rotation_deg=float(np.degrees(np.linalg.norm(full['params'][:3]))),
                      rotation_vector_deg=np.degrees(full['params'][:3]).tolist(),
                      translation_m=full['params'][3:6].tolist(),
                      translation_norm_m=float(np.linalg.norm(full['params'][3:6])),
                      scale=float(np.exp(full['params'][6])), pivot_G=model.c.tolist())
    C.write_json(args.out / 'search.json', dict(
        parameters=vars(args) | {'annotations': str(args.annotations), 'observations': str(args.observations),
                                 'start': str(args.start), 'out': str(args.out),
                                 'extra_annotations': [str(p) for p in (args.extra_annotations or [])]},
        correction=correction, grid_loss=full['grid_loss'], polished_loss=full['polished_loss'],
        grid_params=full['grid_params'].tolist(), polished_params=full['polished_params'].tolist(),
        final_params=full['params'].tolist(), levels=full['history'],
        similarity=dict(R=R.tolist(), t=t.tolist(), **{'lambda': lam})))
    C.write_json(args.out / 'metrics.json', metrics)
    with (args.out / 'residuals.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature', 'run', 'index', 'kind', 'seed_index', 'steps', 'u', 'v', 'sigma_px', 'start_px', 'final_px'])
        for i in range(n):
            w.writerow([obs['feature'][i], obs['run'][i], int(obs['index'][i]), obs['kind'][i], int(obs['seed_index'][i]),
                        int(obs['steps'][i]), f'{obs["pixel"][i, 0]:.2f}', f'{obs["pixel"][i, 1]:.2f}',
                        f'{obs["sigma_px"][i]:.2f}', f'{e0[i]:.3f}', f'{e1[i]:.3f}'])
    print(json.dumps(dict(correction=correction, start=metrics['start']['all'], final=metrics['final']['all']), indent=1))
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
