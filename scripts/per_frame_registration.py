#!/usr/bin/env python3
"""Brute-force pose search for every frame individually on the fixed features.

Each accepted frame starts at its pose under the fitted global registration
and receives its own rigid correction (rotation vector omega, translation
delta in the scene frame):

    R_GC' = exp([omega]_x) R_GC,    p_G' = p_G + delta.

The correction is found on nested grids scored by the frame's own feature
observations (Huber, sigma-normalised, as in the global search) plus a weak
prior that keeps weakly constrained frames close to the global pose.
Frames without any observation take the correction interpolated in time
between their fitted neighbours (within --max-gap-s), otherwise the global
pose. The dataset lists, for every frame, the mode, the observation count,
the correction and the residuals before and after. With --render the frames
are rendered for the comparison video. Run in the CUDA environment.
"""
from __future__ import annotations

import project_config as PC

import argparse
import csv
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

BEHIND = 50.0


def grid_offsets(hr, ht, steps):
    axes = [np.linspace(-hr, hr, steps)] * 3 + [np.linspace(-ht, ht, steps)] * 3
    mesh = np.meshgrid(*axes, indexing='ij')
    return np.stack([m.ravel() for m in mesh], axis=1)


def frame_loss_numpy(T, observation, params, prior_sigma):
    """Same objective for a single Powell candidate, without a GPU round trip."""
    R = Rotation.from_rotvec(params[:3]).as_matrix() @ T[:3, :3]
    q = (observation['X'] - T[:3, 3] - params[3:6]) @ R
    h = q @ C.K.T
    safe = np.where(np.abs(h[:, 2]) < 1e-6, 1e-6, h[:, 2])
    error = np.linalg.norm(h[:, :2] / safe[:, None] - observation['u'], axis=1) / observation['sigma']
    rho = np.where(error <= 1, 0.5 * error**2, error - 0.5)
    rho = np.where(q[:, 2] > .05, rho, BEHIND)
    weight = observation['w']; W = max(float(weight.sum()), 1e-9)
    return float((rho @ weight + 0.5 * np.sum((params / prior_sigma)**2)) / W)


class FrameSet:
    """Observations of one run, padded per frame, on the GPU."""

    def __init__(self, T_GC, obs_by_frame, torch, device):
        self.torch, self.dev = torch, device
        F = len(T_GC)
        n_max = max((len(o['X']) for o in obs_by_frame.values()), default=1)
        X = np.zeros((F, n_max, 3)); u = np.zeros((F, n_max, 2)); sig = np.ones((F, n_max)); w = np.zeros((F, n_max))
        for j, o in obs_by_frame.items():
            n = len(o['X'])
            X[j, :n] = o['X']; u[j, :n] = o['u']; sig[j, :n] = o['sigma']; w[j, :n] = o['w']
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        self.R0, self.p0 = f(T_GC[:, :3, :3]), f(T_GC[:, :3, 3])
        self.X, self.u, self.sig, self.w = f(X), f(u), f(sig), f(w)
        self.W = self.w.sum(dim=1).clamp_min(1e-9)
        self.K = f(C.K)
        self.n_max = n_max

    def loss(self, frames, params, prior_sigma):
        """frames: (B,) indices; params: (B, N, 6) -> (B, N) losses."""
        torch = self.torch
        B, N, _ = params.shape
        omega = params[..., :3].reshape(-1, 3).cpu().numpy()
        Rc = torch.as_tensor(Rotation.from_rotvec(omega).as_matrix(), dtype=torch.float32, device=self.dev).reshape(B, N, 3, 3)
        R = Rc @ self.R0[frames][:, None]                              # B,N,3,3
        p = self.p0[frames][:, None] + params[..., 3:6]               # B,N,3
        X = self.X[frames]                                             # B,M,3
        q = torch.einsum('bnji,bnmj->bnmi', R, X[:, None] - p[:, :, None])   # B,N,M,3
        h = q @ self.K.T
        z = h[..., 2]
        safe = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
        pix = h[..., :2] / safe[..., None]
        e = torch.linalg.norm(pix - self.u[frames][:, None], dim=-1) / self.sig[frames][:, None]
        rho = torch.where(e <= 1, 0.5 * e ** 2, e - 0.5)
        rho = torch.where(z > 0.05, rho, torch.full_like(rho, BEHIND))
        data = (rho * self.w[frames][:, None]).sum(dim=-1) / self.W[frames][:, None]
        prior = 0.5 * ((params / torch.as_tensor(prior_sigma, dtype=torch.float32, device=self.dev)) ** 2).sum(dim=-1) / self.W[frames][:, None]
        return data + prior

    def residuals(self, frames, params):
        """Per-observation pixel errors for one correction per frame: (B, M) with NaN padding."""
        torch = self.torch
        B = len(frames)
        losses_unused = None  # noqa
        omega = params[:, :3]
        Rc = torch.as_tensor(Rotation.from_rotvec(np.asarray(omega.cpu())).as_matrix(), dtype=torch.float32, device=self.dev)
        R = Rc @ self.R0[frames]
        p = self.p0[frames] + params[:, 3:6]
        X = self.X[frames]
        q = torch.einsum('bji,bmj->bmi', R, X - p[:, None])
        h = q @ self.K.T
        z = h[..., 2]
        safe = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
        pix = h[..., :2] / safe[..., None]
        e = torch.linalg.norm(pix - self.u[frames], dim=-1)
        e = torch.where(self.w[frames] > 0, e, torch.full_like(e, float('nan')))
        return e.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='run_20260730T183828')
    ap.add_argument('--annotations', type=Path, default=C.ANNOTATIONS)
    ap.add_argument('--extra-annotations', type=Path, nargs='*', default=[C.ROOT / 'research/fixed_features/auto_features.json'],
                    help='further annotation files whose confirmed features are merged in (default: the automatic feature maker output)')
    ap.add_argument('--observations', type=Path, default=C.FF_OUT / 'observations_extended.npz')
    ap.add_argument('--fit', type=Path, default=C.FF_OUT / 'registration')
    ap.add_argument('--out', type=Path, default=C.FF_OUT / 'per_frame')
    ap.add_argument('--rot-deg', type=float, default=4.0, help='level-0 half width per rotation axis')
    ap.add_argument('--trans-m', type=float, default=0.25, help='level-0 half width per translation axis')
    ap.add_argument('--prior-rot-deg', type=float, default=2.0)
    ap.add_argument('--prior-trans-m', type=float, default=0.10)
    ap.add_argument('--max-gap-s', type=float, default=3.0, help='interpolate corrections across gaps up to this length')
    ap.add_argument('--smooth-frames', type=int, default=0, help='running weighted mean of fitted corrections over +/- this many frames')
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--pinhole-render', action='store_true', help='render undistorted frames instead of lens-distorted ones')
    ap.add_argument('--render-only', action='store_true', help='re-render from the saved per-frame dataset')
    args = ap.parse_args()
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    run = args.run
    out = args.out / run
    out.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        with np.load(out / 'poses_per_frame.npz', allow_pickle=False) as z:
            idx, T_SC, T_CS = z['index'], z['T_SC'], z['T_CS']
        folder, index, arrays = C.frame_store(run)
        args.render = True
        render_frames(args, run, out, idx, index, T_SC, T_CS)
        return
    store = Store(args.annotations, gpu=False, extra=args.extra_annotations)
    points = store.feature_points()
    reg = C.load_registration(args.fit / 'registration.json')
    folder, index, arrays = C.frame_store(run)
    idx = np.array(index['accepted_indices'])
    times = np.array(index['time_s'])
    T_GC = C.apply_registration(arrays['T_MC'], reg['R'], reg['t'], reg['lam'])
    pos = {int(i): j for j, i in enumerate(idx)}
    with np.load(args.observations, allow_pickle=False) as d:
        obs = {k: d[k].copy() for k in d.files if k not in ('feature_ids', 'coords')}
    if 'pixel_pinhole' in obs:
        obs['pixel'] = obs.pop('pixel_pinhole')      # fit in pinhole coordinates
    sel = obs['run'] == run
    feats = {f: np.asarray(p['X']) for f, p in points.items() if p['X'] is not None and p['method'] == 'triangulation'}
    by_frame = {}
    for n in np.flatnonzero(sel):
        f = str(obs['feature'][n])
        if f not in feats or int(obs['index'][n]) not in pos:
            continue
        j = pos[int(obs['index'][n])]
        o = by_frame.setdefault(j, dict(X=[], u=[], sigma=[], w=[], features=[], kinds=[]))
        o['X'].append(feats[f]); o['u'].append(obs['pixel'][n]); o['sigma'].append(float(obs['sigma_px'][n]))
        o['w'].append(1.0); o['features'].append(f); o['kinds'].append(str(obs['kind'][n]))
    for o in by_frame.values():
        for k in ('X', 'u', 'sigma', 'w'):
            o[k] = np.asarray(o[k], dtype=float)
    fitted = np.array(sorted(by_frame))
    print(f'{run}: {len(idx)} frames, {len(fitted)} with observations ({sum(len(o["X"]) for o in by_frame.values())} observations, '
          f'{len(feats)} features); device {dev}', flush=True)
    fs = FrameSet(T_GC, by_frame, torch, dev)
    prior = np.r_[np.full(3, np.radians(args.prior_rot_deg)), np.full(3, args.prior_trans_m)]
    # --- nested grids, all fitted frames in batches
    schedule = [(np.radians(args.rot_deg), args.trans_m, 7)]
    for k in range(1, 5):
        schedule.append((schedule[-1][0] * 0.35, schedule[-1][1] * 0.35, 7))
    params = np.zeros((len(idx), 6))
    t0 = time.time()
    B = 8
    for level, (hr, ht, steps) in enumerate(schedule):
        off = torch.as_tensor(grid_offsets(hr, ht, steps), dtype=torch.float32, device=dev)   # N,6
        for k in range(0, len(fitted), B):
            frames = fitted[k:k + B]
            centre = torch.as_tensor(params[frames], dtype=torch.float32, device=dev)          # B,6
            cand = centre[:, None] + off[None]                                                  # B,N,6
            with torch.no_grad():
                L = fs.loss(torch.as_tensor(frames, device=dev), cand, prior)
            best = L.argmin(dim=1)
            params[frames] = cand[torch.arange(len(frames)), best].cpu().numpy()
        print(f'  level {level}: +/-{np.degrees(hr):.3f} deg, +/-{ht:.3f} m ({time.time() - t0:.0f} s)', flush=True)
    # --- Powell polish per frame on the same objective (CPU, few observations)
    def frame_loss(j, p):
        return frame_loss_numpy(T_GC[j], by_frame[j], np.asarray(p), prior)
    hr, ht = schedule[-1][0], schedule[-1][1]
    for j in fitted:
        span = np.r_[np.full(3, 4 * hr), np.full(3, 4 * ht)]
        res = minimize(lambda p: frame_loss(int(j), p), params[j], method='Powell',
                       bounds=list(zip(params[j] - span, params[j] + span)), options=dict(xtol=1e-6, ftol=1e-8, maxfev=600))
        if res.fun <= frame_loss(int(j), params[j]):
            params[j] = res.x
    print(f'  polish done ({time.time() - t0:.0f} s)', flush=True)
    # --- residuals before/after on fitted frames
    zeros = torch.zeros((len(fitted), 6), dtype=torch.float32, device=dev)
    with torch.no_grad():
        e_before = fs.residuals(torch.as_tensor(fitted, device=dev), zeros)
        e_after = fs.residuals(torch.as_tensor(fitted, device=dev), torch.as_tensor(params[fitted], dtype=torch.float32, device=dev))
    mode = np.array(['global'] * len(idx), dtype='<U12')
    mode[fitted] = 'fitted'
    n_obs = np.zeros(len(idx), int); n_feat = np.zeros(len(idx), int)
    for j, o in by_frame.items():
        n_obs[j] = len(o['X']); n_feat[j] = len(set(o['features']))
    raw = params.copy()
    # --- optional smoothing of fitted corrections (weighted running mean)
    if args.smooth_frames > 0:
        sm = params.copy()
        for j in fitted:
            lo, hi = max(0, j - args.smooth_frames), min(len(idx), j + args.smooth_frames + 1)
            nb = [k for k in range(lo, hi) if mode[k] == 'fitted' and abs(times[k] - times[j]) <= 1.0]
            wts = np.array([n_obs[k] for k in nb], float)
            sm[j] = (params[nb] * wts[:, None]).sum(0) / wts.sum()
        params = sm
    # --- interpolate corrections across gaps
    for j in range(len(idx)):
        if mode[j] == 'fitted':
            continue
        left = fitted[fitted < j]; right = fitted[fitted > j]
        lj = left[-1] if len(left) else None; rj = right[0] if len(right) else None
        cands = [(abs(times[j] - times[k]), k) for k in (lj, rj) if k is not None and abs(times[j] - times[k]) <= args.max_gap_s]
        if lj is not None and rj is not None and times[rj] - times[lj] <= args.max_gap_s:
            a = (times[j] - times[lj]) / (times[rj] - times[lj])
            params[j] = (1 - a) * params[lj] + a * params[rj]
            mode[j] = 'interpolated'
        elif cands:
            _, k = min(cands)
            # fade to the global pose with distance in time
            a = 1 - min(cands)[0] / args.max_gap_s
            params[j] = a * params[k]
            mode[j] = 'interpolated'
    # --- compose poses
    Rc = Rotation.from_rotvec(params[:, :3]).as_matrix()
    T_new = T_GC.copy()
    T_new[:, :3, :3] = Rc @ T_GC[:, :3, :3]
    T_new[:, :3, 3] = T_GC[:, :3, 3] + params[:, 3:6]
    T_SC, T_CS = C.checkpoint_views(T_new)
    rms_b = np.array([np.sqrt(np.nanmean(e ** 2)) for e in e_before]); rms_a = np.array([np.sqrt(np.nanmean(e ** 2)) for e in e_after])
    med_b = np.array([np.nanmedian(e) for e in e_before]); med_a = np.array([np.nanmedian(e) for e in e_after])
    all_b = np.concatenate([e[~np.isnan(e)] for e in e_before]); all_a = np.concatenate([e[~np.isnan(e)] for e in e_after])
    np.savez_compressed(out / 'poses_per_frame.npz', index=idx, frame_id=np.array(index['frame_id']), time_s=times,
                        mode=mode, n_obs=n_obs, n_features=n_feat, correction=params, correction_raw=raw,
                        T_GC_global=T_GC, T_GC=T_new, T_SC=T_SC, T_CS=T_CS, K=C.K)
    with (out / 'poses_per_frame.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['index', 'frame_id', 'time_s', 'mode', 'n_obs', 'n_features', 'rot_deg', 'trans_cm',
                    'rms_before_px', 'rms_after_px', 'median_before_px', 'median_after_px',
                    *[f'T_GC_{r}{c}' for r in range(3) for c in range(4)]])
        fpos = {j: k for k, j in enumerate(fitted)}
        for j in range(len(idx)):
            k = fpos.get(j)
            w.writerow([int(idx[j]), int(index['frame_id'][j]), f'{times[j]:.3f}', mode[j], n_obs[j], n_feat[j],
                        f'{np.degrees(np.linalg.norm(params[j, :3])):.3f}', f'{100 * np.linalg.norm(params[j, 3:6]):.2f}',
                        f'{rms_b[k]:.2f}' if k is not None else '', f'{rms_a[k]:.2f}' if k is not None else '',
                        f'{med_b[k]:.2f}' if k is not None else '', f'{med_a[k]:.2f}' if k is not None else '',
                        *[f'{T_new[j, r, c]:.6f}' for r in range(3) for c in range(4)]])
    summary = dict(
        run=run, frames=int(len(idx)), fitted=int((mode == 'fitted').sum()), interpolated=int((mode == 'interpolated').sum()),
        global_only=int((mode == 'global').sum()), observations=int(n_obs.sum()), features=sorted(feats),
        residual_px=dict(before=dict(rms=float(np.sqrt(np.mean(all_b ** 2))), median=float(np.median(all_b))),
                         after=dict(rms=float(np.sqrt(np.mean(all_a ** 2))), median=float(np.median(all_a)))),
        per_frame_residual_median_px=dict(before=float(np.median(med_b)), after=float(np.median(med_a))),
        correction=dict(rot_deg_median=float(np.median(np.degrees(np.linalg.norm(raw[fitted, :3], axis=1)))),
                        rot_deg_p90=float(np.percentile(np.degrees(np.linalg.norm(raw[fitted, :3], axis=1)), 90)),
                        trans_cm_median=float(np.median(100 * np.linalg.norm(raw[fitted, 3:6], axis=1))),
                        trans_cm_p90=float(np.percentile(100 * np.linalg.norm(raw[fitted, 3:6], axis=1), 90))),
        by_feature_count={str(k): dict(frames=int((n_feat[fitted] == k).sum()) if k < 4 else int((n_feat[fitted] >= 4).sum()),
                                       rms_after=float(np.sqrt(np.nanmean(np.concatenate([e_after[m] for m in range(len(fitted)) if (n_feat[fitted[m]] == k if k < 4 else n_feat[fitted[m]] >= 4)]) ** 2)))
                                       if ((n_feat[fitted] == k) if k < 4 else (n_feat[fitted] >= 4)).any() else None)
                          for k in (1, 2, 3, 4)},
        parameters=dict(rot_deg=args.rot_deg, trans_m=args.trans_m, prior_rot_deg=args.prior_rot_deg, prior_trans_m=args.prior_trans_m,
                        max_gap_s=args.max_gap_s, smooth_frames=args.smooth_frames, polish='same objective, NumPy float64 on CPU', levels=[(float(np.degrees(h)), float(t), s) for h, t, s in schedule]),
        global_registration=str(args.fit / 'registration.json'), observations_file=str(args.observations))
    C.write_json(out / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('fitted', 'interpolated', 'global_only', 'residual_px', 'per_frame_residual_median_px', 'correction', 'by_feature_count')}, indent=1))
    if args.render:
        render_frames(args, run, out, idx, index, T_SC, T_CS)


def render_frames(args, run, out, idx, index, T_SC, T_CS):
    if True:
        from PIL import Image
        from prepare_fixed_feature_frames import Renderer
        renderer = Renderer(distorted=not args.pinhole_render)
        print('render method:', renderer.method, flush=True)
        rdir = out / 'render' / run
        rdir.mkdir(parents=True, exist_ok=True)
        records = []
        for j, i in enumerate(idx):
            Image.fromarray(renderer.rgb(T_CS[j])).save(rdir / f'frame_{int(i):06d}.png')
            records.append(dict(run=run, index=int(i), frame_id=int(index['frame_id'][j]), png_path=str(rdir / f'frame_{int(i):06d}.png')))
        with (out / 'render' / 'render_index.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
        C.write_json(out / 'render' / 'render_metadata.json', dict(
            status='complete_per_frame_brute_force', run=run, pose_sha256=C.sha256(index['pose_source']),
            registration=dict(status='provisional_landmark_fit', label='per-frame brute-force fit on fixed features',
                              global_registration=str(args.fit / 'registration.json'), per_frame=True),
            camera_model='fisheye' if renderer.native else 'pinhole', distortion='none' if args.pinhole_render else renderer.method,
            scene_edited=False, checkpoint=str(C.CHECKPOINT),
            selection=[dict(run=run, index=int(i), frame_id=int(fid)) for i, fid in zip(idx, index['frame_id'])],
            per_frame_dataset=str(out / 'poses_per_frame.npz')))
        np.savez_compressed(out / 'render' / 'render_poses.npz', camera_to_checkpoint=T_SC, checkpoint_to_camera=T_CS, index=idx)
        print(f'rendered {len(records)} frames -> {rdir}')


if __name__ == '__main__':
    main()
