#!/usr/bin/env python3
"""Automatic fixed-feature maker: real frames matched to 3DGS renders.

Stages (run all, or one at a time):

  render  Render every accepted frame of a run at the fitted global registration,
          in the camera's raw geometry (gsplat); the registration used is copied
          next to the renders so later refits cannot change what the render
          keypoints mean.
  match   On every k-th rendered frame (key frames, --stride, default every
          frame): SuperPoint keypoints + LightGlue on the real frame vs its
          render (MINIMA's cross-modal LightGlue weights) and render vs the key
          renders 1, 2 and 5 steps ahead (stock LightGlue), to chain the same
          scene point across frames.
  build   Chain render keypoints into tracks, triangulate each track from the
          exactly known render poses, merge duplicates, then attach every
          real<->render match whose render keypoint lies where a candidate is
          expected (this recovers matches the chaining missed), re-triangulate,
          validate every candidate against the global registration and against
          moved objects, then pick candidates greedily for frame coverage (each
          pick adds most to the frames that still have fewer than --min-per-frame,
          until no frame can gain or --top is reached) and write them for review.
  track   Propagate each candidate's matched observations through the frames in
          between and beyond with CoTracker3 (bidirectional windows between
          matched frames, consistency-checked), so every accepted frame in a
          candidate's span carries a position; written to propagated.json and
          into the candidate file as clicks of method auto_cotracker3.

A candidate is a physical point that (a) is seen by the matcher in both the
real frame and the render of several frames, (b) triangulates consistently
from the renders, and (c) reprojects consistently into the real frames under
the global registration; points on moved objects fail (c) because their real
observations triangulate to a different place than the render point.
Nothing is accepted automatically: candidates are written with status
"proposed" to research/fixed_features/auto_features.json (a separate file
from the manual annotations); scripts/auto_feature_viewer.py records decisions.
"""
from __future__ import annotations

import project_config as PC

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_features_common as C  # noqa: E402

OUT = C.ROOT / 'output/auto_features'
AUTO_ANNOTATIONS = C.ROOT / 'research/fixed_features/auto_features.json'
FLOOR_ABC = np.array(PC.CONFIG['floor_plane_abc'])


def run_dir(run):
    d = OUT / run
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------
def stage_render(args):
    from PIL import Image
    from prepare_fixed_feature_frames import Renderer
    reg = C.load_registration(args.registration)
    folder, index, arrays = C.frame_store(args.run)
    idx = np.array(index['accepted_indices'])
    T_GC = C.apply_registration(arrays['T_MC'], reg['R'], reg['t'], reg['lam'])
    T_SC, T_CS = C.checkpoint_views(T_GC)
    renderer = Renderer(distorted=True)
    d = run_dir(args.run)
    (d / 'renders').mkdir(exist_ok=True)
    used = d / 'registration_used.json'
    used.write_text(Path(args.registration).read_text())
    t0 = time.time()
    for n, i in enumerate(idx):
        path = d / 'renders' / f'{int(i):06d}.png'
        if not path.exists() or args.overwrite:
            Image.fromarray(renderer.rgb(T_CS[n])).save(path)
        if n % 200 == 0:
            print(f'{n + 1}/{len(idx)} frames rendered ({time.time() - t0:.0f} s)', flush=True)
    np.savez_compressed(d / 'frames.npz', index=idx, time_s=np.asarray(index['time_s']), T_GC=T_GC, T_CS=T_CS)
    C.write_json(d / 'frames.json', dict(run=args.run, registration=str(args.registration),
                                          registration_used=str(used.relative_to(C.ROOT)),
                                          frames=[int(i) for i in idx], render_method=renderer.method,
                                          time_s=[float(t) for t in index['time_s']]))
    print(f'{len(idx)} frames rendered ({renderer.method}) in {time.time() - t0:.0f} s -> {d / "renders"}', flush=True)


# ----------------------------------------------------------------------------
class Matcher:
    """SuperPoint keypoints + LightGlue (lightglue package).

    real<->render pairs use MINIMA's cross-modal fine-tune of LightGlue when its
    checkpoint is present (output/auto_features/minima_weights/minima_lightglue.pth),
    which gave the fewest wrong matches in scripts/benchmark_matchers.py;
    render<->render pairs use the stock LightGlue.
    """

    MINIMA = PC.path('minima_weights')

    def __init__(self, max_keypoints=2048, device='cuda'):
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
        self.torch, self.rbd, self.dev = torch, rbd, device
        self.ext = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        self.lg = LightGlue(features='superpoint').eval().to(device)
        self.lg_cross = self.lg
        self.name = 'SuperPoint + LightGlue'
        if PC.CONFIG.get('cross_matcher', 'minima') == 'minima':
            if not self.MINIMA.is_file():
                raise FileNotFoundError(f'{self.MINIMA}: run python pipeline.py fetch-weights, or explicitly select stock cross_matcher')
            lg = LightGlue(features='superpoint')
            sd = torch.load(self.MINIMA, map_location='cpu', weights_only=False)
            sd = sd.get('state_dict', sd)
            for i in range(lg.conf.n_layers):
                sd = {k.replace(f'self_attn.{i}', f'transformers.{i}.self_attn'): v for k, v in sd.items()}
                sd = {k.replace(f'cross_attn.{i}', f'transformers.{i}.cross_attn'): v for k, v in sd.items()}
            lg.load_state_dict(sd, strict=False)
            self.lg_cross = lg.eval().to(device)
            self.name = 'SuperPoint + LightGlue (MINIMA cross-modal weights for real vs render)'

    def features(self, gray):
        t = self.torch.from_numpy(np.array(gray, copy=True)).float()[None, None].to(self.dev) / 255.0
        with self.torch.no_grad():
            return self.ext.extract(t.repeat(1, 3, 1, 1)[0])

    def match(self, fa, fb, cross=False):
        with self.torch.no_grad():
            out = (self.lg_cross if cross else self.lg)({'image0': fa, 'image1': fb})
        out = self.rbd(out)
        m = out['matches'].cpu().numpy().astype(np.int64)
        s = out['scores'].cpu().numpy() if 'scores' in out else np.ones(len(m))
        return m, s

    @staticmethod
    def keypoints(f):
        return f['keypoints'][0].cpu().numpy() if f['keypoints'].dim() == 3 else f['keypoints'].cpu().numpy()


def stage_match(args):
    import cv2
    d = run_dir(args.run)
    meta = json.loads((d / 'frames.json').read_text())
    frames = meta['frames'][::args.stride]          # key frames
    matcher = Matcher(args.max_keypoints)
    (d / 'matches').mkdir(exist_ok=True)
    feats = {}
    t0 = time.time()

    def get(i):
        if i not in feats:
            real = cv2.imread(str(C.real_raw_path(args.run, i)), cv2.IMREAD_GRAYSCALE)
            rend = cv2.imread(str(d / 'renders' / f'{i:06d}.png'), cv2.IMREAD_GRAYSCALE)
            feats[i] = (matcher.features(real), matcher.features(rend))
            if len(feats) > max(args.chain_steps) + 2:
                feats.pop(next(iter(feats)))
        return feats[i]
    total_rr = 0
    print('matcher:', matcher.name, flush=True)
    for n, i in enumerate(frames):
        f_real, f_rend = get(i)
        rr_idx, rr_score = matcher.match(f_real, f_rend, cross=True)
        total_rr += len(rr_idx)
        out = dict(kp_real=matcher.keypoints(f_real), kp_rend=matcher.keypoints(f_rend),
                   rr_idx=rr_idx, rr_score=rr_score)
        for step in args.chain_steps:
            if n + step < len(frames):
                j = frames[n + step]
                _, g_rend = get(j)
                idx, score = matcher.match(f_rend, g_rend)
                out[f'next{step}_frame'] = np.array(j)
                out[f'next{step}_idx'] = idx
                out[f'next{step}_score'] = score
            np.savez_compressed(d / 'matches' / f'{i:06d}.npz', **out)
        if n % 50 == 0:
            print(f'{n + 1}/{len(frames)} frames matched ({time.time() - t0:.0f} s)', flush=True)
    meta['matcher'] = matcher.name
    meta['key_frames'] = [int(i) for i in frames]
    meta['stride'] = args.stride
    meta['chain_steps'] = list(args.chain_steps)
    meta['max_keypoints'] = args.max_keypoints
    C.write_json(d / 'frames.json', meta)
    print(f'real<->render matches: {total_rr} over {len(frames)} key frames -> {d / "matches"}', flush=True)


# ----------------------------------------------------------------------------
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def parallax_deg(T, uv):
    d = np.array([Ti[:3, :3] @ np.linalg.solve(C.K, np.r_[u, 1.0]) for Ti, u in zip(T, uv)])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    cos = np.clip(d @ d.T, -1, 1)
    return float(np.degrees(np.arccos(cos.min())))


def robust_triangulate(T, uv, max_resid=1.5):
    keep = list(range(len(T)))
    while True:
        X, err, depth = C.triangulate(T[keep], uv[keep])
        if len(keep) <= 2 or err.max() <= max_resid:
            break
        keep.pop(int(np.argmax(err)))
    return X, err, depth, keep


def stage_build(args):
    from scipy.spatial import cKDTree
    d = run_dir(args.run)
    meta = json.loads((d / 'frames.json').read_text())
    frames = meta['frames']
    key = meta.get('key_frames', frames)
    with np.load(d / 'frames.npz', allow_pickle=False) as z:
        T_GC = z['T_GC']
        times = z['time_s']
    fpos = {int(i): n for n, i in enumerate(frames)}
    M = {}
    for i in key:
        with np.load(d / 'matches' / f'{i:06d}.npz', allow_pickle=False) as z:
            M[i] = {k: z[k] for k in z.files}
    # --- 1. chain render keypoints into tracks through the key frames
    uf = UnionFind()
    for i in key:
        m = M[i]
        for name in [k for k in m if k.startswith('next') and k.endswith('_idx')]:
            j = int(m[name.replace('_idx', '_frame')])
            for a, b in m[name]:
                uf.union((i, int(a)), (j, int(b)))
    members = {}
    for i in key:
        for k in range(len(M[i]['kp_rend'])):
            members.setdefault(uf.find((i, k)), []).append((i, k))
    tracks = [sorted(v) for v in members.values() if len({f for f, _ in v}) >= 2]
    print(f'{len(tracks)} render tracks spanning >= 2 key frames', flush=True)
    real_of = {}
    for i in key:
        m = M[i]
        for (a, b), sc in zip(m['rr_idx'], m['rr_score']):
            real_of[(i, int(b))] = (int(a), float(sc))

    def triangulate_nodes(nodes):
        """nodes: {frame: render keypoint index}. Robust triangulation from the render poses."""
        items = sorted(nodes.items())
        T = np.array([T_GC[fpos[f]] for f, _ in items])
        uv_raw = np.array([M[f]['kp_rend'][k] for f, k in items], dtype=float)
        uv = C.raw_to_pinhole(uv_raw)
        ok = np.isfinite(uv).all(axis=1)
        if ok.sum() < 2:
            return None
        items = [it for it, o in zip(items, ok) if o]
        T, uv, uv_raw = T[ok], uv[ok], uv_raw[ok]
        X, err, depth, keep = robust_triangulate(T, uv)
        if len(keep) < 2 or np.any(depth <= 0):
            return None
        par = parallax_deg(T[keep], uv[keep])
        if par < args.min_parallax:
            return None
        return dict(X=X, nodes=[items[n] for n in keep], uv_raw=[uv_raw[n] for n in keep], err=float(err.max()), par=par)

    # --- 2. triangulate every track
    tri = []
    for nodes in tracks:
        seen = {}
        for f, k in nodes:
            seen.setdefault(f, k)
        t = triangulate_nodes(seen)
        if t is not None:
            tri.append(t)
    print(f'{len(tri)} tracks triangulated with parallax >= {args.min_parallax} deg', flush=True)
    # --- 3. merge duplicates (the same physical point in separate chains)
    uf2 = UnionFind()
    for a, b in cKDTree(np.array([t['X'] for t in tri])).query_pairs(args.merge_m):
        uf2.union(int(a), int(b))
    groups = {}
    for n in range(len(tri)):
        groups.setdefault(uf2.find(n), []).append(n)
    cands = []
    for mem in groups.values():
        nodes = {}
        for n in sorted(mem, key=lambda n: -len(tri[n]['nodes'])):
            for f, k in tri[n]['nodes']:
                nodes.setdefault(f, k)
        t = triangulate_nodes(nodes)
        if t is not None:
            cands.append(dict(nodes=dict(t['nodes']), X=t['X']))
    print(f'{len(cands)} candidates after merging within {args.merge_m} m', flush=True)
    # --- 4. attach unchained matches whose render keypoint lies where a candidate is expected
    owner = {}
    for n, c in enumerate(cands):
        for f, k in c['nodes'].items():
            owner[(f, k)] = n
    Xc = np.array([c['X'] for c in cands])
    assigned = 0
    for i in key:
        uv, z = C.project(np.repeat(T_GC[fpos[i]][None], len(Xc), 0), Xc)
        raw = C.pinhole_to_raw(uv)
        inview = (z > 0.3) & np.isfinite(raw).all(axis=1) & (raw[:, 0] > -5) & (raw[:, 0] < C.WIDTH + 5) & (raw[:, 1] > -5) & (raw[:, 1] < C.HEIGHT + 5)
        ids = np.flatnonzero(inview)
        if len(ids) == 0:
            continue
        tree = cKDTree(raw[ids])
        m = M[i]
        for (a, b), sc in zip(m['rr_idx'], m['rr_score']):
            b = int(b)
            if (i, b) in owner:
                continue
            dist, nn = tree.query(m['kp_rend'][b], k=2, distance_upper_bound=2 * args.assign_px)
            if not np.isfinite(dist[0]) or dist[0] > args.assign_px or np.isfinite(dist[1]):
                continue            # nothing expected here, or two candidates expected: ambiguous
            n = int(ids[nn[0]])
            if i in cands[n]['nodes']:
                continue
            cands[n]['nodes'][i] = b
            owner[(i, b)] = n
            assigned += 1
    print(f'{assigned} matches attached by expected position (within {args.assign_px} px)', flush=True)
    # --- 5. re-triangulate, collect real observations, check
    candidates = []
    for c in cands:
        t = triangulate_nodes(c['nodes'])
        if t is None:
            continue
        X = t['X']
        obs = []
        for (f, k), uv_raw in zip(t['nodes'], t['uv_raw']):
            if (f, k) in real_of:
                kr, sc = real_of[(f, k)]
                obs.append(dict(frame=int(f), xy=[float(x) for x in M[f]['kp_real'][kr]], score=round(sc, 3),
                                render_xy=[float(x) for x in uv_raw]))
        if len(obs) < args.min_obs:
            continue
        To = np.array([T_GC[fpos[o['frame']]] for o in obs])
        pred = C.pinhole_to_raw(C.project(To, np.repeat(X[None], len(obs), 0))[0])
        e = np.linalg.norm(pred - np.array([o['xy'] for o in obs]), axis=1)
        ok = e <= args.max_obs_reproj                       # a grossly wrong real match is dropped on its own
        obs = [o for o, k in zip(obs, ok) if k]
        e, To = e[ok], To[ok]
        if len(obs) < args.min_obs or times[fpos[obs[-1]['frame']]] - times[fpos[obs[0]['frame']]] < args.min_span_s:
            continue
        uv_real = C.raw_to_pinhole(np.array([o['xy'] for o in obs]))
        fin = np.isfinite(uv_real).all(axis=1)
        par_real, X_real = 0.0, X
        if fin.sum() >= 2:
            X_real, _, _ = C.triangulate(To[fin], uv_real[fin])
            par_real = parallax_deg(To[fin], uv_real[fin])
        dX = float(np.linalg.norm(X_real - X)) if par_real >= 2 else None
        floor_z = FLOOR_ABC[0] * X[0] + FLOOR_ABC[1] * X[1] + FLOOR_ABC[2]
        checks = dict(render_residual_px=t['err'], parallax_deg=t['par'],
                      real_reproj_median_px=float(np.median(e)), real_reproj_max_px=float(e.max()),
                      real_vs_render_point_m=dX, real_parallax_deg=par_real,
                      on_floor=bool(abs(X[2] - floor_z) < 0.15), n_obs=len(obs), n_render_nodes=len(t['nodes']),
                      span_s=float(times[fpos[obs[-1]['frame']]] - times[fpos[obs[0]['frame']]]))
        reasons = []
        if getattr(args, 'exclude_floor', False) and checks['on_floor']:
            reasons.append('floor excluded by the run protocol')
        if np.linalg.norm(X[:2]) < getattr(args, 'exclude_central_radius', 0.0):
            reasons.append('central movable gate region excluded by the run protocol')
        if checks['real_reproj_median_px'] > args.max_reproj:
            reasons.append('real observations disagree with the global registration')
        if dX is not None and dX > args.max_point_shift:
            reasons.append('real observations triangulate elsewhere (moved object?)')
        candidates.append(dict(X=[float(x) for x in X],
                               nodes=[(int(f), [float(x) for x in uv]) for (f, _), uv in zip(t['nodes'], t['uv_raw'])],
                               observations=obs, checks=checks, rejected=reasons))
    print(f'{len(candidates)} candidates with >= {args.min_obs} real observations', flush=True)
    valid = [c for c in candidates if not c['rejected']]
    # --- 6. select for coverage: each pick adds the most to frames that still have few candidates
    # (a floor point counts half); picks stay at least min_separation apart
    frame_count = {f: 0 for f in key}
    remaining = list(valid)
    kept = []
    while remaining and len(kept) < args.top:
        best, best_gain = None, 0.0
        for c in remaining:
            if not any(frame_count[o['frame']] < args.min_per_frame for o in c['observations']):
                continue                    # adds nothing to a frame that still needs candidates
            gain = sum(1.0 / (1.0 + frame_count[o['frame']]) for o in c['observations']) * (0.5 if c['checks']['on_floor'] else 1.0)
            if gain > best_gain:
                best, best_gain = c, gain
        if best is None:
            break                           # every frame that can reach the target has reached it
        best['score'] = round(best_gain, 2)
        kept.append(best)
        X = np.array(best['X'])
        remaining = [c for c in remaining if c is not best and np.linalg.norm(np.array(c['X']) - X) > args.min_separation]
        for o in best['observations']:
            frame_count[o['frame']] += 1
    covered = {k: sum(1 for f in key if frame_count[f] >= k) for k in (1, 3, args.min_per_frame)}
    print(f'coverage of the {len(key)} key frames by the {len(kept)} selected candidates: '
          f'{covered[1]} frames with >= 1 matched candidate, {covered[3]} with >= 3, {covered[args.min_per_frame]} with >= {args.min_per_frame}', flush=True)
    # --- write outputs, preserving earlier decisions by 3-D position
    previous = json.loads(AUTO_ANNOTATIONS.read_text()) if AUTO_ANNOTATIONS.exists() else None
    prev_by_X = []
    if previous:
        for f in previous.get('features', []):
            if f.get('status') in ('confirmed', 'rejected') and f.get('point_G'):
                prev_by_X.append((np.array(f['point_G']), f))
    reg_used = meta.get('registration_used') or str(Path(meta['registration']).resolve().relative_to(C.ROOT))
    features = []
    for n, c in enumerate(kept):
        X = np.array(c['X'])
        prior = next((f for Xp, f in prev_by_X if np.linalg.norm(Xp - X) < args.merge_m), None)
        fid = prior['id'] if prior else f'auto_{n + 1:03d}'
        status = prior['status'] if prior else 'proposed'
        ch = c['checks']
        features.append(dict(
            id=fid, status=status, source='auto_features', created=dt.datetime.now().isoformat(timespec='seconds'),
            description=f"auto: {ch['n_obs']} matched frames over {ch['span_s']:.1f} s, parallax {ch['parallax_deg']:.1f} deg, "
                        f"real reprojection median {ch['real_reproj_median_px']:.1f} px" + (', near floor plane' if ch['on_floor'] else ''),
            point_G=c['X'], score=c['score'], checks=ch, run=args.run,
            render_clicks=[dict(run=args.run, index=f, xy=[round(x, 2) for x in xy], sigma_px=1.0, method='auto_superpoint',
                                coords='raw', render_registration=reg_used) for f, xy in c['nodes']],
            real_clicks=[dict(run=args.run, index=o['frame'], xy=[round(x, 2) for x in o['xy']], sigma_px=2.5,
                              method='auto_lightglue', match_score=o['score'], coords='raw') for o in c['observations']]))
        c['id'] = fid
    C.write_json(AUTO_ANNOTATIONS, dict(
        version=1, coords='raw', coordinate_convention='Raw recorded pixel coordinates (see annotations.json)',
        source='scripts/auto_features.py', run=args.run, registration=meta['registration'], registration_used=reg_used,
        matcher=meta.get('matcher', ''), features=features, updated=dt.datetime.now().isoformat(timespec='seconds')))
    C.write_json(d / 'candidates.json', dict(
        run=args.run, frames=key, registration=meta['registration'], registration_used=reg_used,
        parameters=vars(args) | {'registration': str(args.registration)},
        counts=dict(tracks=len(tracks), triangulated=len(tri), merged=len(cands), assigned=assigned, checked=len(candidates),
                    valid=len(valid), kept=len(kept), rejected=len(candidates) - len(valid),
                    key_frames_with_1=covered[1], key_frames_with_3=covered[3], key_frames_with_target=covered[args.min_per_frame]),
        kept=kept, valid_unselected=[dict(X=c['X'], frames=[o['frame'] for o in c['observations']], checks=c['checks'])
                                     for c in valid if c not in kept],
        rejected=[c for c in candidates if c['rejected']][:400]))
    print(f'kept {len(kept)} candidates ({len(valid)} valid of {len(candidates)} checked) -> {AUTO_ANNOTATIONS}')


# ----------------------------------------------------------------------------
def stage_track(args):
    """CoTracker3 propagation of every kept candidate through the accepted frames."""
    import cv2
    import torch
    d = run_dir(args.run)
    meta = json.loads((d / 'frames.json').read_text())
    frames = meta['frames']
    with np.load(d / 'frames.npz', allow_pickle=False) as z:
        T_GC = z['T_GC']
        times = z['time_s']
    fpos = {int(i): n for n, i in enumerate(frames)}
    cands = json.loads((d / 'candidates.json').read_text())
    model = PC.load_cotracker('cuda')
    gray = {}

    def load(i):
        if i not in gray:
            gray[i] = cv2.imread(str(C.real_raw_path(args.run, i)), cv2.IMREAD_GRAYSCALE)
        return gray[i]

    def run_window(js, queries):
        """js: contiguous frame positions; queries: [(t, x, y)]. Tracks [Q, N, 2], visibility [Q, N]."""
        video = torch.from_numpy(np.stack([np.repeat(load(frames[j])[..., None], 3, axis=2) for j in js]))
        video = video.permute(0, 3, 1, 2)[None].float().cuda()
        q = torch.tensor([[[float(t), float(x), float(y)] for t, x, y in queries]], dtype=torch.float32, device='cuda')
        with torch.no_grad():
            tr, vis = model(video, queries=q, backward_tracking=True)
        return tr[0].permute(1, 0, 2).cpu().numpy(), vis[0].permute(1, 0).cpu().numpy()

    def inside(xy):
        return 4 <= xy[0] <= C.WIDTH - 5 and 4 <= xy[1] <= C.HEIGHT - 5

    gap_ok = np.diff(times) <= 0.25              # gap_ok[j]: positions j and j+1 are consecutive in time
    propagated, stats = {}, {}
    t0 = time.time()
    for c in cands['kept']:
        cid = c['id']
        X = np.array(c['X'])
        anchors = {fpos[o['frame']]: np.array(o['xy'], dtype=float) for o in c['observations']}
        apos = sorted(anchors)
        uv, z = C.project(T_GC, np.repeat(X[None], len(T_GC), 0))
        raw = C.pinhole_to_raw(uv)
        inview = (z > 0.3) & np.isfinite(raw).all(axis=1) & (raw[:, 0] > 2) & (raw[:, 0] < C.WIDTH - 3) & (raw[:, 1] > 2) & (raw[:, 1] < C.HEIGHT - 3)
        out = {}
        anchor_err = []

        def accept(j, xy, gap, err, seed):
            if j not in out or gap < out[j][1]:
                out[j] = (np.asarray(xy, dtype=float), int(gap), float(err), int(seed))

        def one_sided(a, direction):
            js = [a]
            while len(js) <= args.max_gap:
                j = js[-1] + direction
                if j < 0 or j >= len(frames) or j in anchors or not inview[j] or not gap_ok[min(j, js[-1])]:
                    break
                js.append(j)
            if len(js) < 2:
                return
            tr, vis = run_window(js, [(0, *anchors[a])])
            fwd, v = tr[0], vis[0]
            t_end = 0
            while t_end + 1 < len(js) and v[t_end + 1] > 0.5 and inside(fwd[t_end + 1]):
                t_end += 1
            if t_end == 0:
                return
            back = run_window(js, [(t_end, *fwd[t_end])])[0][0]      # consistency pass from the far end
            for t in range(1, t_end + 1):
                err = float(np.linalg.norm(fwd[t] - back[t]))
                if err > args.track_tol:
                    break
                accept(js[t], fwd[t], t, err, frames[a])

        for a, b in zip(apos[:-1], apos[1:]):
            if b - a < 2:
                continue
            contiguous = bool(np.all(gap_ok[a:b])) and bool(np.all(inview[a:b + 1]))
            if not contiguous or b - a - 1 > 2 * args.max_gap:
                one_sided(a, +1)
                one_sided(b, -1)
                continue
            js = list(range(a, b + 1))
            tr, vis = run_window(js, [(0, *anchors[a]), (len(js) - 1, *anchors[b])])
            fa, fb, va, vb = tr[0], tr[1], vis[0], vis[1]
            anchor_err += [float(np.linalg.norm(fa[-1] - anchors[b])), float(np.linalg.norm(fb[0] - anchors[a]))]
            for t in range(1, len(js) - 1):
                ga, gb = t, len(js) - 1 - t
                near_a = ga <= gb
                xy, v = (fa[t], va[t]) if near_a else (fb[t], vb[t])
                err = float(np.linalg.norm(fa[t] - fb[t]))
                if v > 0.5 and err <= args.track_tol and inside(xy):
                    accept(js[t], xy, min(ga, gb), err, frames[a] if near_a else frames[b])
        one_sided(apos[0], -1)
        one_sided(apos[-1], +1)
        propagated[cid] = {str(frames[j]): [round(float(xy[0]), 2), round(float(xy[1]), 2), gap, round(err, 2), seed]
                           for j, (xy, gap, err, seed) in sorted(out.items())}
        stats[cid] = dict(matched=len(anchors), propagated=len(out), in_view=int(inview.sum()),
                          anchor_err_median_px=float(np.median(anchor_err)) if anchor_err else None)
        print(f'{cid}: {len(anchors)} matched, {len(out)} propagated of {int(inview.sum())} frames in view'
              + (f', tracker reaches the next match within {np.median(anchor_err):.1f} px' if anchor_err else '')
              + f' ({time.time() - t0:.0f} s)', flush=True)
    C.write_json(d / 'propagated.json', dict(
        run=args.run, tracker='CoTracker3 offline, bidirectional windows between matched frames',
        parameters=dict(max_gap=args.max_gap, track_tol_px=args.track_tol, sigma_growth_px_per_frame=args.sigma_growth), format='frame -> [x, y, gap, consistency_px, seed_frame]',
        candidates=propagated, stats=stats))
    # the candidate file carries the propagated positions as clicks (replaced on every run)
    ann = json.loads(AUTO_ANNOTATIONS.read_text())
    for f in ann['features']:
        f['real_clicks'] = [c for c in f.get('real_clicks', []) if c.get('method') != 'auto_cotracker3']
        for fr, (x, y, gap, err, seed) in propagated.get(f['id'], {}).items():
            f['real_clicks'].append(dict(run=args.run, index=int(fr), xy=[x, y], sigma_px=round(float(np.hypot(2.5, args.sigma_growth * gap)), 2),
                                         method='auto_cotracker3', seed_index=int(seed), steps=int(gap), consistency_px=err, coords='raw'))
        f['real_clicks'].sort(key=lambda c: c['index'])
    ann['updated'] = dt.datetime.now().isoformat(timespec='seconds')
    C.write_json(AUTO_ANNOTATIONS, ann)
    total = sum(len(v) for v in propagated.values())
    print(f'{total} propagated positions for {len(propagated)} candidates -> {d / "propagated.json"} and {AUTO_ANNOTATIONS}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['render', 'match', 'build', 'track', 'all'])
    ap.add_argument('--run', default='run_20260730T183828')
    ap.add_argument('--registration', type=Path, default=C.START_REGISTRATION)
    ap.add_argument('--stride', type=int, default=1, help='match every k-th rendered frame (key frames)')
    ap.add_argument('--chain-steps', type=lambda v: tuple(int(x) for x in v.split(',')), default=(1, 2, 5),
                    help='key-frame offsets matched render-to-render for chaining')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--max-keypoints', type=int, default=4096, help='SuperPoint budget per image (2048 starves textured walls)')
    ap.add_argument('--min-parallax', type=float, default=3.0)
    ap.add_argument('--min-obs', type=int, default=3)
    ap.add_argument('--min-span-s', type=float, default=0.3, help='real observations must span at least this time')
    ap.add_argument('--max-reproj', type=float, default=12.0, help='median real reprojection error allowed (px)')
    ap.add_argument('--max-point-shift', type=float, default=0.25, help='real-vs-render triangulation gap allowed (m)')
    ap.add_argument('--merge-m', type=float, default=0.05)
    ap.add_argument('--min-separation', type=float, default=0.10)
    ap.add_argument('--top', type=int, default=250, help='cap on the number of candidates kept')
    ap.add_argument('--min-per-frame', type=int, default=5, help='selection continues until every frame that can see this many candidates does')
    ap.add_argument('--assign-px', type=float, default=3.0, help='a match is attached to a candidate expected within this radius (px)')
    ap.add_argument('--max-obs-reproj', type=float, default=25.0, help='single real observations farther than this from the expected position are dropped (px)')
    ap.add_argument('--exclude-floor', action='store_true', help='do not use points near the room floor plane')
    ap.add_argument('--exclude-central-radius', type=float, default=0.0, help='exclude points this close to the scene XY origin, in nominal metres')
    ap.add_argument('--max-gap', type=int, default=30, help='track: frames propagated beyond a matched frame')
    ap.add_argument('--track-tol', type=float, default=2.0, help='track: forward/backward disagreement allowed (px)')
    ap.add_argument('--sigma-growth', type=float, default=0.05,
                    help='track: allowance growth per frame of distance from the matched position (px); CoTracker3 error is flat, so a small margin')
    args = ap.parse_args()
    for stage in (['render', 'match', 'build', 'track'] if args.stage == 'all' else [args.stage]):
        {'render': stage_render, 'match': stage_match, 'build': stage_build, 'track': stage_track}[stage](args)


if __name__ == '__main__':
    main()
