#!/usr/bin/env python3
"""Local browser tool for pointing out permanently fixed features.

Serves each timing-accepted Crazyflie frame exactly as recorded beside its
3DGS render at the same corrected pose, warped with the supplied lens model
into the same raw pixel geometry. Clicks are stored in raw pixel coordinates
and converted to pinhole coordinates for all geometry. Clicks on the render
define a scene point (triangulated from two or more render clicks, or lifted
with rendered depth); clicks on real frames are the observations the
brute-force registration search fits. Everything is written to
research/fixed_features/annotations.json as you work. --pinhole serves the
older undistorted frames and pinhole renders instead.

    PYTHONPATH=.deps/python python3 scripts/fixed_feature_picker.py --open

Run it from the CUDA environment instead to get depth-lifted predictions after a
single render click:

    python \
        scripts/fixed_feature_picker.py --open
"""
from __future__ import annotations

import project_config as PC

import argparse
import datetime as dt
import io
import json
import shutil
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_features_common as C  # noqa: E402

HTML = Path(__file__).with_name('fixed_feature_picker.html')
BACKUPS = C.FF_OUT / 'annotation_backups'
from migrate_annotations_to_raw import RAW_CONVENTION, migrate  # noqa: E402

CONVENTION = RAW_CONVENTION


def now():
    return dt.datetime.now().isoformat(timespec='seconds')


def empty_annotations():
    return {'version': 1, 'coordinate_convention': CONVENTION,
            'start_registration': PC.relative_or_absolute(C.START_REGISTRATION),
            'features': [], 'rejected_proposals': [], 'updated': now()}


class Store:
    """Frame index, poses, annotations and (optional) GPU depth for lifting."""

    def __init__(self, annotations_path, gpu, pinhole=False, extra=()):
        self.path = Path(annotations_path)
        self.pinhole = pinhole      # serve undistorted frames / pinhole renders and keep pinhole clicks
        self.extra = [Path(e) for e in extra]
        self.lock = threading.Lock()
        self.runs = {}
        for run in C.RUNS:
            folder, index, arrays = C.frame_store(run)
            index['position'] = {int(i): j for j, i in enumerate(index['accepted_indices'])}
            self.runs[run] = dict(folder=folder, index=index, arrays=arrays)
        self.by_short = {C.SHORT[r]: r for r in C.RUNS}
        self.annotations = json.loads(self.path.read_text()) if self.path.exists() else empty_annotations()
        if not pinhole and self.annotations.get('coords') != 'raw':
            n = migrate(self.annotations)
            if n:
                print(f'{n} clicks converted from undistorted to raw pixel coordinates', flush=True)
                self.save(self.annotations)
        # Confirmed features from extra files (e.g. the automatic feature maker) are merged
        # read-only, so the pipeline can use them without mixing the annotation files.
        have = {f['id'] for f in self.annotations['features']}
        for path in self.extra:
            if not path.exists():
                continue
            extra = json.loads(path.read_text())
            for f in extra.get('features', []):
                if f.get('status') == 'confirmed' and f['id'] not in have:
                    self.annotations['features'].append(dict(f, external=str(path)))
                    have.add(f['id'])
        self.depth_cache = {}
        self.gpu = None
        self.gpu_status = 'disabled'
        if gpu:
            self.gpu_status = 'loading'
            threading.Thread(target=self._load_gpu, daemon=True).start()

    # -- GPU depth --------------------------------------------------------
    def _load_gpu(self):
        try:
            import torch
            from gsplat.rendering import rasterization
            from render_baseline import load_scene
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            scene, info = load_scene(C.CHECKPOINT, 'cuda')
            self.gpu = dict(torch=torch, rasterization=rasterization, scene=scene, info=info)
            self.gpu_status = 'ready'
            print('GPU depth rendering ready', flush=True)
        except Exception as error:  # noqa: BLE001
            self.gpu_status = f'off ({error.__class__.__name__}: {error})'
            print('Depth lifting off:', error, '- click each feature in two rendered frames instead', flush=True)

    def depth(self, run, i):
        key = (run, int(i))
        if key in self.depth_cache:
            return self.depth_cache[key]
        path = C.depth_path(run, int(i))
        if not path.exists():
            if self.gpu is None:
                return None
            from prepare_fixed_feature_frames import render_depth
            torch = self.gpu['torch']
            r = self.runs[run]
            j = r['index']['position'][int(i)]
            view = torch.tensor(r['arrays']['T_CS'][j:j + 1], device='cuda', dtype=torch.float32)
            with torch.no_grad():
                depth, std, alpha = render_depth(self.gpu['scene'], self.gpu['info'], view,
                                                 self.gpu['rasterization'], torch)
            m = C.metres_per_checkpoint_unit()
            c = C.distortion_canvas()
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, depth_m=(depth * m).astype(np.float32),
                                depth_std_m=(std * m).astype(np.float32),
                                alpha=alpha.astype(np.float32), T_GC=r['arrays']['T_GC'][j],
                                K=c['K_canvas'], canvas_offset=np.array(c['offset']))
        with np.load(path, allow_pickle=False) as d:
            value = {k: d[k].copy() for k in d.files}
        self.depth_cache[key] = value
        return value

    # -- coordinates --------------------------------------------------------
    def to_pinhole(self, xy):
        """Stored click -> pinhole pixel (identity in --pinhole mode)."""
        if self.pinhole:
            return np.asarray(xy, dtype=float)
        return C.raw_to_pinhole(np.asarray(xy, dtype=float)[None])[0]

    def to_display(self, xy):
        """Pinhole pixel -> coordinates of the displayed images."""
        if self.pinhole:
            return np.asarray(xy, dtype=float)
        return C.pinhole_to_raw(np.asarray(xy, dtype=float)[None])[0]

    # -- geometry ---------------------------------------------------------
    def pose(self, run, i, click=None):
        """Camera pose in G of the render a click was made on. Renders served by the picker
        were made under the start registration (poses.npz); a click from another source
        names its registration in 'render_registration'."""
        r = self.runs[run]
        j = r['index']['position'][int(i)]
        reg_path = (click or {}).get('render_registration')
        if not reg_path:
            return r['arrays']['T_GC'][j]
        cache = self.__dict__.setdefault('_registrations', {})
        if reg_path not in cache:
            path = Path(reg_path)
            cache[reg_path] = C.load_registration(path if path.is_absolute() else C.ROOT / path)
        reg = cache[reg_path]
        return C.apply_registration(r['arrays']['T_MC'][j][None], reg['R'], reg['t'], reg['lam'])[0]

    def lift_with_depth(self, run, i, xy):
        """Scene point for one render click: intersect its ray with a local plane.

        Pixels around the click that are opaque and not depth-mixed are
        unprojected with the rendered expected depth; a RANSAC plane through
        them is refined by SVD and intersected with the click ray. This keeps
        a corner on its own surface even though the corner pixel itself mixes
        the depths of the surfaces meeting there.
        """
        d = self.depth(run, i)
        if d is None:
            return None
        T = self.pose(run, i)
        pin = self.to_pinhole(xy)
        if not np.isfinite(pin).all():
            return None
        Kd = np.asarray(d.get('K', C.K), dtype=float)
        off = d.get('canvas_offset', np.zeros(2))
        u, v = float(pin[0] + off[0]), float(pin[1] + off[1])
        H, W = d['depth_m'].shape
        x0, y0 = int(round(u)), int(round(v))
        r = 6
        xs, ys = np.meshgrid(np.arange(x0 - r, x0 + r + 1), np.arange(y0 - r, y0 + r + 1))
        inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xs, ys = xs[inside], ys[inside]
        z = d['depth_m'][ys, xs]
        good = (d['alpha'][ys, xs] >= 0.95) & (d['depth_std_m'][ys, xs] <= np.maximum(0.03, 0.05 * z))
        xs, ys, z = xs[good], ys[good], z[good]
        ray = np.linalg.solve(Kd, np.array([u, v, 1.0]))
        ray_G = T[:3, :3] @ ray
        result = None
        if len(z) >= 12:
            rays = np.linalg.solve(Kd, np.vstack([xs, ys, np.ones_like(xs)]).astype(float)).T
            P = (z[:, None] * rays) @ T[:3, :3].T + T[:3, 3]
            rng = np.random.default_rng(0)
            best = (0, None)
            for _ in range(200):
                a, b, c = P[rng.choice(len(P), 3, replace=False)]
                n = np.cross(b - a, c - a)
                if np.linalg.norm(n) < 1e-9:
                    continue
                n /= np.linalg.norm(n)
                dist = np.abs((P - a) @ n)
                count = int((dist < 0.02).sum())
                if count > best[0]:
                    best = (count, (n, a))
            if best[1] is not None and best[0] >= 12:
                n, a = best[1]
                inl = np.abs((P - a) @ n) < 0.02
                centre = P[inl].mean(axis=0)
                _, _, vt = np.linalg.svd(P[inl] - centre, full_matrices=False)
                n = vt[-1]
                denom = n @ ray_G
                if abs(denom) > 1e-6:
                    t = ((centre - T[:3, 3]) @ n) / denom
                    if t > 0.1:
                        X = T[:3, 3] + t * ray_G
                        rms = float(np.sqrt(np.mean(((P[inl] - centre) @ n) ** 2)))
                        result = dict(X=X.tolist(), method='depth_plane', depth_m=float(t * ray[2] / np.linalg.norm(ray)),
                                      plane_inliers=int(inl.sum()), plane_rms_m=round(rms, 4),
                                      note='single render click: ray intersected with the local rendered surface plane; '
                                           'a second render click from another position replaces this by triangulation')
        if result is None:
            # Fallback: the least depth-mixed pixel next to the click.
            cands = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    x, y = x0 + dx, y0 + dy
                    if 0 <= x < W and 0 <= y < H and d['alpha'][y, x] >= 0.9:
                        cands.append((float(d['depth_std_m'][y, x]), float(d['depth_m'][y, x])))
            if not cands:
                return None
            std, zc = min(cands)
            X = T[:3, :3] @ (zc * ray) + T[:3, 3]
            result = dict(X=X.tolist(), method='depth', depth_m=zc, depth_std_m=std,
                          note='single render click lifted with a mixed depth pixel; add a second render click')
        return result

    def feature_points(self):
        result = {}
        for f in self.annotations['features']:
            clicks = f.get('render_clicks', [])
            entry = dict(n_render=len(clicks), n_real=len(f.get('real_clicks', [])), X=None,
                         method=None, residual_px=[], note='')
            good = [c for c in clicks if c['run'] in self.runs and int(c['index']) in self.runs[c['run']]['index']['position']]
            if len(good) >= 2:
                T_all = np.array([self.pose(c['run'], c['index'], c) for c in good])
                xy_all = np.array([self.to_pinhole(c['xy']) for c in good], dtype=float)
                finite = np.isfinite(xy_all).all(axis=1)
                good = [c for c, ok in zip(good, finite) if ok]
                T_all, xy_all = T_all[finite], xy_all[finite]
                if len(good) < 2:
                    entry['note'] = 'render clicks lie outside the valid lens region'
                    result[f['id']] = entry
                    continue
                # Robust triangulation: with three or more render clicks, drop the
                # click that disagrees most until all reproject within 3 px.
                keep = list(range(len(good)))
                dropped = []
                while True:
                    X, err, depth = C.triangulate(T_all[keep], xy_all[keep])
                    if len(keep) <= 2 or err.max() <= 3.0:
                        break
                    worst = keep[int(np.argmax(err))]
                    dropped.append(f"{C.SHORT[good[worst]['run']]}:{good[worst]['index']}")
                    keep.remove(worst)
                T, xy = T_all[keep], xy_all[keep]
                entry.update(X=X.tolist(), method='triangulation', residual_px=np.round(err, 2).tolist(),
                             render_clicks_used=[f"{C.SHORT[good[k]['run']]}:{good[k]['index']}" for k in keep],
                             dropped_render_clicks=dropped,
                             note='' if np.all(depth > 0) else 'a render click sees the point behind the camera')
                if dropped:
                    entry['note'] = f'render clicks at {", ".join(dropped)} disagree with the others and are ignored; check them'
                elif len(keep) == 2 and err.max() > 3.0:
                    entry['note'] = 'the two render clicks disagree; one is off, check both or add a third'
                # Largest pairwise ray angle; shallow angles leave the distance poorly determined.
                dirs = [T[k, :3, :3] @ np.linalg.solve(C.K, np.r_[xy[k], 1.0]) for k in range(len(T))]
                dirs = [v / np.linalg.norm(v) for v in dirs]
                angle = max(float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
                            for m, a in enumerate(dirs) for b in dirs[m + 1:])
                entry['parallax_deg'] = round(angle, 2)
                rng = float(np.mean(np.linalg.norm(T[:, :3, 3] - X, axis=1)))
                sigma_depth = rng * (1.0 / C.K[0, 0]) / max(np.sin(np.radians(angle)), 1e-6)
                entry['depth_uncertainty_m'] = round(sigma_depth, 3)
                if angle < 8 and not entry['note']:
                    entry['note'] = (f'weak parallax ({angle:.1f} deg between render clicks, about +/-{sigma_depth:.2f} m '
                                     'along the ray): add a render click from a frame taken several seconds away, '
                                     'from a clearly different position')
            elif len(good) == 1:
                lifted = self.lift_with_depth(good[0]['run'], good[0]['index'], good[0]['xy'])
                if lifted is not None:
                    entry.update(lifted)
                else:
                    entry['note'] = 'need a second render click (or run in the CUDA environment for depth lifting)'
            result[f['id']] = entry
        return result

    def predictions(self, run, i, points=None):
        points = self.feature_points() if points is None else points
        T = self.pose(run, i)[None]
        out = {}
        for fid, p in points.items():
            if p['X'] is None:
                continue
            uv, z = C.project(T, np.asarray(p['X'])[None])
            u, v = float(uv[0, 0]), float(uv[0, 1])
            if z[0] > 0.05 and -40 <= u <= C.WIDTH + 40 and -40 <= v <= C.HEIGHT + 40:
                du, dv = self.to_display([u, v])
                if np.isfinite(du) and np.isfinite(dv):
                    out[fid] = dict(xy=[round(float(du), 2), round(float(dv), 2)], depth_m=round(float(z[0]), 3))
        return out

    # -- persistence ------------------------------------------------------
    def save(self, annotations):
        if not isinstance(annotations, dict) or 'features' not in annotations:
            raise ValueError('annotations must contain a feature list')
        ids = [f['id'] for f in annotations['features']]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate feature ids')
        for f in annotations['features']:
            for kind in ('render_clicks', 'real_clicks'):
                for c in f.get(kind, []):
                    if c['run'] not in self.runs or int(c['index']) not in self.runs[c['run']]['index']['position']:
                        raise ValueError(f'{f["id"]}: unknown frame {c["run"]}:{c["index"]}')
                    u, v = c['xy']
                    if not (-1 <= u <= C.WIDTH and -1 <= v <= C.HEIGHT):
                        raise ValueError(f'{f["id"]}: click outside the image')
        annotations['coordinate_convention'] = CONVENTION if not self.pinhole else annotations.get('coordinate_convention', CONVENTION)
        if not self.pinhole:
            annotations['coords'] = 'raw'
            for f in annotations['features']:
                for kind in ('render_clicks', 'real_clicks'):
                    for c in f.get(kind, []):
                        c.setdefault('coords', 'raw')
        annotations['updated'] = now()
        with self.lock:
            if self.path.exists():
                folder = BACKUPS / self.path.stem
                folder.mkdir(parents=True, exist_ok=True)
                stamp = dt.datetime.now().strftime('%Y%m%dT%H%M%S')
                shutil.copy2(self.path, folder / f'{self.path.stem}_{stamp}.json')
                backups = sorted(folder.glob(f'{self.path.stem}_*.json'))
                # Keep every 20th old backup so the history stays recoverable.
                for k, old in enumerate(backups[:-60]):
                    if k % 20:
                        old.unlink()
            tmp = self.path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(annotations, indent=2) + '\n')
            tmp.replace(self.path)
            self.annotations = annotations

    def index_payload(self):
        runs = []
        for run in C.RUNS:
            idx = self.runs[run]['index']
            runs.append(dict(run=run, short=C.SHORT[run], indices=idx['accepted_indices'],
                             time_s=idx['time_s'], frame_id=idx['frame_id'],
                             total_frames=idx['total_frames'],
                             has_render=(self.runs[run]['folder'] / self.render_dir).exists()))
        return dict(runs=runs, K=C.K.tolist(), width=C.WIDTH, height=C.HEIGHT,
                    gpu_depth=self.gpu_status, annotations_path=str(self.path),
                    proposals_available=C.PROPOSALS.exists(),
                    registration=PC.relative_or_absolute(C.START_REGISTRATION),
                    mode='pinhole' if self.pinhole else 'raw',
                    real_label='Real Crazyflie image (undistorted)' if self.pinhole else 'Real Crazyflie image (as recorded)',
                    render_label='3DGS render, pinhole, same corrected pose' if self.pinhole else '3DGS render, lens model applied, same corrected pose')

    @property
    def real_dir(self):
        return 'real' if self.pinhole else 'real_raw'

    @property
    def render_dir(self):
        return 'render' if self.pinhole else 'render_distorted'

    def image_path(self, run, i, kind):
        if kind == 'real':
            return C.real_path(run, i) if self.pinhole else C.real_raw_path(run, i)
        return C.render_path(run, i) if self.pinhole else C.render_distorted_path(run, i)

    def crop(self, run, i, kind, u, v, radius, scale):
        from PIL import Image
        path = self.image_path(run, i, 'real' if kind == 'real' else 'render')
        im = Image.open(path).convert('RGB')
        box = (int(round(u)) - radius, int(round(v)) - radius, int(round(u)) + radius + 1, int(round(v)) + radius + 1)
        crop = im.crop(box).resize(((2 * radius + 1) * scale, (2 * radius + 1) * scale), Image.NEAREST)
        buf = io.BytesIO()
        crop.save(buf, format='PNG')
        return buf.getvalue()


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter log
            if self.path.startswith('/api/annotations') or self.path == '/':
                super().log_message(fmt, *args)

        def send(self, code, body, ctype='application/json'):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def run_of(self, q):
            key = q['run'][0]
            return store.by_short.get(key, key)

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path == '/':
                    return self.send(200, HTML.read_text(), 'text/html; charset=utf-8')
                if url.path == '/api/index':
                    return self.send(200, store.index_payload())
                if url.path == '/api/annotations':
                    return self.send(200, store.annotations)
                if url.path == '/api/points':
                    return self.send(200, store.feature_points())
                if url.path == '/api/predict':
                    run, i = self.run_of(q), int(q['index'][0])
                    points = store.feature_points()
                    return self.send(200, dict(points=points, predictions=store.predictions(run, i, points)))
                if url.path == '/api/depth':
                    run, i = self.run_of(q), int(q['index'][0])
                    d = store.depth(run, i)
                    return self.send(200, dict(available=d is not None, gpu=store.gpu_status))
                if url.path == '/api/proposals':
                    if not C.PROPOSALS.exists():
                        return self.send(200, dict(proposals=[], note='no proposals file'))
                    props = json.loads(C.PROPOSALS.read_text())
                    if not store.pinhole and props.get('coords', 'pinhole') != 'raw':
                        for pr in props.get('proposals', []):
                            for c in [pr['render'], *pr['real']]:
                                raw = C.pinhole_to_raw(np.array([c['xy']], dtype=float))[0]
                                c['xy'] = [round(float(raw[0]), 2), round(float(raw[1]), 2)]
                                c['coords'] = 'raw'
                        props['coords'] = 'raw'
                    return self.send(200, props)
                if url.path == '/api/crop':
                    run, i = self.run_of(q), int(q['index'][0])
                    body = store.crop(run, i, q['kind'][0], float(q['u'][0]), float(q['v'][0]),
                                      int(q.get('r', ['16'])[0]), int(q.get('s', ['4'])[0]))
                    return self.send(200, body, 'image/png')
                if url.path.startswith('/img/'):
                    _, _, run, kind, name = url.path.split('/')
                    run = store.by_short.get(run, run)
                    i = int(Path(name).stem)
                    path = store.image_path(run, i, 'real' if kind == 'real' else 'render')
                    if not path.exists():
                        return self.send(404, dict(error='missing image'))
                    return self.send(200, path.read_bytes(), 'image/png')
                return self.send(404, dict(error='not found'))
            except Exception as error:  # noqa: BLE001
                return self.send(500, dict(error=f'{error.__class__.__name__}: {error}'))

        def do_POST(self):
            url = urlparse(self.path)
            length = int(self.headers.get('Content-Length', 0))
            payload = json.loads(self.rfile.read(length) or b'{}')
            try:
                if url.path == '/api/annotations':
                    store.save(payload['annotations'])
                    response = dict(ok=True, updated=store.annotations['updated'])
                    if 'run' in payload and 'index' in payload:
                        run = store.by_short.get(payload['run'], payload['run'])
                        points = store.feature_points()
                        response.update(points=points, predictions=store.predictions(run, int(payload['index']), points))
                    return self.send(200, response)
                return self.send(404, dict(error='not found'))
            except Exception as error:  # noqa: BLE001
                return self.send(400, dict(error=f'{error.__class__.__name__}: {error}'))

    return Handler


CUDA_PYTHON = Path(sys.executable)


def relaunch_with_gsplat():
    """Keep this interpreter; Store already reports unavailable optional GPU depth."""
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--annotations', type=Path, default=C.ANNOTATIONS)
    ap.add_argument('--no-gpu', action='store_true', help='Do not try to load the scene for depth lifting')
    ap.add_argument('--open', action='store_true', help='Open the page in the default browser')
    ap.add_argument('--pinhole', action='store_true', help='Serve undistorted frames and pinhole renders (old convention)')
    args = ap.parse_args()
    if not (C.FRAMES / 'manifest.json').exists():
        sys.exit('Run scripts/prepare_fixed_feature_frames.py frames (and render) first')
    if not args.no_gpu:
        relaunch_with_gsplat()
    store = Store(args.annotations, gpu=not args.no_gpu, pinhole=args.pinhole)
    if not (C.FRAMES / C.RUNS[0] / store.render_dir).exists():
        sys.exit(f'Missing {store.render_dir} renders: run scripts/prepare_fixed_feature_frames.py render'
                 + (' --pinhole' if args.pinhole else '') + ' first')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(store))
    url = f'http://127.0.0.1:{args.port}/'
    print(f'Fixed-feature picker: {url}\nAnnotations: {store.path}\nPress Ctrl-C to stop.', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
