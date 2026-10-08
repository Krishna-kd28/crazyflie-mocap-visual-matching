#!/usr/bin/env python3
"""Viewer for the automatic feature maker (separate from the manual picker).

Shows, for every matched frame, the raw real frame and its 3DGS render side by
side with the matches drawn between them: grey for plain matches, coloured for
matches that belong to a candidate feature. Candidates are listed with their
statistics; accept or reject decisions are written to
research/fixed_features/auto_features.json. A candidate's detail strip shows
its real and render crops in every frame it was seen in.

    python scripts/auto_feature_viewer.py --open
"""
from __future__ import annotations

import project_config as PC

import argparse
import datetime as dt
import io
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_features_common as C  # noqa: E402
from auto_features import AUTO_ANNOTATIONS, OUT  # noqa: E402

HTML = Path(__file__).with_name('auto_feature_viewer.html')


class Store:
    def __init__(self, run):
        self.run = run
        self.dir = OUT / run
        self.meta = json.loads((self.dir / 'frames.json').read_text())
        self.frames = self.meta['frames']                      # every rendered (accepted) frame
        self.key = set(self.meta.get('key_frames', self.frames))  # frames with matcher output
        with np.load(self.dir / 'frames.npz', allow_pickle=False) as z:
            self.T_GC, self.times = z['T_GC'], z['time_s']
        self.fpos = {int(i): n for n, i in enumerate(self.frames)}
        self.cands = json.loads((self.dir / 'candidates.json').read_text())
        self.ann = json.loads(AUTO_ANNOTATIONS.read_text())
        self.by_id = {f['id']: f for f in self.ann['features']}
        # CoTracker3 propagations from scripts/auto_features.py track (optional)
        self.prop_by_frame, self.prop_stats, self.prop_frames = {}, {}, {}
        prop_path = self.dir / 'propagated.json'
        if prop_path.exists():
            prop = json.loads(prop_path.read_text())
            self.prop_stats = prop.get('stats', {})
            for cid, per in prop['candidates'].items():
                self.prop_frames[cid] = sorted(int(f) for f in per)
                for f, (x, y, gap, err, seed) in per.items():
                    self.prop_by_frame.setdefault(int(f), {})[cid] = dict(xy=[x, y], gap=gap, err=err, seed=seed)
        # candidate membership per (frame, render keypoint xy) for drawing
        self.member = {}
        for c in self.cands['kept']:
            for f, xy in c['nodes']:
                self.member[(int(f), round(xy[0], 1), round(xy[1], 1))] = c['id']

    def frame_payload(self, i):
        matches, kp_real, kp_rend, rr_idx, rr_score = [], [], [], [], []
        if i in self.key and (self.dir / 'matches' / f'{i:06d}.npz').exists():
            with np.load(self.dir / 'matches' / f'{i:06d}.npz', allow_pickle=False) as z:
                kp_real, kp_rend, rr_idx, rr_score = z['kp_real'], z['kp_rend'], z['rr_idx'], z['rr_score']
        for (a, b), s in zip(rr_idx, rr_score):
            xr, yr = kp_real[a]; xg, yg = kp_rend[b]
            cid = self.member.get((int(i), round(float(xg), 1), round(float(yg), 1)))
            matches.append(dict(real=[round(float(xr), 2), round(float(yr), 2)], render=[round(float(xg), 2), round(float(yg), 2)],
                                score=round(float(s), 3), candidate=cid))
        # predicted positions of candidates in this frame under the global registration
        preds = {}
        T = self.T_GC[self.fpos[int(i)]][None]
        for c in self.cands['kept']:
            uv, z = C.project(T, np.array(c['X'])[None])
            if z[0] > 0.3:
                raw = C.pinhole_to_raw(uv)[0]
                if -20 <= raw[0] <= C.WIDTH + 20 and -20 <= raw[1] <= C.HEIGHT + 20:
                    preds[c['id']] = [round(float(raw[0]), 2), round(float(raw[1]), 2)]
        return dict(index=int(i), time_s=float(self.times[self.fpos[int(i)]]), key=i in self.key, matches=matches,
                    predictions=preds, propagated=self.prop_by_frame.get(int(i), {}),
                    n_real=int(len(kp_real)), n_render=int(len(kp_rend)))

    def candidates_payload(self):
        out = []
        for c in self.cands['kept']:
            f = self.by_id.get(c['id'], {})
            out.append(dict(id=c['id'], status=f.get('status', 'proposed'), X=[round(x, 3) for x in c['X']], score=c['score'],
                            checks=c['checks'], frames=[o['frame'] for o in c['observations']],
                            prop_frames=self.prop_frames.get(c['id'], []), prop_stats=self.prop_stats.get(c['id'], {}),
                            description=f.get('description', '')))
        return dict(run=self.run, candidates=out, counts=self.cands['counts'], frames=self.frames, key_frames=sorted(self.key),
                    stride=self.meta.get('stride'), times=[float(t) for t in self.times], matcher=self.ann.get('matcher', ''),
                    registration=self.cands['registration'], propagated=bool(self.prop_by_frame))

    def set_status(self, cid, status):
        if cid not in self.by_id or status not in ('proposed', 'confirmed', 'rejected'):
            raise ValueError('unknown candidate or status')
        self.by_id[cid]['status'] = status
        self.by_id[cid]['decided'] = dt.datetime.now().isoformat(timespec='seconds')
        self.ann['updated'] = dt.datetime.now().isoformat(timespec='seconds')
        tmp = AUTO_ANNOTATIONS.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self.ann, indent=2) + '\n')
        tmp.replace(AUTO_ANNOTATIONS)

    def crops(self, cid, radius=16, scale=3, max_frames=12):
        """Strip of crops for a candidate: one column per frame (up to max_frames, evenly
        spread over its track), real crop on top, render crop below, crosshair on the point."""
        from PIL import Image, ImageDraw
        c = next(x for x in self.cands['kept'] if x['id'] == cid)
        obs = c['observations']
        pick = sorted({int(round(k)) for k in np.linspace(0, len(obs) - 1, min(max_frames, len(obs)))})
        obs = [obs[k] for k in pick]
        size = (2 * radius + 1) * scale
        band = 14
        sheet = Image.new('RGB', (size * len(obs), 2 * size + band), 'black')
        draw = ImageDraw.Draw(sheet)
        for col, o in enumerate(obs):
            i = o['frame']
            for row, (kind, xy) in enumerate((('real', o['xy']), ('render', o['render_xy']))):
                path = C.real_raw_path(self.run, i) if kind == 'real' else self.dir / 'renders' / f'{i:06d}.png'
                im = Image.open(path).convert('RGB')
                x, y = int(round(xy[0])), int(round(xy[1]))
                crop = im.crop((x - radius, y - radius, x + radius + 1, y + radius + 1)).resize((size, size), Image.NEAREST)
                ox, oy = col * size, band + row * size
                sheet.paste(crop, (ox, oy))
                cx = ox + (xy[0] - x + radius + 0.5) * scale
                cy = oy + (xy[1] - y + radius + 0.5) * scale
                draw.line([(cx - 10, cy), (cx - 4, cy)], fill=(255, 80, 60), width=2)
                draw.line([(cx + 4, cy), (cx + 10, cy)], fill=(255, 80, 60), width=2)
                draw.line([(cx, cy - 10), (cx, cy - 4)], fill=(255, 80, 60), width=2)
                draw.line([(cx, cy + 4), (cx, cy + 10)], fill=(255, 80, 60), width=2)
            draw.text((col * size + 3, 1), f'frame {i}', fill=(230, 230, 230))
        buf = io.BytesIO()
        sheet.save(buf, format='PNG')
        return buf.getvalue(), size, len(obs), [o['frame'] for o in obs]


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

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

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path == '/':
                    return self.send(200, HTML.read_text(), 'text/html; charset=utf-8')
                if url.path == '/api/candidates':
                    return self.send(200, store.candidates_payload())
                if url.path == '/api/frame':
                    return self.send(200, store.frame_payload(int(q['index'][0])))
                if url.path == '/api/crops':
                    png, size, rows, frames = store.crops(q['id'][0])
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/png')
                    self.send_header('X-Crop-Size', str(size))
                    self.send_header('X-Crop-Frames', ','.join(map(str, frames)))
                    self.send_header('Content-Length', str(len(png)))
                    self.end_headers()
                    self.wfile.write(png)
                    return
                if url.path.startswith('/img/'):
                    _, _, kind, name = url.path.split('/')
                    i = int(Path(name).stem)
                    path = C.real_raw_path(store.run, i) if kind == 'real' else store.dir / 'renders' / f'{i:06d}.png'
                    return self.send(200, path.read_bytes(), 'image/png')
                return self.send(404, dict(error='not found'))
            except Exception as error:  # noqa: BLE001
                return self.send(500, dict(error=f'{error.__class__.__name__}: {error}'))

        def do_POST(self):
            url = urlparse(self.path)
            length = int(self.headers.get('Content-Length', 0))
            payload = json.loads(self.rfile.read(length) or b'{}')
            try:
                if url.path == '/api/status':
                    for cid in payload.get('ids', [payload.get('id')]):
                        store.set_status(cid, payload['status'])
                    return self.send(200, dict(ok=True, updated=store.ann['updated']))
                return self.send(404, dict(error='not found'))
            except Exception as error:  # noqa: BLE001
                return self.send(400, dict(error=f'{error.__class__.__name__}: {error}'))
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='run_20260730T183828')
    ap.add_argument('--port', type=int, default=8777)
    ap.add_argument('--open', action='store_true')
    args = ap.parse_args()
    if not (OUT / args.run / 'candidates.json').exists():
        sys.exit('Run scripts/auto_features.py all first')
    store = Store(args.run)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(store))
    url = f'http://127.0.0.1:{args.port}/'
    print(f'Auto-feature viewer: {url}\nDecisions: {AUTO_ANNOTATIONS}\nPress Ctrl-C to stop.', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
