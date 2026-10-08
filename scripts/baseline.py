#!/usr/bin/env python3
"""Prepare recorded frames/poses and score externally rendered Step-1 RGB images.

Preparation is not a completed rendering baseline. All timestamps are provisional
host-arrival times; scene registration and camera orientation require verification.
"""

import project_config as PC
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

ROOT = PC.WORKSPACE
K = np.asarray(PC.read_camera()['K'], dtype=float)
DIST = np.asarray(PC.read_camera()['distortion_opencv_k1_k2_p1_p2_k3'], dtype=float)
R_BC = np.asarray(PC.read_camera()['camera_to_body_rotation_assumed'], dtype=float)
P_BC = np.asarray(PC.read_camera()['camera_origin_in_body_m'], dtype=float)


def read_table(path):
    return {k: np.asarray(v) for k, v in pq.read_table(path).to_pydict().items()}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def raw_frame(run, table, i):
    with (run / 'camera_frames.bin').open('rb') as f:
        f.seek(int(table['blob_offset'][i]))
        payload = f.read(int(table['size'][i]))
    if len(payload) != int(table['size'][i]):
        raise ValueError('Truncated camera payload')
    if table['pixel_format'][i] == 1:
        im = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
    elif table['pixel_format'][i] == 0 and table['depth'][i] == 1:
        im = np.frombuffer(payload, np.uint8).reshape(int(table['height'][i]), int(table['width'][i]))
    else:
        raise ValueError('Unsupported camera format; refusing to guess')
    if im is None or im.shape != (240, 320):
        raise ValueError('Unexpected frame shape')
    return im


def clock_diagnostic(device_s, host_ns):
    """Robust centered affine fit; intercept includes unknown transport latency."""
    x = device_s - device_s[0]
    y = (host_ns - host_ns[0]) * 1e-9
    A = np.column_stack((x, np.ones_like(x)))
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(10):
        r = y - A @ beta
        scale = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-6)
        w = np.minimum(1., 1.345 * scale / np.maximum(np.abs(r), 1e-12))
        beta = np.linalg.lstsq(A * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
    r_ms = (y - A @ beta) * 1000
    return {'slope': float(beta[0]), 'drift_ppm': float((beta[0]-1)*1e6),
            'centered_intercept_s': float(beta[1]),
            'abs_residual_p50_ms': float(np.percentile(abs(r_ms), 50)),
            'abs_residual_p95_ms': float(np.percentile(abs(r_ms), 95)),
            'abs_residual_max_ms': float(max(abs(r_ms)))}


def interpolate_pose(m, query_ns, max_gap_ms, max_angle_deg=45.):
    mt = m['host_ns'].astype(np.int64)
    if np.any(np.diff(mt) <= 0):
        raise ValueError('Non-increasing host timestamps')
    p = np.column_stack([m[k] for k in ('x', 'y', 'z')])
    q = np.column_stack([m[k] for k in ('qx', 'qy', 'qz', 'qw')])
    norms = np.linalg.norm(q, axis=1)
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(abs(norms-1) > .01):
        raise ValueError('Invalid mocap position/quaternion; explicit cleaning required')
    right = np.searchsorted(mt, query_ns, side='right')
    outside = (query_ns < mt[0]) | (query_ns > mt[-1])
    right = right.clip(1, len(mt)-1)
    left = right - 1
    gap_ms = (mt[right]-mt[left])*1e-6
    qn=q/norms[:,None]
    angle_deg=np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(qn[left]*qn[right],axis=1)),0,1)))
    valid = ~outside & (gap_ms <= max_gap_ms) & (angle_deg <= max_angle_deg)
    reason = np.where(outside, 'outside_mocap_support',
                      np.where(gap_ms > max_gap_ms, 'mocap_gap',
                               np.where(angle_deg > max_angle_deg,'mocap_rotation_jump','accepted')))
    pp = np.full((len(query_ns), 3), np.nan)
    qq = np.full((len(query_ns), 4), np.nan)
    fraction = (query_ns[valid]-mt[left[valid]])/(mt[right[valid]]-mt[left[valid]])
    pp[valid] = (1-fraction[:, None])*p[left[valid]] + fraction[:, None]*p[right[valid]]
    times = (mt-mt[0])*1e-9
    qq[valid] = Slerp(times, Rotation.from_quat(q))((query_ns[valid]-mt[0])*1e-9).as_quat()
    return pp, qq, valid, reason, gap_ms


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(args.data.glob('run_*'))
    if len(runs) != 4:
        raise ValueError('This predeclared split expects the four supplied runs')
    summaries, all_rows, hashes = [], [], {}
    prepared = []
    for ri, run in enumerate(runs):
        split = ['train', 'train', 'validation', 'test'][ri]
        c, m = read_table(run/'camera.parquet'), read_table(run/'mocap.parquet')
        manifest = json.loads((run/'manifest.json').read_text())
        n = len(c['seq'])
        assert n == manifest['streams']['camera']['rows']
        assert len(m['seq']) == manifest['streams']['mocap']['rows']
        assert np.all(np.diff(c['host_ns']) > 0)
        offsets, sizes = c['blob_offset'], c['size']
        assert offsets[0] == 0 and np.all(offsets[1:] == offsets[:-1]+sizes[:-1])
        assert offsets[-1]+sizes[-1] == (run/'camera_frames.bin').stat().st_size
        pp, qq, valid, reason, gaps = interpolate_pose(m, c['host_ns'], args.max_gap_ms)
        cam_clock=clock_diagnostic(c['deck_ms']*.001,c['host_ns'])
        clock_residual_ms=((c['host_ns']-c['host_ns'][0])*1e-9-
                           cam_clock['slope']*(c['deck_ms']-c['deck_ms'][0])*.001-
                           cam_clock['centered_intercept_s'])*1000
        unstable=abs(clock_residual_ms)>args.max_clock_residual_ms
        reason=np.where(valid & unstable,'camera_arrival_outlier',reason)
        valid &= ~unstable
        pp[~valid]=np.nan;qq[~valid]=np.nan
        # Body forward +x, left +y, up +z and zero camera roll are assumptions.
        # These matrices are explicitly provisional and not render authorization.
        cam_to_mocap = np.full((n, 4, 4), np.nan)
        rb = Rotation.from_quat(qq[valid]).as_matrix()
        cam_to_mocap[valid] = np.eye(4)
        cam_to_mocap[valid, :3, :3] = rb @ R_BC
        cam_to_mocap[valid, :3, 3] = pp[valid] + (rb @ P_BC)
        folder = out/run.name
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'poses.npz', camera_to_mocap_assumed=cam_to_mocap,
                            body_position=pp, body_quaternion_xyzw=qq, accepted=valid,
                            host_ns=c['host_ns'], frame_id=c['frame_id'], K=K,
                            distortion_opencv=DIST)
        means, saturated, black, sharpness = [], [], [], []
        for i in range(n):
            im = raw_frame(run, c, i)
            means.append(float(im.mean()/255))
            saturated.append(float(np.mean(im==255)))
            black.append(float(np.mean(im==0)))
            sharpness.append(float(cv2.Laplacian(im, cv2.CV_64F).var()))
        for i in np.linspace(0, n-1, 6, dtype=int):
            cv2.imwrite(str(folder/f'frame_{i:06d}.png'), raw_frame(run, c, i))
        for i in range(n):
            row = {'run': run.name, 'index': i, 'frame_id': int(c['frame_id'][i]),
                   'split': split, 'host_ns': int(c['host_ns'][i]),
                   'deck_ms': int(c['deck_ms'][i]), 'accepted': bool(valid[i]),
                   'reason': str(reason[i]), 'mocap_bracket_ms': float(gaps[i]),
                   'camera_clock_residual_ms':float(clock_residual_ms[i]),
                   'mean_intensity': means[i], 'fraction_255': saturated[i],
                   'fraction_0': black[i], 'laplacian_variance': sharpness[i]}
            for j, k in enumerate(['body_x','body_y','body_z']):
                row[k] = float(pp[i,j]) if valid[i] else None
            for j, k in enumerate(['body_qx','body_qy','body_qz','body_qw']):
                row[k] = float(qq[i,j]) if valid[i] else None
            all_rows.append(row)
        summary = {'run': run.name, 'split': split, 'frames': n,
                   'mocap_rows': len(m['seq']), 'accepted': int(valid.sum()),
                   'excluded_outside': int(np.sum(reason=='outside_mocap_support')),
                   'excluded_gap': int(np.sum(reason=='mocap_gap')),
                   'excluded_rotation_jump':int(np.sum(reason=='mocap_rotation_jump')),
                   'excluded_camera_arrival':int(np.sum(reason=='camera_arrival_outlier')),
                   'camera_duration_s': float((c['host_ns'][-1]-c['host_ns'][0])*1e-9),
                   'camera_rate_hz': float((n-1)/((c['host_ns'][-1]-c['host_ns'][0])*1e-9)),
                   'mocap_start_minus_camera_start_s': float((m['host_ns'][0]-c['host_ns'][0])*1e-9),
                   'mocap_max_host_gap_ms': float(np.max(np.diff(m['host_ns']))*1e-6),
                   'mocap_source_seq_nonincreasing': int(np.sum(np.diff(m['source_seq'])<=0)),
                   'mean_fraction_255': float(np.mean(saturated)),
                   'mean_fraction_0': float(np.mean(black)),
                   'camera_clock': cam_clock,
                   'mocap_clock': clock_diagnostic(m['source_ts_s'],m['host_ns'])}
        summaries.append(summary)
        prepared.append((run.name,split,pp,qq,valid))
        for f in sorted(run.iterdir()):
            if f.is_file(): hashes[str(f.relative_to(args.data))] = sha256(f)
    # A held-out run is not automatically an unseen pose: flag spatial/angular overlap.
    tp = np.concatenate([p[v] for _,s,p,q,v in prepared if s=='train'])
    tq = np.concatenate([q[v] for _,s,p,q,v in prepared if s=='train'])
    tree = cKDTree(tp)
    unseen = {}
    for name,s,p,q,v in prepared:
        ids = np.flatnonzero(v)
        for i in ids:
            close = tree.query_ball_point(p[i], .20)
            overlap = bool(close) and bool(np.any(2*np.arccos(np.clip(np.abs(tq[close]@q[i]),0,1)) < np.deg2rad(10)))
            unseen[(name,int(i))] = not overlap if s != 'train' else False
    for row in all_rows:
        row['unseen_pose_20cm_10deg'] = unseen.get((row['run'],row['index']),False)
    pq.write_table(pa.Table.from_pylist(all_rows),out/'frame_index.parquet')
    with (out/'frame_index.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    for s in summaries:
        s['unseen_accepted'] = sum(r['unseen_pose_20cm_10deg'] for r in all_rows if r['run']==s['run'])
    write_json(out/'audit.json',{'status':'prepared; rendering baseline not measured',
                               'alignment':'host arrival; unknown differential sensor/transport latency',
                               'max_mocap_gap_ms':args.max_gap_ms,
                               'max_camera_clock_residual_ms':args.max_clock_residual_ms,
                               'max_bracket_rotation_deg':45.,'runs':summaries})
    write_json(out/'source_sha256.json',hashes)
    write_json(out/'camera_config.json',{
        'width':320,'height':240,'K':K.tolist(),
        'distortion_opencv_k1_k2_p1_p2_k3':DIST.tolist(),
        'baseline_distortion':'none: supplied K, unwarped raw target per Step 1',
        'grayscale':'cv2.COLOR_RGB2GRAY on float32 RGB in [0,1]',
        'camera_to_body_rotation_assumed':R_BC.tolist(),
        'camera_origin_in_body_m':P_BC.tolist(),
        'assumed_body_axes':'forward x, left y, up z; camera right x, down y, forward z',
        'quaternion_assumption':'active body-to-mocap, xyzw',
        'conventions_verified':False,'scene_to_mocap_registration':None,
        'scene_checkpoint':None})
    print(json.dumps(summaries,indent=2))


def score_gray(real, pred):
    if real.shape != pred.shape or real.ndim != 2:
        raise ValueError('Metric arrays must be equal-size grayscale images')
    real, pred = real.astype(np.float64), pred.astype(np.float64)
    if not np.isfinite(real).all() or not np.isfinite(pred).all():
        raise ValueError('Nonfinite pixels')
    if min(real.min(),pred.min()) < 0 or max(real.max(),pred.max()) > 1:
        raise ValueError('Metrics require intensities in [0,1]')
    err = pred-real; mse = float(np.mean(err**2))
    blur = lambda x: cv2.GaussianBlur(x,(11,11),1.5,borderType=cv2.BORDER_REFLECT_101)[5:-5,5:-5]
    a,b = blur(real),blur(pred)
    va,vb,cov = blur(real**2)-a*a,blur(pred**2)-b*b,blur(real*pred)-a*b
    ssim = ((2*a*b+.01**2)*(2*cov+.03**2))/((a*a+b*b+.01**2)*(va+vb+.03**2))
    return {'mae':float(np.mean(abs(err))),'rmse':float(np.sqrt(mse)),
            'mse':mse,'psnr_db':float(-10*np.log10(mse)) if mse>0 else None,
            'perfect_match':mse==0,'ssim':float(ssim.mean())}


def evaluate(args):
    records = pq.read_table(args.prepared/'frame_index.parquet').to_pylist()
    index = {(r['run'],r['index']):r for r in records}
    with args.render_index.open() as f: renders = list(csv.DictReader(f))
    if not renders: raise ValueError('No rendered pairs provided')
    seen, tables, rows = set(), {}, []
    for r in renders:
        key=(r['run'],int(r['index']))
        if key in seen: raise ValueError(f'Duplicate rendered pair {key}')
        seen.add(key)
        src=index[key]
        if not src['accepted']: raise ValueError(f'Excluded camera frame {key}')
        run=args.data/key[0]
        if key[0] not in tables: tables[key[0]]=read_table(run/'camera.parquet')
        real=raw_frame(run,tables[key[0]],key[1]).astype(np.float32)/255
        path=Path(r['rgb_path'])
        if not path.is_absolute(): path=args.render_index.parent/path
        # NPY holds float RGB; PNG is read as OpenCV BGR and explicitly reordered.
        if path.suffix.lower()=='.npy': rgb=np.load(path,allow_pickle=False)
        else:
            bgr=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            if bgr is None or bgr.dtype!=np.uint8 or bgr.ndim!=3 or bgr.shape[2]!=3:
                raise ValueError('Use 8-bit 3-channel PNG or float RGB NPY')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        if rgb.shape!=(240,320,3) or not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:
            raise ValueError('Rendered RGB must have shape (240,320,3), range [0,1]')
        gray=cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2GRAY)
        rows.append({'run':key[0],'index':key[1],'split':src['split'],
                     'unseen_pose_20cm_10deg':src['unseen_pose_20cm_10deg'],**score_gray(real,gray)})
    args.output.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows),args.output/'metrics.parquet')
    with (args.output/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    meta_path=args.render_index.parent/'render_metadata.json'
    metadata=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    registration_status=metadata.get('registration',{}).get('status','unspecified')
    summary={'status':'diagnostic: world registration unverified' if registration_status!='verified' else 'scored; timing assumptions still apply',
             'registration_status':registration_status,'render_metadata':str(meta_path) if meta_path.exists() else None,'groups':{}}
    for split in ['train','validation','test']:
        for only_unseen in [False,True]:
            rr=[r for r in rows if r['split']==split and (not only_unseen or r['unseen_pose_20cm_10deg'])]
            if not rr: continue
            mse=float(np.mean([r['mse'] for r in rr]))
            summary['groups'][split+('_unseen' if only_unseen else '')]={
                'scored_frames':len(rr),'accepted_available':sum(r['accepted'] and r['split']==split and (not only_unseen or r['unseen_pose_20cm_10deg']) for r in records),
                'mae':float(np.mean([r['mae'] for r in rr])),
                'pooled_rmse':float(np.sqrt(mse)),
                'pooled_psnr_db':float(-10*np.log10(mse)) if mse>0 else None,
                'ssim':float(np.mean([r['ssim'] for r in rr]))}
    write_json(args.output/'metrics_summary.json',summary)
    print(json.dumps(summary,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.set_defaults(func=prepare)
    p.add_argument('--data',type=Path,default=PC.DATA)
    p.add_argument('--output',type=Path,default=ROOT/'output/baseline')
    p.add_argument('--max-gap-ms',type=float,default=50.)
    p.add_argument('--max-clock-residual-ms',type=float,default=50.)
    e=sub.add_parser('evaluate'); e.set_defaults(func=evaluate)
    e.add_argument('--data',type=Path,default=PC.DATA)
    e.add_argument('--prepared',type=Path,default=ROOT/'output/baseline')
    e.add_argument('--render-index',type=Path,required=True)
    e.add_argument('--output',type=Path,default=ROOT/'output/baseline/evaluation')
    args=ap.parse_args(); args.func(args)


if __name__=='__main__': main()
