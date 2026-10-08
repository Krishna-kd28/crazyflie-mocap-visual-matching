#!/usr/bin/env python3
"""Render prepared Crazyflie poses through the unedited source GSplat.

R_GM, o_M, and m come from an explicit registration JSON. With A = D W,
p_SC = s (R_A R_GM (p_MC - o_M) / m + t_A),
R_SC = R_A R_GM R_MC, and T_CS = inverse(T_SC).
M: mocap frame; G: metric scene frame; S: checkpoint frame; C: OpenCV camera.
D, s: dataparser rigid transform and scale; W: source world transform.
Prepared camera poses already contain the Crazyflie lever arm and mount.
"""
from __future__ import annotations

import project_config as PC

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = PC.WORKSPACE
SCENE = PC.path('scene_dir')
C0 = 0.28209479177387814


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_rotation(R, name):
    R = np.asarray(R, dtype=np.float64)
    if R.shape[-2:] != (3, 3) or not np.isfinite(R).all():
        raise ValueError(f'{name}: expected finite 3x3 rotation(s)')
    error = np.max(np.abs(np.swapaxes(R, -1, -2) @ R - np.eye(3)))
    if error > 2e-5 or np.max(np.abs(np.linalg.det(R) - 1)) > 2e-5:
        raise ValueError(f'{name}: invalid rigid rotation, orthogonality error={error}')
    return R


def homogeneous(matrix, name):
    T = np.asarray(matrix, dtype=np.float64)
    if T.shape == (3, 4):
        T = np.vstack((T, [0, 0, 0, 1]))
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f'{name}: expected finite 3x4 or 4x4 transform')
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError(f'{name}: invalid homogeneous row')
    check_rotation(T[:3, :3], name)
    return T


def map_camera_poses(camera_to_mocap, registration, dataparser, world):
    """Return (camera-to-checkpoint, checkpoint-to-camera), without axis flips.

    Input cameras already use x right, y down, z forward. Source get_viewmat's
    OpenGL-to-OpenCV flip is therefore unnecessary here.
    """
    T_MC = np.asarray(camera_to_mocap, dtype=np.float64)
    if T_MC.ndim != 3 or T_MC.shape[1:] != (4, 4):
        raise ValueError('Camera poses must have shape (N,4,4)')
    if not np.isfinite(T_MC).all() or not np.allclose(T_MC[:, 3], [0, 0, 0, 1]):
        raise ValueError('Camera poses contain nonfinite values or invalid last row')
    check_rotation(T_MC[:, :3, :3], 'camera-to-mocap')
    R_GM = check_rotation(registration['R_scene_metric_from_mocap'], 'registration')
    o_M = np.asarray(registration['mocap_origin_m'], dtype=float)
    m = float(registration['meters_per_scene_unit'])
    s = float(dataparser['scale'])
    if o_M.shape != (3,) or not np.isfinite(o_M).all() or not (m > 0 and s > 0):
        raise ValueError('Registration origin and scales are invalid')
    A = homogeneous(dataparser['transform'], 'dataparser') @ homogeneous(
        world['world_transform'], 'world frame')
    p_G = (T_MC[:, :3, 3] - o_M) @ R_GM.T
    T_SC = np.repeat(np.eye(4)[None], len(T_MC), axis=0)
    T_SC[:, :3, :3] = A[:3, :3] @ R_GM @ T_MC[:, :3, :3]
    T_SC[:, :3, 3] = s * ((p_G / m) @ A[:3, :3].T + A[:3, 3])
    T_CS = np.repeat(np.eye(4)[None], len(T_MC), axis=0)
    T_CS[:, :3, :3] = T_SC[:, :3, :3].transpose(0, 2, 1)
    T_CS[:, :3, 3] = -np.einsum('nij,nj->ni', T_CS[:, :3, :3], T_SC[:, :3, 3])
    residual = np.max(np.abs(T_CS @ T_SC - np.eye(4)))
    if residual > 2e-5:
        raise ValueError(f'Pose inverse check failed: {residual}')
    # A point one meter forward must project onto the supplied principal point.
    center = np.c_[T_SC[:, :3, 3], np.ones(len(T_SC))]
    forward = center.copy()
    forward[:, :3] += (s / m) * T_SC[:, :3, 2]
    center_cam = np.einsum('nij,nj->ni', T_CS, center)
    forward_cam = np.einsum('nij,nj->ni', T_CS, forward)
    if not np.allclose(center_cam[:, :3], 0, atol=2e-5):
        raise ValueError('Camera center does not map to the view origin')
    if not np.allclose(forward_cam[:, :3], [0, 0, s / m], atol=2e-5):
        raise ValueError('Camera forward axis has an incorrect sign or scale')
    return T_SC, T_CS, float(residual)


def select_frames(args):
    with (args.prepared / 'frame_index.csv').open() as f:
        rows = list(csv.DictReader(f))
    available = sorted({r['run'] for r in rows})
    if args.run and set(args.run) - set(available):
        raise ValueError(f'Unknown run; available: {available}')
    selected = []
    for run in available:
        if args.run and run not in args.run:
            continue
        rr = [r for r in rows if r['run'] == run and r['accepted'].lower() == 'true'
              and (args.split == 'all' or r['split'] == args.split)]
        if args.indices:
            wanted = {int(x) for x in args.indices.split(',')}
            rr = [r for r in rr if int(r['index']) in wanted]
            if wanted != {int(r['index']) for r in rr}:
                raise ValueError(f'{run}: requested indices excluded or outside chosen split')
        elif not args.all_frames and rr:
            picks = np.unique(np.linspace(0, len(rr) - 1, min(args.per_run, len(rr))).astype(int))
            rr = [rr[i] for i in picks]
        selected.extend(rr)
    if not selected:
        raise ValueError('No accepted frames selected')
    return selected


def prepare_geometry(args, camera, registration, selected):
    matrices = []
    max_prepared_error = 0.0
    R_BC = check_rotation(camera['camera_to_body_rotation_assumed'], 'camera mount')
    p_BC = np.asarray(camera['camera_origin_in_body_m'], dtype=float)
    hashes = {}
    for run in sorted({r['run'] for r in selected}):
        path = args.prepared / run / 'poses.npz'
        with np.load(path, allow_pickle=False) as data:
            for row in [r for r in selected if r['run'] == run]:
                i = int(row['index'])
                if not data['accepted'][i]:
                    raise ValueError(f'Pose file excludes {run}/{i}')
                T = data['camera_to_mocap_assumed'][i]
                R_MB = Rotation.from_quat(data['body_quaternion_xyzw'][i]).as_matrix()
                expected = np.eye(4)
                expected[:3, :3] = R_MB @ R_BC
                expected[:3, 3] = data['body_position'][i] + R_MB @ p_BC
                error = float(np.max(np.abs(T - expected)))
                if error > 1e-8:
                    raise ValueError(f'{run}/{i}: prepared camera pose differs from declared mount/lever')
                max_prepared_error = max(max_prepared_error, error)
                if not np.allclose(data['K'], camera['K'], atol=1e-9):
                    raise ValueError('Prepared K differs from supplied camera configuration')
                matrices.append(T)
        hashes[str(path)] = sha256(path)
    dp = read_json(args.scene / 'dataparser_transforms.json')
    wf = read_json(args.scene / 'world_frame.json')
    T_SC, T_CS, inverse_error = map_camera_poses(np.asarray(matrices), registration, dp, wf)
    return T_SC, T_CS, {'prepared_mount_max_abs_error': max_prepared_error,
                       'view_inverse_max_abs_error': inverse_error,
                       'forward_axis_check': 'passed', 'pose_sha256': hashes}


def load_scene(path, device):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)['pipeline']
    def get(name, allow_infinite=False):
        value = state['_model.gauss_params.' + name].detach().float()
        if torch.isnan(value).any() or (not allow_infinite and not torch.isfinite(value).all()):
            raise ValueError(f'Nonfinite source Gaussian tensor: {name}')
        return value
    means, quats, logs, logits = [get(k) for k in ('means', 'quats', 'scales', 'opacities')]
    # Degree-zero Splatfacto stores logits; +/-infinity encodes exact 1/0
    # after sigmoid. Preserve that valid limiting value, while rejecting NaNs.
    dc, rest = get('features_dc', allow_infinite=True), get('features_rest')
    n = len(means)
    if means.shape != (n, 3) or quats.shape != (n, 4) or logs.shape != (n, 3):
        raise ValueError('Invalid Gaussian geometry shapes')
    if dc.shape != (n, 3) or rest.ndim != 3 or rest.shape[0] != n or rest.shape[-1] != 3:
        raise ValueError('Invalid source color tensor shapes')
    if torch.any(torch.linalg.vector_norm(quats, dim=1) <= 1e-12):
        raise ValueError('Source has zero-length Gaussian quaternions')
    scales, opacity = torch.exp(logs), torch.sigmoid(logits).reshape(-1)
    if not torch.isfinite(scales).all() or not torch.all(scales > 0) or len(opacity) != n:
        raise ValueError('Invalid activated Gaussian scales or opacities')
    bands = rest.shape[1] + 1
    degree = int(round(np.sqrt(bands))) - 1
    if (degree + 1) ** 2 != bands:
        raise ValueError('Color coefficient count is not a complete SH degree')
    if degree == 0:
        colors = ((torch.sigmoid(dc) - 0.5) / C0)[:, None, :]
        error = float(torch.max(torch.abs(C0 * colors[:, 0] + 0.5 - torch.sigmoid(dc))))
        if error > 1e-6:
            raise ValueError('Degree-zero sigmoid/SH conversion check failed')
        encoding = 'sigmoid(dc), re-expressed as SH0'
    else:
        if not torch.isfinite(dc).all():
            raise ValueError('Higher-degree SH coefficients must be finite')
        colors = torch.cat((dc[:, None, :], rest), dim=1)
        error, encoding = 0.0, 'all source SH coefficients retained'
    result = dict(means=means, quats=quats, scales=scales, opacities=opacity, colors=colors)
    result = {k: v.to(device).contiguous() for k, v in result.items()}
    info = {'gaussians': n, 'sh_degree': degree, 'sh_coefficients': bands,
            'color_encoding': encoding, 'color_conversion_max_abs_error': error,
            'source_dc_positive_infinity': int(torch.isposinf(dc).sum()),
            'source_dc_negative_infinity': int(torch.isneginf(dc).sum())}
    return result, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared', type=Path, default=ROOT / 'output/baseline')
    ap.add_argument('--scene', type=Path, default=SCENE)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--registration', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'output/render_preview')
    ap.add_argument('--split', choices=['train', 'validation', 'test', 'all'], default='train')
    ap.add_argument('--run', action='append')
    ap.add_argument('--per-run', type=int, default=4)
    ap.add_argument('--indices', help='Comma-separated accepted frame row indices; requires --run')
    ap.add_argument('--all-frames', action='store_true')
    ap.add_argument('--check-only', action='store_true', help='Validate camera/scene transforms without rendering')
    ap.add_argument('--inspect-scene', action='store_true', help='Also load and validate checkpoint on CPU in check-only mode')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if args.per_run < 1 or (args.indices and not args.run):
        ap.error('--per-run must be positive; --indices requires --run')
    args.prepared, args.out, args.scene = [p.resolve() for p in (args.prepared, args.out, args.scene)]
    checkpoint = args.checkpoint or args.scene / 'nerfstudio_models/step-000129999.ckpt'
    camera_path = args.prepared / 'camera_config.json'
    camera, registration = read_json(camera_path), read_json(args.registration)
    K = np.asarray(camera['K'], dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError('Invalid K')
    if K[0, 0] <= 0 or K[1, 1] <= 0 or K[0, 1] != 0 or K[1, 0] != 0:
        raise ValueError('gsplat baseline expects positive focal lengths and zero skew')
    selected = select_frames(args)
    T_SC, T_CS, checks = prepare_geometry(args, camera, registration, selected)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [checkpoint, camera_path, args.registration, args.scene / 'dataparser_transforms.json',
               args.scene / 'world_frame.json', Path(__file__)]
    metadata = {
        'status': 'geometry_checks_passed', 'rendered_frames': 0,
        'registration': registration, 'camera_config': camera,
        'checkpoint': str(checkpoint.resolve()), 'camera_model': 'pinhole',
        'distortion': 'none', 'background_rgb': [0.0, 0.0, 0.0],
        'rasterize_mode': 'classic', 'eps2d': 0.3,
        'near_plane_checkpoint_units': 0.01, 'far_plane_checkpoint_units': 1e10,
        'scene_edited': False, 'camera_axes': 'OpenCV: right, down, forward',
        'selection': [{'run': r['run'], 'index': int(r['index']), 'split': r['split']} for r in selected],
        'checks': checks, 'sha256': {str(p.resolve()): sha256(p) for p in sources},
    }
    if args.check_only:
        if args.inspect_scene:
            _, metadata['scene'] = load_scene(checkpoint, 'cpu')
        write_json(args.out / 'geometry_check.json', metadata)
        print(json.dumps({'status': metadata['status'], 'selected_frames': len(selected),
                          'checks': checks, 'scene': metadata.get('scene')}, indent=2))
        return
    if (args.out / 'render_index.csv').exists() and not args.overwrite:
        raise FileExistsError('render_index.csv already exists; use a new output directory or --overwrite')
    import torch
    from gsplat import __version__ as gsplat_version
    from gsplat.rendering import rasterization
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this process; no images rendered. Geometry check can run on CPU.')
    device = 'cuda'
    scene, scene_info = load_scene(checkpoint, device)
    metadata.update(status='rendering', scene=scene_info, torch_version=torch.__version__,
                    gsplat_version=gsplat_version, gpu=torch.cuda.get_device_name(0))
    write_json(args.out / 'render_metadata.json', metadata)
    np.savez_compressed(args.out / 'render_poses.npz', camera_to_checkpoint=T_SC,
                        checkpoint_to_camera=T_CS, K=K,
                        run=np.array([r['run'] for r in selected]),
                        index=np.array([int(r['index']) for r in selected]))
    K_tensor = torch.tensor(K, dtype=torch.float32, device=device)[None]
    records = []
    with torch.no_grad():
        for j, row in enumerate(selected):
            rgb, alpha, _ = rasterization(
                **scene, viewmats=torch.tensor(T_CS[j:j + 1], dtype=torch.float32, device=device),
                Ks=K_tensor, width=int(camera['width']), height=int(camera['height']),
                packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB',
                sh_degree=scene_info['sh_degree'], rasterize_mode='classic', eps2d=0.3,
                camera_model='pinhole', backgrounds=torch.zeros((1, 3), device=device))
            im = rgb[0].clamp(0, 1).cpu().numpy()
            al = alpha[0, ..., 0].cpu().numpy()
            if not np.isfinite(im).all() or not np.isfinite(al).all():
                raise ValueError(f'Nonfinite render for {row["run"]}/{row["index"]}')
            folder = args.out / row['run']
            folder.mkdir(exist_ok=True)
            stem = folder / f'frame_{int(row["index"]):06d}'
            np.save(str(stem) + '.rgb.npy', im, allow_pickle=False)
            np.save(str(stem) + '.alpha.npy', al, allow_pickle=False)
            Image.fromarray(np.round(im * 255).astype(np.uint8)).save(str(stem) + '.png')
            record = {k: row[k] for k in ('run', 'index', 'frame_id', 'split')}
            record.update(rgb_path=str(stem) + '.rgb.npy', png_path=str(stem) + '.png',
                          alpha_path=str(stem) + '.alpha.npy', mean_alpha=float(al.mean()),
                          fraction_alpha_below_0_5=float(np.mean(al < 0.5)))
            records.append(record)
            print(f'Rendered {j + 1}/{len(selected)}: {row["run"]}/{row["index"]}', flush=True)
    with (args.out / 'render_index.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata.update(status='render_complete_registration_' + registration.get('status', 'unspecified'),
                    rendered_frames=len(records))
    write_json(args.out / 'render_metadata.json', metadata)
    print(f'Saved {len(records)} renders to {args.out}; registration status: {registration.get("status")}')


if __name__ == '__main__':
    main()
