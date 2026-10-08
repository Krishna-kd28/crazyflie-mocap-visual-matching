#!/usr/bin/env python3
"""Shared data and geometry for the fixed-feature registration pipeline.

Frames: M is the recorded mocap frame, G the nominal metric scene frame in
metres (0.85 m per checkpoint unit), S the checkpoint frame passed to gsplat,
and C the OpenCV camera frame (x right, y down, z forward).

A registration is the similarity p_G = lambda R p_M + t, R_GC = R R_MC.
The renderer JSON (research/registration_landmarks_rigid.json) stores the
same map as R_scene_metric_from_mocap = R, meters_per_scene_unit = 0.85/lambda
and mocap_origin_m = -R^T t / lambda; both directions are implemented here.
"""
from __future__ import annotations

import project_config as PC

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = PC.WORKSPACE
DATA = PC.DATA
POSES_DIR = ROOT / 'output/motion_sync/candidate_pairs'
SCENE = PC.path('scene_dir')
CHECKPOINT = PC.path('checkpoint')
START_REGISTRATION = PC.path('seed_registration')
CAMERA_CONFIG = PC.path('camera')
FF_OUT = ROOT / 'output/fixed_features'
FRAMES = FF_OUT / 'frames'
ANNOTATIONS = ROOT / 'research/fixed_features/annotations.json'
PROPOSALS = FF_OUT / 'proposals/proposals.json'
RUNS = PC.RUNS
SHORT = {r: r[-6:] for r in RUNS}
NOMINAL_M_PER_UNIT = float(PC.CONFIG['nominal_meters_per_scene_unit'])
WIDTH, HEIGHT = 320, 240
K = np.asarray(PC.read_camera()['K'], dtype=float)
DIST = np.asarray(PC.read_camera()['distortion_opencv_k1_k2_p1_p2_k3'], dtype=float)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_rotation(R, name='rotation'):
    R = np.asarray(R, dtype=np.float64)
    err = np.max(np.abs(np.swapaxes(R, -1, -2) @ R - np.eye(3)))
    if R.shape[-2:] != (3, 3) or not np.isfinite(R).all() or err > 2e-5 \
            or np.max(np.abs(np.linalg.det(R) - 1)) > 2e-5:
        raise ValueError(f'{name}: not a proper rotation (error {err})')
    return R


# ----------------------------------------------------------------------------
# Registration JSON <-> (R, t, lambda)
# ----------------------------------------------------------------------------
def load_registration(path=START_REGISTRATION):
    """Return dict(R, t, lam) with p_G = lam R p_M + t in nominal metres."""
    j = read_json(path)
    R = check_rotation(j['R_scene_metric_from_mocap'], 'registration')
    m = float(j['meters_per_scene_unit'])
    o_M = np.asarray(j['mocap_origin_m'], dtype=float)
    lam = NOMINAL_M_PER_UNIT / m
    t = -lam * R @ o_M
    return dict(R=R, t=t, lam=lam, label=j.get('label', ''), path=str(path))


def registration_json(R, t, lam, label, provenance):
    """Renderer-compatible JSON for the similarity p_G = lam R p_M + t."""
    R = check_rotation(R)
    t = np.asarray(t, dtype=float)
    lam = float(lam)
    o_M = -R.T @ t / lam
    return {
        'status': 'provisional_landmark_fit',
        'R_scene_metric_from_mocap': R.tolist(),
        'mocap_origin_m': o_M.tolist(),
        'meters_per_scene_unit': NOMINAL_M_PER_UNIT / lam,
        'label': label,
        'nominal_scene_similarity': {
            'equation': 'p_G = lambda R p_M + t', 'lambda': lam, 'R_GM': R.tolist(),
            'translation_G_nominal_m': t.tolist(),
            'source_nominal_meters_per_scene_unit': NOMINAL_M_PER_UNIT},
        'renderer_adapter': 'm_eff=0.85/lambda; o_M=-R^T t/lambda',
        **provenance,
    }


# ----------------------------------------------------------------------------
# Poses
# ----------------------------------------------------------------------------
def load_run_poses(run):
    """Corrected (timing-adjusted) camera-to-mocap poses for one run."""
    path = POSES_DIR / run / 'poses.npz'
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k].copy() for k in d.files}
    out['path'] = str(path)
    T = out['camera_to_mocap_assumed']
    acc = out['accepted']
    if not np.isfinite(T[acc]).all():
        raise ValueError(f'{run}: nonfinite accepted poses')
    check_rotation(T[acc][:, :3, :3], f'{run} camera-to-mocap')
    t0 = out['host_ns'][0]
    out['time_s'] = (out['pose_query_host_ns'] - t0) * 1e-9
    return out


def apply_registration(T_MC, R, t, lam):
    """Camera-to-scene poses T_GC (metres) for camera-to-mocap poses T_MC."""
    T_MC = np.asarray(T_MC, dtype=np.float64)
    T_GC = np.repeat(np.eye(4)[None], len(T_MC), axis=0)
    T_GC[:, :3, :3] = R @ T_MC[:, :3, :3]
    T_GC[:, :3, 3] = lam * (T_MC[:, :3, 3] @ R.T) + t
    return T_GC


def scene_transforms():
    dp = read_json(SCENE / 'dataparser_transforms.json')
    wf = read_json(SCENE / 'world_frame.json')
    D = np.vstack((np.asarray(dp['transform'], dtype=float), [0, 0, 0, 1]))
    W = np.asarray(wf['world_transform'], dtype=float)
    A = D @ W
    check_rotation(A[:3, :3], 'dataparser*world')
    return A, float(dp['scale'])


def checkpoint_views(T_GC):
    """(T_SC, T_CS): camera-to-checkpoint and the gsplat view matrices."""
    A, s = scene_transforms()
    T_GC = np.asarray(T_GC, dtype=np.float64)
    T_SC = np.repeat(np.eye(4)[None], len(T_GC), axis=0)
    T_SC[:, :3, :3] = A[:3, :3] @ T_GC[:, :3, :3]
    T_SC[:, :3, 3] = s * ((T_GC[:, :3, 3] / NOMINAL_M_PER_UNIT) @ A[:3, :3].T + A[:3, 3])
    T_CS = np.repeat(np.eye(4)[None], len(T_GC), axis=0)
    T_CS[:, :3, :3] = T_SC[:, :3, :3].transpose(0, 2, 1)
    T_CS[:, :3, 3] = -np.einsum('nij,nj->ni', T_CS[:, :3, :3], T_SC[:, :3, 3])
    if np.max(np.abs(T_CS @ T_SC - np.eye(4))) > 2e-5:
        raise ValueError('View inverse check failed')
    return T_SC, T_CS


def metres_per_checkpoint_unit():
    return NOMINAL_M_PER_UNIT / scene_transforms()[1]


# ----------------------------------------------------------------------------
# Projection helpers
# ----------------------------------------------------------------------------
def project(T_GC, X, K=K):
    """Pixels and depths of scene points X[N,3] seen from poses T_GC[N,4,4]."""
    q = np.einsum('nji,nj->ni', T_GC[:, :3, :3], X - T_GC[:, :3, 3])
    h = q @ K.T
    z = h[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return h[:, :2] / safe[:, None], q[:, 2]


# ----------------------------------------------------------------------------
# Lens model: pinhole (undistorted) pixels <-> raw (recorded) pixels, same K.
# Forward model (OpenCV k1, k2, p1, p2, k3) on normalised coordinates (x, y):
#   r2 = x^2 + y^2, f = 1 + k1 r2 + k2 r2^2 + k3 r2^3,
#   xd = f x + 2 p1 x y + p2 (r2 + 2 x^2),  yd = f y + p1 (r2 + 2 y^2) + 2 p2 x y.
# The supplied polynomial folds over at large radius, so beyond the pinhole
# radius r_c where its radial slope drops to MIN_SLOPE the radial function is
# continued as a straight line with that slope (C1 continuous, monotonic).
# Pixels that need the continuation are "extrapolated" and flagged as such.
# ----------------------------------------------------------------------------
MIN_SLOPE = 0.25
_RADIAL = {}


def radial_model():
    """Cut-over radius of the supplied radial model and its linear continuation.

    Returns dict(r_c, rd_c, slope, a, b, rd_c_px): pinhole radius r_c where
    d r_d/d r = MIN_SLOPE, the raw radius rd_c there (also in px), and the
    continuation r_d = a + b r used for r > r_c.
    """
    if _RADIAL:
        return _RADIAL
    k1, k2, p1, p2, k3 = DIST
    r = np.linspace(0, 3, 30001)
    r2 = r * r
    slope = 1 + 3 * k1 * r2 + 5 * k2 * r2 ** 2 + 7 * k3 * r2 ** 3
    bad = np.flatnonzero(slope < MIN_SLOPE)
    r_c = float(r[bad[0] - 1]) if len(bad) else float(r[-1])
    rd_c = float(r_c * (1 + k1 * r_c ** 2 + k2 * r_c ** 4 + k3 * r_c ** 6))
    b = float(1 + 3 * k1 * r_c ** 2 + 5 * k2 * r_c ** 4 + 7 * k3 * r_c ** 6)
    _RADIAL.update(r_c=r_c, rd_c=rd_c, slope=b, a=rd_c - b * r_c, b=b,
                   rd_c_px=rd_c * float(np.sqrt(K[0, 0] * K[1, 1])), min_slope=MIN_SLOPE)
    return _RADIAL


def radial_validity(min_slope=MIN_SLOPE):
    """Kept for callers: raw/pinhole radius up to which the supplied model itself is used."""
    m = radial_model()
    return dict(r_max=m['r_c'], rd_max=m['rd_c'], rd_max_px=m['rd_c_px'], min_slope=m['min_slope'])


def _distort_normalised(n):
    k1, k2, p1, p2, k3 = DIST
    m = radial_model()
    x, y = n[:, 0], n[:, 1]
    r2 = x * x + y * y
    f = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = f * x + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = f * y + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    r = np.sqrt(r2)
    ext = r > m['r_c']
    if ext.any():
        # Radial factor continued linearly in r_d; tangential terms unchanged.
        g = (m['a'] / r[ext] + m['b'])            # r_d / r on the continuation
        xe, ye, r2e = x[ext], y[ext], r2[ext]
        xd[ext] = g * xe + 2 * p1 * xe * ye + p2 * (r2e + 2 * xe * xe)
        yd[ext] = g * ye + p1 * (r2e + 2 * ye * ye) + 2 * p2 * xe * ye
    return np.stack([xd, yd], axis=1)


def _distort_jacobian(n):
    k1, k2, p1, p2, k3 = DIST
    m = radial_model()
    x, y = n[:, 0], n[:, 1]
    r2 = x * x + y * y
    f = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    df = 2 * k1 + 4 * k2 * r2 + 6 * k3 * r2 ** 2
    J = np.empty((len(n), 2, 2))
    J[:, 0, 0] = f + x * x * df + 2 * p1 * y + 6 * p2 * x
    J[:, 0, 1] = x * y * df + 2 * p1 * x + 2 * p2 * y
    J[:, 1, 0] = x * y * df + 2 * p1 * x + 2 * p2 * y
    J[:, 1, 1] = f + y * y * df + 6 * p1 * y + 2 * p2 * x
    r = np.sqrt(r2)
    ext = r > m['r_c']
    if ext.any():
        xe, ye, re = x[ext], y[ext], r[ext]
        g = m['a'] / re + m['b']
        c = -m['a'] / re ** 3                      # (d g / d r) / r
        J[ext, 0, 0] = g + c * xe * xe + 2 * p1 * ye + 6 * p2 * xe
        J[ext, 0, 1] = c * xe * ye + 2 * p1 * xe + 2 * p2 * ye
        J[ext, 1, 0] = c * xe * ye + 2 * p1 * xe + 2 * p2 * ye
        J[ext, 1, 1] = g + c * ye * ye + 6 * p1 * ye + 2 * p2 * xe
    return J


LENS_MODEL_PATH = PC.path('lens')
_LENS = {}


def lens_model():
    """Active lens model: the calibrated angle-based model when
    research/lens_model.json exists, otherwise the supplied polynomial."""
    if _LENS:
        return _LENS
    if LENS_MODEL_PATH.exists():
        j = read_json(LENS_MODEL_PATH)
        _LENS.update(type='angle', k=[float(x) for x in j['k']], source=str(LENS_MODEL_PATH),
                     fx=float(j.get('fx', K[0, 0])), fy=float(j.get('fy', K[1, 1])),
                     cx=float(j.get('cx', K[0, 2])), cy=float(j.get('cy', K[1, 2])))
    else:
        _LENS.update(type='polynomial', source='supplied K and D')
    return _LENS


def _angle_rd(theta, k):
    rd = theta.copy()
    for m, km in enumerate(k):
        rd = rd + km * theta ** (2 * m + 3)
    return rd


def _angle_rd_slope(theta, k):
    d = np.ones_like(theta)
    for m, km in enumerate(k):
        d = d + (2 * m + 3) * km * theta ** (2 * m + 2)
    return d


def pinhole_to_raw(xy):
    """Undistorted (pinhole, supplied K) pixel coordinates -> raw recorded pixels."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = np.c_[(xy[:, 0] - K[0, 2]) / K[0, 0], (xy[:, 1] - K[1, 2]) / K[1, 1]]
    L = lens_model()
    if L['type'] == 'angle':
        r = np.hypot(n[:, 0], n[:, 1])
        theta = np.arctan(r)
        rd = _angle_rd(theta, L['k'])
        scale = np.where(r > 1e-12, rd / np.maximum(r, 1e-12), 1.0)
        return np.c_[L['fx'] * n[:, 0] * scale + L['cx'], L['fy'] * n[:, 1] * scale + L['cy']]
    d = _distort_normalised(n)
    return np.c_[d[:, 0] * K[0, 0] + K[0, 2], d[:, 1] * K[1, 1] + K[1, 2]]


def supplied_pinhole_to_raw(xy):
    """The supplied polynomial (with its linear continuation), regardless of the active model."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = np.c_[(xy[:, 0] - K[0, 2]) / K[0, 0], (xy[:, 1] - K[1, 2]) / K[1, 1]]
    d = _distort_normalised(n)
    return np.c_[d[:, 0] * K[0, 0] + K[0, 2], d[:, 1] * K[1, 1] + K[1, 2]]


def raw_to_pinhole(xy, iterations=40):
    """Raw recorded pixel coordinates -> undistorted (pinhole, supplied K) pixels.

    Angle model: Newton solve of r_d(theta) for theta (monotonic), then
    r = tan(theta). Polynomial model: Newton inversion with the linear
    continuation beyond the fold-over radius.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    L = lens_model()
    if L['type'] == 'angle':
        xd = (xy[:, 0] - L['cx']) / L['fx']
        yd = (xy[:, 1] - L['cy']) / L['fy']
        rd = np.hypot(xd, yd)
        theta = rd.copy()
        for _ in range(iterations):
            step = (_angle_rd(theta, L['k']) - rd) / _angle_rd_slope(theta, L['k'])
            theta = theta - step
            if np.max(np.abs(step)) < 1e-13:
                break
        r = np.tan(theta)
        scale = np.where(rd > 1e-12, r / np.maximum(rd, 1e-12), 1.0)
        return np.c_[xd * scale * K[0, 0] + K[0, 2], yd * scale * K[1, 1] + K[1, 2]]
    target = np.c_[(xy[:, 0] - K[0, 2]) / K[0, 0], (xy[:, 1] - K[1, 2]) / K[1, 1]]
    m = radial_model()
    rd = np.hypot(target[:, 0], target[:, 1])
    n = target.copy()
    far = rd > m['rd_c']
    if far.any():
        r0 = (rd[far] - m['a']) / m['b']
        n[far] = target[far] * (r0 / rd[far])[:, None]
    for _ in range(iterations):
        resid = _distort_normalised(n) - target
        J = _distort_jacobian(n)
        step = np.linalg.solve(J, resid[..., None])[..., 0]
        n = n - step
        if np.max(np.abs(step)) < 1e-12:
            break
    return np.c_[n[:, 0] * K[0, 0] + K[0, 2], n[:, 1] * K[1, 1] + K[1, 2]]


def is_extrapolated(xy_raw):
    """True beyond the conservative ~199 px polynomial cut-over.

    The angle fit has almost no correspondences there. The original calibration
    target coverage is unknown; this flag does not establish where it was measured.
    """
    xy = np.asarray(xy_raw, dtype=np.float64).reshape(-1, 2)
    rd = np.hypot((xy[:, 0] - K[0, 2]) / K[0, 0], (xy[:, 1] - K[1, 2]) / K[1, 1])
    return rd > radial_model()['rd_c']


def undistort_points(xy):
    """Raw pixel coordinates -> undistorted pixel coordinates under the same K."""
    return raw_to_pinhole(xy)


_CANVAS = {}


def distortion_canvas():
    """Padded pinhole canvas that covers the raw image after distortion.

    Returns dict(width, height, K_canvas, offset, map_x, map_y): render the
    pinhole image at (width, height) with K_canvas, then
    cv2.remap(image, map_x, map_y) gives the render in raw pixel coordinates.
    """
    if _CANVAS:
        return _CANVAS
    us, vs = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    raw = np.c_[us.ravel(), vs.ravel()]
    pin = raw_to_pinhole(raw)
    modelled = ~is_extrapolated(raw)          # historical evaluation support, independent of active lens
    x0 = int(np.floor(pin[:, 0].min())) - 2
    y0 = int(np.floor(pin[:, 1].min())) - 2
    x1 = int(np.ceil(pin[:, 0].max())) + 2
    y1 = int(np.ceil(pin[:, 1].max())) + 2
    Kc = K.copy()
    Kc[0, 2] -= x0
    Kc[1, 2] -= y0
    _CANVAS.update(width=x1 - x0 + 1, height=y1 - y0 + 1, K_canvas=Kc, offset=(-x0, -y0),
                   map_x=(pin[:, 0] - x0).reshape(HEIGHT, WIDTH).astype(np.float32),
                   map_y=(pin[:, 1] - y0).reshape(HEIGHT, WIDTH).astype(np.float32),
                   valid=modelled.reshape(HEIGHT, WIDTH), valid_fraction=float(modelled.mean()),
                   radial=radial_model(), lens=lens_model())
    return _CANVAS


def distort_image(canvas_image):
    """Warp a render made on the padded pinhole canvas into raw pixel coordinates."""
    import cv2
    c = distortion_canvas()
    if canvas_image.shape[0] != c['height'] or canvas_image.shape[1] != c['width']:
        raise ValueError('image is not on the padded pinhole canvas')
    return cv2.remap(canvas_image, c['map_x'], c['map_y'], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def triangulate(T_GC, xy, K=K):
    """Least-squares point from >=2 rays; returns X, per-view reprojection px."""
    T_GC = np.asarray(T_GC, dtype=float)
    xy = np.asarray(xy, dtype=float)
    A_rows, b_rows = [], []
    for T, uv in zip(T_GC, xy):
        d = T[:3, :3] @ np.linalg.solve(K, np.r_[uv, 1.])
        d /= np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A_rows.append(P)
        b_rows.append(P @ T[:3, 3])
    A = np.vstack(A_rows)
    b = np.concatenate(b_rows)
    X, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred, depth = project(T_GC, np.repeat(X[None], len(T_GC), 0), K)
    err = np.linalg.norm(pred - xy, axis=1)
    return X, err, depth


# ----------------------------------------------------------------------------
# Real frames
# ----------------------------------------------------------------------------
def read_camera_table(run):
    import pyarrow.parquet as pq
    return {k: np.asarray(v) for k, v in pq.read_table(DATA / run / 'camera.parquet').to_pydict().items()}


def raw_frame(run, table, i):
    import cv2
    with (DATA / run / 'camera_frames.bin').open('rb') as f:
        f.seek(int(table['blob_offset'][i]))
        payload = f.read(int(table['size'][i]))
    if len(payload) != int(table['size'][i]):
        raise ValueError('Truncated camera payload')
    if table['pixel_format'][i] == 1:
        im = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
    elif table['pixel_format'][i] == 0 and table['depth'][i] == 1:
        im = np.frombuffer(payload, np.uint8).reshape(int(table['height'][i]), int(table['width'][i]))
    else:
        raise ValueError('Unsupported camera format')
    if im is None or im.shape != (HEIGHT, WIDTH):
        raise ValueError('Unexpected frame shape')
    return im


def undistort_image(im):
    import cv2
    return cv2.undistort(im, K, DIST, None, K)


# ----------------------------------------------------------------------------
# Prepared frame store (output/fixed_features/frames)
# ----------------------------------------------------------------------------
def frame_store(run):
    """Index and pose arrays written by prepare_fixed_feature_frames.py."""
    folder = FRAMES / run
    index = read_json(folder / 'index.json')
    with np.load(folder / 'poses.npz', allow_pickle=False) as d:
        arrays = {k: d[k].copy() for k in d.files}
    return folder, index, arrays


def real_path(run, i):
    return FRAMES / run / 'real' / f'{i:06d}.png'


def real_raw_path(run, i):
    return FRAMES / run / 'real_raw' / f'{i:06d}.png'


def render_path(run, i):
    return FRAMES / run / 'render' / f'{i:06d}.png'


def render_distorted_path(run, i):
    return FRAMES / run / 'render_distorted' / f'{i:06d}.png'


def depth_path(run, i):
    return FRAMES / run / 'depth' / f'{i:06d}.npz'


def small_rotation(omega):
    return Rotation.from_rotvec(np.asarray(omega, dtype=float)).as_matrix()
