"""Fixed-support image comparison; intensities are always in [0, 1].

The composite is a declared engineering score, not a calibrated perceptual
probability. Return its components and the support coverage alongside it.
"""
from __future__ import annotations

import project_config as PC
import cv2
import numpy as np

GRAY = np.array([0.299, 0.587, 0.114], np.float32)


def ssim_map(a, b):
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT_101)
    ma, mb = blur(a), blur(b)
    va = np.maximum(0, blur(a*a)-ma*ma)
    vb = np.maximum(0, blur(b*b)-mb*mb)
    cov = blur(a*b)-ma*mb
    return ((2*ma*mb+0.01**2)*(2*cov+0.03**2)
            / ((ma*ma+mb*mb+0.01**2)*(va+vb+0.03**2)))


def edge_map(a):
    x = np.round(255*np.clip(a, 0, 1)).astype(np.uint8)
    x = cv2.GaussianBlur(x, (5, 5), 0.8)
    return cv2.Canny(x, 15, 35, L2gradient=True) > 0


def compare(a, b, mask=None, tolerance=2, intensity_scale=0.10):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected equal, finite grayscale images')
    if min(a.min(), b.min()) < -1e-6 or max(a.max(), b.max()) > 1+1e-6:
        raise ValueError('Images must use the fixed [0,1] intensity range')
    m = np.ones_like(a, bool) if mask is None else np.asarray(mask, bool).copy()
    if m.shape != a.shape:
        raise ValueError('Support shape differs from image')
    # An SSIM window must not run beyond the original image. Support is otherwise
    # fixed: it is never picked by an appearance-model residual.
    m[:5] = False; m[-5:] = False; m[:, :5] = False; m[:, -5:] = False
    if not m.any():
        return None
    delta = (a-b)[m]
    mae = float(np.abs(delta).mean()); rmse = float(np.sqrt(np.mean(delta*delta)))
    ss = float(np.clip(ssim_map(a, b)[m].mean(), -1, 1))
    ea, eb = edge_map(a), edge_map(b)
    na, nb = int((ea&m).sum()), int((eb&m).sum())
    # Matches may lie just outside the support; only edge query pixels count.
    da = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    db = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    precision = float((da[eb&m] <= tolerance).mean()) if nb else float(na == 0)
    recall = float((db[ea&m] <= tolerance).mean()) if na else float(nb == 0)
    ef = 2*precision*recall/max(precision+recall, 1e-12)
    intensity = float(np.exp(-mae/intensity_scale))
    structure = (1+ss)/2
    q = 100*float((intensity*structure*ef)**(1/3))
    grad = lambda x: np.hypot(cv2.Sobel(x, cv2.CV_32F, 1, 0)/8,
                              cv2.Sobel(x, cv2.CV_32F, 0, 1)/8)
    ga, gb = grad(a)[m], grad(b)[m]
    return dict(score=q, mae=mae, rmse=rmse, ssim=ss, edge_f1=ef,
                edge_precision=precision, edge_recall=recall,
                gradient_ratio=float(gb.mean()/max(ga.mean(), 1e-8)),
                coverage=float(m.mean()), pixels=int(m.sum()), real_edges=na, render_edges=nb)


def aggregate(rows):
    rows = [r for r in rows if r is not None]
    if not rows:
        return dict(frames=0)
    keys = ['score', 'mae', 'rmse', 'ssim', 'edge_f1', 'edge_precision',
            'edge_recall', 'gradient_ratio', 'coverage']
    # Every frame has equal influence, regardless of how many support pixels it has.
    return dict(frames=len(rows), **{k:float(np.mean([r[k] for r in rows])) for k in keys},
                score_median=float(np.median([r['score'] for r in rows])),
                score_p10=float(np.percentile([r['score'] for r in rows], 10)))
