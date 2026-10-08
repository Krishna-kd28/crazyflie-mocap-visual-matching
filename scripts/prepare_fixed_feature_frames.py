#!/usr/bin/env python3
"""Prepare real/rendered frame pairs for fixed-feature picking and checking.

`frames` (system python, needs pyarrow) writes, for every timing-accepted
frame of all four Crazyflie runs, the raw recorded image (real_raw/), its
undistorted version (real/, kept for reference) and the pose arrays.
`render` (CUDA environment with gsplat) renders the same poses through the
unedited scene under a registration; by default on the padded pinhole canvas
and warped with the supplied lens model into raw pixel coordinates
(render_distorted/), with --pinhole into the undistorted 320x240 frame
(render/). `depth` writes expected depth maps on the pinhole canvas so a render
click can be lifted to a scene point.
"""
from __future__ import annotations

import project_config as PC

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_features_common as C  # noqa: E402


def stage_frames(args):
    import cv2
    manifest = {'registration': str(args.registration), 'runs': {}, 'source_sha256': {}}
    reg = C.load_registration(args.registration)
    for run in C.RUNS:
        poses = C.load_run_poses(run)
        table = C.read_camera_table(run)
        acc = np.flatnonzero(poses['accepted'])
        folder = C.FRAMES / run
        (folder / 'real').mkdir(parents=True, exist_ok=True)
        (folder / 'real_raw').mkdir(parents=True, exist_ok=True)
        T_MC = poses['camera_to_mocap_assumed'][acc]
        T_GC = C.apply_registration(T_MC, reg['R'], reg['t'], reg['lam'])
        T_SC, T_CS = C.checkpoint_views(T_GC)
        np.savez_compressed(folder / 'poses.npz', index=acc, frame_id=poses['frame_id'][acc],
                            time_s=poses['time_s'][acc], T_MC=T_MC, T_GC=T_GC, T_CS=T_CS, K=C.K,
                            distortion=C.DIST)
        n_written = 0
        for i in acc:
            path = C.real_path(run, int(i))
            raw_path = C.real_raw_path(run, int(i))
            if path.exists() and raw_path.exists() and not args.overwrite:
                continue
            raw = C.raw_frame(run, table, int(i))
            cv2.imwrite(str(raw_path), raw)
            cv2.imwrite(str(path), C.undistort_image(raw))
            n_written += 1
        index = {'run': run, 'short': C.SHORT[run], 'accepted_indices': acc.tolist(),
                 'frame_id': poses['frame_id'][acc].tolist(),
                 'time_s': np.round(poses['time_s'][acc], 3).tolist(),
                 'total_frames': int(len(poses['accepted'])), 'width': C.WIDTH, 'height': C.HEIGHT,
                 'real_images': 'real_raw/: recorded pixels; real/: OpenCV undistort with supplied K and D, output K unchanged; grayscale',
                 'pose_source': poses['path'], 'pose_sha256': C.sha256(poses['path'])}
        C.write_json(folder / 'index.json', index)
        manifest['runs'][run] = {'accepted': int(len(acc)), 'written': n_written}
        manifest['source_sha256'][poses['path']] = index['pose_sha256']
        print(f'{run}: {len(acc)} accepted frames, {n_written} images written', flush=True)
    manifest['source_sha256'][str(args.registration)] = C.sha256(args.registration)
    manifest['start_registration'] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in reg.items()}
    C.write_json(C.FRAMES / 'manifest.json', manifest)


def load_scene(device):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_baseline import load_scene as _load
    return _load(C.CHECKPOINT, device)


class Renderer:
    """Renders RGB in raw pixel geometry (default) or on the undistorted 320x240 frame.

    Raw geometry is produced natively by gsplat's distorted camera path
    (camera_model='fisheye' with the calibrated angle model's coefficients,
    3DGUT: with_ut and with_eval3d) when the active lens model is the angle
    model and the installed gsplat supports it; otherwise by rendering on the
    padded pinhole canvas and warping with the lens model (overscan + remap).
    """

    def __init__(self, distorted=True, native=None):
        import inspect
        import torch
        from gsplat.rendering import rasterization
        assert torch.cuda.is_available(), 'CUDA required for rendering'
        torch.set_num_threads(4)
        self.torch, self.rasterization = torch, rasterization
        self.scene, self.info = load_scene('cuda')
        self.distorted = distorted
        lens = C.lens_model()
        supports = all(k in inspect.signature(rasterization).parameters for k in ('with_ut', 'with_eval3d', 'radial_coeffs'))
        self.native = bool(distorted and lens['type'] == 'angle' and supports) if native is None else bool(native)
        c = C.distortion_canvas()
        self.K_canvas, self.width_canvas, self.height_canvas = c['K_canvas'], c['width'], c['height']
        if distorted and not self.native:
            self.K, self.width, self.height = self.K_canvas, self.width_canvas, self.height_canvas
        else:
            self.K, self.width, self.height = C.K, C.WIDTH, C.HEIGHT
        self.Ks = torch.tensor(self.K, device='cuda', dtype=torch.float32)[None]
        self.Ks_canvas = torch.tensor(self.K_canvas, device='cuda', dtype=torch.float32)[None]
        self.bg = torch.zeros((1, 3), device='cuda')
        self.radial = None
        if self.native:
            k = list(lens['k']) + [0.0] * (4 - len(lens['k']))
            self.radial = torch.tensor([k[:4]], device='cuda', dtype=torch.float32)
        self.method = ('native fisheye render (gsplat 3DGUT, calibrated angle model)' if self.native else
                       'overscan pinhole render warped with the lens model' if distorted else 'pinhole render')

    def rgb_canvas(self, T_CS):
        """uint8 RGB pinhole render on the padded overscan canvas (no lens warp)."""
        torch = self.torch
        view = torch.as_tensor(np.asarray(T_CS, dtype=np.float32)[None], device='cuda')
        with torch.no_grad():
            rgb, alpha, _ = self.rasterization(**self.scene, viewmats=view, Ks=self.Ks_canvas, width=self.width_canvas,
                                               height=self.height_canvas, packed=False, near_plane=.01, far_plane=1e10,
                                               render_mode='RGB', sh_degree=self.info['sh_degree'],
                                               rasterize_mode='classic', eps2d=.3, camera_model='pinhole',
                                               backgrounds=self.bg)
        im = rgb[0].clamp(0, 1).cpu().numpy()
        return np.round(im * 255).astype(np.uint8)

    def rgb(self, T_CS):
        """uint8 RGB image (raw pixel coordinates when distorted) for one view matrix."""
        torch = self.torch
        view = torch.as_tensor(np.asarray(T_CS, dtype=np.float32)[None], device='cuda')
        common = dict(viewmats=view, Ks=self.Ks, width=self.width, height=self.height, packed=False,
                      near_plane=.01, far_plane=1e10, render_mode='RGB', sh_degree=self.info['sh_degree'],
                      rasterize_mode='classic', eps2d=.3, backgrounds=self.bg)
        with torch.no_grad():
            if self.native:
                rgb, alpha, _ = self.rasterization(**self.scene, camera_model='fisheye', radial_coeffs=self.radial,
                                                   with_ut=True, with_eval3d=True, **common)
            else:
                rgb, alpha, _ = self.rasterization(**self.scene, camera_model='pinhole', **common)
        im = rgb[0].clamp(0, 1).cpu().numpy()
        if not np.isfinite(im).all():
            raise ValueError('Nonfinite render')
        im = np.round(im * 255).astype(np.uint8)
        if self.distorted and not self.native:
            return C.distort_image(im)
        return im


def stage_render(args):
    from PIL import Image
    renderer = Renderer(distorted=not args.pinhole)
    print('render method:', renderer.method, flush=True)
    runs = args.run or C.RUNS
    for run in runs:
        folder, index, arrays = C.frame_store(run)
        out = folder / (args.subdir or ('render' if args.pinhole else 'render_distorted'))
        out.mkdir(exist_ok=True)
        T_CS = arrays['T_CS']
        if args.registration is not None:
            reg = C.load_registration(args.registration)
            T_GC = C.apply_registration(arrays['T_MC'], reg['R'], reg['t'], reg['lam'])
            _, T_CS = C.checkpoint_views(T_GC)
        t0 = time.time()
        done = 0
        for j, i in enumerate(index['accepted_indices']):
            path = out / f'{i:06d}.png'
            if path.exists() and not args.overwrite:
                continue
            Image.fromarray(renderer.rgb(T_CS[j])).save(path)
            done += 1
            if done % 250 == 0:
                print(f'{run}: {done} rendered ({time.time() - t0:.0f} s)', flush=True)
        print(f'{run}: rendered {done} new frames into {out} in {time.time() - t0:.0f} s', flush=True)


def render_depth(scene, info, view, rasterization, torch):
    """Expected z-depth, its mixture spread and alpha for one view (checkpoint
    units) on the padded pinhole canvas (see fixed_features_common.distortion_canvas)."""
    c = C.distortion_canvas()
    Ks = torch.tensor(c['K_canvas'], device='cuda', dtype=torch.float32)[None]
    common = dict(viewmats=view, Ks=Ks, width=c['width'], height=c['height'], packed=False,
                  near_plane=.01, far_plane=1e10, rasterize_mode='classic', eps2d=.3,
                  camera_model='pinhole')
    rgbd, alpha, _ = rasterization(**scene, **common, render_mode='RGB+ED',
                                   sh_degree=info['sh_degree'],
                                   backgrounds=torch.zeros((1, 3), device='cuda'))
    z = scene['means'] @ view[0, 2, :3] + view[0, 2, 3]
    second, _, _ = rasterization(**dict(scene, colors=z.square()[:, None]), **common,
                                 render_mode='RGB', sh_degree=None,
                                 backgrounds=torch.zeros((1, 1), device='cuda'))
    depth = rgbd[0, :, :, 3]
    a = alpha[0, :, :, 0]
    var = (second[0, :, :, 0] / a.clamp_min(1e-8) - depth.square()).clamp_min(0)
    return depth.cpu().numpy(), var.sqrt().cpu().numpy(), a.cpu().numpy()


def stage_depth(args):
    import torch
    from gsplat.rendering import rasterization
    assert torch.cuda.is_available()
    scene, info = load_scene('cuda')
    m = C.metres_per_checkpoint_unit()
    for spec in args.frames:
        run_short, idx = spec.split(':')
        run = next(r for r in C.RUNS if r.endswith(run_short))
        folder, index, arrays = C.frame_store(run)
        j = index['accepted_indices'].index(int(idx))
        (folder / 'depth').mkdir(exist_ok=True)
        view = torch.tensor(arrays['T_CS'][j:j + 1], device='cuda', dtype=torch.float32)
        with torch.no_grad():
            depth, std, alpha = render_depth(scene, info, view, rasterization, torch)
        c = C.distortion_canvas()
        np.savez_compressed(C.depth_path(run, int(idx)), depth_m=(depth * m).astype(np.float32),
                            depth_std_m=(std * m).astype(np.float32), alpha=alpha.astype(np.float32),
                            T_GC=arrays['T_GC'][j], K=c['K_canvas'], canvas_offset=np.array(c['offset']))
        print(f'depth {run}/{idx} written', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='stage', required=True)
    f = sub.add_parser('frames')
    f.add_argument('--registration', type=Path, default=C.START_REGISTRATION)
    f.add_argument('--overwrite', action='store_true')
    f.set_defaults(func=stage_frames)
    r = sub.add_parser('render')
    r.add_argument('--run', action='append')
    r.add_argument('--registration', type=Path, help='Render under another registration')
    r.add_argument('--subdir', help='Output subfolder name (default render_distorted, or render with --pinhole)')
    r.add_argument('--pinhole', action='store_true', help='Render the undistorted 320x240 frame instead')
    r.add_argument('--overwrite', action='store_true')
    r.set_defaults(func=stage_render)
    d = sub.add_parser('depth')
    d.add_argument('frames', nargs='+', help='run-suffix:index, e.g. 183828:602')
    d.set_defaults(func=stage_depth)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
