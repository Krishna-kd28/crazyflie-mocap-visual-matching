"""Numerical contracts, copied-fixture parity, and portable-path regressions."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
from scipy.spatial.transform import Rotation
import fixed_features_common as C
from baseline import clock_diagnostic, interpolate_pose
from image_similarity import compare


class GeometryTests(unittest.TestCase):
    def test_raw_lens_roundtrip_including_corners(self):
        u,v = np.meshgrid(np.linspace(0,319,33), np.linspace(0,239,25))
        pixels = np.c_[u.ravel(),v.ravel()]
        np.testing.assert_allclose(C.pinhole_to_raw(C.raw_to_pinhole(pixels)),pixels,atol=1e-7)

    def test_overscan_contains_every_sensor_ray(self):
        import cv2
        canvas = C.distortion_canvas()
        self.assertGreaterEqual(canvas['map_x'].min(),1)
        self.assertLess(canvas['map_x'].max(),canvas['width']-1)
        white = np.full((canvas['height'],canvas['width']),255,np.uint8)
        self.assertTrue(np.all(C.distort_image(white)==255))

    def test_triangulation_recovers_known_scene_point(self):
        T = np.repeat(np.eye(4)[None],3,axis=0)
        T[:,0,3] = [-.7,0,.8]
        point = np.array([.2,-.15,3.2])
        uv,depth = C.project(T,np.repeat(point[None],3,axis=0))
        found,errors,z = C.triangulate(T,uv)
        np.testing.assert_allclose(found,point,atol=1e-10)
        self.assertLess(errors.max(),1e-8)
        self.assertTrue(np.all(z>0))

    def test_registration_and_checkpoint_conventions(self):
        R = Rotation.from_euler('zyx',[.3,-.1,.2]).as_matrix()
        t,scale = np.array([1.,-.5,.2]),1.04
        j = C.registration_json(R,t,scale,'test',{})
        self.assertAlmostEqual(j['meters_per_scene_unit'],.85/scale)
        np.testing.assert_allclose(-scale*R@j['mocap_origin_m'],t)
        T = np.eye(4)[None]
        T[0,:3,3] = [.3,.4,-.1]
        G = C.apply_registration(T,R,t,scale)
        np.testing.assert_allclose(G[0,:3,3],scale*R@T[0,:3,3]+t)
        A = np.eye(4);A[:3,:3]=Rotation.from_euler('x',.4).as_matrix();A[:3,3]=[.1,-.1,.2]
        with patch.object(C,'scene_transforms',return_value=(A,1.7)):
            T_SC,T_CS = C.checkpoint_views(G)
        np.testing.assert_allclose(T_SC@T_CS,np.eye(4)[None],atol=1e-10)
        np.testing.assert_allclose(T_SC[0,:3,3],1.7*(A[:3,:3]@(G[0,:3,3]/.85)+A[:3,3]),atol=1e-10)

    def test_clock_recovery_with_arrival_outliers(self):
        x = np.arange(300)*.05
        y = 1.00015*x+.012
        y[::31] += .2
        host = 1700000000000000000+np.rint(y*1e9).astype(np.int64)
        fit = clock_diagnostic(x,host)
        self.assertLess(abs(fit['slope']-1.00015),1e-6)

    def test_pose_interpolation_rejects_outages(self):
        t = np.array([0,10,20,200,210])*1_000_000
        m = dict(host_ns=t,x=t/1e9,y=np.zeros(5),z=np.zeros(5),
                 qx=np.zeros(5),qy=np.zeros(5),qz=np.zeros(5),qw=np.ones(5))
        p,q,ok,reason,gaps=interpolate_pose(m,np.array([5,100,220])*1_000_000,50)
        self.assertEqual(ok.tolist(),[True,False,False])
        self.assertEqual(reason.tolist(),['accepted','mocap_gap','outside_mocap_support'])
        self.assertAlmostEqual(p[0,0],.005)
        self.assertTrue(np.isnan(p[1:]).all())

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch not installed')
    def test_per_frame_tensor_loss_matches_numpy(self):
        import torch
        from per_frame_registration import FrameSet,frame_loss_numpy
        rng = np.random.default_rng(40)
        X = np.c_[rng.normal(0,.4,(15,2)),rng.uniform(2,4,15)]
        T = np.eye(4)
        uv,_ = C.project(np.repeat(T[None],15,axis=0),X)
        obs=dict(X=X,u=uv,sigma=np.full(15,1.5),w=np.full(15,.5))
        prior=np.array([.04]*3+[.1]*3)
        params=np.r_[.01,-.005,.002,.01,.02,-.01]
        expected=frame_loss_numpy(T,obs,params,prior)
        frame=FrameSet(T[None],{0:obs},torch,'cpu')
        got=frame.loss(torch.tensor([0]),torch.tensor(params[None,None],dtype=torch.float32),prior).item()
        self.assertAlmostEqual(got,expected,places=4)
        self.assertLess(frame_loss_numpy(T,obs,np.zeros(6),prior),expected)


class AppearanceTests(unittest.TestCase):
    def test_saved_appearance_is_pixel_identical(self):
        from PIL import Image
        import project_config as PC
        from fit_appearance import apply
        base=ROOT/'examples/minimal_pair'
        model=json.loads((ROOT/'configs/appearance_selection.json').read_text())['selected']
        for row in json.loads((base/'manifest.json').read_text())['frames']:
            rgb=np.asarray(Image.open(base/row['render']).convert('RGB'))
            expected=np.asarray(Image.open(base/row['expected']).convert('L'))
            pose=np.load(base/row['pose'])['T_GC']
            result=np.rint(255*apply(rgb,model,row['run'],pose[:3,:3])).astype(np.uint8)
            np.testing.assert_array_equal(result,expected)

    def test_image_metric_identity_and_range(self):
        rng=np.random.default_rng(3);im=rng.uniform(0,1,(240,320)).astype(np.float32)
        result=compare(im,im)
        self.assertAlmostEqual(result['score'],100,places=5)
        self.assertEqual(result['rmse'],0)
        self.assertEqual(result['edge_f1'],1)
        with self.assertRaises(ValueError):compare(im*255,im)

    def test_empty_support_not_perfect_score(self):
        im=np.zeros((240,320),np.float32)
        self.assertIsNone(compare(im,im,np.zeros_like(im,bool)))


class PackagingTests(unittest.TestCase):
    def test_commands_resolve_outside_repo_working_directory(self):
        result=subprocess.run([sys.executable,str(ROOT/'pipeline.py'),'--dry-run','align','--track'],
                              cwd='/tmp',capture_output=True,text=True,check=True)
        self.assertIn(str(ROOT/'scripts/run_alignment.py'),result.stdout)

    def test_rejects_output_workspace_outside_package(self):
        config=json.loads((ROOT/'configs/project.json').read_text())
        config['workspace_dir']='/tmp/forbidden-cf-output'
        with tempfile.NamedTemporaryFile(mode='w',suffix='.json') as f:
            json.dump(config,f);f.flush()
            result=subprocess.run([sys.executable,str(ROOT/'pipeline.py'),'--config',f.name,'--dry-run','prepare'],
                                  capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('must be a subdirectory',result.stderr)

    def test_every_algorithm_parses_and_imports_from_package(self):
        import ast
        import importlib
        for p in (ROOT/'scripts').glob('*.py'):
            ast.parse(p.read_text())
            module=importlib.import_module(p.stem)
            self.assertTrue(Path(module.__file__).resolve().is_relative_to(ROOT))


if __name__=='__main__':unittest.main()
