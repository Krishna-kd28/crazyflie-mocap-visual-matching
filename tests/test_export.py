"""Exercise corrected-PNG/video export from a minimal accepted-pose dataset."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


class ExportIntegration(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'ffmpeg/ffprobe not installed')
    def test_export_preserves_pixels_and_frame_mapping(self):
        base=ROOT/'examples/minimal_pair'
        row=json.loads((base/'manifest.json').read_text())['frames'][0];run=row['run']
        with tempfile.TemporaryDirectory(prefix='workspace_test_',dir=ROOT) as directory:
            workspace=Path(directory)
            config=json.loads((ROOT/'configs/project.json').read_text())
            config.update(workspace_dir=str(workspace),runs=[run],exposure={run:config['exposure'][run]})
            cp=workspace/'config.json';cp.write_text(json.dumps(config))
            out=workspace/'output/captures'
            frames=out/'frames'/run;raw=frames/'real_raw';raw.mkdir(parents=True)
            pf=out/run/'fixed_features/per_frame'/run;renders=pf/'render'/run;renders.mkdir(parents=True)
            # Three source frames, deliberately irregular times to test playback holds.
            for index in [0,1,2]:
                shutil.copy2(base/row['real'],raw/f'{index:06d}.png')
                shutil.copy2(base/row['render'],renders/f'frame_{index:06d}.png')
            np.savez_compressed(frames/'raw_timing.npz',time_s=np.array([0.,.2,.5]))
            T=np.load(base/row['pose'])['T_GC']
            np.savez_compressed(pf/'poses_per_frame.npz',index=np.arange(3),frame_id=np.array([10,11,12]),
                                T_GC=np.repeat(T[None],3,axis=0),mode=np.array(['fitted','interpolated','fitted']))
            env=dict(os.environ,CF_MATCH_CONFIG=str(cp),PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable,str(ROOT/'pipeline.py'),'appearance-export'],cwd='/tmp',env=env,
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
            dest=out/'appearance_export'/run
            expected=np.asarray(Image.open(base/row['expected']))
            for index in [0,1,2]:np.testing.assert_array_equal(np.asarray(Image.open(dest/'renders'/f'{index:06d}.png')),expected)
            result=json.loads((dest/'manifest.json').read_text())
            self.assertEqual(result['source_frames'],3)
            self.assertEqual(result['encoded_frames'],6)
            import csv
            with (dest/'video_frame_index.csv').open() as stream:
                mapping=list(csv.DictReader(stream))
            self.assertEqual([int(x['source_index']) for x in mapping],[0,0,1,1,1,2])
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                'stream=width,height,nb_frames','-of','json',str(dest/'real_sim_overlay.mp4')]))['streams'][0]
            self.assertEqual((probe['width'],probe['height'],int(probe['nb_frames'])),(960,270,6))
