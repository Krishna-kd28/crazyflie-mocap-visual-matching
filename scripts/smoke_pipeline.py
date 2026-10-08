"""Real-fixture parity check; optional fresh rendering and matching."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import project_config as PC
import fixed_features_common as C
from fit_appearance import apply
from image_similarity import GRAY, compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--match', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    base = PC.CODE_ROOT/'examples/minimal_pair'
    rows = json.loads((base/'manifest.json').read_text())['frames']
    selected = json.loads(PC.path('appearance_model').read_text())['selected']
    out = PC.WORKSPACE/'smoke'
    out.mkdir(parents=True, exist_ok=True)
    result = dict(kind='packaging parity on three saved research examples, not a new accuracy study', frames=[])
    sheet = Image.new('RGB', (1280, len(rows)*274), 'white')
    draw = ImageDraw.Draw(sheet)
    for n, row in enumerate(rows):
        real = np.asarray(Image.open(base/row['real']).convert('L'))
        rgb = np.asarray(Image.open(base/row['render']).convert('RGB'))
        expected = np.asarray(Image.open(base/row['expected']).convert('L'))
        pose = np.load(base/row['pose'])
        fit = np.rint(255*apply(rgb, selected, row['run'], pose['T_GC'][:3,:3])).astype(np.uint8)
        difference = np.abs(fit.astype(int)-expected.astype(int))
        if np.max(difference) > 0:
            raise AssertionError(f'Appearance changed for {row["run"]}: max pixel difference {difference.max()}')
        Image.fromarray(fit).save(out/f'{row["run"]}_fitted.png')
        panels = [real, np.rint(rgb@GRAY).astype(np.uint8), fit, cv2.addWeighted(real,.5,fit,.5,0)]
        for col, (label, panel) in enumerate(zip(['Real', 'Grayscale sim', 'Fitted sim', 'Overlay'], panels)):
            draw.text((col*320+7,n*274+4), f'{label} | {row["run"][-6:]} frame {row["index"]}', fill='black')
            sheet.paste(Image.fromarray(panel).convert('RGB'), (col*320,n*274+30))
        metrics = {'run':row['run'], 'index':row['index'], 'max_parity_difference_255':int(difference.max()),
                   'before':compare(real/255., rgb.astype(np.float32)@GRAY/255.), 'after':compare(real/255.,fit/255.)}
        if n == 0 and args.render:
            from prepare_fixed_feature_frames import Renderer
            renderer = Renderer()
            _, views = C.checkpoint_views(pose['T_GC'][None])
            fresh = renderer.rgb(views[0])
            Image.fromarray(fresh).save(out/'fresh_render.png')
            metrics['fresh_render_mae_255'] = float(np.abs(fresh.astype(float)-rgb).mean())
            metrics['render_method'] = renderer.method
            if metrics['fresh_render_mae_255'] > 1:
                raise AssertionError('Fresh render differs materially from saved scene/pose fixture')
        if n == 0 and args.match:
            import torch
            from auto_features import Matcher
            torch.set_num_threads(4)
            matcher = Matcher(1024, device='cuda' if torch.cuda.is_available() else 'cpu')
            pairs, scores = matcher.match(matcher.features(real), matcher.features(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)), cross=True)
            metrics['matching'] = dict(model=matcher.name, correspondences=len(pairs), median_confidence=float(np.median(scores)))
            if len(pairs) < 8:
                raise AssertionError('Insufficient fixture correspondences to exercise the matcher')
        result['frames'].append(metrics)
    sheet.save(out/'real_sim_fitted_overlay.png')
    result['status'] = 'passed'
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
