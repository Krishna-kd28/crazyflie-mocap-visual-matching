"""Apply a frozen response to aligned RGB; keep real pixels and poses unchanged."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw
import project_config as PC
import fixed_features_common as C
from image_similarity import GRAY, compare
from fit_appearance import apply


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', choices=PC.RUNS)
    p.add_argument('--model', type=Path, default=PC.path('appearance_model'))
    p.add_argument('--out', type=Path, default=PC.OUT/'appearance_export')
    p.add_argument('--no-video', action='store_true')
    args = p.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(PC.WORKSPACE):
        raise ValueError('Export destination must be below the configured workspace')
    chosen = json.loads(args.model.read_text())
    model = chosen.get('selected', chosen)
    cv2.setNumThreads(1)
    for run in [args.run] if args.run else PC.RUNS:
        folder = PC.OUT/run/'fixed_features/per_frame'/run
        poses = np.load(folder/'poses_per_frame.npz', allow_pickle=False)
        times = np.load(PC.OUT/'frames'/run/'raw_timing.npz')['time_s'][poses['index']]
        dest = out/run
        (dest/'renders').mkdir(parents=True, exist_ok=True)
        proc = None
        if not args.no_video:
            proc = subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','rgb24',
                '-video_size','960x270','-framerate','10','-i','-','-an','-c:v','libx264','-crf','18',
                '-pix_fmt','yuv420p',str(dest/'real_sim_overlay.mp4')], stdin=subprocess.PIPE)
        records, index_rows = [], []
        tick = float(times[0])
        try:
            for j, idx in enumerate(poses['index']):
                real = np.asarray(Image.open(PC.OUT/'frames'/run/'real_raw'/f'{idx:06d}.png').convert('L'))
                rgb = np.asarray(Image.open(folder/'render'/run/f'frame_{idx:06d}.png').convert('RGB'))
                # Unknown run IDs have no inherited gain; exposure metadata is explicit.
                settings = PC.EXPOSURE[run]
                relative = settings['exposure_ms']*settings['digital_gain']/(8.33*1.5)
                fitted = np.rint(255*apply(rgb,model,run,poses['T_GC'][j,:3,:3],relative_exposure=relative)).astype(np.uint8)
                Image.fromarray(fitted).save(dest/'renders'/f'{idx:06d}.png')
                for name, pred in [('aligned_only',rgb.astype(np.float32)@GRAY/255), ('appearance',fitted/255.)]:
                    records.append(dict(run=run,index=int(idx),camera_time_s=float(times[j]),method=name,**compare(real/255.,pred)))
                if proc:
                    canvas = Image.new('RGB',(960,270),'white'); draw = ImageDraw.Draw(canvas)
                    for col, (title, arr) in enumerate(zip(['Real','Fitted sim','Overlay'],[real,fitted,cv2.addWeighted(real,.5,fitted,.5,0)])):
                        title += f' | {times[j]:.1f}s | {idx}' if col==0 else ''
                        draw.text((col*320+6,8),title,fill='black');canvas.paste(Image.fromarray(arr).convert('RGB'),(col*320,30))
                    stop = times[j+1] if j+1<len(times) else times[j]+.1
                    buf = canvas.tobytes()
                    while tick < stop-1e-8:
                        proc.stdin.write(buf)
                        index_rows.append(dict(encoded_index=len(index_rows),source_index=int(idx),camera_time_s=float(tick)))
                        tick += .1
        finally:
            if proc:
                proc.stdin.close()
                if proc.wait(): raise RuntimeError('Video encoding failed')
        for name, rows in [('frame_metrics.csv',records),('video_frame_index.csv',index_rows)]:
            if rows:
                with (dest/name).open('w') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        np.savez_compressed(dest/'frames.npz',index=poses['index'],frame_id=poses['frame_id'],camera_time_s=times,pose_mode=poses['mode'])
        summary=dict(run=run,source_frames=len(times),encoded_frames=len(index_rows),model_sha256=C.sha256(args.model),
            pose_sha256=C.sha256(folder/'poses_per_frame.npz'),metrics='descriptive same-recording scores; no fitting in this export',
            time_basis='camera host time; hold previous usable frame across excluded intervals',model=model)
        (dest/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(run,'exported to',dest,flush=True)


if __name__ == '__main__':
    main()
