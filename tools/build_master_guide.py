#!/usr/bin/env python3
"""Rebuild the simplified master PDF entirely from packaged source images."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'docs/master_assets'
TMP = ROOT/'workspace/pdf_build'
GRAY = np.array([.299,.587,.114])


def font(size):
    try: return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError: return ImageFont.load_default(size=size)


def gray(path):
    im=np.asarray(Image.open(path).convert('RGB'))
    return Image.fromarray(np.rint(im@GRAY).astype(np.uint8)).convert('RGB')


def row(images, labels, name, scale=2):
    w,h=320*scale,240*scale
    panel=Image.new('RGB',(len(images)*w,h+50),'white');draw=ImageDraw.Draw(panel)
    for j,(im,label) in enumerate(zip(images,labels)):
        draw.text((j*w+10,10),label,font=font(24),fill='#173943')
        panel.paste(im.resize((w,h),Image.Resampling.LANCZOS),(j*w,50))
    panel.save(ASSETS/name)


def main():
    ASSETS.mkdir(exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
    base=ROOT/'examples/minimal_pair'
    main=base/'run_20260925T122400'
    real=Image.open(main/'real.png').convert('RGB')
    final=Image.open(main/'expected_fitted.png').convert('RGB')
    aligned=gray(main/'render_rgb.png')
    row([real,aligned],['Real camera image','3DGS at the fitted viewpoint'],'overview.png')
    pair=json.loads((ASSETS/'feature_pairs.json').read_text())
    # Only the two wall/fixture candidates are drawn; the crate candidate is
    # omitted because a movable object is a poor illustration of a fixed feature.
    selected=pair['pairs'][:2]
    ims=[real,gray(ASSETS/pair['render'])]
    factor=3;w,h=320*factor,240*factor
    canvas=Image.new('RGB',(2*w+90,h+60),'white');draw=ImageDraw.Draw(canvas)
    offsets=[0,w+90]
    for j,(im,label) in enumerate(zip(ims,['Real image','Render used for matching'])):
        draw.text((offsets[j]+12,12),label,font=font(30),fill='#173943')
        canvas.paste(im.resize((w,h),Image.Resampling.LANCZOS),(offsets[j],60))
    colors=['#00e6ed','#ffb000']
    for n,(p,color) in enumerate(zip(selected,colors)):
        pts=[]
        for j,key in enumerate(['real','render']):
            x,y=np.array(p[key])*factor+[offsets[j],60];pts.append((x,y))
        draw.line(pts,fill=color,width=4)
        for x,y in pts:
            draw.ellipse((x-12,y-12,x+12,y+12),fill='black')
            draw.ellipse((x-7,y-7,x+7,y+7),fill=color)
            draw.text((x+14,y-26),chr(65+n),font=font(32),fill=color,stroke_width=3,stroke_fill='black')
    canvas.save(ASSETS/'feature_matches.png')
    global_render=gray(ASSETS/'feature_render_rgb.png')
    row([Image.blend(real,global_render,.5),Image.blend(real,aligned,.5)],
        ['Overlay at the starting pose','Overlay after per-frame pose fitting'],'pose_overlays.png')
    for suffix,run in [('dark','run_20260925T123004'),('bright','run_20260925T125125')]:
        d=base/run
        row([Image.open(d/'real.png').convert('RGB'),gray(d/'render_rgb.png'),Image.open(d/'expected_fitted.png').convert('RGB')],
            ['Real','Aligned grayscale sim','Appearance-corrected sim'],f'appearance_{suffix}.png')
    row([real,final,Image.blend(real,final,.5)],['Real','Fitted simulation','Overlay'],'final_example.png')
    cmd=['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(TMP),str(ROOT/'docs/master_pipeline_guide.tex')]
    for _ in range(2):
        result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        (TMP/'build_stdout.txt').write_text(result.stdout+result.stderr)
        if result.returncode:
            print(result.stdout[-6000:]);raise SystemExit(result.returncode)
    shutil.copy2(TMP/'master_pipeline_guide.pdf',ROOT/'docs/master_pipeline_guide.pdf')
    print(ROOT/'docs/master_pipeline_guide.pdf')


if __name__=='__main__':main()
