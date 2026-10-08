#!/usr/bin/env python3
"""Fit and evaluate a shared appearance transform with all camera poses frozen.

Images: the saved per-frame run 183828 renders and raw grayscale camera frames.
Primary support: small, fixed disks around manually named room fixtures tracked
with conservative LK checks. Changed floor mats, movable fan and whiteboard are
excluded by feature identity before fitting, not by their photometric errors.
Fit / select / audit use disjoint 10 s time blocks with one-second guard bands.
"""
from __future__ import annotations

import project_config as PC
import argparse, csv, hashlib, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from scipy.optimize import least_squares
from PIL import Image, ImageDraw
from image_similarity import GRAY, compare, aggregate

ROOT=PC.WORKSPACE
RUN='run_20260730T183828'
OUT=ROOT/'output/appearance_183828_v1'
STABLE={'2','3','4','6','backwallupsquare','fire','backwallbotsquare',
        'ladderwallboardtoprightcorner','whitebotleftladderwall',
        'toprightcornerdoorladder','doorladderfire','doortopright','doortopelft'}
SIGMAS=[0,0.4,0.7,1.0,1.4,2.0,2.8,4.0]


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log(event, **data):
    row=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),event=event,**data)
    with (OUT/'journal.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(event, json.dumps(data)[:800],flush=True)


def split(t):
    phase=t%10
    if phase<1 or phase>9:return 'guard'
    block=int(t//10)%5
    return 'train' if block<3 else 'validation' if block==3 else 'test'


def data():
    pose=ROOT/f'output/fixed_features/per_frame/{RUN}/poses_per_frame.npz'
    z=np.load(pose); idx=z['index']; times=z['time_s']; pos={int(i):j for j,i in enumerate(idx)}
    tr=np.load(OUT/'manual_support_tracks.npz')
    mask=np.zeros((len(idx),240,320),np.uint8)
    obs=[]
    for f,r,i,xy,kind in zip(tr['feature'],tr['run'],tr['index'],tr['pixel'],tr['kind']):
        if str(f) not in STABLE or str(r)!=RUN or int(i) not in pos:continue
        j=pos[int(i)]
        if z['n_features'][j]<4 or z['mode'][j]!='fitted':continue
        x,y=np.round(xy).astype(int)
        if not (12<=x<308 and 12<=y<228):continue
        cv2.circle(mask[j],(x,y),12,1,-1)
        obs.append(dict(feature=str(f),index=int(i),xy=list(map(float,xy)),kind=str(kind)))
    def read(i):
        a=ROOT/f'output/fixed_features/frames/{RUN}/real_raw/{i:06d}.png'
        b=ROOT/f'output/fixed_features/per_frame/{RUN}/render/{RUN}/frame_{i:06d}.png'
        return np.asarray(Image.open(a).convert('L')), np.asarray(Image.open(b).convert('RGB'))
    with ThreadPoolExecutor(max_workers=8) as pool: pairs=list(pool.map(read,idx))
    real=np.stack([p[0] for p in pairs]); rgb=np.stack([p[1] for p in pairs])
    splits=np.array([split(t) for t in times])
    protocol=dict(run=RUN,created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        input_pose=str(pose),pose_sha256=sha(pose),lens_sha256=sha(PC.path('lens')),
        real_pixels_sha256=hashlib.sha256(real.tobytes()).hexdigest(),
        render_pixels_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        annotations_sha256=sha(ROOT/'research/fixed_features/annotations.json'),
        support_source=str(OUT/'manual_support_tracks.npz'),support_sha256=sha(OUT/'manual_support_tracks.npz'),
        support_radius_px=12,stable_features=sorted(STABLE),support_observations=len(obs),
        split='10 s blocks modulo 5: 0,1,2 train; 3 validation; 4 test; first/last second guarded',
        counts={s:dict(all_frames=int((splits==s).sum()),supported_frames=int(((splits==s)&(mask.sum((1,2))>0)).sum()))
                for s in ['train','validation','test','guard']},
        metric=dict(intensity_scale=0.10,edge_tolerance_px=2,edge_canny=[15,35],
                    analysis_blur_sigma=0.8,ssim_window=11,ssim_sigma=1.5,
                    formula='100 * (exp(-MAE/0.10) * (1+SSIM)/2 * edge_F1)**(1/3)',
                    scope='declared engineering score, not a validated human-perception probability'),
        selection='highest mean validation stable-support score; all components and whole-frame scores reported',
        render_source='saved native fisheye per-frame renders; poses, K and lens frozen',
        limitations=['same recording and reconstruction; no independent scene validation',
                    'support uses user-designated fixtures and tracking; no new semantic ground truth',
                    'appearance is not a physical camera calibration'])
    if (OUT/'protocol.json').exists():
        old=json.loads((OUT/'protocol.json').read_text())
        for k in ['pose_sha256','lens_sha256','annotations_sha256','support_sha256','real_pixels_sha256','render_pixels_sha256']:
            if old[k]!=protocol[k]:raise RuntimeError('Frozen input changed: '+k)
        protocol=old
    else:write(OUT/'protocol.json',protocol)
    np.savez_compressed(OUT/'support.npz',index=idx,mask=mask,split=splits,time_s=times)
    return idx,times,real,rgb,mask.astype(bool),splits,protocol


def radius_map():
    y,x=np.mgrid[:240,:320].astype(np.float32)
    K=np.asarray(PC.read_camera()['K'])
    return ((x-K[0,2])/200)**2+((y-K[1,2])/200)**2


R2=radius_map()


def apply(rgb, model):
    """A single frozen model uses only simulated RGB, never the paired real frame."""
    x=np.asarray(rgb,np.float32)/255 if rgb.dtype==np.uint8 else np.asarray(rgb,np.float32)
    p=model.get('params',{})
    weights=np.array(p.get('weights',GRAY),np.float32)
    g=x@weights
    sx,sy=model.get('sigma_x',0),model.get('sigma_y',0)
    if sx>0 or sy>0:
        g=cv2.GaussianBlur(g,(0,0),sigmaX=max(sx,0.01),sigmaY=max(sy,0.01),borderType=cv2.BORDER_REFLECT_101)
    g=np.maximum(g,0)**p.get('gamma',1)
    g=p.get('gain',1)*g+p.get('bias',0)
    if 'vignette' in p:g=g*np.exp(-p['vignette']*R2)
    return np.clip(g,0,1).astype(np.float32)


def unpack(family,p):
    if family=='grayscale':return {}
    out=dict(gain=float(p[0]),bias=float(p[1]))
    if family!='affine':out['gamma']=float(p[2])
    if family in {'vignette','spectral'}:out['vignette']=float(p[3])
    if family=='spectral':
        v=np.exp(np.r_[p[4:6],0]);out['weights']=(v/v.sum()).tolist()
    return out


def fit_one(real,rgb,mask,ids,family,sx,sy):
    if family=='grayscale':return dict(family=family,sigma_x=sx,sigma_y=sy,params={})
    # A deterministic equal-sized sample per frame prevents large masks dominating.
    rng=np.random.default_rng(928)
    xx=[];yy=[];rr=[]
    for j in ids:
        im=rgb[j].astype(np.float32)/255
        if sx>0 or sy>0:im=cv2.GaussianBlur(im,(0,0),sigmaX=max(sx,0.01),sigmaY=max(sy,0.01))
        pixels=np.flatnonzero(mask[j])
        pixels=rng.choice(pixels,512,replace=len(pixels)<512)
        xx.append(im.reshape(-1,3)[pixels]);yy.append(real[j].ravel()[pixels]/255);rr.append(R2.ravel()[pixels])
    x,y,r=np.concatenate(xx),np.concatenate(yy),np.concatenate(rr)
    p0=[1,0];lo=[0.2,-0.3];hi=[2.5,0.3]
    if family!='affine':p0+=[1];lo+=[0.35];hi+=[2.8]
    if family in {'vignette','spectral'}:p0+=[0.1];lo+=[0];hi+=[1.4]
    if family=='spectral':
        p0+=list(np.log(GRAY[:2]/GRAY[2]));lo+=[-4,-4];hi+=[4,4]
    def residual(p):
        q=unpack(family,p); g=x@np.array(q.get('weights',GRAY))
        g=q['gain']*np.maximum(g,0)**q.get('gamma',1)+q['bias']
        g*=np.exp(-q.get('vignette',0)*r)
        return np.clip(g,0,1)-y
    t0=time.time()
    fit=least_squares(residual,p0,bounds=(lo,hi),loss='soft_l1',f_scale=0.05,
                      max_nfev=150,ftol=1e-6,xtol=1e-6,gtol=1e-6,diff_step=0.001)
    return dict(family=family,sigma_x=sx,sigma_y=sy,params=unpack(family,fit.x),
                train_robust_cost=float(fit.cost),nfev=fit.nfev,seconds=time.time()-t0)


def score_model(model,real,rgb,mask,ids,whole=False):
    def one(j):return compare(real[j].astype(np.float32)/255,apply(rgb[j],model),None if whole else mask[j])
    with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(one,ids))
    return aggregate(rows),rows


def sheet(path,ids,idx,real,rgb,mask,model):
    titles=['Real camera','Grayscale baseline','Fitted appearance','Support / absolute error']
    im=Image.new('RGB',(4*320,len(ids)*264),'white');dr=ImageDraw.Draw(im)
    for n,j in enumerate(ids):
        g=rgb[j]@GRAY/255;f=apply(rgb[j],model);a=real[j]/255
        err=cv2.applyColorMap(np.minimum(np.abs(a-f)/0.3*255,255).astype(np.uint8),cv2.COLORMAP_INFERNO)[...,::-1]
        err[cv2.morphologyEx(mask[j].astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0]=(0,255,180)
        imgs=[np.repeat(real[j][...,None],3,2),np.repeat(np.uint8(np.round(g*255))[...,None],3,2),np.repeat(np.uint8(np.round(f*255))[...,None],3,2),err]
        for k,x in enumerate(imgs):
            dr.text((k*320+5,n*264+5),f'{titles[k]} | frame {idx[j]}',fill='black');im.paste(Image.fromarray(x),(k*320,n*264+24))
    im.save(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--export-only',action='store_true');args=ap.parse_args()
    cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
    idx,times,real,rgb,mask,splits,protocol=data()
    supported=mask.sum((1,2))>=200
    train=np.flatnonzero((splits=='train')&supported)
    if len(train)>96:train=train[np.linspace(0,len(train)-1,96).astype(int)]
    val=np.flatnonzero((splits=='validation')&supported)
    if min(len(train),len(val))<5:raise RuntimeError('Insufficient support in the frozen splits')
    log('appearance_protocol',counts=protocol['counts'],fit_frames=len(train),validation_frames=len(val))
    if not args.export_only:
        candidates=[]
        specs=[('grayscale',0,0)]
        for family in ['affine','gamma','vignette','spectral']:
            for sigma in SIGMAS:specs.append((family,sigma,sigma))
        # Directional PSF candidates: limited grid, compared on the same validation support.
        specs += [('vignette',a,b) for a,b in [(0.7,1.4),(1.4,0.7),(1,2.8),(2.8,1)]]
        for family,sx,sy in specs:
            m=fit_one(real,rgb,mask,train,family,sx,sy)
            m['id']=f'{family}_{sx:g}_{sy:g}'
            m['validation'],_=score_model(m,real,rgb,mask,val)
            candidates.append(m);write(OUT/'candidates.json',candidates)
            log('appearance_candidate',id=m['id'],params=m['params'],validation=m['validation'])
        best=max(candidates,key=lambda m:m['validation']['score'])
        write(OUT/'selected_model.json',best)
        log('appearance_selection',id=best['id'],rule=protocol['selection'])
    candidates=json.loads((OUT/'candidates.json').read_text());best=json.loads((OUT/'selected_model.json').read_text());base=candidates[0]
    summary=dict(protocol=protocol,selected=best,baseline=base,models={},count=len(candidates))
    rows=[]
    for m in [base,best]:
        summary['models'][m['id']]={}
        for s in ['train','validation','test','guard','all']:
            si=np.arange(len(idx)) if s=='all' else np.flatnonzero(splits==s)
            ids=si[supported[si]]
            stable,rs=score_model(m,real,rgb,mask,ids)
            full,rf=score_model(m,real,rgb,mask,si,whole=True)
            summary['models'][m['id']][s]=dict(stable=stable,full=full)
            if s!='all':
                for j,r in zip(ids,rs):rows.append(dict(model=m['id'],scope='stable',split=s,index=int(idx[j]),**r))
                for j,r in zip(si,rf):rows.append(dict(model=m['id'],scope='full',split=s,index=int(idx[j]),**r))
    write(OUT/'summary.json',summary)
    with (OUT/'per_frame_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    test=np.flatnonzero((splits=='test')&supported)
    pick=test[np.linspace(0,len(test)-1,min(8,len(test))).astype(int)]
    sheet(OUT/'held_out_examples.jpg',pick,idx,real,rgb,mask,best)
    # Predeclared stress examples include views with changed content and poor initial coverage.
    pick=np.linspace(0,len(idx)-1,8).astype(int)
    sheet(OUT/'whole_run_examples.jpg',pick,idx,real,rgb,mask,best)
    renderdir=OUT/'corrected_renders';renderdir.mkdir(exist_ok=True)
    for j,i in enumerate(idx):cv2.imwrite(str(renderdir/f'{i:06d}.png'),np.uint8(np.round(255*apply(rgb[j],best))))
    # Metric sanity/sensitivity: identity and controlled disturbances of real images.
    sanity={}
    rng=np.random.default_rng(928)
    for name in ['identity','gain_0.7','offset_0.15','blur_2','blur_5','shift_4','noise_0.10','constant','black']:
        rr=[]
        for j in pick:
            a=real[j].astype(np.float32)/255;b=a.copy()
            if name=='gain_0.7':b*=0.7
            elif name=='offset_0.15':b=np.clip(b+0.15,0,1)
            elif name.startswith('blur_'):b=cv2.GaussianBlur(b,(0,0),float(name.split('_')[1]))
            elif name=='shift_4':b=cv2.warpAffine(b,np.array([[1,0,4],[0,1,0]],np.float32),(320,240),borderMode=cv2.BORDER_REFLECT_101)
            elif name=='noise_0.10':b=np.clip(b+rng.normal(0,0.1,b.shape),0,1).astype(np.float32)
            elif name=='constant':b[:]=a.mean()
            elif name=='black':b[:]=0
            rr.append(compare(a,b))
        sanity[name]=aggregate(rr)
    write(OUT/'metric_sanity.json',sanity)
    log('appearance_complete',selected=best['id'],held_out=summary['models'][best['id']]['test'])


if __name__=='__main__':main()
