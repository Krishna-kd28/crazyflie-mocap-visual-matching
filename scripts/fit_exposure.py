#!/usr/bin/env python3
"""Fit exposure and gain response from aligned recordings.

Use equal run influence, temporal partitions, and a leave-one-setting-out check.
"""

import project_config as PC
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares
from image_similarity import GRAY, compare, aggregate
from grayscale_appearance import R2, apply as baseline_apply, split
from capture_paths import C, ROOT, OUT, RUNS, EXPOSURE, write, event

D=OUT/'exposure'
E=np.array([EXPOSURE[r]['exposure_ms']*EXPOSURE[r]['digital_gain']/(8.33*1.5) for r in RUNS])


def apply(rgb, model, run):
    if model['family']=='baseline':return baseline_apply(rgb,model['baseline_model'])
    x=np.asarray(rgb,np.float32)/255 if rgb.dtype==np.uint8 else np.asarray(rgb,np.float32)
    g=x@GRAY;sig=model.get('sigma',0)
    if sig:g=cv2.GaussianBlur(g,(0,0),sig)
    if model['family']=='grayscale':return g
    ri=RUNS.index(run);p=model['params'][ri] if model['family']=='per_run' else model['params']
    a,b,gamma,v=p[:4]
    beta=0 if model['family'] in ['blind','per_run'] else 1 if model['family']=='metadata_linear' else p[4]
    return np.clip(b+a*E[ri]**beta*np.maximum(g,0)**gamma*np.exp(-v*R2),0,1).astype(np.float32)


def load(*, write_protocol=True):
    rows=[];hashes={}
    for run in RUNS:
        base=OUT/run/'fixed_features';pf=base/'per_frame'/run
        z=np.load(pf/'poses_per_frame.npz');obs=np.load(base/'observations_extended.npz')
        camera_times=np.load(OUT/'frames'/run/'raw_timing.npz')['time_s']
        ann=json.loads((OUT/run/'annotations.json').read_text())
        # Geometry-only foreground exclusion: central arena/gate and the floor.
        stable=set()
        for f in ann['features']:
            X=np.array(f['point_G']);abc=PC.CONFIG['floor_plane_abc'];floor=abc[0]*X[0]+abc[1]*X[1]+abc[2]
            if f['status']=='confirmed' and np.linalg.norm(X[:2])>PC.CONFIG.get('exclude_central_radius_m',0) and (not PC.CONFIG.get('exclude_floor',False) or abs(X[2]-floor)>.3):stable.add(f['id'])
        obs_by={}
        for i,f,uv in zip(obs['index'],obs['feature'],obs['pixel']):
            if f in stable:obs_by.setdefault(int(i),[]).append(uv)
        hashes[run]=dict(poses=C.sha256(pf/'poses_per_frame.npz'),observations=C.sha256(base/'observations_extended.npz'),annotations=C.sha256(OUT/run/'annotations.json'))
        for j in range(0,len(z['index']),5):
            i=int(z['index'][j]);real=cv2.imread(str(OUT/'frames'/run/'real_raw'/f'{i:06d}.png'),0)
            rgb=np.asarray(Image.open(pf/'render'/run/f'frame_{i:06d}.png').convert('RGB'))
            mask=np.zeros((240,320),np.uint8)
            if z['mode'][j]=='fitted' and z['n_features'][j]>=4:
                for xy in obs_by.get(i,[]):
                    x,y=np.rint(xy).astype(int)
                    if 12<=x<308 and 12<=y<228:cv2.circle(mask,(x,y),6,1,-1)
            rows.append(dict(run=run,index=i,time_s=float(camera_times[i]),split=split(float(camera_times[i])),
                real=real,rgb=rgb,mask=mask.astype(bool),supported=bool(mask.sum()>=180)))
    protocol=dict(source_hashes=hashes,lens_sha256=C.sha256(PC.path('lens')),
        exposure=EXPOSURE,relative_exposure_gain=dict(zip(RUNS,E.tolist())),
        exposure_source='supplied run notes; no per-frame exposure telemetry',sampling='every fifth accepted frame, fixed before appearance fitting',
        split='recorded camera-host elapsed time; 10 s blocks modulo 5: 0,1,2 train; 3 validation; 4 test; first and last second guarded',
        support='radius 6 px around geometrically confirmed background features; >=4 features at fitted pose; >=180 support pixels; exclude central 1 m and floor +/-0.3 m',
        training='256 unsaturated pixels per frame; max 72 frames per run; runs weighted equally; soft-L1 scale 0.05',
        models=['grayscale','frozen baseline','shared without settings','shared exposure*gain, beta=1','shared exposure*gain, fitted beta','separate fitted model per run'],
        selection='lowest equal-run validation support MAE; settings model selected among beta=1 and fitted beta, sigma in {0,0.8,1.4}; test never used for selection',
        transfer='leave each entire setting/run out; sigma fixed at 0.8 before checking transfer; compare blind vs fitted-beta family; refit using only other runs training pixels; score held run validation/test support',
        caveats=['3DGS RGB is not calibrated linear radiance; beta is an empirical response exponent',
            'setting and recording are confounded; four runs do not establish a physical camera calibration',
            'alignment uses all images; appearance hold-out freezes geometry and support but is not an independent end-to-end localization test',
            'automatically matched background is not manually verified semantic ground truth',
            'real images and segmentation weights are unchanged; no per-frame histogram normalization'])
    if write_protocol:write(D/'protocol.json',protocol)
    return rows


def fit(rows,family,sigma,excluded=None):
    rng=np.random.default_rng(25928);xx=[];yy=[];rr=[];ee=[];run_index=[];weights=[]
    for ri,run in enumerate(RUNS):
        if run==excluded:continue
        selected=[r for r in rows if r['run']==run and r['split']=='train' and r['supported']]
        if len(selected)>72:selected=[selected[i] for i in np.linspace(0,len(selected)-1,72,dtype=int)]
        if not selected:raise RuntimeError('No training support for '+run)
        added=0
        for r in selected:
            g=r['rgb'].astype(np.float32)@GRAY/255
            if sigma:g=cv2.GaussianBlur(g,(0,0),sigma)
            valid=r['mask']&(r['real']>5)&(r['real']<250)&(g>.01)
            pix=np.flatnonzero(valid)
            if len(pix)<32:continue
            pix=rng.choice(pix,256,replace=len(pix)<256)
            xx.append(g.ravel()[pix]);yy.append(r['real'].ravel()[pix]/255);rr.append(R2.ravel()[pix]);ee.append(np.full(256,E[ri]));run_index.append(np.full(256,ri));added+=1
        if not added:raise RuntimeError('No unsaturated training pixels for '+run)
        weights.extend([1/np.sqrt(added)]*added)
    x,y,r,e,ri=map(np.concatenate,(xx,yy,rr,ee,run_index));w=np.repeat(weights,256);w/=np.sqrt(np.mean(w*w))
    p0=[1,0,1,.5];lo=[.1,-.2,.25,0];hi=[4,.35,3,2.5]
    if family=='metadata_response':p0+=[1];lo+=[.05];hi+=[2.5]
    if family=='per_run':p0*=len(RUNS);lo*=len(RUNS);hi*=len(RUNS)
    def residual(p):
        if family=='per_run':q=p.reshape(len(RUNS),4)[ri];a,b,gamma,v=q.T
        else:a,b,gamma,v=p[:4]
        beta=0 if family in ['blind','per_run'] else 1 if family=='metadata_linear' else p[4]
        pred=np.clip(b+a*e**beta*x**gamma*np.exp(-v*r),0,1)
        return pred-y
    def weighted_soft_l1(squared_scaled_residual):
        # Weight outside the robust penalty: every run has equal influence
        # without changing the 0.05 intensity transition from run to run.
        root=np.sqrt(1+squared_scaled_residual)
        return np.array([2*(root-1),1/root,-.5/root**3])*w[None]**2
    result=least_squares(residual,p0,bounds=(lo,hi),loss=weighted_soft_l1,f_scale=.05,max_nfev=160,diff_step=.001)
    return dict(id=f'{family}_{sigma:g}',family=family,sigma=sigma,params=(result.x.reshape(len(RUNS),4) if family=='per_run' else result.x).tolist(),
        fit_cost=float(result.cost),nfev=result.nfev,excluded_run=excluded)


def score(model,rows,ids,scope='support'):
    def one(i):
        r=rows[i];return compare(r['real'].astype(np.float32)/255,apply(r['rgb'],model,r['run']),r['mask'] if scope=='support' else None)
    with ThreadPoolExecutor(max_workers=6) as pool:values=list(pool.map(one,ids))
    per_run={run:aggregate([v for i,v in zip(ids,values) if rows[i]['run']==run]) for run in RUNS}
    available=[v for v in per_run.values() if v['frames']]
    overall={key:float(np.mean([a[key] for a in available])) for key in ['mae','rmse','ssim','edge_f1','score']} if available else {}
    return dict(equal_run=overall,per_run=per_run,frames=len(ids)),values


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--export-only',action='store_true');args=ap.parse_args()
    cv2.setNumThreads(1);D.mkdir(exist_ok=True);rows=load()
    val=[i for i,r in enumerate(rows) if r['split']=='validation' and r['supported']]
    if not args.export_only:
        models=[dict(id='grayscale',family='grayscale',sigma=0),dict(id='frozen_baseline',family='baseline',baseline_model=json.loads((PC.path('baseline_appearance')).read_text()))]
        for family in ['blind','metadata_linear','metadata_response','per_run']:
            for sigma in [0,.8,1.4]:
                model=fit(rows,family,sigma);model['validation'],_=score(model,rows,val);models.append(model)
                write(D/'candidates.json',models);print(model['id'],model['params'],model['validation']['equal_run'],flush=True)
        selected=min([m for m in models if m['family'].startswith('metadata')],key=lambda m:m['validation']['equal_run']['mae'])
        blind=min([m for m in models if m['family']=='blind'],key=lambda m:m['validation']['equal_run']['mae'])
        perrun=min([m for m in models if m['family']=='per_run'],key=lambda m:m['validation']['equal_run']['mae'])
        fixed=min([m for m in models if m['family']=='metadata_linear'],key=lambda m:m['validation']['equal_run']['mae'])
        response=min([m for m in models if m['family']=='metadata_response'],key=lambda m:m['validation']['equal_run']['mae'])
        write(D/'selection.json',dict(selected=selected,blind=blind,per_run=perrun,metadata_linear=fixed,metadata_response=response))
    models=json.loads((D/'candidates.json').read_text());selection=json.loads((D/'selection.json').read_text())
    chosen=models[:2]+[selection[k] for k in ['blind','metadata_linear','metadata_response','per_run']]
    report=dict(selection=selection,models={},counts={run:{sp:sum(r['run']==run and r['split']==sp and r['supported'] for r in rows) for sp in ['train','validation','test']} for run in RUNS})
    metrics=[]
    for model in chosen:
        report['models'][model['id']]={}
        for sp in ['validation','test']:
            report['models'][model['id']][sp]={}
            for scope in ['support','full']:
                ids=[i for i,r in enumerate(rows) if r['split']==sp and (scope=='full' or r['supported'])]
                value,details=score(model,rows,ids,scope);report['models'][model['id']][sp][scope]=value
                for i,v in zip(ids,details):metrics.append(dict(model=model['id'],split=sp,scope=scope,run=rows[i]['run'],index=rows[i]['index'],**v))
    if not args.export_only:
        report['leave_one_setting_out']={}
        for run in RUNS:
            ids=[i for i,r in enumerate(rows) if r['run']==run and r['split'] in ['validation','test'] and r['supported']]
            result={}
            for key,family in [('blind','blind'),('selected','metadata_response')]:
                # Do not reuse the primary hyperparameters, which saw every run's
                # validation frames. This transfer comparison fixes blur a priori.
                model=fit(rows,family,.8,excluded=run)
                result[key]=dict(model=model,metrics=score(model,rows,ids)[0])
            report['leave_one_setting_out'][run]=result
            write(D/'transfer.json',report['leave_one_setting_out']);print('leave-one-setting-out',run,flush=True)
    else:report['leave_one_setting_out']=json.loads((D/'transfer.json').read_text())
    write(D/'summary.json',report)
    with (D/'per_frame_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=metrics[0].keys());w.writeheader();w.writerows(metrics)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for run in RUNS:
        test=[i for i,r in enumerate(rows) if r['run']==run and r['split']=='test' and r['supported']]
        if not test:continue
        pick=[test[j] for j in np.linspace(0,len(test)-1,4,dtype=int)]
        sheet=Image.new('RGB',(1280,len(pick)*292+60),'#f3f5f6');draw=ImageDraw.Draw(sheet)
        draw.text((8,8),f'{run} | exposure {EXPOSURE[run]["exposure_ms"]} ms; gain {EXPOSURE[run]["digital_gain"]}x | held-out samples',font=font,fill='#173943')
        for rownum,i in enumerate(pick):
            r=rows[i];ims=[r['real'],np.uint8(np.round(255*apply(r['rgb'],models[0],run))),np.uint8(np.round(255*apply(r['rgb'],selection['blind'],run))),np.uint8(np.round(255*apply(r['rgb'],selection['selected'],run)))]
            for col,(a,label) in enumerate(zip(ims,['Real','Grayscale sim','Fit without settings','Fit with exposure + gain'])):
                x=320*col;y=60+292*rownum;draw.text((x+5,y),label,font=font,fill='#173943');draw.text((x+5,y+25),f'Frame {r["index"]} | {r["time_s"]:.1f} s',font=small,fill='#526870');sheet.paste(Image.fromarray(a),(x,y+48))
        sheet.save(D/f'{run}_comparison.png')
    event('exposure_study_complete',selected=selection['selected'],heldout=report['models'][selection['selected']['id']]['test'])
    print(json.dumps({m['id']:report['models'][m['id']]['test'] for m in chosen}),flush=True)


if __name__=='__main__':main()
