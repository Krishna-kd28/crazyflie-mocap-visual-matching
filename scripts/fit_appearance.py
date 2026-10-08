#!/usr/bin/env python3
"""Fit a shared simulated-camera response using aligned recordings.

New frames in old guard gaps are scored only after model selection is frozen.
This remains a same-recording appearance check, conditional on fitted poses.
"""

import project_config as PC
import argparse
import copy
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares

from capture_paths import C, ROOT, OUT, RUNS, EXPOSURE, write, event
from grayscale_appearance import R2, split
from fit_exposure import E, apply as old_apply
from appearance_metrics import measure, average
from image_similarity import GRAY

D=OUT/'appearance_v2'
REGIONS=np.digitize(R2,[.25,.81])
_RAYS=None


def camera_rays():
    global _RAYS
    if _RAYS is None:
        yy,xx=np.mgrid[:240,:320]
        uv=C.raw_to_pinhole(np.c_[xx.ravel(),yy.ravel()])
        rays=np.c_[(uv[:,0]-C.K[0,2])/C.K[0,0],(uv[:,1]-C.K[1,2])/C.K[1,1],np.ones(len(uv))]
        _RAYS=rays/np.linalg.norm(rays,axis=1,keepdims=True)
    return _RAYS


def direction_basis(rotation,rays=None):
    rays=camera_rays() if rays is None else rays
    x,y,z=(rays@rotation.T).T
    return np.stack([x,y,z,x*y,x*z,y*z,x*x-y*y,3*z*z-1],axis=-1)


def camera_response_features(rgb,rotation,include_brightness=True):
    """Whole-camera predictors; all come from simulation, never paired real pixels."""
    forward=direction_basis(rotation,np.array([[0.,0.,1.]]))[0]
    if not include_brightness:return forward
    x=np.asarray(rgb,np.float64)/255 if rgb.dtype==np.uint8 else np.asarray(rgb,np.float64)
    g=x@GRAY
    bright=np.maximum(g-.75,0)
    center=np.exp(-R2/.125)
    return np.r_[forward,bright.mean()/.1,(bright*center).sum()/center.sum()/.1]


def freeze_protocol():
    path=D/'protocol.json'
    if path.exists():return json.loads(path.read_text())
    rows=[];hashes={}
    for run in RUNS:
        pf=OUT/run/'fixed_features/per_frame'/run
        z=np.load(pf/'poses_per_frame.npz');times=np.load(OUT/'frames'/run/'raw_timing.npz')['time_s'][z['index']]
        centers=np.arange(40,float(times.max())+1,50)
        hashes[run]={str(p.relative_to(ROOT)):C.sha256(p) for p in [pf/'poses_per_frame.npz',OUT/run/'annotations.json',OUT/run/'fixed_features/observations_extended.npz']}
        for j,(i,t) in enumerate(zip(z['index'],times)):
            distance=float(np.min(abs(t-centers)))
            if distance<=.65:
                assert split(t)=='guard'
                phase='fresh';window=int(centers[np.argmin(abs(t-centers))])
            elif j%5==0 and distance>=3 and split(t)!='guard':
                phase='train' if split(t)=='train' else 'development';window=int(t//10)*10
            else:continue
            rows.append(dict(run=run,index=int(i),time_s=float(t),phase=phase,window=window,
                             fitted=bool(z['mode'][j]=='fitted' and z['n_features'][j]>=4)))
    p=dict(frames=rows,source_hashes=hashes,exposure=EXPOSURE,
        lens_sha256=C.sha256(PC.path('lens')),
        prior_selection_sha256=C.sha256(OUT/'exposure/selection.json'),
        split='Training uses prior train blocks. Development combines already inspected validation/test blocks. Fresh evaluation uses old guard gaps at 40+50n seconds +/-0.65 s; remove development/training samples within 3 s of each center.',
        independence='Fresh frames were excluded from prior appearance objectives and metrics, but nearby views and entire trajectories have been seen. Frozen geometry used all images. This is not independent end-to-end localization or new-scene testing.',
        tone='Shared exposure-conditioned gamma curve; black offset bounded to +/-0.02; radial quadratic/quartic attenuation. Compare fixed vs learned exposure exponent, plus the prior model family refit with broader support.',
        objective='Equal run/frame influence; robust fixed-feature pixels plus spatially uniform 40x40 tile means, signed radial-region means, and dark quantiles. Full-image supervision includes changed objects; robust tile loss limits their influence without claiming semantic ground truth.',
        selection='Development only. J=full MAE + 0.5*support MAE + 0.5*mean absolute radial bias/255 + 0.05*(1-full SSIM) + 0.10*(1-full edge F1) + 0.05*(1-support edge F1). All terms and per-run regressions retained. Engineering weights declared before scoring.',
        blur='Tone fitted with sigma=0.6; evaluate sigma 0,0.4,0.6,0.8 independently. No added noise or sharpening, real-frame histogram matching, pose changes, or per-frame learned parameters.',
        old_results='Preserved. The original selected model is a fixed baseline; it had access to nearby training/selection images under the old split.')
    write(path,p);event('appearance_v2_protocol_frozen',fresh_frames=sum(r['phase']=='fresh' for r in rows),path=str(path.relative_to(ROOT)))
    return p


def load_rows(protocol,phases):
    masks={};rotations={}
    for run in RUNS:
        ann=json.loads((OUT/run/'annotations.json').read_text());stable=set()
        for f in ann['features']:
            X=np.array(f['point_G']);abc=PC.CONFIG['floor_plane_abc'];floor=abc[0]*X[0]+abc[1]*X[1]+abc[2]
            if f['status']=='confirmed' and np.linalg.norm(X[:2])>PC.CONFIG.get('exclude_central_radius_m',0) and (not PC.CONFIG.get('exclude_floor',False) or abs(X[2]-floor)>.3):stable.add(f['id'])
        z=np.load(OUT/run/'fixed_features/observations_extended.npz');obs={}
        for i,f,xy in zip(z['index'],z['feature'],z['pixel']):
            if f in stable:obs.setdefault(int(i),[]).append(xy)
        masks[run]=obs
        poses=np.load(OUT/run/'fixed_features/per_frame'/run/'poses_per_frame.npz')
        rotations[run]={int(i):T[:3,:3] for i,T in zip(poses['index'],poses['T_GC'])}
    def one(meta):
        r=dict(meta);run=r['run'];i=r['index']
        r['R_GC']=rotations[run][i]
        r['real']=cv2.imread(str(OUT/'frames'/run/'real_raw'/f'{i:06d}.png'),0)
        r['rgb']=np.asarray(Image.open(OUT/run/'fixed_features/per_frame'/run/'render'/run/f'frame_{i:06d}.png').convert('RGB'))
        mask=np.zeros((240,320),np.uint8)
        if r['fitted']:
            for xy in masks[run].get(i,[]):
                x,y=np.rint(xy).astype(int)
                if 12<=x<308 and 12<=y<228:cv2.circle(mask,(x,y),6,1,-1)
        r['mask']=mask.astype(bool);r['supported']=bool(mask.sum()>=180)
        return r
    with ThreadPoolExecutor(max_workers=8) as pool:return list(pool.map(one,[r for r in protocol['frames'] if r['phase'] in phases]))


def unpack(model):
    p=np.asarray(model['params'],float)
    # a, black, gamma, v1, v2, beta
    return p


def transform(x,radius,e,p,loglight=0):
    a,b,gamma,v1,v2,beta=p[:6]
    return np.clip(b+a*e**beta*np.maximum(x,1e-7)**gamma*np.exp(-v1*radius-v2*radius**2+loglight),0,1)


def apply(rgb,model,run=None,rotation=None,*,relative_exposure=None):
    if model.get('family') not in ['spatial_response','lighting_response']:return old_apply(rgb,model,run)
    x=np.asarray(rgb,np.float32)/255 if rgb.dtype==np.uint8 else np.asarray(rgb,np.float32)
    g=x@GRAY;sigma=model['sigma']
    if sigma:g=cv2.GaussianBlur(g,(0,0),sigma)
    p=unpack(model);light=0
    if model['family']=='lighting_response':
        if rotation is None:raise ValueError('Directional response requires the simulated camera-to-scene rotation')
        light=(direction_basis(rotation)@p[6:]).reshape(240,320)
    if 'camera_gain_coeffs' in model:
        light+=camera_response_features(rgb,rotation,model['camera_brightness_features'])@np.array(model['camera_gain_coeffs'])
    exposure=E[RUNS.index(run)] if relative_exposure is None else float(relative_exposure)
    if not np.isfinite(exposure) or exposure<=0:raise ValueError('Relative exposure must be finite and positive')
    exposure*=model.get('run_gain',{}).get(run,1.)
    return transform(g,R2,exposure,p,light).astype(np.float32)


def calibrate_run_gain(rows,model):
    """Small constant capture-level correction, learned on training frames only."""
    data=fit_data(rows);p=unpack(model);runs=np.array([d['ri'] for d in data]);n=len(data)
    x=np.stack([d['x'] for d in data]);rr=np.stack([d['radius'] for d in data]);e=np.array([d['e'] for d in data])
    y=np.stack([d['y'] for d in data]);region=np.stack([d['region'] for d in data]);ry=np.stack([d['region_y'] for d in data])
    fx=np.stack([d['feature_x'] for d in data]);fy=np.stack([d['feature_y'] for d in data]);fr=np.stack([d['feature_r'] for d in data])
    dy=np.array([d['dark_y'] for d in data]);counts=np.bincount(runs,minlength=len(RUNS))
    wf=np.array([1/np.sqrt(counts[r]) for r in runs]);wf/=np.sqrt(np.mean(wf**2))
    light=np.stack([d['directions'] for d in data])@p[6:];flight=np.stack([d['feature_directions'] for d in data])@p[6:]
    def soft(v):return np.sign(v)*np.sqrt(2*.07**2*(np.sqrt(1+(v/.07)**2)-1))
    def residual(delta):
        ee=e*np.exp(delta[runs]);pred=transform(x,rr,ee[:,None,None],p,light);fp=transform(fx,fr,ee[:,None],p,flight)
        pr=np.stack([(pred*(region==j)).sum((1,2))/(region==j).sum((1,2)) for j in range(3)],axis=1)
        return np.r_[(soft(fp-fy)*wf[:,None]/np.sqrt(128)).ravel(),
                     (soft(pred.mean(2)-y)*wf[:,None]/np.sqrt(48)).ravel(),
                     ((pr-ry)*wf[:,None]/np.sqrt(3)).ravel(),
                     (np.quantile(pred.reshape(n,-1),.03,axis=1)-dy)*wf*.5,
                     np.sqrt(n/len(RUNS))*.1*delta]
    result=least_squares(residual,np.zeros(len(RUNS)),bounds=(-.2,.2),max_nfev=50,diff_step=.001)
    out=copy.deepcopy(model);out.pop('development',None);out['id']=model['id']+'_run_gain'
    out['run_gain']=dict(zip(RUNS,np.exp(result.x).tolist()))
    out['run_gain_calibration']=dict(log_delta=result.x.tolist(),bound=.2,ridge=.1,training_frames=n,nfev=result.nfev,
        limitation='A constant empirical correction for each known recording. Use the shared model for a new recording until separately calibrated; these are not measured hardware gains.')
    return out


def fit_data(rows):
    """Deterministic quadrature per tile; identical samples for every candidate."""
    rng=np.random.default_rng(25929);result=[]
    # Forty-pixel cells cover the sensor; sixteen quadrature points per cell.
    cells=[]
    for y in range(0,240,40):
        for x in range(0,320,40):
            yy,xx=np.meshgrid(np.arange(y+5,y+40,10),np.arange(x+5,x+40,10),indexing='ij')
            cells.append((yy.ravel()*320+xx.ravel(),slice(y,y+40),slice(x,x+40)))
    ids=np.array([c[0] for c in cells]);rad=R2.ravel()[ids]
    for ri,run in enumerate(RUNS):
        selected=[r for r in rows if r['run']==run and r['phase']=='train' and r['supported']]
        if len(selected)>72:selected=[selected[i] for i in np.linspace(0,len(selected)-1,72,dtype=int)]
        for r in selected:
            gray=r['rgb'].astype(np.float64)@GRAY/255
            gray=cv2.GaussianBlur(gray,(0,0),.6)
            real=r['real'].astype(float)/255
            features=np.flatnonzero(r['mask']&(r['real']>3)&(r['real']<253))
            if len(features)<32:continue
            pix=rng.choice(features,128,replace=len(features)<128)
            target=np.array([real[c[1],c[2]].mean() for c in cells])
            # Region membership belongs to quadrature pixels, never to a fitted residual.
            region=np.digitize(rad,[.25,.81])
            region_target=np.array([real[REGIONS==j].mean() for j in range(3)])
            result.append(dict(ri=ri,e=E[ri],x=gray.ravel()[ids],radius=rad,y=target,
                region=region,region_y=region_target,
                dark_y=float(np.quantile(real,.03)),
                feature_x=gray.ravel()[pix],feature_y=real.ravel()[pix],feature_r=R2.ravel()[pix],
                directions=direction_basis(r['R_GC'],camera_rays()[ids.ravel()]).reshape(48,16,8),
                feature_directions=direction_basis(r['R_GC'],camera_rays()[pix])))
    return result


def fit(rows,name,mode='spatial',beta_free=True,quartic=True,lighting_ridge=None):
    data=fit_data(rows);n=len(data)
    x=np.stack([d['x'] for d in data]);rr=np.stack([d['radius'] for d in data]);e=np.array([d['e'] for d in data])[:,None,None]
    y=np.stack([d['y'] for d in data]);region=np.stack([d['region'] for d in data]);ry=np.stack([d['region_y'] for d in data])
    dy=np.array([d['dark_y'] for d in data])
    fx=np.stack([d['feature_x'] for d in data]);fy=np.stack([d['feature_y'] for d in data]);fr=np.stack([d['feature_r'] for d in data])
    runs=np.array([d['ri'] for d in data]);counts=np.bincount(runs,minlength=len(RUNS))
    wf=np.array([1/np.sqrt(counts[r]) for r in runs]);wf/=np.sqrt(np.mean(wf**2))
    p0=np.array([1.25,0,1,1,.2,1.]);low=np.array([.1,-.02,.3,0,0,.25]);high=np.array([5,.02,3,5,8,2.])
    if mode=='old_form':p0[1]=.08;low[1]=-.2;high[1]=.35
    free=[0,1,2,3]+([4] if quartic else [])+([5] if beta_free else [])
    if not quartic:p0[4]=0
    if lighting_ridge is not None:
        seed=D/('fit_quartic_learned_beta.json' if beta_free else 'fit_quartic_fixed_beta.json')
        if seed.exists():p0=np.array(json.loads(seed.read_text())['params'])
        p0=np.r_[p0,np.zeros(8)];low=np.r_[low,np.full(8,-2.)];high=np.r_[high,np.full(8,2.)];free+=list(range(6,14))
        directions=np.stack([d['directions'] for d in data]);fd=np.stack([d['feature_directions'] for d in data])
    def expand(p):
        q=p0.copy();q[free]=p;return q
    # Residual blocks are normalized for comparable influence. Soft-L1 at each
    # unnormalized residual is implemented explicitly before least_squares.
    def robust_residual(p):
        # The normalized block residual is scaled back to its intensity units
        # before robustification, then receives its declared block weight.
        q=expand(p);light=0;flight=0
        if lighting_ridge is not None:light=directions@q[6:];flight=fd@q[6:]
        pred=transform(x,rr,e,q,light);fp=transform(fx,fr,e[:,:,0],q,flight)
        def soft(v):return np.sign(v)*np.sqrt(2*.07**2*(np.sqrt(1+(v/.07)**2)-1))
        parts=[(soft(fp-fy)*wf[:,None]/np.sqrt(128)).ravel()]
        if mode!='support_only':
            parts.append((soft(pred.mean(axis=2)-y)*wf[:,None]/np.sqrt(48)).ravel())
            pr=np.stack([(pred*(region==j)).sum((1,2))/(region==j).sum((1,2)) for j in range(3)],axis=1)
            parts.append(((pr-ry)*wf[:,None]/np.sqrt(3)).ravel())
            parts.append((np.quantile(pred.reshape(n,-1),.03,axis=1)-dy)*wf*.5)
        if lighting_ridge is not None:parts.append(np.sqrt(n)*lighting_ridge*q[6:])
        return np.concatenate(parts)
    result=least_squares(robust_residual,p0[free],bounds=(low[free],high[free]),max_nfev=100,diff_step=.001)
    return dict(id=name,family='spatial_response' if lighting_ridge is None else 'lighting_response',sigma=.6,params=expand(result.x).tolist(),fit_mode=mode,
        lighting_ridge=lighting_ridge,
        beta_free=beta_free,quartic=quartic,nfev=result.nfev,cost=float(result.cost),training_frames=n,
        converged=bool(result.success),message=result.message)


def score(model,rows,phase):
    ids=[i for i,r in enumerate(rows) if r['phase']==phase]
    def one(i):
        r=rows[i];pred=apply(r['rgb'],model,r['run'],r['R_GC']);result=[]
        for scope in ['full','support']:
            if scope=='support' and not r['supported']:continue
            value=measure(r['real'],pred,r['mask'] if scope=='support' else None)
            value['radial_absolute_bias_255']=float(np.mean([abs(value[k]) for k in ['center_bias_255','middle_bias_255','periphery_bias_255'] if value[k] is not None]))
            result.append(dict(model=model['id'],run=r['run'],index=r['index'],phase=phase,window=r['window'],scope=scope,**value))
        return result
    records=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for values in pool.map(one,ids):records.extend(values)
    result={}
    for scope in ['full','support']:
        per={}
        for run in RUNS:
            rr=[r for r in records if r['scope']==scope and r['run']==run];v=average(rr)
            v['radial_absolute_bias_255']=float(np.mean([r['radial_absolute_bias_255'] for r in rr])) if rr else None
            per[run]=v
        keys=[k for k in next(iter(per.values())) if k!='frames'];equal={}
        for k in keys:
            values=[v[k] for v in per.values() if v.get(k) is not None]
            equal[k]=float(np.mean(values)) if values else None
        result[scope]=dict(per_run=per,equal_run=equal,frames=sum(v['frames'] for v in per.values()))
    f=result['full']['equal_run'];s=result['support']['equal_run']
    result['objective']=f['mae']+.5*s['mae']+.5*f['radial_absolute_bias_255']/255+.05*(1-f['ssim'])+.10*(1-f['edge_f1'])+.05*(1-s['edge_f1'])
    return result,records


def save_records(name,records):
    with (D/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)


def make_sheet(rows,indices,old,new,path,title):
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    sheet=Image.new('RGB',(960,316*len(indices)+44),'#f3f5f6');draw=ImageDraw.Draw(sheet)
    draw.text((6,8),title,font=font,fill='#173943')
    for j,idx in enumerate(indices):
        r=rows[idx];y=44+j*316
        for col,(arr,label) in enumerate([(r['real'],'Real'),(np.uint8(np.round(255*apply(r['rgb'],old,r['run'],r['R_GC']))),'Previous fit'),(np.uint8(np.round(255*apply(r['rgb'],new,r['run'],r['R_GC']))),'Revised fit')]):
            x=320*col
            draw.text((x+5,y),label,font=font,fill='#173943')
            draw.text((x+5,y+24),f'{r["run"][-6:]} | frame {r["index"]} | {r["time_s"]:.1f} s',font=small,fill='#526870')
            draw.text((x+5,y+44),f'Mean {arr[5:-5,5:-5].mean():.1f}/255',font=small,fill='#526870')
            sheet.paste(Image.fromarray(arr),(x,y+70))
    sheet.save(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','lighting','run-gain','evaluate','sheets']);args=ap.parse_args()
    cv2.setNumThreads(1);D.mkdir(exist_ok=True);protocol=freeze_protocol()
    old=json.loads((OUT/'exposure/selection.json').read_text())['selected']
    if args.stage=='fit':
        if (D/'fresh_metrics.json').exists():raise RuntimeError('Fresh set was already evaluated. Start a new study for further selection.')
        rows=load_rows(protocol,{'train','development'});models=[]
        specs=[('bounded_support','support_only',True,True),('old_form_spatial','old_form',True,False),
               ('radial_fixed_beta','spatial',False,False),('radial_learned_beta','spatial',True,False),
               ('quartic_fixed_beta','spatial',False,True),('quartic_learned_beta','spatial',True,True)]
        for name,mode,beta,quartic in specs:
            path=D/f'fit_{name}.json'
            model=json.loads(path.read_text()) if path.exists() else fit(rows,name,mode,beta,quartic)
            write(path,model);print('fit',name,model['params'],flush=True)
            for sigma in [0,.4,.6,.8]:
                m=copy.deepcopy(model);m.update(id=f'{name}_blur_{sigma:g}',sigma=sigma)
                m['development'],_=score(m,rows,'development');models.append(m)
                write(D/'candidates.json',models)
                v=m['development'];print(m['id'],'J',round(v['objective'],5),'full',v['full']['equal_run']['mae'],'edges',v['full']['equal_run']['edge_f1'],'bias',v['full']['equal_run']['signed_bias_255'],flush=True)
        baseline,_=score(old,rows,'development')
        selected=min(models,key=lambda m:m['development']['objective'])
        write(D/'selection.json',dict(selected=selected,baseline=old,baseline_development=baseline,protocol_sha256=C.sha256(D/'protocol.json')))
        event('appearance_v2_selected',model=selected['id'],objective=selected['development']['objective'],fresh_images_read=False)
        # Only development images are shown until the parameters are frozen.
        pick=[]
        for run in RUNS:
            eligible=[i for i,r in enumerate(rows) if r['run']==run and r['phase']=='development']
            pick.extend(eligible[j] for j in np.linspace(0,len(eligible)-1,3,dtype=int))
        make_sheet(rows,pick,old,selected,D/'development_comparison.png','Development frames - before fresh evaluation')
        print('SELECTED',selected['id'],selected['params'],flush=True)
    elif args.stage=='lighting':
        if (D/'fresh_metrics.json').exists():raise RuntimeError('Fresh set was already evaluated. Do not tune further on it.')
        if not (D/'selection_stage1.json').exists():write(D/'selection_stage1.json',json.loads((D/'selection.json').read_text()))
        extension=dict(reason='Spatial-only candidates improve edges but retain viewpoint-dependent brightness error on development frames, especially toward windows.',
            basis='Viewing ray d=R_GC*normalized(K^-1*undistort(raw_pixel)); B(d)=[x,y,z,xy,xz,yz,x^2-y^2,3z^2-1]. Multiply the positive radiance term by exp(B(d)h) before clipping.',
            regularization='Eight coefficients bounded +/-2 with mean objective penalty ridge^2*||h||^2. Compare ridge 0.03,0.08,0.15 and fixed/learned exposure beta.',
            purpose='Empirical correction of room illumination shared across runs, using only simulation pose/rays at inference. Not a new physical light transport reconstruction.',
            selection='Same declared development objective and untouched 160 fresh frames; preserve stage-one candidate results.')
        write(D/'lighting_extension.json',extension);event('appearance_v2_lighting_extension',fresh_images_read=False)
        rows=load_rows(protocol,{'train','development'});models=json.loads((D/'candidates.json').read_text())
        models=[m for m in models if m['family']!='lighting_response']
        for beta in [False,True]:
            for ridge in [.03,.08,.15]:
                name=f'lighting_{"free" if beta else "fixed"}_beta_ridge_{ridge:g}';path=D/f'fit_{name}.json'
                model=json.loads(path.read_text()) if path.exists() else fit(rows,name,'spatial',beta,True,ridge)
                write(path,model);print('fit',name,model['params'],flush=True)
                for sigma in [0,.4,.6,.8]:
                    m=copy.deepcopy(model);m.update(id=f'{name}_blur_{sigma:g}',sigma=sigma)
                    m['development'],_=score(m,rows,'development');models.append(m);write(D/'candidates.json',models)
                    v=m['development'];print(m['id'],'J',round(v['objective'],5),'MAE',v['full']['equal_run']['mae'],'edges',v['full']['equal_run']['edge_f1'],'abs_bias',v['full']['equal_run']['absolute_frame_bias_255'],flush=True)
        previous=json.loads((D/'selection_stage1.json').read_text());selected=min(models,key=lambda m:m['development']['objective'])
        write(D/'selection.json',dict(selected=selected,baseline=old,baseline_development=previous['baseline_development'],protocol_sha256=C.sha256(D/'protocol.json'),lighting_extension_sha256=C.sha256(D/'lighting_extension.json')))
        pick=[]
        for run in RUNS:
            eligible=[i for i,r in enumerate(rows) if r['run']==run and r['phase']=='development']
            pick.extend(eligible[j] for j in np.linspace(0,len(eligible)-1,3,dtype=int))
        make_sheet(rows,pick,old,selected,D/'development_lighting.png','Development frames - lighting correction selected without fresh frames')
        event('appearance_v2_selected_after_lighting',model=selected['id'],objective=selected['development']['objective'],fresh_images_read=False)
        print('SELECTED',selected['id'],selected['params'],flush=True)
    elif args.stage=='run-gain':
        if (D/'fresh_metrics.json').exists():raise RuntimeError('Fresh set already evaluated; no further selection.')
        path=D/'selection_shared.json'
        if not path.exists():write(path,json.loads((D/'selection.json').read_text()))
        previous=json.loads(path.read_text());model=previous['selected']
        extension=dict(reason='The shared lighting fit retains opposite mean bias in the first two recordings. Test a small constant residual exposure multiplier for each known capture.',
            fit='Only the four log gain corrections are fitted on original training blocks, bounded +/-0.2 with ridge 0.1. Shared parameters stay frozen. Evaluate by the unchanged development objective.',
            transfer='This optional correction is specific to known recordings. It is not an exposure-only calibration for an unseen run. The unadjusted shared model remains separately available.',fresh_images_read=False)
        write(D/'run_gain_extension.json',extension)
        rows=load_rows(protocol,{'train','development'});adjusted=calibrate_run_gain(rows,model)
        adjusted['development'],_=score(adjusted,rows,'development');write(D/'run_gain_candidate.json',adjusted)
        selected=min([model,adjusted],key=lambda m:m['development']['objective'])
        previous['selected']=selected;previous['shared']=model;previous['run_gain_extension_sha256']=C.sha256(D/'run_gain_extension.json')
        write(D/'selection.json',previous)
        print('RUN GAINS',adjusted['run_gain'],'J',adjusted['development']['objective'],'selected',selected['id'],flush=True)
        event('appearance_v2_known_run_calibration',run_gain=adjusted['run_gain'],selected=selected['id'],fresh_images_read=False)
    elif args.stage=='evaluate':
        selection=json.loads((D/'selection.json').read_text());rows=load_rows(protocol,{'fresh'})
        selected=selection['selected'];results={};records=[]
        comparisons=[('previous',old)]
        if 'shared' in selection:comparisons.append(('shared',selection['shared']))
        comparisons.append(('revised',selected))
        for name,model in comparisons:
            results[name],rr=score(model,rows,'fresh');records.extend(rr)
        write(D/'fresh_metrics.json',dict(selection_sha256=C.sha256(D/'selection.json'),protocol_sha256=C.sha256(D/'protocol.json'),results=results))
        save_records('fresh_frame_metrics.csv',records)
        for run in RUNS:
            eligible=[i for i,r in enumerate(rows) if r['run']==run]
            pick=[eligible[j] for j in np.linspace(0,len(eligible)-1,4,dtype=int)]
            make_sheet(rows,pick,old,selected,D/f'{run}_fresh.png','Previously excluded guard frames - frozen comparison')
        pick=[next(i for i,r in enumerate(rows) if r['run']==run) for run in RUNS]
        make_sheet(rows,pick,old,selected,D/'fresh_overview.png','First previously excluded guard frame from each run')
        event('appearance_v2_fresh_evaluated',frames=len(rows),results=results)
        print(json.dumps(results),flush=True)
    else:
        # Revisit the user's original examples after selection; clearly development.
        selection=json.loads((D/'selection.json').read_text())
        extra=copy.deepcopy(protocol)
        # These already-inspected original examples are outside the fresh gaps.
        targets=dict(zip(RUNS,[398,396,404,392]));extra['frames']=[]
        for run,index in targets.items():
            time=np.load(OUT/'frames'/run/'raw_timing.npz')['time_s'][index]
            extra['frames'].append(dict(run=run,index=index,time_s=float(time),phase='original',fitted=True,window=0))
        rows=load_rows(extra,{'original'})
        targets=dict(zip(RUNS,[398,396,404,392]));pick=[next(i for i,r in enumerate(rows) if r['run']==run and r['index']==index) for run,index in targets.items()]
        make_sheet(rows,pick,old,selection['selected'],D/'original_examples.png','Original examples revisited - not fresh test images')


if __name__=='__main__':main()
