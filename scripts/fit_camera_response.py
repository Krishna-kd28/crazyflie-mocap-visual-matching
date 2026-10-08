#!/usr/bin/env python3
"""Test a view-level camera response after the user's bright-source observation.

V2's reserved check is now a development diagnostic. New exact frames in other
old guard gaps are reserved before fitting this added term. No causal claim of
automatic exposure versus flare is inferred from image intensities alone.
"""

import project_config as PC
import argparse,copy,csv,json
import numpy as np
import cv2
from scipy.optimize import least_squares
from pathlib import Path
import fit_appearance as A
from capture_paths import OUT,ROOT,RUNS,C,write,event

D=OUT/'appearance_v3';PREV=OUT/'appearance_v2'


def protocol():
    path=D/'protocol.json'
    if path.exists():return json.loads(path.read_text())
    prior=json.loads((PREV/'protocol.json').read_text());p=copy.deepcopy(prior);rows=[]
    for run in RUNS:
        z=np.load(OUT/run/'fixed_features/per_frame'/run/'poses_per_frame.npz');t=np.load(OUT/'frames'/run/'raw_timing.npz')['time_s'][z['index']]
        centers=np.arange(50,float(t.max())+1,50)
        for j,(i,time) in enumerate(zip(z['index'],t)):
            dist=float(np.min(abs(time-centers)))
            if dist<=.65:phase='fresh';window=int(centers[np.argmin(abs(time-centers))]);assert A.split(time)=='guard'
            elif j%5==0 and dist>=3 and A.split(time)!='guard':phase='train' if A.split(time)=='train' else 'development';window=int(time//10)*10
            else:continue
            rows.append(dict(run=run,index=int(i),time_s=float(time),phase=phase,window=window,fitted=bool(z['mode'][j]=='fitted' and z['n_features'][j]>=4)))
    p.update(frames=rows,parent_selection_sha256=C.sha256(PREV/'selection.json'),
        reason='User reports excessive brightening toward windows/lights and darkening away. Separate a frame-level camera/view response from pixel-ray directional correction.',
        split='Added response fitted using prior training blocks, with 3 s exclusions around new guard centers 50+50n seconds. Old validation/test blocks are development. New exact guard-gap frames +/-0.65 s have not been scored before; V2 fresh windows are no longer an independent test for development decisions.',
        inherited='V2 tone/directional parameters are frozen and had access to nearby training views under the earlier split. New frames are newly scored, not an independent recording or a fully new end-to-end split.',
        camera_response='Multiply the radiance term by exp(c^T F). F contains quadratic basis of the central viewing ray and optionally two simulated bright-content indicators above 0.75: whole image and a central Gaussian weighting. Coefficients bounded +/-0.6; ridge 0.05 or 0.15. Tone, per-ray response and known-run gains stay fixed.',
        selection='Same development objective as V2. Compare the unchanged parent against four camera-response candidates. Do not reselect after the new check.',
        causal_limit='Brightness changes alone do not distinguish light transport, sensor clipping, bloom/flare, tone mapping or automatic exposure/gain. Supplied exposure/gain are run notes, not per-frame telemetry.')
    write(path,p);event('camera_response_protocol_frozen',fresh_frames=sum(r['phase']=='fresh' for r in rows),parent='appearance_v2',hypothesis=p['reason'])
    return p


def fit(rows,base,bright,ridge):
    data=A.fit_data(rows);by={}
    for run in RUNS:
        selected=[r for r in rows if r['run']==run and r['phase']=='train' and r['supported']]
        if len(selected)>72:selected=[selected[i] for i in np.linspace(0,len(selected)-1,72,dtype=int)]
        selected=[r for r in selected if np.sum(r['mask']&(r['real']>3)&(r['real']<253))>=32]
        by[run]=selected
    ordered=[r for run in RUNS for r in by[run]];assert len(ordered)==len(data)
    F=np.stack([A.camera_response_features(r['rgb'],r['R_GC'],bright) for r in ordered]);p=np.array(base['params']);n=len(data)
    runs=np.array([d['ri'] for d in data]);counts=np.bincount(runs,minlength=len(RUNS))
    wf=np.array([1/np.sqrt(counts[r]) for r in runs]);wf/=np.sqrt(np.mean(wf**2))
    x=np.stack([d['x'] for d in data]);rr=np.stack([d['radius'] for d in data]);e=np.array([d['e']*base.get('run_gain',{}).get(RUNS[d['ri']],1) for d in data])
    y=np.stack([d['y'] for d in data]);region=np.stack([d['region'] for d in data]);ry=np.stack([d['region_y'] for d in data]);dy=np.array([d['dark_y'] for d in data])
    fx=np.stack([d['feature_x'] for d in data]);fy=np.stack([d['feature_y'] for d in data]);fr=np.stack([d['feature_r'] for d in data])
    light=np.stack([d['directions'] for d in data])@p[6:];flight=np.stack([d['feature_directions'] for d in data])@p[6:]
    def soft(v):return np.sign(v)*np.sqrt(2*.07**2*(np.sqrt(1+(v/.07)**2)-1))
    def residual(c):
        gain=F@c
        pred=A.transform(x,rr,e[:,None,None],p,light+gain[:,None,None]);fp=A.transform(fx,fr,e[:,None],p,flight+gain[:,None])
        pr=np.stack([(pred*(region==j)).sum((1,2))/(region==j).sum((1,2)) for j in range(3)],axis=1)
        return np.r_[(soft(fp-fy)*wf[:,None]/np.sqrt(128)).ravel(),(soft(pred.mean(2)-y)*wf[:,None]/np.sqrt(48)).ravel(),
            ((pr-ry)*wf[:,None]/np.sqrt(3)).ravel(),(np.quantile(pred.reshape(n,-1),.03,axis=1)-dy)*wf*.5,np.sqrt(n)*ridge*c]
    result=least_squares(residual,np.zeros(F.shape[1]),bounds=(-.6,.6),max_nfev=100,diff_step=.001)
    model=copy.deepcopy(base);model.pop('development',None)
    model.update(id=f'camera_{"bright" if bright else "direction"}_ridge_{ridge:g}',camera_gain_coeffs=result.x.tolist(),camera_brightness_features=bright,
        camera_fit=dict(ridge=ridge,training_frames=n,nfev=result.nfev,converged=bool(result.success)))
    return model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','evaluate']);args=ap.parse_args();cv2.setNumThreads(1);D.mkdir(exist_ok=True)
    A.D=D;p=protocol();prior=json.loads((PREV/'selection.json').read_text());base=prior['selected'];old=prior['baseline']
    if args.stage=='fit':
        if (D/'fresh_metrics.json').exists():raise RuntimeError('Do not tune after the reserved check.')
        rows=A.load_rows(p,{'train','development'});models=[]
        base=copy.deepcopy(base);base['development'],_=A.score(base,rows,'development');models.append(base)
        for bright in [False,True]:
            for ridge in [.05,.15]:
                m=fit(rows,base,bright,ridge);m['development'],_=A.score(m,rows,'development');models.append(m);write(D/'candidates.json',models)
                print(m['id'],m['camera_gain_coeffs'],'J',m['development']['objective'],'MAE',m['development']['full']['equal_run']['mae'],'bias',m['development']['full']['equal_run']['absolute_frame_bias_255'],flush=True)
        selected=min(models,key=lambda m:m['development']['objective'])
        write(D/'selection.json',dict(selected=selected,baseline=old,parent=base,protocol_sha256=C.sha256(D/'protocol.json')))
        print('SELECTED',selected['id'],flush=True);event('camera_response_selected',model=selected['id'],fresh_images_read=False)
        pick=[]
        for run in RUNS:
            ids=[i for i,r in enumerate(rows) if r['run']==run and r['phase']=='development'];pick.extend(ids[j] for j in np.linspace(0,len(ids)-1,3,dtype=int))
        A.make_sheet(rows,pick,base,selected,D/'development.png','Camera-response extension - development images only')
    else:
        s=json.loads((D/'selection.json').read_text());rows=A.load_rows(p,{'fresh'});result={};records=[]
        for name,m in [('previous',old),('v2',s['parent']),('revised',s['selected'])]:
            result[name],rr=A.score(m,rows,'fresh');records+=rr
        write(D/'fresh_metrics.json',dict(selection_sha256=C.sha256(D/'selection.json'),results=result));A.save_records('fresh_frame_metrics.csv',records)
        for run in RUNS:
            ids=[i for i,r in enumerate(rows) if r['run']==run];pick=[ids[j] for j in np.linspace(0,len(ids)-1,4,dtype=int)]
            A.make_sheet(rows,pick,old,s['selected'],D/f'{run}_fresh.png','New guard-gap check - frozen camera-response model')
        A.make_sheet(rows,[next(i for i,r in enumerate(rows) if r['run']==run) for run in RUNS],old,s['selected'],D/'fresh_overview.png','First new guard-gap frame per run')
        print(json.dumps(result),flush=True);event('camera_response_fresh_evaluated',frames=len(rows),results=result)


if __name__=='__main__':main()
