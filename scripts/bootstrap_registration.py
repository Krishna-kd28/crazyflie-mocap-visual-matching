#!/usr/bin/env python3
"""Localize recording keyframes against a known-pose 3DGS render atlas.

Initialize scene and camera orientation with MINIMA SuperPoint/LightGlue,
rendered expected-depth lifting, robust PnP, then inspect a shared hand-eye fit.
Depth-derived points are bootstrap evidence, not permanent triangulated points.
"""

import project_config as PC
import argparse
import json
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from capture_paths import C, ROOT, OUT, RUNS, configure, write, event
from auto_features import Matcher
from prepare_fixed_feature_frames import Renderer, render_depth

D=OUT/'bootstrap'


def atlas(renderer, matcher, count=64):
    # Compact camera poses copied into the package; render the atlas here.
    # No source-workspace images or prior full-run output folders are needed.
    z=np.load(PC.path('reference_atlas'),allow_pickle=False)
    T=z['T_GC'];count=min(count,len(T));selected=list(range(count))
    _,views=C.checkpoint_views(T)
    d=D/'atlas';d.mkdir(parents=True,exist_ok=True);records=[]
    for n,j in enumerate(selected):
        i=int(z['index'][j]);path=d/f'{i:06d}.npz'
        rgb=renderer.rgb(views[j])
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);feat=matcher.features(gray);uv=matcher.keypoints(feat)
        if path.exists():
            zz=np.load(path);depth,spread,alpha=zz['depth'],zz['spread'],zz['alpha']
        else:
            view=renderer.torch.as_tensor(views[j][None],dtype=renderer.torch.float32,device='cuda')
            with renderer.torch.no_grad():depth,spread,alpha=render_depth(renderer.scene,renderer.info,view,renderer.rasterization,renderer.torch)
            np.savez_compressed(path,depth=depth,spread=spread,alpha=alpha)
        pin=C.raw_to_pinhole(uv);off=C.distortion_canvas()['offset'];xy=np.rint(pin+off).astype(int)
        valid=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<depth.shape[1])&(xy[:,1]<depth.shape[0])&np.isfinite(pin).all(1)
        xx=np.clip(xy[:,0],0,depth.shape[1]-1);yy=np.clip(xy[:,1],0,depth.shape[0]-1)
        dep=depth[yy,xx]*C.metres_per_checkpoint_unit()
        valid&=(alpha[yy,xx]>.95)&(dep>.1)&(dep<15)&(spread[yy,xx]*C.metres_per_checkpoint_unit()<.25)
        rays=np.c_[pin,np.ones(len(pin))]@np.linalg.inv(C.K).T
        X=(rays*dep[:,None])@T[j,:3,:3].T+T[j,:3,3]
        records.append(dict(index=i,T=T[j],features=feat,uv=uv,X=X,valid=valid,rgb=rgb))
        if n%16==0:print('atlas',n+1,'/',count,flush=True)
    write(d/'selection.json',dict(source=str(PC.path('reference_atlas')),indices=[r['index'] for r in records],rule='packaged known-pose atlas; rerendered in the configured scene'))
    return records


def pnp(X, raw):
    if len(X)<12:return None
    pin=C.raw_to_pinhole(raw).astype(np.float64)
    finite=np.isfinite(pin).all(1)&np.isfinite(X).all(1)
    if finite.sum()<12:return None
    ok,rv,tv,idx=cv2.solvePnPRansac(np.ascontiguousarray(X[finite],np.float64),np.ascontiguousarray(pin[finite]),C.K,None,
        iterationsCount=1500,reprojectionError=3,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or idx is None or len(idx)<12:return None
    ids=np.flatnonzero(finite)[idx[:,0]]
    rv,tv=cv2.solvePnPRefineLM(X[ids].astype(np.float64),pin[ids],C.K,None,rv,tv)
    R=cv2.Rodrigues(rv)[0];T=np.eye(4);T[:3,:3]=R.T;T[:3,3]=-R.T@tv[:,0]
    projected,depth=C.project(np.repeat(T[None],len(X),axis=0),X)
    error=np.linalg.norm(C.pinhole_to_raw(projected)-raw,axis=1)
    ids=np.flatnonzero((error<3)&(depth>.1)&finite)
    hull=cv2.contourArea(cv2.convexHull(raw[ids].astype(np.float32)))/(320*240) if len(ids)>2 else 0
    if len(ids)<12 or hull<.025:return None
    return dict(T=T,ids=ids,error=error,hull=hull,score=len(ids)*np.sqrt(hull))


def localize(args):
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(2);cv2.setRNGSeed(25928)
    D.mkdir(parents=True,exist_ok=True);matcher=Matcher(2048);renderer=Renderer()
    refs=atlas(renderer,matcher,args.atlas)
    records=[]
    for run in RUNS:
        configure(run);_,_,z=C.frame_store(run)
        ids=np.linspace(0,len(z['index'])-1,args.queries,dtype=int)
        for num,j in enumerate(ids):
            i=int(z['index'][j]);cache=D/f'{run}_{i:06d}.json'
            if cache.exists():
                row=json.loads(cache.read_text());records.append(row);continue
            real=cv2.imread(str(C.real_raw_path(run,i)),0);features=matcher.features(real);uv=matcher.keypoints(features)
            best=None
            for ref in refs:
                match,score=matcher.match(features,ref['features'],cross=True)
                keep=ref['valid'][match[:,1]]&(score>.1);match=match[keep]
                result=pnp(ref['X'][match[:,1]],uv[match[:,0]])
                if result is not None and (best is None or result['score']>best['score']):
                    best={**result,'ref':ref,'match':match,'raw':uv[match[:,0]],'X':ref['X'][match[:,1]]}
            row=dict(run=run,index=i,time_s=float(z['time_s'][j]),T_MC=z['T_MC'][j].tolist(),localized=best is not None)
            if best:
                row.update(T_GC=best['T'].tolist(),inliers=len(best['ids']),median_px=float(np.median(best['error'][best['ids']])),
                    hull_fraction=float(best['hull']),score=float(best['score']),reference_index=best['ref']['index'])
                np.savez_compressed(cache.with_suffix('.npz'),X=best['X'][best['ids']],raw=best['raw'][best['ids']])
                _,V=C.checkpoint_views(best['T'][None]);b=renderer.rgb(V[0]);Image.fromarray(b).save(cache.with_suffix('.png'))
            write(cache,row);records.append(row)
            print(run,num+1,'/',len(ids),'frame',i,'inliers',row.get('inliers',0),'error',row.get('median_px'),flush=True)
        write(D/'localizations.json',records)
    event('bootstrap_localized',queries=len(records),localized=sum(r['localized'] for r in records))


def fit():
    rows=json.loads((D/'localizations.json').read_text());good=[r for r in rows if r['localized'] and r['inliers']>=18 and r['hull_fraction']>=.04]
    # Reuse visual PnP solutions after a timing update, but query the new mocap poses.
    latest={}
    for run in RUNS:
        configure(run);_,_,z=C.frame_store(run)
        latest[run]={int(i):T for i,T in zip(z['index'],z.get('T_MC_initial',z['T_MC']))}
    good=[r for r in good if r['index'] in latest[r['run']]]
    for r in good:r['T_MC']=latest[r['run']][r['index']].tolist()
    M=np.array([r['T_MC'] for r in good]);G=np.array([r['T_GC'] for r in good])
    if len(good)<12:raise RuntimeError('Not enough independent visual poses')
    def unpack(p):return Rotation.from_rotvec(p[:3]).as_matrix(),Rotation.from_rotvec(p[3:6]).as_matrix(),p[6:9],np.exp(p[9])
    def residual(p,ids):
        A,B,t,s=unpack(p);R=A@M[ids,:3,:3]@B
        rot=Rotation.from_matrix(np.swapaxes(R,1,2)@G[ids,:3,:3]).as_rotvec()
        pos=s*(M[ids,:3,3]@A.T)+t-G[ids,:3,3]
        return np.c_[rot/.05,pos/.15].ravel()
    allids=np.arange(len(M));best=None
    # Broad initial yaw grid; do not assume the recorded rigid-body axes stayed the same.
    for yaw in [0,90,180,270]:
        A=Rotation.from_euler('zyx',[yaw,0,180],degrees=True).as_matrix()
        B=Rotation.from_matrix(np.swapaxes(M[:,:3,:3],1,2)@A.T@G[:,:3,:3]).mean().as_matrix()
        p=np.r_[Rotation.from_matrix(A).as_rotvec(),Rotation.from_matrix(B).as_rotvec(),np.median(G[:,:3,3]-M[:,:3,3]@A.T,axis=0),0]
        res=least_squares(residual,p,args=(allids,),loss='soft_l1',max_nfev=400)
        if best is None or res.cost<best.cost:best=res
    p=best.x
    err=np.linalg.norm(residual(p,allids).reshape(-1,6),axis=1);keep=err<5
    if keep.sum()>=12:p=least_squares(residual,p,args=(allids[keep],),loss='soft_l1',max_nfev=400).x
    A,B,t,s=unpack(p)
    re=np.linalg.norm(Rotation.from_matrix(np.swapaxes(A@M[:,:3,:3]@B,1,2)@G[:,:3,:3]).as_rotvec(),axis=1)*180/np.pi
    pe=np.linalg.norm(s*(M[:,:3,3]@A.T)+t-G[:,:3,3],axis=1)
    out=dict(status='bootstrap; inspect rerenders before use',localized=len(good),inlier_poses=int(keep.sum()),R_GM=A.tolist(),right_camera_rotation=B.tolist(),translation_G=t.tolist(),scale=float(s),
        rotation_error_deg_median=float(np.median(re[keep])),position_error_m_median=float(np.median(pe[keep])),
        rows=[dict(run=r['run'],index=r['index'],kept=bool(k),rotation_error_deg=float(a),position_error_m=float(b)) for r,k,a,b in zip(good,keep,re,pe)])
    out['leave_one_run_out']={}
    for run in RUNS:
        tr=np.array([r['run']!=run for r in good]);te=~tr
        if tr.sum()<12 or not te.any():
            out['leave_one_run_out'][run]={'status':'insufficient independent poses'}
            continue
        q=least_squares(residual,p,args=(allids[tr],),loss='soft_l1',max_nfev=300).x
        a,b,tv,sc=unpack(q)
        angle=np.degrees(np.linalg.norm(Rotation.from_matrix(np.swapaxes(a@M[te,:3,:3]@b,1,2)@G[te,:3,:3]).as_rotvec(),axis=1))
        dist=np.linalg.norm(sc*(M[te,:3,3]@a.T)+tv-G[te,:3,3],axis=1)
        out['leave_one_run_out'][run]=dict(poses=int(te.sum()),rotation_median_deg=float(np.median(angle)),position_median_m=float(np.median(dist)),rotation_p90_deg=float(np.percentile(angle,90)),position_p90_m=float(np.percentile(dist,90)))
    write(D/'hand_eye.json',out);print(json.dumps(out),flush=True)


def apply_mount():
    h=json.loads((D/'hand_eye.json').read_text());A=np.array(h['R_GM']);B=np.array(h['right_camera_rotation']);t=np.array(h['translation_G']);s=h['scale']
    renderer=Renderer();font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for run in RUNS:
        configure(run);folder,index,z=C.frame_store(run)
        arrays={k:z[k].copy() for k in z.files} if hasattr(z,'files') else {k:v.copy() for k,v in z.items()}
        # Idempotent: preserve and always start from the original assumed mount.
        arrays.setdefault('T_MC_initial',arrays['T_MC'].copy());arrays['T_MC']=arrays['T_MC_initial'].copy();arrays['T_MC'][:,:3,:3]=arrays['T_MC'][:,:3,:3]@B
        arrays['T_GC']=C.apply_registration(arrays['T_MC'],A,t,s);_,arrays['T_CS']=C.checkpoint_views(arrays['T_GC'])
        np.savez_compressed(folder/'poses.npz',**arrays)
        pp=OUT/'prepared'/run/'poses.npz';p=np.load(pp);full={k:p[k].copy() for k in p.files};key='camera_to_mocap_assumed'
        full.setdefault('camera_to_mocap_initial',full[key].copy());full[key]=full['camera_to_mocap_initial'].copy();full[key][full['accepted'],:3,:3]=full[key][full['accepted'],:3,:3]@B
        np.savez_compressed(pp,**full)
        rawindex=json.loads((folder/'index.json').read_text());rawindex['pose_sha256']=C.sha256(pp);rawindex['camera_correction']=str(D/'hand_eye.json');write(folder/'index.json',rawindex)
        reg=C.registration_json(A,t,s,'Image-localized scene bootstrap',dict(camera_rotation_correction=B.tolist(),bootstrap_source=str(D/'hand_eye.json'),source_sha256=C.sha256(D/'hand_eye.json'),camera_lever_arm='configured lever arm retained; mount rotation inferred from visual poses'))
        write(OUT/run/'bootstrap_registration.json',reg)
        sh=Image.new('RGB',(1280,4*274+44),'#f3f5f6');draw=ImageDraw.Draw(sh);draw.text((8,8),run+' | real and bootstrap 3DGS (camera convention corrected)',font=font,fill='#173943')
        for n,j in enumerate(np.linspace(0,len(arrays['index'])-1,8,dtype=int)):
            i=int(arrays['index'][j]);x=n%2*640;y=44+n//2*274
            rgb=renderer.rgb(arrays['T_CS'][j]);Image.fromarray(rgb).save(D/f'{run}_corrected_{i:06d}.png')
            for col,(image,label) in enumerate([(Image.open(C.real_raw_path(run,i)),'Real'),(Image.fromarray(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)),'3DGS')]):
                draw.text((x+col*320+5,y+4),label+' | '+str(i),font=font,fill='#173943');sh.paste(image,(x+col*320,y+28))
        sh.save(D/f'{run}_corrected_sheet.png')
    event('bootstrap_applied',mount_rotation=B.tolist(),registration_R=A.tolist(),scale=s,source_sha256=C.sha256(D/'hand_eye.json'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['localize','fit','apply','all']);ap.add_argument('--atlas',type=int,default=64);ap.add_argument('--queries',type=int,default=24);a=ap.parse_args()
    if a.stage in ['localize','all']:localize(a)
    if a.stage in ['fit','all']:fit()
    if a.stage=='apply':apply_mount()


if __name__=='__main__':main()
