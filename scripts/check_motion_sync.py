#!/usr/bin/env python3
"""Estimate a provisional real-image/mocap lag without a scene reconstruction.

Use only original train/validation runs. Test run 184244 remains untouched.
Negative lag means look up an earlier mocap pose than the affine camera clock.
"""

import project_config as PC
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from baseline import ROOT, K, DIST, R_BC, P_BC, read_table, raw_frame, clock_diagnostic, write_json

OUT=ROOT/'output/motion_sync'
LAGS=np.linspace(-.75,.75,301)
FOCAL=np.sqrt(K[0,0]*K[1,1])


def epipolar_scores(r1,p1,r2,p2,x,y):
    """Equal-weight per-pair median Sampson distance in approximate pixel units."""
    r=r2.transpose(0,2,1)@r1
    t=np.einsum('nij,nj->ni',r2.transpose(0,2,1),p1-p2)
    length=np.linalg.norm(t,axis=1)
    t=t/np.maximum(length[:,None],1e-12)
    skew=np.zeros_like(r)
    skew[:,0,1]=-t[:,2];skew[:,0,2]=t[:,1]
    skew[:,1,0]=t[:,2];skew[:,1,2]=-t[:,0]
    skew[:,2,0]=-t[:,1];skew[:,2,1]=t[:,0]
    E=skew@r
    Ex=np.einsum('nij,nkj->nki',E,x)
    Ety=np.einsum('nji,nkj->nki',E,y)
    denom=np.sum(Ex[:,:,:2]**2,axis=2)+np.sum(Ety[:,:,:2]**2,axis=2)
    err=abs(np.sum(y*Ex,axis=2))/np.sqrt(np.maximum(denom,1e-15))*FOCAL
    return np.nanmedian(err,axis=1),length


def track(a,b):
    mask=np.zeros_like(a)
    mask[32:-8,10:-10]=255  # Fixed top platform-occlusion exclusion.
    mask[(a>=250)|(a<=3)]=0
    p=cv2.goodFeaturesToTrack(a,300,.02,8,mask=mask)
    if p is None or len(p)<30:return None
    q,status,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(21,21),maxLevel=3)
    back,bs,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(21,21),maxLevel=3)
    p,q,back=p[:,0],q[:,0],back[:,0]
    ok=(status[:,0]>0)&(bs[:,0]>0)&(np.linalg.norm(back-p,axis=1)<.75)
    ok&=(q[:,0]>=10)&(q[:,0]<310)&(q[:,1]>=32)&(q[:,1]<232)
    p,q=p[ok],q[ok]
    if len(p)<30:return None
    qi=np.rint(q).astype(int)
    ok=(b[qi[:,1],qi[:,0]]<250)&(b[qi[:,1],qi[:,0]]>3)
    p,q=p[ok],q[ok]
    if len(p)<30 or np.median(np.linalg.norm(p-q,axis=1))<2:return None
    # Remove mismatches from images alone; never refit tracks for a trial lag.
    pu=cv2.undistortPoints(p[:,None,:],K,DIST,P=K)[:,0,:]
    qu=cv2.undistortPoints(q[:,None,:],K,DIST,P=K)[:,0,:]
    _,inlier=cv2.findFundamentalMat(pu,qu,cv2.FM_RANSAC,1.0,.999,2000)
    if inlier is None:return None
    ok=inlier[:,0].astype(bool);pu,qu,p=pu[ok],qu[ok],p[ok]
    if len(pu)<25 or np.ptp(p[:,0])<80 or np.ptp(p[:,1])<50:return None
    pick=np.linspace(0,len(pu)-1,min(100,len(pu)),dtype=int)
    x=np.c_[pu[pick],np.ones(len(pick))]@np.linalg.inv(K).T
    y=np.c_[qu[pick],np.ones(len(pick))]@np.linalg.inv(K).T
    return x,y


def run_one(run):
    c,m=read_table(run/'camera.parquet'),read_table(run/'mocap.parquet')
    origin=c['host_ns'][0]
    mt=(m['host_ns']-origin)*1e-9
    ch=(c['host_ns']-origin)*1e-9
    fit=clock_diagnostic(c['deck_ms']*.001,c['host_ns'])
    ct=fit['slope']*(c['deck_ms']-c['deck_ms'][0])*.001+fit['centered_intercept_s']
    residual=ch-ct
    pos=np.column_stack([m[k] for k in ['x','y','z']])
    quat=np.column_stack([m[k] for k in ['qx','qy','qz','qw']])
    rot=Rotation.from_quat(quat);slerp=Slerp(mt,rot)
    angle=2*np.arccos(np.clip(abs(np.sum(quat[1:]*quat[:-1],axis=1)),0,1))
    bad=np.flatnonzero((np.diff(mt)>.050)|(angle>np.pi/4))
    pairs=[];xs=[];ys=[];counts=[]
    for i in range(0,len(c['seq'])-1,3):
        a,b=ct[i]-.75,ct[i+1]+.75
        if a<mt[0]+.2 or b>mt[-1]-.2:continue
        if max(abs(residual[i:i+2]))>.05 or ct[i+1]-ct[i]>.2:continue
        if np.any((mt[bad]<=b)&(mt[bad+1]>=a)):continue
        match=track(raw_frame(run,c,i),raw_frame(run,c,i+1))
        if match is None:continue
        x,y=match;n=len(x)
        xp=np.full((100,3),np.nan);yp=xp.copy();xp[:n]=x;yp[:n]=y
        xs.append(xp);ys.append(yp);pairs.append([i,i+1]);counts.append(n)
    pairs=np.asarray(pairs);x=np.asarray(xs);y=np.asarray(ys)
    if len(pairs)<20:raise ValueError(f'{run.name}: only {len(pairs)} informative pairs')
    def poses(times):
        R=slerp(times).as_matrix()
        p=np.column_stack([np.interp(times,mt,pos[:,j]) for j in range(3)])
        return R@R_BC,p+np.einsum('nij,j->ni',R,P_BC)
    scores=[];baselines=[]
    for lag in LAGS:
        r1,p1=poses(ct[pairs[:,0]]+lag);r2,p2=poses(ct[pairs[:,1]]+lag)
        error,length=epipolar_scores(r1,p1,r2,p2,x,y)
        baselines.append(length);scores.append(error)
    scores=np.asarray(scores);baselines=np.asarray(baselines)
    valid=np.min(baselines,axis=0)>=.005  # Same pairs for all lags; avoid pure-rotation E degeneracy.
    scores=scores[:,valid];pairs=pairs[valid];counts=np.asarray(counts)[valid]
    if len(pairs)<20:raise ValueError('Too few pairs with translation across all tested lags')
    times=ct[pairs[:,0]]
    fold=(times//10).astype(int)%2 # alternating 10 s blocks, no adjacent random leakage
    curves={name:np.median(scores[:,mask],axis=1) for name,mask in {
        'all':np.ones(len(pairs),bool),'fit_blocks':fold==0,'heldout_blocks':fold==1}.items()}
    best={name:int(np.argmin(curve)) for name,curve in curves.items()}
    result={'run':run.name,'pairs':len(pairs),'tracks':int(counts.sum()),
            'lag_sign':'mocap query time = affine camera time + lag',
            'best_lag_s':{name:float(LAGS[i]) for name,i in best.items()},
            'zero_lag_score':float(curves['all'][150]),
            'best_score':float(curves['all'][best['all']]),
            'heldout_score_zero_lag':float(curves['heldout_blocks'][150]),
            'heldout_score_fit_lag':float(curves['heldout_blocks'][best['fit_blocks']]),
            'score_units':'median pairwise Sampson residual, approximate undistorted pixels',
            'clock_fit':fit,'time_quarter_best_lag_s':[]}
    for lo,hi in zip(np.linspace(times.min(),times.max()+1e-9,5)[:-1],np.linspace(times.min(),times.max()+1e-9,5)[1:]):
        sel=(times>=lo)&(times<hi)
        result['time_quarter_best_lag_s'].append(float(LAGS[np.argmin(np.median(scores[:,sel],axis=1))]) if sel.sum()>=10 else None)
    # 10-second block bootstrap; describes this run and model, not hardware exposure uncertainty.
    blocks=(times//10).astype(int);unique=np.unique(blocks);rng=np.random.default_rng(1729)
    estimates=[]
    for _ in range(300):
        sample=rng.choice(unique,len(unique),replace=True)
        sel=np.concatenate([np.flatnonzero(blocks==b) for b in sample])
        estimates.append(LAGS[np.argmin(np.median(scores[:,sel],axis=1))])
    result['block_bootstrap_95pct_lag_s']=np.percentile(estimates,[2.5,97.5]).tolist()
    np.savez_compressed(OUT/(run.name+'.npz'),lags=LAGS,pairs=pairs,pair_time=times,
                        scores=scores,track_count=counts,**curves)
    write_json(OUT/(run.name+'.json'),result)
    print(json.dumps(result,indent=2),flush=True)
    return result,curves


def save_candidate_pairing(lag):
    """Save a separate suggested alignment; preserve the original baseline."""
    from baseline import interpolate_pose
    import pyarrow as pa
    import pyarrow.parquet as pq
    all_rows=[];stats=[]
    for run in sorted((PC.DATA).glob('run_*')):
        c,m=read_table(run/'camera.parquet'),read_table(run/'mocap.parquet')
        fit=clock_diagnostic(c['deck_ms']*.001,c['host_ns'])
        clock_s=fit['slope']*(c['deck_ms']-c['deck_ms'][0])*.001+fit['centered_intercept_s']
        query_ns=c['host_ns'][0]+np.rint((clock_s+lag)*1e9).astype(np.int64)
        p,q,valid,reason,gaps=interpolate_pose(m,query_ns,50)
        residual=(c['host_ns']-c['host_ns'][0])*1e-9-clock_s
        bad=abs(residual)>.05
        reason=np.where(valid&bad,'camera_arrival_outlier',reason);valid&=~bad
        quat=np.column_stack([m[k] for k in ['qx','qy','qz','qw']])
        angle=2*np.arccos(np.clip(abs(np.sum(quat[1:]*quat[:-1],axis=1)),0,1))
        jumps=np.flatnonzero(angle>np.pi/4)
        near_jump=np.zeros(len(c['seq']),bool)
        for j in jumps:
            near_jump|=(query_ns>=m['host_ns'][j]-500_000_000)&(query_ns<=m['host_ns'][j+1]+500_000_000)
        reason=np.where(valid&near_jump,'within_0_5s_of_tracking_jump',reason);valid&=~near_jump
        p[~valid]=np.nan;q[~valid]=np.nan
        T=np.full((len(valid),4,4),np.nan)
        rb=Rotation.from_quat(q[valid]).as_matrix()
        T[valid]=np.eye(4);T[valid,:3,:3]=rb@R_BC
        T[valid,:3,3]=p[valid]+rb@P_BC
        dest=OUT/'candidate_pairs'/run.name;dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'poses.npz',camera_to_mocap_assumed=T,body_position=p,
            body_quaternion_xyzw=q,accepted=valid,host_ns=c['host_ns'],pose_query_host_ns=query_ns,
            frame_id=c['frame_id'],K=K,distortion_opencv=DIST)
        for i in range(len(c['seq'])):
            all_rows.append({'run':run.name,'index':i,'frame_id':int(c['frame_id'][i]),
                'accepted':bool(valid[i]),'reason':str(reason[i]),'camera_host_ns':int(c['host_ns'][i]),
                'pose_query_host_ns':int(query_ns[i]),'mocap_bracket_ms':float(gaps[i]),
                'clock_residual_ms':float(residual[i]*1000),
                'lag_s':float(lag),'provisional':True})
        stats.append({'run':run.name,'frames':len(valid),'candidate_accepted':int(valid.sum())})
    pq.write_table(pa.Table.from_pylist(all_rows),OUT/'candidate_pairs/frame_index.parquet')
    import csv
    with (OUT/'candidate_pairs/frame_index.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=all_rows[0].keys());w.writeheader();w.writerows(all_rows)
    write_json(OUT/'candidate_pairs/alignment.json',{
        'status':'provisional; corrected effective timing, camera conventions still assumed',
        'equation':'pose_time_ns = camera_host_ns[0] + round(1e9 * (a*(deck_ms-deck_ms[0])/1000 + b + lag_s))',
        'lag_s':float(lag),'fit':'shared lag selected on alternating training blocks of first two runs',
        'tracking_jump_guard_s':.5,'run_184244':'fixed lag applied using timestamps only; image validation not performed',
        'runs':stats})


def main():
    cv2.setRNGSeed(1729);cv2.setNumThreads(4)
    OUT.mkdir(exist_ok=True,parents=True)
    results=[];curves=[]
    for run in sorted((PC.DATA).glob('run_*'))[:3]:
        result,curve=run_one(run);results.append(result);curves.append(curve)
    train_curve=(curves[0]['fit_blocks']+curves[1]['fit_blocks'])/2
    best=int(np.argmin(train_curve))
    final={'status':'diagnostic lag search; no synchronized dataset certified',
           'training_selected_shared_lag_s':float(LAGS[best]),
           'validation_zero_lag_score':float(curves[2]['all'][150]),
           'validation_training_lag_score':float(curves[2]['all'][best]),
           'runs':results,'reserved_test_run_used':False}
    write_json(OUT/'summary.json',final)
    save_candidate_pairing(float(LAGS[best]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for ax,result,curve in zip(axes,results,curves):
        for name,style in [('fit_blocks','-'),('heldout_blocks','--')]:
            ax.plot(LAGS*1000,curve[name],style,label=name.replace('_',' '))
        ax.axvline(0,color='gray',lw=.8);ax.axvline(LAGS[best]*1000,color='tab:green',lw=.8,label='shared train estimate')
        ax.set(title=result['run'][-6:],xlabel='Mocap time adjustment (ms)',ylabel='Epipolar error (approx. pixels)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(OUT/'lag_profiles.png',dpi=180)
    print(json.dumps(final,indent=2))


if __name__=='__main__':main()
