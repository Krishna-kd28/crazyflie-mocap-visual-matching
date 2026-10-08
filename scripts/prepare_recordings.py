#!/usr/bin/env python3
"""Validate raw files, preserve pixels, estimate each run's timing and stage poses."""

import project_config as PC
import argparse
import csv
import json
import time
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation

from baseline import read_table,raw_frame,clock_diagnostic,interpolate_pose,R_BC,P_BC
from capture_paths import C,DATA,OUT,RUNS,EXPOSURE,event,write,snapshot_seeds,seed_path


def ingest():
    snapshot_seeds();summaries=[];sheet=Image.new('RGB',(1920,len(RUNS)*284+60),'white');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for ri,run in enumerate(RUNS):
        path=DATA/run;manifest=json.loads((path/'manifest.json').read_text())
        c,m=read_table(path/'camera.parquet'),read_table(path/'mocap.parquet')
        n=len(c['seq']);assert n==manifest['streams']['camera']['rows']
        assert len(m['seq'])==manifest['streams']['mocap']['rows']
        assert np.all(np.diff(c['host_ns'])>0) and np.all(np.diff(m['host_ns'])>0)
        assert np.all((c['width']==320)&(c['height']==240))
        assert c['blob_offset'][0]==0 and np.all(c['blob_offset'][1:]==c['blob_offset'][:-1]+c['size'][:-1])
        assert int(c['blob_offset'][-1]+c['size'][-1])==(path/'camera_frames.bin').stat().st_size
        dest=OUT/'frames'/run/'real_raw';dest.mkdir(parents=True,exist_ok=True)
        times=(c['host_ns']-c['host_ns'][0])*1e-9;metrics=[]
        ids=np.linspace(0,n-1,6,dtype=int).tolist()
        draw.text((6,ri*284+3),f'{run}  |  exposure {EXPOSURE[run]["exposure_ms"]:.2f} ms, gain {EXPOSURE[run]["digital_gain"]:.2f}x',font=font,fill='#173943')
        for i in range(n):
            a=raw_frame(path,c,i)
            cv2.imwrite(str(dest/f'{i:06d}.png'),a,[cv2.IMWRITE_PNG_COMPRESSION,2])
            metrics.append(dict(index=i,frame_id=int(c['frame_id'][i]),time_s=float(times[i]),mean_intensity=float(a.mean()/255),
                p05=float(np.percentile(a,5)/255),p95=float(np.percentile(a,95)/255),fraction_black=float(np.mean(a==0)),
                fraction_white=float(np.mean(a==255)),laplacian_variance=float(cv2.Laplacian(a,cv2.CV_32F).var())))
            if i in ids:
                col=ids.index(i);draw.text((col*320+6,ri*284+26),f'Frame {i} | {times[i]:.1f} s',font=font,fill='#53636a')
                sheet.paste(Image.fromarray(a),(col*320,ri*284+49))
        with (dest.parent/'raw_metrics.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=metrics[0].keys());w.writeheader();w.writerows(metrics)
        np.savez_compressed(dest.parent/'raw_timing.npz',host_ns=c['host_ns'],frame_id=c['frame_id'],time_s=times)
        stats=dict(run=run,frames=n,duration_s=float(times[-1]),image_size=[320,240],reported_settings=EXPOSURE[run],
            exposure_source='data/25th sep data gate/Run Details.txt; run-level settings, not per-frame telemetry',
            camera_clock=clock_diagnostic(c['deck_ms']*.001,c['host_ns']),mocap_rows=len(m['seq']),
            mean_intensity=float(np.mean([x['mean_intensity'] for x in metrics])),mean_white_fraction=float(np.mean([x['fraction_white'] for x in metrics])),
            mean_black_fraction=float(np.mean([x['fraction_black'] for x in metrics])),raw_pixels='unchanged; no exposure or brightness normalization')
        summaries.append(stats);write(dest.parent/'raw_audit.json',stats)
        print(json.dumps(stats),flush=True)
    draw.text((8,len(RUNS)*284+20),'Raw Crazyflie recordings. Exposure/gain from supplied notes; no brightness normalization.',font=font,fill='#173943')
    sheet.save(OUT/'raw_preview.png')
    write(OUT/'inventory.json',dict(runs=summaries,total_frames=sum(x['frames'] for x in summaries),all_raw_frames_decoded=True))
    event('ingest_complete',runs=len(RUNS),frames=sum(x['frames'] for x in summaries))


def timing_tracks(a,b):
    """Same conservative optical-flow check, using the active lens inversion."""
    mask=np.zeros_like(a);mask[32:-8,10:-10]=255;mask[(a>=250)|(a<=3)]=0
    p=cv2.goodFeaturesToTrack(a,300,.02,8,mask=mask)
    if p is None or len(p)<30:return None
    q,ok,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(21,21),maxLevel=3)
    back,bo,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(21,21),maxLevel=3)
    p,q,back=p[:,0],q[:,0],back[:,0]
    keep=(ok[:,0]>0)&(bo[:,0]>0)&(np.linalg.norm(back-p,axis=1)<.75)
    keep&=(q[:,0]>=10)&(q[:,0]<310)&(q[:,1]>=32)&(q[:,1]<232)
    p,q=p[keep],q[keep]
    if len(p)<30:return None
    qi=np.rint(q).astype(int);keep=(b[qi[:,1],qi[:,0]]<250)&(b[qi[:,1],qi[:,0]]>3)
    p,q=p[keep],q[keep]
    if len(p)<30 or np.median(np.linalg.norm(p-q,axis=1))<2:return None
    pu,qu=C.raw_to_pinhole(p),C.raw_to_pinhole(q)
    _,inlier=cv2.findFundamentalMat(pu,qu,cv2.FM_RANSAC,1.0,.999,2000)
    if inlier is None:return None
    keep=inlier[:,0].astype(bool);pu,qu,p=pu[keep],qu[keep],p[keep]
    if len(pu)<25 or np.ptp(p[:,0])<80 or np.ptp(p[:,1])<50:return None
    pick=np.linspace(0,len(pu)-1,min(100,len(pu)),dtype=int)
    return np.c_[pu[pick],np.ones(len(pick))]@np.linalg.inv(C.K).T,np.c_[qu[pick],np.ones(len(pick))]@np.linalg.inv(C.K).T


def sync(run, camera_correction=None):
    import check_motion_sync as S
    S.OUT=OUT/'sync';S.OUT.mkdir(exist_ok=True);S.track=timing_tracks
    S.R_BC=R_BC if camera_correction is None else R_BC@np.asarray(camera_correction)
    cv2.setRNGSeed(1729)
    result,curves=S.run_one(DATA/run)
    lag=result['best_lag_s']['fit_blocks'];bound=abs(lag)>=.745
    improvement=result['heldout_score_zero_lag']-result['heldout_score_fit_lag']
    # Require a non-boundary solution that improves alternating held-out blocks.
    trusted=not bound and improvement>0
    used_lag=float(lag if trusted else 0)
    c,m=read_table(DATA/run/'camera.parquet'),read_table(DATA/run/'mocap.parquet')
    fit=clock_diagnostic(c['deck_ms']*.001,c['host_ns'])
    clock_s=fit['slope']*(c['deck_ms']-c['deck_ms'][0])*.001+fit['centered_intercept_s']
    query=c['host_ns'][0]+np.rint((clock_s+used_lag)*1e9).astype(np.int64)
    p,q,valid,reason,gaps=interpolate_pose(m,query,50)
    residual=(c['host_ns']-c['host_ns'][0])*1e-9-clock_s
    bad=abs(residual)>.05;reason=np.where(valid&bad,'camera_arrival_outlier',reason);valid&=~bad
    mq=np.column_stack([m[k] for k in ['qx','qy','qz','qw']])
    angle=2*np.arccos(np.clip(abs(np.sum(mq[1:]*mq[:-1],axis=1)),0,1))
    near=np.zeros(len(c['seq']),bool)
    for j in np.flatnonzero(angle>np.pi/4):near|=(query>=m['host_ns'][j]-500_000_000)&(query<=m['host_ns'][j+1]+500_000_000)
    reason=np.where(valid&near,'within_0_5s_of_tracking_jump',reason);valid&=~near
    p[~valid]=np.nan;q[~valid]=np.nan
    T=np.full((len(valid),4,4),np.nan);rb=Rotation.from_quat(q[valid]).as_matrix()
    T[valid]=np.eye(4);T[valid,:3,:3]=rb@R_BC;T[valid,:3,3]=p[valid]+rb@P_BC
    d=OUT/'prepared'/run;d.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(d/'poses.npz',camera_to_mocap_assumed=T,body_position=p,body_quaternion_xyzw=q,
        accepted=valid,host_ns=c['host_ns'],pose_query_host_ns=query,frame_id=c['frame_id'],K=C.K,distortion_opencv=C.DIST)
    idx=np.flatnonzero(valid);reg=C.load_registration(seed_path(run))
    T_GC=C.apply_registration(T[idx],reg['R'],reg['t'],reg['lam']);_,T_CS=C.checkpoint_views(T_GC)
    folder=OUT/'frames'/run;times=(query[idx]-c['host_ns'][0])*1e-9
    np.savez_compressed(folder/'poses.npz',index=idx,frame_id=c['frame_id'][idx],time_s=times,T_MC=T[idx],T_GC=T_GC,T_CS=T_CS,K=C.K,distortion=C.DIST)
    write(folder/'index.json',dict(run=run,short=run[-6:],accepted_indices=idx.tolist(),frame_id=c['frame_id'][idx].tolist(),time_s=times.tolist(),
        total_frames=len(valid),width=320,height=240,real_images='real_raw/: unchanged recorded grayscale pixels',pose_source=str(d/'poses.npz'),pose_sha256=C.sha256(d/'poses.npz')))
    record=dict(run=run,lag_s=used_lag,candidate_lag_s=lag,lag_validated_on_alternating_blocks=trusted,
        heldout_improvement_px=float(improvement),accepted=int(valid.sum()),total=len(valid),reasons={s:int(np.sum(reason==s)) for s in np.unique(reason)},
        lens='active Crazyflie angle model',timing_camera_correction=camera_correction,
        pose_convention='Configured camera mount; bootstrap_registration.py apply adds the estimated camera rotation')
    write(d/'timing.json',record);event('timing_prepared',**record);print(json.dumps(record),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['ingest','sync','all']);ap.add_argument('--run',choices=RUNS)
    ap.add_argument('--timing-camera-correction',type=str);args=ap.parse_args()
    cv2.setNumThreads(4);OUT.mkdir(parents=True,exist_ok=True)
    if args.stage in ['ingest','all']:ingest()
    if args.stage in ['sync','all']:
        correction=json.loads(open(args.timing_camera_correction).read())['right_camera_rotation'] if args.timing_camera_correction else None
        for run in ([args.run] if args.run else RUNS):sync(run,correction)


if __name__=='__main__':main()
