#!/usr/bin/env python3
"""Conservative CoTracker propagation using shared clips and two matched anchors.

Each candidate must have two independent matcher anchors in the clip. Both
bidirectional tracks must agree within 2 px and reach each other's anchors.
No single-anchor extrapolation. Existing matches are never replaced.
"""

import project_config as PC
import argparse,json,time
import numpy as np
import cv2
import torch
from capture_paths import C,OUT,RUNS,configure,write,event


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True,choices=RUNS);a=ap.parse_args();run=a.run
    configure(run);torch.set_num_threads(4);cv2.setNumThreads(1)
    d=OUT/run/'auto_features'/run;meta=json.loads((d/'frames.json').read_text());z=np.load(d/'frames.npz')
    frames=meta['frames'];times=z['time_s'];pos={i:j for j,i in enumerate(frames)};path=OUT/run/'annotations.json';ann=json.loads(path.read_text())
    features=[f for f in ann['features'] if f['status']=='confirmed'];anchors={};predictions={};visibility={}
    for f in features:
        fid=f['id'];anchors[fid]={pos[c['index']]:np.array(c['xy']) for c in f['real_clicks'] if c['method']=='auto_lightglue'}
        pin,dep=C.project(z['T_GC'],np.repeat(np.array(f['point_G'])[None],len(frames),axis=0));raw=C.pinhole_to_raw(pin)
        predictions[fid]=raw;visibility[fid]=(dep>.3)&np.isfinite(raw).all(1)&(raw[:,0]>2)&(raw[:,0]<317)&(raw[:,1]>2)&(raw[:,1]<237)
    model=PC.load_cotracker('cuda')
    propagated={f['id']:{} for f in features};audit=[];cache=d/'track_windows';cache.mkdir(exist_ok=True)
    ends=np.r_[0,np.flatnonzero(np.diff(times)>.25)+1,len(times)];windows=[]
    for lo,hi in zip(ends[:-1],ends[1:]):
        for start in range(int(lo),int(hi),20):
            stop=min(start+41,int(hi))
            if stop-start>=3:windows.append((start,stop))
    t0=time.time()
    for wi,(lo,hi) in enumerate(windows):
        pairs=[];queries=[]
        for f in features:
            fid=f['id'];aa=sorted(j for j in anchors[fid] if lo<=j<hi)
            if len(aa)<2:continue
            first,last=aa[0],aa[-1]
            if last-first<2:continue
            pairs.append((fid,first,last));queries.extend([(first-lo,*anchors[fid][first]),(last-lo,*anchors[fid][last])])
        if not pairs:continue
        cp=cache/f'{lo:06d}_{hi:06d}.npz'
        if cp.exists():
            saved=np.load(cp);tracks,vis=saved['tracks'],saved['visibility']
            np.testing.assert_array_equal(saved['queries'],np.asarray(queries))
        else:
            gray=np.stack([cv2.imread(str(C.real_raw_path(run,frames[j])),0) for j in range(lo,hi)])
            video=torch.from_numpy(gray).float().cuda()[None,:,None].repeat(1,1,3,1,1);alltr=[];allvis=[]
            with torch.no_grad():
                for k in range(0,len(queries),128):
                    q=torch.tensor([queries[k:k+128]],dtype=torch.float32,device='cuda');tr,v=model(video,queries=q,backward_tracking=True)
                    alltr.append(tr[0].permute(1,0,2).cpu().numpy());allvis.append(v[0].permute(1,0).cpu().numpy())
            tracks=np.concatenate(alltr);vis=np.concatenate(allvis);np.savez_compressed(cp,tracks=tracks,visibility=vis,queries=np.asarray(queries))
        good_pairs=0;added=0;anchor_errors=[]
        for k,(fid,first,last) in enumerate(pairs):
            fa,fb=tracks[2*k:2*k+2];va,vb=vis[2*k:2*k+2]
            ea=np.linalg.norm(fa[last-lo]-anchors[fid][last]);eb=np.linalg.norm(fb[first-lo]-anchors[fid][first]);anchor_errors.extend([float(ea),float(eb)])
            if max(ea,eb)>3:continue
            good_pairs+=1
            for t,j in enumerate(range(lo,hi)):
                if j in anchors[fid] or not visibility[fid][j]:continue
                da,db=abs(j-first),abs(j-last);gap=min(da,db)
                if gap>30 or min(va[t],vb[t])<=.5:continue
                xy=fa[t] if da<=db else fb[t];err=float(np.linalg.norm(fa[t]-fb[t]))
                if err>2 or not(4<=xy[0]<=315 and 4<=xy[1]<=235):continue
                if np.linalg.norm(xy-predictions[fid][j])>25:continue
                key=str(frames[j]);seed=frames[first if da<=db else last];v=[round(float(xy[0]),2),round(float(xy[1]),2),gap,round(err,2),seed]
                if key not in propagated[fid] or gap<propagated[fid][key][2]:propagated[fid][key]=v;added+=1
        audit.append(dict(start_frame=frames[lo],stop_frame=frames[hi-1],candidate_pairs=len(pairs),consistent_pairs=good_pairs,added=added,anchor_error_median_px=float(np.median(anchor_errors))))
        if wi%10==0:print(run,wi+1,'/',len(windows),'clips',sum(map(len,propagated.values())),'propagations',round(time.time()-t0),'s',flush=True)
    for f in ann['features']:
        f['real_clicks']=[c for c in f['real_clicks'] if c['method']!='auto_cotracker3']
        for frame,(x,y,gap,err,seed) in propagated.get(f['id'],{}).items():
            f['real_clicks'].append(dict(run=run,index=int(frame),xy=[x,y],sigma_px=float(np.hypot(2.5,.05*gap)),method='auto_cotracker3',seed_index=seed,steps=gap,consistency_px=err,coords='raw'))
        f['real_clicks'].sort(key=lambda c:c['index'])
    stats={f['id']:dict(matched=len(anchors[f['id']]),propagated=len(propagated[f['id']])) for f in features}
    protocol=dict(run=run,tracker='CoTracker3; paired independent matcher anchors, bidirectional, shared overlapping clips',
        parameters=dict(window=41,stride=20,max_gap=30,track_tol_px=2,anchor_tol_px=3,gross_seed_reprojection_px=25,single_anchor_propagation=False),
        format='frame -> [x, y, gap, consistency_px, seed_frame]',candidates=propagated,stats=stats,windows=audit,seconds=time.time()-t0)
    write(d/'propagated.json',protocol);write(path,ann);event('batched_tracking_complete',run=run,total=sum(map(len,propagated.values())),seconds=protocol['seconds'])
    print(run,'tracking complete',sum(map(len,propagated.values())),flush=True)


if __name__=='__main__':main()
