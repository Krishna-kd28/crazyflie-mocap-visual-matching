#!/usr/bin/env python3
"""Resumable feature extraction, global registration, and per-frame alignment."""

import project_config as PC
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from capture_paths import C,ROOT,OUT,RUNS,write,event


def step(run,module,args,sentinel):
    cmd=[sys.executable,str(PC.CODE_ROOT/'scripts/run_stage.py'),run,module,*map(str,args)]
    log=OUT/run/'logs'/(module+'_'+Path(sentinel).parent.name+'_'+Path(sentinel).stem+'.log');log.parent.mkdir(exist_ok=True)
    if Path(sentinel).exists():
        print('exists:',sentinel,flush=True);return
    event('alignment_stage_started',run=run,module=module,command=cmd,log=str(log))
    with log.open('w') as f:result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{module} failed: {log}')
    if not Path(sentinel).exists():raise RuntimeError(f'Stage did not produce {sentinel}')
    event('alignment_stage_complete',run=run,module=module,output=str(sentinel));print(run,module,'complete',flush=True)


def features(run):
    d=OUT/run;ad=d/'auto_features'/run
    args=['--run',run,'--registration',d/'bootstrap_registration.json','--exclude-central-radius',PC.CONFIG.get('exclude_central_radius_m',0)]
    if PC.CONFIG.get('exclude_floor',False):args.append('--exclude-floor')
    for stage,sentinel in [('render',ad/'frames.npz'),('match',ad/'matches_complete.json'),('build',ad/'candidates.json')]:
        # Match has no separate stage marker in the original tool.
        if stage=='match':
            if sentinel.exists():continue
            meta=ad/'frames.json'
            step(run,'auto_features',[stage,*args],meta.with_name('matches')/(json.loads(meta.read_text())['frames'][-1].__format__('06d')+'.npz'))
            write(sentinel,dict(frames=len(json.loads(meta.read_text())['frames']),registration_sha256=C.sha256(d/'bootstrap_registration.json')))
        else:step(run,'auto_features',[stage,*args],sentinel)


def approve_geometric(run):
    path=OUT/run/'annotations.json';ann=json.loads(path.read_text())
    count=0
    for f in ann['features']:
        if f['status']=='rejected':continue
        ch=f['checks'];ok=ch['n_obs']>=3 and ch['parallax_deg']>=3 and ch['render_residual_px']<=1.5 and ch['real_reproj_median_px']<=12
        if ok:
            f['status']='confirmed';f['decision_source']='batch admission after automatic geometric checks';f['manually_reviewed']=False;count+=1
    if count<8:raise RuntimeError('Too few candidates; inspect '+str(path))
    write(path,ann);event('automatic_features_admitted',run=run,count=count,individual_manual_review=False)


def fit(run,track=False,accept_geometric=False):
    d=OUT/run;ff=d/'fixed_features';ann=d/'annotations.json';ad=d/'auto_features'/run
    if accept_geometric:
        approve_geometric(run)
    elif sum(f.get("status") == "confirmed" for f in json.loads(ann.read_text())["features"]) < 8:
        raise RuntimeError("Review/confirm at least 8 features, or explicitly pass --accept-geometric")
    if track:
        if not (ad/'propagated.json').exists():
            log=d/'logs'/'batched_tracker.log';log.parent.mkdir(exist_ok=True)
            command=[sys.executable,str(PC.CODE_ROOT/'scripts/track_features_batch.py'),'--run',run]
            event('alignment_stage_started',run=run,module='batched_cotracker',command=command,log=str(log))
            with log.open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    args=['--annotations',ann,'--extra-annotations']
    step(run,'track_fixed_features',[*args,'--tracker','lk','--out',ff/'observations.npz'],ff/'observations.npz')
    # Global transform is fitted using at most 120 distributed seed frames.
    # All observations remain available for audit and independent per-frame search.
    z=np.load(ff/'observations.npz');ids=np.unique(z['index']);chosen=ids[np.linspace(0,len(ids)-1,min(120,len(ids)),dtype=int)]
    use=np.isin(z['index'],chosen);a={k:(z[k][use] if z[k].ndim and len(z[k])==len(use) and k!='feature_ids' else z[k]) for k in z.files}
    np.savez_compressed(ff/'observations_global_sample.npz',**a)
    write(ff/'global_sampling.json',dict(rule='120 equally spaced observed frame indices; all features on each; full observations kept for per-frame fitting',total=len(use),sample=int(use.sum()),frames=chosen.tolist()))
    for name,extra in [('free',[]),('fixed',['--fix-scale'])]:
        path=ff/f'registration_{name}'
        step(run,'brute_force_registration',[*args,'--observations',ff/'observations_global_sample.npz','--start',d/'bootstrap_registration.json','--out',path,'--min-parallax-deg',3,*extra],path/'metrics.json')
    choices={name:json.loads((ff/f'registration_{name}/metrics.json').read_text()) for name in ['free','fixed']}
    winner=min(choices,key=lambda k:choices[k]['validation']['held_out_frames']['fitted']['rms_px'])
    import shutil
    selected=ff/'registration';selected.mkdir(exist_ok=True)
    for p in (ff/f'registration_{winner}').iterdir():
        if p.is_file():shutil.copy2(p,selected/p.name)
    write(ff/'selected_registration.json',dict(winner=winner,criterion='held-out seed-frame RMS on global sample',rms={k:v['validation']['held_out_frames']['fitted']['rms_px'] for k,v in choices.items()}))
    step(run,'extend_fixed_feature_observations',[*args,'--run',run,'--observations',ff/'observations.npz','--fit',selected,'--out',ff/'observations_extended.npz'],ff/'observations_extended.npz')
    step(run,'per_frame_registration',[*args,'--run',run,'--observations',ff/'observations_extended.npz','--fit',selected,'--out',ff/'per_frame','--render'],ff/'per_frame'/run/'render'/'render_metadata.json')
    event('run_alignment_complete',run=run,summary=str(ff/'per_frame'/run/'summary.json'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['features','fit','all']);ap.add_argument('--run',choices=RUNS);ap.add_argument('--track',action='store_true');ap.add_argument('--accept-geometric',action='store_true');args=ap.parse_args()
    for run in [args.run] if args.run else RUNS:
        if args.stage in ['features','all']:features(run)
        if args.stage in ['fit','all']:fit(run,args.track,args.accept_geometric)


if __name__=='__main__':main()
