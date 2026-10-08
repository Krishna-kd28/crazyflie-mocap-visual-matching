#!/usr/bin/env python3
"""Audit the reported darkness/softness without changing the released fit.

The four displayed images are already-inspected test frames. This is a
post-publication diagnostic, not a new independent test or model selection.
"""

import project_config as PC
import copy
import csv
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fit_exposure import apply, load
from grayscale_appearance import R2
from image_similarity import compare
from capture_paths import C, OUT, ROOT, RUNS, write, event

D=OUT/'exposure_review'
DISPLAYED=dict(zip(RUNS,[398,396,404,392]))


def measure(real,pred,mask=None):
    a=real.astype(np.float32)/255
    m=np.ones(a.shape,bool) if mask is None else mask.copy()
    m[:5]=False;m[-5:]=False;m[:,:5]=False;m[:,-5:]=False
    result=compare(a,pred,m)
    if result is None:return None
    delta=255*(pred-a)
    result.update(real_mean_255=float(a[m].mean()*255),
                  sim_mean_255=float(pred[m].mean()*255),
                  signed_bias_255=float(delta[m].mean()),
                  median_residual_255=float(np.median(delta[m])),
                  absolute_frame_bias_255=float(abs(delta[m].mean())),
                  dark_pixel_fraction=float((delta[m]<-5).mean()),
                  real_saturated_fraction=float((real[m]>=250).mean()),
                  sim_saturated_fraction=float((pred[m]>=250/255).mean()))
    for name,region in [('center',R2<=.25),('middle',(R2>.25)&(R2<=.81)),('periphery',R2>.81)]:
        q=m&region
        result[name+'_bias_255']=float(delta[q].mean()) if q.any() else None
    return result


def average(rows):
    if not rows:return dict(frames=0)
    keys=['mae','rmse','ssim','edge_f1','edge_precision','edge_recall','gradient_ratio','coverage',
          'real_mean_255','sim_mean_255','signed_bias_255','median_residual_255',
          'absolute_frame_bias_255','dark_pixel_fraction','real_saturated_fraction','sim_saturated_fraction',
          'center_bias_255','middle_bias_255','periphery_bias_255']
    out=dict(frames=len(rows))
    for k in keys:
        valid=[r[k] for r in rows if r[k] is not None]
        out[k]=float(np.mean(valid)) if valid else None
    out['fraction_frames_darker_by_5']=float(np.mean([r['signed_bias_255'] < -5 for r in rows]))
    return out


def sheet(rows,models,path,indices):
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    labels=['Real','Published fit: blur 1.4 px','Same tone: blur 0.8 px']
    im=Image.new('RGB',(960,322*len(indices)+64),'#f3f5f6');draw=ImageDraw.Draw(im)
    draw.text((8,7),'Controlled blur check - brightness parameters are identical',font=font,fill='#173943')
    draw.text((8,34),'Diagnostic only. The right column does not correct the remaining darkness.',font=small,fill='#526870')
    for k,i in enumerate(indices):
        r=rows[i];y=64+322*k
        draw.text((5,y),f'{r["run"][-6:]} | frame {r["index"]} | {r["time_s"]:.1f} s',font=font,fill='#173943')
        arrays=[r['real']]+[np.uint8(np.round(255*apply(r['rgb'],m,r['run']))) for m in models[:2]]
        for c,(a,label) in enumerate(zip(arrays,labels)):
            x=c*320
            draw.text((x+5,y+25),label,font=font,fill='#173943')
            draw.text((x+5,y+48),f'Mean intensity {a[5:-5,5:-5].mean():.1f}/255',font=small,fill='#526870')
            im.paste(Image.fromarray(a),(x,y+76))
    im.save(path)


def main():
    cv2.setNumThreads(1);D.mkdir(exist_ok=True)
    protected_paths=[OUT/'exposure'/f'{n}.json' for n in ['protocol','candidates','selection','summary','transfer']]
    protected_paths+=[ROOT/p for p in json.loads((OUT/'protected_originals.json').read_text())]
    before={str(p.relative_to(ROOT)):C.sha256(p) for p in protected_paths}
    rows=load(write_protocol=False)
    selected=json.loads((OUT/'exposure/selection.json').read_text())['selected']
    models=[]
    for sigma in [1.4,.8,0]:
        m=copy.deepcopy(selected);m.update(id=f'frozen_tone_sigma_{sigma:g}',sigma=sigma)
        models.append(m)
    write(D/'protocol.json',dict(question='Is the published exposure/gain image too dark or blurred?',
        source_hashes=before,models=models,
        design='Frozen poses, support and tone parameters; vary only Gaussian blur. No fitting or replacement of the published model.',
        splits='Original validation and test temporal blocks. The test figures have already been inspected; this is a diagnostic, not a fresh independent test.',
        displayed_frames=DISPLAYED,
        brightness='Signed bias = mean(sim - real) in 0-255 units; negative means darker. Also report absolute per-frame bias to avoid cancellation.',
        sharpness='Known applied Gaussian sigma; gradient ratio and edge recall are detail/contrast diagnostics, not independent optical blur measurements. Noise, contrast and scene changes affect them.',
        scopes='Same fixed matched-background support and full image, excluding 5 border pixels. Radial regions use distance to the calibrated principal point: <=100, 100-180, >180 px.'))
    records=[]
    def one(pair):
        i,model=pair;r=rows[i];pred=apply(r['rgb'],model,r['run']);out=[]
        for scope in ['full','support']:
            if scope=='support' and not r['supported']:continue
            value=measure(r['real'],pred,r['mask'] if scope=='support' else None)
            out.append(dict(model=model['id'],run=r['run'],index=r['index'],split=r['split'],scope=scope,**value))
        return out
    ids=[i for i,r in enumerate(rows) if r['split'] in ['validation','test']]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for value in pool.map(one,[(i,m) for m in models for i in ids]):records.extend(value)
    summary=dict(models={})
    for m in models:
        data={}
        for sp in ['validation','test']:
            data[sp]={}
            for scope in ['support','full']:
                per_run={run:average([r for r in records if r['model']==m['id'] and r['split']==sp and r['scope']==scope and r['run']==run]) for run in RUNS}
                keys=[k for k in next(iter(per_run.values())) if k!='frames']
                equal={}
                for k in keys:
                    values=[v[k] for v in per_run.values() if v.get(k) is not None]
                    equal[k]=float(np.mean(values)) if values else None
                data[sp][scope]=dict(equal_run=equal,per_run=per_run,frames=sum(v['frames'] for v in per_run.values()))
        summary['models'][m['id']]=data
    displayed=[];display_ids=[]
    for run,index in DISPLAYED.items():
        i=next(i for i,r in enumerate(rows) if r['run']==run and r['index']==index)
        display_ids.append(i)
        displayed.append(dict(next(r for r in records if r['run']==run and r['index']==index and r['model']==models[0]['id'] and r['scope']=='full')))
    summary['displayed_frames']=displayed
    summary['existing_validation_candidates']=[dict(id=m['id'],sigma=m['sigma'],**m['validation']['equal_run']) for m in json.loads((OUT/'exposure/candidates.json').read_text()) if m['family'].startswith('metadata')]
    summary['protected_inputs_unchanged']=all(C.sha256(ROOT/p)==h for p,h in before.items())
    assert summary['protected_inputs_unchanged']
    write(D/'summary.json',summary)
    with (D/'per_frame_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    sheet(rows,models,D/'blur_check.png',display_ids)
    for run in RUNS:
        eligible=[i for i,r in enumerate(rows) if r['run']==run and r['split']=='test' and r['supported']]
        chosen=[eligible[j] for j in np.linspace(0,len(eligible)-1,4,dtype=int)]
        sheet(rows,models,D/f'{run}_blur_check.png',chosen)
    event('appearance_darkness_blur_audit',path=str(D.relative_to(ROOT)),protected_inputs_unchanged=True,
          reason='User noticed the selected MAE winner looks darker and blurrier than real footage.',
          no_model_selection=True)
    for m in models:
        print(m['id'],json.dumps(summary['models'][m['id']]['test']['full']['equal_run']),flush=True)


if __name__=='__main__':main()
