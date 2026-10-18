"""Aggregate offline quality and per-run timings; generate fixed, declared crops."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from streaming_common import ROOT,dump

OUT=ROOT/'results/natural-2026-10-15'


def csvwrite(path,rows):
    if not rows:return
    with open(path,'w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def percentile(values,p):
    return float(np.percentile(values,p)) if values else None


def main():
    quality=[]
    for path in sorted(OUT.glob('*/w*-quality.json')):
        quality.extend(json.loads(path.read_text())['summary'])
    dump(OUT/'quality-summary.json',quality);csvwrite(OUT/'quality-summary.csv',quality)
    exclusions=json.loads((OUT/'excluded-runs.json').read_text()) if (OUT/'excluded-runs.json').exists() else {}
    rows=[]
    failures=[]
    for path in sorted(OUT.glob('stream*/**/client.json')):
        c=json.loads(path.read_text());cfg=c['config'];frames=c['frames']
        if c['status']!='ok':
            failures.append(dict(run=str(path.parent.relative_to(OUT)),status=c['status'],errors=c['errors'],decoded=c['decoded_frames'],submitted=c['unique_render_submissions']))
            continue
        rendered=[f for f in frames if not f['dropped']]
        lat=[f['application_latency_s']*1000 for f in rendered]
        ages=[f['queue_age_s']*1000 for f in frames]
        last=max([f['render_event'] for f in rendered] or [c['header']['epoch']])
        first=min([f['render_event'] for f in rendered] or [last])
        duration=len(frames)/30
        row=dict(run=str(path.parent.relative_to(OUT)),excluded_reason=exclusions.get(str(path.parent.relative_to(OUT)),''),clip=Path(cfg['encoded_asset']).parent.name,mode=c['mode'],
             crf=cfg['crf'],warmup_chunks=cfg.get('warmup_chunks',0),warmup_pipeline=cfg.get('warmup_pipeline',False),verification=cfg.get('verification',True),
             headless=c['headless'],buffer_ms=cfg['playout_delay_s']*1000,frames=len(frames),submitted=c['unique_render_submissions'],
             displayed_submissions=c['unique_displayed_frames'],drops=c['dropped_frames'],completeness=c['unique_render_submissions']/len(frames),
             submitted_per_source_second=c['unique_render_submissions']/duration,
             active_submission_fps=(len(rendered)-1)/(last-first) if last>first else 0,
             session_submission_fps=len(rendered)/(last-c['header']['epoch']) if last>c['header']['epoch'] else 0,
             latency_p50_ms=percentile(lat,50),latency_p95_ms=percentile(lat,95),latency_max_ms=max(lat) if lat else None,
             first_frame_latency_ms=frames[0]['application_latency_s']*1000 if frames[0]['application_latency_s'] is not None else None,
             queue_age_p50_ms=percentile(ages,50),queue_age_p95_ms=percentile(ages,95),queue_age_max_ms=max(ages),queue_peak=c['queue_peak'],
             processing_fps=c['processing_throughput_fps'],processing_ms_per_frame=sum(x['processing_s'] for x in c['chunks'])*1000/len(frames),
             model_ms_per_frame=sum(x['model_ms'] for x in c['chunks'])/len(frames),resize_ms_per_frame=sum(x['resize_s'] for x in c['chunks'])*1000/len(frames),
             first_chunk_ms=c['chunks'][0]['processing_s']*1000,model_load_s=c.get('model_load_s',0),warmup_s=c.get('warmup_s',0),
             peak_gpu_allocated=c['gpu_peak_allocated_bytes'],peak_gpu_reserved=c['gpu_peak_reserved_bytes'],
             payload_bytes=c['payload_bytes'],protocol_bytes=c['protocol_bytes'])
        rows.append(row)
    dump(OUT/'failed-runs.json',failures)
    dump(OUT/'streaming-summary.json',rows);csvwrite(OUT/'streaming-summary.csv',rows)
    # Fixed frame 150 (5 s), coordinates chosen on the reference before metric ranking.
    boxes={'flowers':(80,210,192,128),'road':(320,170,192,128),'waterfall':(420,110,192,128)}
    match=json.loads((ROOT/'configs/natural-matched-crf.json').read_text()) if (ROOT/'configs/natural-matched-crf.json').exists() else {}
    cropmeta=[]
    for clip,box in boxes.items():
        folder=OUT/clip;x,y,w,h=box;frame=150
        paths=[('Reference',folder/'reference.npy'),('Bicubic CRF23',folder/'w320-crf23-bicubic.npy'),
               ('NanoVSR CRF23',folder/'w320-crf23-nanovsr.npy')]
        if clip in match: paths.append((f'Direct CRF{match[clip]}',folder/f'w640-crf{match[clip]}-direct.npy'))
        if not all(p.exists() for _,p in paths):continue
        panels=[]
        for label,path in paths:
            a=np.load(path,mmap_mode='r')[frame,y:y+h,x:x+w]
            image=cv2.resize(a,(w*2,h*2),interpolation=cv2.INTER_NEAREST)
            image=cv2.copyMakeBorder(image,32,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
            cv2.putText(image,label,(5,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            panels.append(image)
        cv2.imwrite(str(folder/'comparison-crops.png'),np.concatenate(panels,axis=1)[...,::-1])
        cropmeta.append(dict(clip=clip,frame=frame,reference_box=box,display_enlargement='2x nearest; not scored',methods=[p[0] for p in paths]))
    dump(OUT/'crop-selection.json',cropmeta)
    print(f'{len(quality)} quality rows; {len(rows)} streaming runs summarized')


if __name__=='__main__':main()
