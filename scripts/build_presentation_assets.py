"""Consolidate immutable study evidence; no model execution or metric recomputation."""
import csv,hashlib,json,shutil
from collections import Counter
from pathlib import Path
import numpy as np
from streaming_common import ROOT,dump
out=ROOT/'presentation_assets'; study=ROOT/'results/natural-2026-10-15'
def csvwrite(path,rows):
    with open(path,'w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
quality=json.loads((study/'quality-summary.json').read_text());runs=json.loads((study/'streaming-summary.json').read_text())
valid=[r for r in runs if not r['verification'] and not r['excluded_reason']]
assert len(valid)==39
inventory=[]
for r in runs:
    inventory.append(dict(run=r['run'],status='excluded' if r['excluded_reason'] else 'verification' if r['verification'] else 'completed',reason=r['excluded_reason']))
for r in json.loads((study/'failed-runs.json').read_text()):inventory.append(dict(run=r['run'],status='failed',reason=str(r['errors'])))
csvwrite(out/'data/run-inventory.csv',inventory)
counts=Counter((r['clip'],r['mode'],r['warmup_chunks'],r['buffer_ms']) for r in valid)
assert len(counts)==13 and all(n==3 for n in counts.values())
assert any('continuation/flowers-direct' in r['run'] for r in valid)
dump(out/'data/inventory-summary.json',dict(counts=dict(Counter(r['status'] for r in inventory)),missing_repetitions=0,groups=[dict(clip=k[0],mode=k[1],warmup_chunks=k[2],buffer_ms=k[3],repetitions=n) for k,n in counts.items()],replacement=[r['run'] for r in valid if r['clip']=='flowers' and r['mode']=='direct']))
# Independently check summary values against raw records before exporting.
for r in valid:
    c=json.loads((study/r['run']/'client.json').read_text());s=json.loads((study/r['run']/'server.json').read_text())
    lat=[f['application_latency_s']*1000 for f in c['frames'] if not f['dropped']]
    for key,p in [('latency_p50_ms',50),('latency_p95_ms',95),('latency_max_ms',100)]:assert abs(r[key]-np.percentile(lat,p))<1e-8
    assert r['payload_bytes']==c['payload_bytes']==s['payload_bytes']
    assert c['config']['playback_size']==[640,360]
    assert r['submitted']==c['unique_render_submissions'] and r['drops']==c['dropped_frames']
    asset=json.loads((ROOT/c['config']['encoded_asset']).with_suffix('.json').read_text())
    assert s['asset_sha256']==asset['sha256']
    if r['mode']!='direct':assert c['config']['encoded_asset']==f"results/natural-2026-10-15/{r['clip']}/w320-crf23.ts"
csvwrite(out/'data/valid-repetitions.csv',valid)
csvwrite(out/'data/quality-all-points.csv',quality)
matched=json.loads((ROOT/'configs/natural-matched-crf.json').read_text())
rows=[]
for clip in ('flowers','road','waterfall'):
    low=next(q for q in quality if q['clip']==clip and q['method']=='bicubic' and q['crf']==23)
    for mode in ('direct','bicubic','nanovsr'):
        crf=matched[clip] if mode=='direct' else 23
        q=next(q for q in quality if q['clip']==clip and q['method']==mode and q['crf']==crf)
        for warm,buffer in sorted(set((r['warmup_chunks'],r['buffer_ms']) for r in valid if r['clip']==clip and r['mode']==mode)):
            rs=[r for r in valid if r['clip']==clip and r['mode']==mode and r['warmup_chunks']==warm and r['buffer_ms']==buffer]
            assert len(rs)==3
            row=dict(config_id=f'{clip}-{mode}-crf{crf}-'+('pipelinewarm3' if warm else 'cold')+f'-buffer{buffer:g}ms',clip=clip,mode=mode,crf=crf,warmup_chunks=warm,buffer_ms=buffer,tx_width_px=q['width'],tx_height_px=q['width']*9//16,playback_width_px=640,playback_height_px=360,chunk_frames=3 if mode=='nanovsr' else 1,ts_bytes=q['bytes'],ts_kbps=q['kbps'],rate_delta_vs_lr_percent=(q['bytes']/low['bytes']-1)*100,psnr_db=q['psnr_db'],ssim=q['ssim'],repetitions=3,submissions_total=sum(r['submitted'] for r in rs),drops_total=sum(r['drops'] for r in rs),completeness_percent=100*sum(r['submitted'] for r in rs)/sum(r['frames'] for r in rs))
            for key in ('processing_fps','processing_ms_per_frame','model_ms_per_frame','resize_ms_per_frame','active_submission_fps','latency_p50_ms','latency_p95_ms','latency_max_ms','queue_age_p95_ms','queue_age_max_ms','model_load_s','warmup_s','first_chunk_ms'):
                row[key+'_mean']=float(np.mean([r[key] for r in rs]));row[key+'_min']=min(r[key] for r in rs);row[key+'_max']=max(r[key] for r in rs)
            row['peak_gpu_allocated_MiB']=max(r['peak_gpu_allocated'] for r in rs)/2**20
            row['peak_gpu_reserved_MiB']=max(r['peak_gpu_reserved'] for r in rs)/2**20
            rows.append(row)
csvwrite(out/'data/comparison-detailed.csv',rows)
concise=[]
for r in rows:
    x={k:r[k] for k in ('config_id','clip','mode','crf','warmup_chunks','buffer_ms','ts_bytes','ts_kbps','rate_delta_vs_lr_percent','psnr_db','ssim','repetitions','submissions_total','drops_total','completeness_percent','peak_gpu_allocated_MiB','peak_gpu_reserved_MiB')}
    for k in ('processing_ms_per_frame','model_ms_per_frame','resize_ms_per_frame','processing_fps'):
        x[k+'_mean']=r[k+'_mean']
    for k in ('active_submission_fps','latency_p50_ms','latency_p95_ms','latency_max_ms','queue_age_max_ms'):
        x[k+'_range']=f"{r[k+'_min']:.3f} to {r[k+'_max']:.3f}"
    concise.append(x)
csvwrite(out/'data/comparison.csv',concise)
for name in ('ATTRIBUTION.md','provenance.json','crop-selection.json','excluded-runs.json','failed-runs.json'):
    shutil.copy2(study/name,out/('ATTRIBUTION.md' if name=='ATTRIBUTION.md' else 'data/'+name))
for clip in ('flowers','road','waterfall'):shutil.copy2(study/clip/'comparison-crops.png',out/f'crops/{clip}.png')
# Summary by ten-second source window, including dropped frames separately.
p=out/'evidence/stability/client.json'
if p.exists():
 c=json.loads(p.read_text());windows=[]
 for start in range(0,len(c['frames']),300):
    fs=c['frames'][start:start+300];shown=[f for f in fs if not f['dropped']];lat=[f['application_latency_s']*1000 for f in shown];ages=[f['queue_age_s']*1000 for f in fs]
    row=dict(source_start_s=start/30,frames=len(fs),submissions=len(shown),drops=len(fs)-len(shown),queue_age_p95_ms=float(np.percentile(ages,95)),queue_age_max_ms=max(ages),latency_p50_ms=float(np.percentile(lat,50)) if lat else None,latency_p95_ms=float(np.percentile(lat,95)) if lat else None,latency_max_ms=max(lat) if lat else None,active_submission_fps=(len(shown)-1)/(shown[-1]['render_event']-shown[0]['render_event']) if len(shown)>1 else 0)
    windows.append(row)
 csvwrite(out/'data/stability-windows.csv',windows)
 shown=[f for f in c['frames'] if not f['dropped']];lat=[f['application_latency_s']*1000 for f in shown]
 dump(out/'data/stability-summary.json',dict(status=c['status'],errors=c['errors'],frames=len(c['frames']),decoded=c['decoded_frames'],submissions=c['unique_render_submissions'],drops=c['dropped_frames'],submitted_fraction_of_planned_9000_percent=100*c['unique_render_submissions']/9000,processed_prefix_completeness_percent=100*c['unique_render_submissions']/len(c['frames']),active_submission_fps=(len(shown)-1)/(shown[-1]['render_event']-shown[0]['render_event']),processing_fps=c['processing_throughput_fps'],latency_p50_ms=float(np.percentile(lat,50)),latency_p95_ms=float(np.percentile(lat,95)),latency_max_ms=max(lat),queue_peak=c['queue_peak'],queue_age_max_ms=max(f['queue_age_s'] for f in c['frames'])*1000,gpu_peak_allocated_bytes=c['gpu_peak_allocated_bytes'],gpu_peak_reserved_bytes=c['gpu_peak_reserved_bytes'],received_eos=c.get('received_eos',False)))
dump(out/'evidence/consolidation-audit.json',dict(valid_timing_runs=39,quality_rows=31,summary_latency_and_bytes_recomputed=True,common_playback_size=[640,360],bc_identical_asset_verified=True,missing_repetitions=0))
print('Consolidated and independently checked existing comparison.')
