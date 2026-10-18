"""Permit one lower-buffer test only after three viable 350 ms repetitions."""
import json
from streaming_common import ROOT,dump
out=ROOT/'results/natural-2026-10-15'
rows=json.loads((out/'streaming-summary.json').read_text())
rows=[r for r in rows if not r['verification'] and not r['excluded_reason'] and r['buffer_ms']==350 and r['clip']=='road']
evidence=[];chosen=None
for mode,warm in [('nanovsr',3),('direct',0),('bicubic',0)]:
    rr=[r for r in rows if r['mode']==mode and r['warmup_chunks']==warm]
    assert len(rr)==3,(mode,len(rr))
    viable=all(r['drops']==0 and r['latency_p95_ms']<450 for r in rr)
    evidence.append(dict(mode=mode,warmup_chunks=warm,runs=[r['run'] for r in rr],p95_ms=[r['latency_p95_ms'] for r in rr],drops=[r['drops'] for r in rr],viable=viable))
    if chosen is None and viable:chosen=(mode,warm)
assert chosen is not None,'No viable lower-buffer candidate; stop here'
mode,warm=chosen
crfs=json.loads((ROOT/'configs/natural-matched-crf.json').read_text())
w,q=(640,crfs['road']) if mode=='direct' else (320,23)
point=dict(id=f'road-{mode}-lower150',asset=f'results/natural-2026-10-15/road/w{w}-crf{q}.ts',mode=mode,warmup_chunks=warm,warmup_pipeline=bool(warm),buffer_s=.15,repetitions=3)
dump(ROOT/'configs/natural-lower-buffer-selection.json',[point])
dump(out/'lower-buffer-selection.json',dict(criterion='All 3 initial runs: zero drops and latency p95 <450 ms; prefer positive NanoVSR road case if viable, otherwise direct, then bicubic',evidence=evidence,selected=point))
print(json.dumps(point,indent=2))
