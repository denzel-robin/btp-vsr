"""Build the focused timed comparison from completed quality screening."""
import json
from pathlib import Path
from streaming_common import ROOT,dump

out=ROOT/'results/natural-2026-10-15'
matched=json.loads((ROOT/'configs/natural-matched-crf.json').read_text())
points=[]
for clip in ('flowers','road','waterfall'):
    for mode,w,q,warm in [('direct',640,matched[clip],0),('bicubic',320,23,0),('nanovsr',320,23,0),('nanovsr',320,23,3)]:
        points.append(dict(id=f'{clip}-{mode}-'+('warm' if warm else 'cold'),asset=f'results/natural-2026-10-15/{clip}/w{w}-crf{q}.ts',
                           mode=mode,warmup_chunks=warm,warmup_pipeline=bool(warm),repetitions=3,buffer_s=.35))
# Interleave method order across repetitions in the saved expanded selection.
expanded=[]
for rep in range(3):
    ordered=points[rep:]+points[:rep]
    for point in ordered:
        expanded.append(dict(point,id=point['id']+f'-repeat{rep+1}',repetitions=1))
dump(ROOT/'configs/natural-final-selection.json',expanded)
