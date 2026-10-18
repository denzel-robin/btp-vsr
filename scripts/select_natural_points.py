"""Choose one extra direct CRF per clip using measured log-rate interpolation.
Targets each clip's low-resolution CRF23 bytes; no exhaustive sweep.
"""
import json
import math
from pathlib import Path
from streaming_common import ROOT,dump

out=ROOT/'results/natural-2026-10-15'
matched={};evidence=[]
for clip in ('flowers','road','waterfall'):
    folder=out/clip
    target=json.loads((folder/'w320-crf23-quality.json').read_text())['summary'][0]['kbps']
    rates={q:json.loads((folder/f'w640-crf{q}-quality.json').read_text())['summary'][0]['kbps'] for q in (23,28)}
    q=round(28+5*math.log(target/rates[28])/math.log(rates[28]/rates[23]),1)
    assert 18<=q<=40
    matched[clip]=q
    evidence.append(dict(clip=clip,target_lr23_kbps=target,direct23_kbps=rates[23],direct28_kbps=rates[28],extra_crf=q))
dump(ROOT/'configs/natural-matched-crf.json',matched)
dump(out/'matched-rate-selection.json',evidence)
print(json.dumps(evidence,indent=2))
