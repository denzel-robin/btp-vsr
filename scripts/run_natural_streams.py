"""Small repeatable study runner. Serial runs avoid GPU/CPU interference.
Selection file controls points, warmup, repetitions, buffer and verification.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from streaming_common import ROOT,dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('selection');p.add_argument('--output',required=True);p.add_argument('--headless',action='store_true')
    a=p.parse_args();selection=json.loads(Path(a.selection).read_text())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    dump(out/'selection.json',selection)
    for point in selection:
        for rep in range(1,point.get('repetitions',3)+1):
            c=json.loads((ROOT/'configs/streaming-smoke.json').read_text())
            asset=ROOT/point['asset'];meta=json.loads(asset.with_suffix('.json').read_text())
            c.update(source=str(asset.parent.relative_to(ROOT)/'reference.nut'),source_kind='natural_reference_from_compressed_source',
                     frames=meta['frames'],transmission_size=[320,180],playback_size=[640,360],chunk_size=3,
                     crf=meta['crf'],encoded_asset=point['asset'],verification=point.get('verification',False),
                     warmup_chunks=point.get('warmup_chunks',0),warmup_pipeline=point.get('warmup_pipeline',False),playout_delay_s=point.get('buffer_s',.35))
            tag=f'{point["id"]}-r{rep}';cfg=out/(tag+'.json');dump(cfg,c)
            cmd=[sys.executable,str(ROOT/'scripts/run_streaming_demo.py'),'--mode',point['mode'],
                 '--config',str(cfg),'--output',str(out/tag),'--port','8775']
            if a.headless:cmd.append('--headless')
            gpu_cmd=['nvidia-smi','--query-gpu=name,driver_version,memory.used,memory.total,temperature.gpu,clocks.sm,power.draw','--format=csv']
            with open(out/(tag+'-gpu-before.csv'),'w') as log:
                subprocess.run(gpu_cmd,stdout=log,stderr=log,check=False)
            subprocess.run(cmd,check=True,cwd=ROOT,timeout=200)
            with open(out/(tag+'-gpu-after.csv'),'w') as log:
                subprocess.run(gpu_cmd,stdout=log,stderr=log,check=False)
            if c['verification']:
                subprocess.run([sys.executable,str(ROOT/'scripts/verify_natural_stream.py'),str(out/tag)],check=True,cwd=ROOT)


if __name__=='__main__':main()
