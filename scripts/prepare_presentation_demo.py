"""Prepare a five-minute stream-copy loop and labelled demo configurations."""
import hashlib,json,subprocess
from pathlib import Path
from streaming_common import ROOT,dump
out=ROOT/'presentation_assets'; base=json.loads((ROOT/'configs/natural-road-nanovsr.json').read_text())
for mode in ('direct','bicubic','nanovsr'):
    cfg=dict(base,demo_labels=True)
    if mode!='nanovsr':cfg.update(warmup_chunks=0,warmup_pipeline=False)
    if mode=='direct':cfg.update(encoded_asset='results/natural-2026-10-15/road/w640-crf30.4.ts',crf=30.4)
    dump(out/f'configs/{mode}.json',cfg)
p=out/'demo/road-loop-300s.ts'
cmd=['ffmpeg','-v','error','-stream_loop','29','-i',str(ROOT/base['encoded_asset']),'-map','0:v:0','-an','-c:v','copy','-frames:v','9000','-mpegts_copyts','1','-muxdelay','0','-muxpreload','0',str(p)]
if not p.exists():subprocess.run(cmd,check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_packets','-select_streams','v:0','-of','json',str(p)]))
packets=probe['packets'];pts=[float(v['pts_time']) for v in packets];pos=[int(v['pos']) for v in packets]
assert len(pts)==9000 and all(abs(v-pts[0]-i/30)<2e-6 for i,v in enumerate(pts))
assert pos==sorted(set(pos)) and all(v%188==0 for v in pos)
meta=dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,width=320,height=180,frames=9000,pts=pts,offsets=[0]+pos[1:]+[p.stat().st_size],command=cmd,crf=23)
dump(p.with_suffix('.json'),meta)
cfg=dict(base,source=str(p.relative_to(ROOT)),encoded_asset=str(p.relative_to(ROOT)),frames=9000,stability_monitor=True)
dump(out/'configs/stability.json',cfg)
dump(out/'evidence/loop-protocol.json',dict(command=cmd,frames=9000,duration_s=300,loop_boundary_frames=list(range(300,9000,300)),state='Native recurrent state resets every chunk3, including each 300-frame loop boundary; decoder remains connected, each loop starts on original IDR. No model reload or pacing reset.',logging='At most 9000 scalar frame records and 3000 chunk records; 30 progress windows; no RGB frames, hashes or payload capture in timed path.',early_stop='Three consecutive 300-frame windows with >1.5s end schedule age or >150 drops each.'))
print('Prepared 9000-frame stream-copy loop; exact CFR timestamps checked.')
