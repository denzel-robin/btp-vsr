"""Audit the single native720 comparison and create a diagnostic replay config."""
import csv,json,hashlib,sys
import numpy as np
from streaming_common import ROOT,dump
from evaluate_x4_video import metrics
out=ROOT/'results/native720-road-2026-10-18';pub=ROOT/'presentation_assets/native720'
r=json.loads((pub/'quality.json').read_text());assert r['playback_size']==[1280,720]
rows=list(csv.DictReader(open(pub/'frame-metrics.csv')));assert len(rows)==900
hr=np.load(out/'reference.npy',mmap_mode='r');assert hr.shape==(300,720,1280,3)
checks=0
for summary in r['summary']:
    mode=summary['method'];width=1280 if mode=='direct' else 320
    asset=out/f'w{width}-crf23.ts';meta=json.loads(asset.with_suffix('.json').read_text())
    assert hashlib.sha256(asset.read_bytes()).hexdigest()==meta['sha256']==summary['asset_sha256']
    assert asset.stat().st_size==summary['bytes'] and summary['kbps']==summary['bytes']*8/10/1000
    subset=[v for v in rows if v['method']==mode];assert [int(v['frame']) for v in subset]==list(range(300))
    assert abs(np.mean([float(v['psnr_db']) for v in subset])-summary['psnr_db'])<1e-10
    assert abs(np.mean([float(v['ssim']) for v in subset])-summary['ssim'])<1e-10
    a=np.load(out/f'w{width}-crf23-{mode}.npy',mmap_mode='r');assert a.shape==hr.shape
    for i in (0,2,3,299):
        p,s=metrics(a[i],hr[i]);assert abs(p-float(subset[i]['psnr_db']))<1e-10 and abs(s-float(subset[i]['ssim']))<1e-10;checks+=1
    from demo_stats import DemoStats
    cfg=json.loads((pub/f'{mode}.json').read_text());panel=DemoStats(cfg,mode);assert len(panel.quality)==3
assert r['summary'][1]['asset_sha256']==r['summary'][2]['asset_sha256']
cfg=json.loads((pub/'nanovsr.json').read_text());cfg.update(verification=True,presentation_stats=False)
dump(out/'verify-config.json',cfg)
dump(pub/'verification.json',dict(status='pass',summary_rows=3,all_frame_rows=900,independent_boundary_frame_rechecks=checks,lr_bitstream_identical=True,native_resolution=[1280,720],streaming_verification='see separately retained live audit'))
print('PASS: source-sized reference, 900 metric rows, exact LR bitstream sharing, aggregates and 12 boundary metric rechecks; native stats load for all three modes.')
