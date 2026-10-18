"""Independently recheck counts, timestamps and sampled saved-array RGB scores."""
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
from evaluate_x4_video import metrics,metric_checks
from streaming_common import ROOT

out=ROOT/'results/natural-2026-10-15'
print(metric_checks())
checked=0
for file in sorted(out.glob('*/w*-quality.json')):
    q=json.loads(file.read_text());meta=q['asset'];n=meta['frames'];folder=file.parent
    frames=list(csv.DictReader(file.with_name(file.name.replace('-quality.json','-frames.csv')).open()))
    hr=np.load(folder/'reference.npy',mmap_mode='r')
    assert len(hr)==n
    for summary in q['summary']:
        method=summary['method'];rows=[r for r in frames if r['method']==method]
        assert [int(r['frame']) for r in rows]==list(range(n))
        assert abs(float(np.mean([float(r['psnr_db']) for r in rows]))-summary['psnr_db'])<1e-10
        assert abs(float(np.mean([float(r['ssim']) for r in rows]))-summary['ssim'])<1e-10
        recon=np.load(folder/f'w{meta["width"]}-crf{meta["crf"]}-{method}.npy',mmap_mode='r')
        assert recon.shape==hr.shape
        for i in (0,2,3,149,150,n-1):
            psnr,ssim=metrics(recon[i],hr[i]);r=rows[i]
            assert abs(psnr-float(r['psnr_db']))<1e-10 and abs(ssim-float(r['ssim']))<1e-10
            checked+=1
print(f'PASS: {checked} independent metric rechecks including chunk boundaries and final frames')
