"""Focused offline natural-video screening. Lossless RGB references, shared TS assets.
Run prepare first, inspect contact sheets, then score. No timed playback here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np
from streaming_common import ROOT, dump
sys.path.insert(0,str(ROOT/'client'))
from streaming_client import load_model
from evaluate_x4_video import metrics, metric_checks

CFG=json.loads((ROOT/'configs/natural-study.json').read_text())
OUT=ROOT/CFG['output']
COMMANDS=[]


def run(cmd):
    cmd=list(map(str,cmd)); COMMANDS.append(cmd)
    dump(OUT/'offline-commands.json',COMMANDS)
    return subprocess.check_output(cmd,stderr=subprocess.PIPE)


def sha(path):
    with open(path,'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def decode(path,w,h,n):
    raw=run(['ffmpeg','-v','error','-i',path,'-an','-fps_mode','passthrough','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
    assert len(raw)==n*w*h*3, (path,len(raw),n*w*h*3)
    return np.frombuffer(raw,np.uint8).reshape(n,h,w,3)


def csvwrite(path,rows):
    with open(path,'w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def prepare():
    provenance=[]
    for clip in CFG['clips']:
        folder=OUT/clip['id']; folder.mkdir(exist_ok=True)
        source=ROOT/clip['source']
        probe=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_format','-of','json',source]))
        s=probe['streams'][0]; assert s['width']>=640 and s['height']>=360
        n=clip['frames']; ref=folder/'reference.nut'
        # All selected sources are 16:9. Reject an accidental aspect ratio change.
        assert abs(s['width']/s['height']-16/9)<.01
        cmd=['ffmpeg','-v','error','-ss',clip['start'],'-i',source,'-map','0:v:0','-an',
             '-vf','setpts=PTS-STARTPTS,fps=30,scale=640:360:flags=bicubic,format=bgr0',
             '-frames:v',n,'-c:v','ffv1','-level','3','-threads','1',ref]
        if not ref.exists(): run(cmd)
        rgb=decode(ref,640,360,n)
        np.save(folder/'reference.npy',rgb)
        alignment=[dict(frame=i,reference_pts=i/30,source_nominal_time=clip['start']+i/30,
                        reference_rgb_sha256=hashlib.sha256(f.tobytes()).hexdigest()) for i,f in enumerate(rgb)]
        csvwrite(folder/'reference-alignment.csv',alignment)
        panels=[]
        for i in (0,n//2,n-1):
            panel=rgb[i].copy();cv2.putText(panel,f'{clip["id"]} frame {i}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            panels.append(panel)
        cv2.imwrite(str(folder/'contact.png'),np.concatenate(panels,axis=1)[...,::-1])
        motion=float(np.abs(rgb[1:].astype(np.int16)-rgb[:-1].astype(np.int16)).mean())
        provenance.append(dict(**clip,sha256=sha(source),bytes=source.stat().st_size,probe=probe,
            reference_command=list(map(str,cmd)),reference_sha256=sha(ref),reference_array_sha256=sha(folder/'reference.npy'),
            frame_count=n,duration=n/30,mean_abs_temporal_rgb_difference=motion,
            modifications='trim; drop audio; normalize PTS; nearest-frame fps conversion to 30; bicubic downsample to 640x360; lossless FFV1 RGB storage. Already compressed source, not pristine camera ground truth.'))
    dump(OUT/'provenance.json',provenance)


def encode(clip,width,crf):
    folder=OUT/clip['id']; path=folder/f'w{width}-crf{crf}.ts'; n=clip['frames']
    cmd=['ffmpeg','-v','error','-i',folder/'reference.nut','-an','-frames:v',n,
         '-vf',f'scale={width}:{width*9//16}:flags=bicubic:in_range=full:out_range=limited:out_color_matrix=bt709,format=yuv420p',
         '-c:v','libx264','-preset',CFG['preset'],'-tune','zerolatency','-crf',crf,'-threads','1',
         '-g','30','-keyint_min','30','-sc_threshold','0','-bf','0','-refs','1',
         '-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
         '-mpegts_copyts','1','-muxdelay','0','-muxpreload','0','-flush_packets','1','-f','mpegts',path]
    if not path.exists():run(cmd)
    probe=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_packets','-show_streams','-of','json',path]))
    packets=probe['packets']; assert len(packets)==n
    pts=[float(p['pts_time']) for p in packets]
    np.testing.assert_allclose(np.array(pts)-pts[0],np.arange(n)/30,atol=2e-6,rtol=0)
    pos=[int(p['pos']) for p in packets];assert pos==sorted(set(pos)) and all(v%188==0 for v in pos)
    meta=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size,
              packet_bytes=sum(int(p['size']) for p in packets),width=width,height=width*9//16,
              frames=n,pts=pts,offsets=[0]+pos[1:]+[path.stat().st_size],command=list(map(str,cmd)),crf=crf)
    dump(path.with_suffix('.json'),meta)
    return path,meta


def score(only_extra=None):
    torch,model=load_model(CFG)
    dump(OUT/'metric-checks.json',dict(checks=metric_checks(),protocol='RGB uint8 PSNR float64; SSIM per-channel 11x11 Gaussian sigma1.5 valid interior; mean frame metrics; no extra crop'))
    for clip in CFG['clips']:
        folder=OUT/clip['id']; n=clip['frames']; hr=np.load(folder/'reference.npy',mmap_mode='r')
        points=[(w,q) for w in (320,640) for q in CFG['crfs']] if only_extra is None else [(640,only_extra[clip['id']])]
        for w,crf in points:
            target=folder/f'w{w}-crf{crf}-quality.json'
            if target.exists(): continue
            path,meta=encode(clip,w,crf)
            lr=decode(path,w,w*9//16,n)
            rows=[];chunks=[]
            methods=['direct'] if w==640 else ['bicubic','nanovsr']
            recon={m:np.lib.format.open_memmap(folder/f'w{w}-crf{crf}-{m}.npy',mode='w+',dtype=np.uint8,shape=(n,360,640,3)) for m in methods}
            for first in range(0,n,3):
                last=min(first+3,n)
                if w==640:
                    recon['direct'][first:last]=lr[first:last]
                else:
                    recon['bicubic'][first:last]=[cv2.resize(f,(640,360),interpolation=cv2.INTER_CUBIC) for f in lr[first:last]]
                    tick=time.monotonic()
                    x=torch.from_numpy(lr[first:last].copy()).permute(0,3,1,2).unsqueeze(0).float().cuda()/255
                    torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    with torch.inference_mode():
                        a.record(); y=model(x); b.record();b.synchronize()
                        assert y.shape==(1,last-first,3,720,1280) and torch.isfinite(y).all()
                        native=y[0].clamp(0,1).mul(255).round().byte().permute(0,2,3,1).cpu().numpy()
                    r=time.monotonic()
                    recon['nanovsr'][first:last]=[cv2.resize(f,(640,360),interpolation=cv2.INTER_CUBIC) for f in native]
                    chunks.append(dict(first=first,model_ms=a.elapsed_time(b),resize_s=time.monotonic()-r,processing_s=time.monotonic()-tick))
                    del x,y,native
                for m in methods:
                    for i in range(first,last):
                        psnr,ssim=metrics(recon[m][i],hr[i])
                        rows.append(dict(clip=clip['id'],width=w,crf=crf,method=m,frame=i,pts=i/30,psnr_db=psnr,ssim=ssim))
            for arr in recon.values():arr.flush()
            csvwrite(folder/f'w{w}-crf{crf}-frames.csv',rows)
            summary=[]
            for m in methods:
                subset=[r for r in rows if r['method']==m]
                summary.append(dict(clip=clip['id'],width=w,crf=crf,method=m,frames=n,bytes=meta['bytes'],
                    kbps=meta['bytes']*8/(n/30)/1000,psnr_db=float(np.mean([r['psnr_db'] for r in subset])),
                    ssim=float(np.mean([r['ssim'] for r in subset]))))
            dump(target,dict(asset=meta,summary=summary,chunks=chunks,peak_gpu_allocated=torch.cuda.max_memory_allocated()))
            print(json.dumps(summary),flush=True)
    summaries=[]
    for p in OUT.glob('*/w*-quality.json'):summaries.extend(json.loads(p.read_text())['summary'])
    csvwrite(OUT/'quality-summary.csv',summaries)
    dump(OUT/'quality-summary.json',summaries)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','score','extra']);p.add_argument('--extra-config')
    a=p.parse_args()
    if (OUT/'offline-commands.json').exists():COMMANDS=json.loads((OUT/'offline-commands.json').read_text())
    if a.stage=='prepare':prepare()
    else:score(None if a.stage=='score' else json.loads(Path(a.extra_config).read_text()))
