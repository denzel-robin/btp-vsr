"""Lossy offline comparison preview from already-scored arrays; NEVER a live capture."""
import json,subprocess
from pathlib import Path
import cv2,numpy as np
from streaming_common import ROOT,dump
out=ROOT/'presentation_assets';folder=ROOT/'results/natural-2026-10-15/road'
paths=[folder/'reference.npy',folder/'w640-crf30.4-direct.npy',folder/'w320-crf23-bicubic.npy',folder/'w320-crf23-nanovsr.npy']
labels=['Reference 640x360','Direct 640x360 CRF30.4','Bicubic 320x180 -> 640x360','NanoVSR native x4 -> resize 640x360']
arrays=[np.load(p,mmap_mode='r') for p in paths]
target=out/'demo/OFFLINE-road-comparison.mp4'
if target.exists():raise SystemExit('Refusing to overwrite backup')
cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x820','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-threads','1',str(target)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
try:
 for i in range(300):
    canvas=np.zeros((820,1280,3),np.uint8)
    cv2.putText(canvas,'OFFLINE comparison - precomputed frames, NOT live playback',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
    for j,(a,label) in enumerate(zip(arrays,labels)):
        x=(j%2)*640;y=50+(j//2)*385
        cv2.putText(canvas,label,(x+8,y+18),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        canvas[y+25:y+385,x:x+640]=a[i]
    p.stdin.write(canvas.tobytes())
finally:p.stdin.close()
assert p.wait()==0
dump(out/'demo/backup-protocol.json',dict(kind='offline precomputed comparison; not screen capture or live playback',frames=300,fps=30,source_arrays=[str(p.relative_to(ROOT)) for p in paths],command=cmd,scoring='Preview lossy encoding never scored; metrics from original RGB arrays.'))
print(target)
