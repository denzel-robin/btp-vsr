"""Short native-x4 320x180 chunk3 capacity check; no resolution/buffer fallback."""
import sys
import time
from pathlib import Path
import json
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'client'))
from streaming_client import load_model
from streaming_common import dump

c = dict(checkpoint='external/nanovsr/checkpoints/nanovsr_226k.pth')
start=time.monotonic()
torch, model=load_model(c)
load_s=time.monotonic()-start
rng=np.random.default_rng(2026)
frames=rng.integers(0,256,(3,180,320,3),dtype=np.uint8)
rows=[]
for i in range(12):
    t=time.monotonic()
    x=torch.from_numpy(frames.copy()).permute(0,3,1,2).unsqueeze(0).float().cuda()/255
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        a.record(); y=model(x); b.record(); b.synchronize()
        assert y.shape==(1,3,3,720,1280) and torch.isfinite(y).all()
        output=y[0].clamp(0,1).mul(255).round().byte().permute(0,2,3,1).cpu().numpy()
    r=time.monotonic()
    resized=[cv2.resize(f,(640,360),interpolation=cv2.INTER_CUBIC) for f in output]
    end=time.monotonic()
    rows.append(dict(iteration=i,model_ms=a.elapsed_time(b),application_processing_ms=1000*(end-t),resize_ms=1000*(end-r)))
    del x,y,output,resized
report=dict(input='synthetic seeded random; capacity only',input_shape=[1,3,3,180,320],native_output=[1,3,3,720,1280],playback=[640,360],gpu=torch.cuda.get_device_name(),torch=torch.__version__,load_s=load_s,rows=rows,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),device_total=torch.cuda.get_device_properties(0).total_memory)
report['warm_processing_fps']=3000/np.mean([v['application_processing_ms'] for v in rows[2:]])
report['maintains_30fps_processing']=bool(report['warm_processing_fps']>=30)
dump('results/natural-2026-10-15/capacity.json',report)
print(json.dumps(report,indent=2))
