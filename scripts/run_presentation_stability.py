"""One serial GUI stability trial, sampled host telemetry; no screen capture."""
import json,subprocess,time
from pathlib import Path
from streaming_common import ROOT,dump
out=ROOT/'presentation_assets/evidence/stability'
if out.exists():raise SystemExit('Refusing to overwrite prior stability trial')
telemetry=ROOT/'presentation_assets/evidence/gpu-telemetry.csv'
cmd=[str(ROOT/'.venv/bin/python'),'scripts/run_streaming_demo.py','--mode','nanovsr','--config','presentation_assets/configs/stability.json','--output',str(out),'--timeout','360']
with open(ROOT/'presentation_assets/evidence/stability-launch.log','w') as log,open(telemetry,'w') as gpu:
    gpu.write('monotonic_s,gpu_timestamp,temperature_C,utilization_percent,memory_used_MiB,power_W\n');gpu.flush()
    p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    while p.poll() is None:
        r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
        gpu.write(f'{time.monotonic()},'+r.stdout.strip()+'\n');gpu.flush()
        time.sleep(5)
    dump(ROOT/'presentation_assets/evidence/stability-launch.json',dict(command=cmd,returncode=p.returncode))
raise SystemExit(p.returncode)
