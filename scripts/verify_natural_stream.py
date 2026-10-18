"""Verify a diagnostic natural replay against offline decoded/reconstructed frames."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

folder=Path(sys.argv[1]); c=json.loads((folder/'client.json').read_text());s=json.loads((folder/'server.json').read_text())
assert c['status']==s['status']=='ok' and c['received_eos']
asset=Path(c['config']['encoded_asset']);meta=json.loads(asset.with_suffix('.json').read_text())
assert c['payload_bytes']==s['payload_bytes']==meta['bytes']
assert c['protocol_bytes']==s['protocol_bytes']
assert c['payload_sha256']==s['payload_sha256']==meta['sha256']
assert c['decoded_frames']==len(c['frames'])==meta['frames']
assert c['queue_peak']<=c['config']['queue_frames']
w,h=c['header']['size'];n=meta['frames']
raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(asset),'-an','-fps_mode','passthrough','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
assert len(raw)==n*w*h*3
frames=np.frombuffer(raw,np.uint8).reshape(n,h,w,3)
recon=np.load(asset.with_name(asset.stem+'-'+c['mode']+'.npy'),mmap_mode='r')
for i,f in enumerate(c['frames']):
    assert f['index']==i and abs(f['source_pts']-i/30)<2e-6
    assert f['decoded_sha256']==hashlib.sha256(frames[i].tobytes()).hexdigest()
    assert f['output_sha256']==hashlib.sha256(recon[i].tobytes()).hexdigest(), ('reconstruction mismatch',i)
print(f'PASS {folder}: {n} frame timestamps, bitstream bytes, decoded hashes, all reconstruction hashes match offline outputs')
