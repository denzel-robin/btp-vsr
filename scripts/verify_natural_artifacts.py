"""Final integrity/count audit; does not rerun GPU work or performance trials."""
import hashlib
import json
from pathlib import Path
from streaming_common import ROOT

out=ROOT/'results/natural-2026-10-15'
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for clip in json.loads((out/'provenance.json').read_text()):
    assert sha(ROOT/clip['source'])==clip['sha256']
    folder=out/clip['id']
    assert sha(folder/'reference.nut')==clip['reference_sha256']
    assert sha(folder/'reference.npy')==clip['reference_array_sha256']
quality=json.loads((out/'quality-summary.json').read_text())
assert len(quality)==31 and sum(r['frames'] for r in quality)==9000
for path in out.glob('*/w*-quality.json'):
    asset=json.loads(path.read_text())['asset']; p=ROOT/asset['path']
    assert sha(p)==asset['sha256'] and p.stat().st_size==asset['bytes']
rows=json.loads((out/'streaming-summary.json').read_text())
valid=[r for r in rows if not r['verification'] and not r['excluded_reason']]
assert len(valid)==39
for r in valid:
    folder=out/r['run'];c=json.loads((folder/'client.json').read_text());s=json.loads((folder/'server.json').read_text())
    meta=json.loads((ROOT/c['config']['encoded_asset']).with_suffix('.json').read_text())
    n=c['config']['frames']
    assert c['status']==s['status']=='ok' and c['received_eos']
    assert c['payload_bytes']==s['payload_bytes']==meta['bytes']
    assert c['protocol_bytes']==s['protocol_bytes']
    assert c['decoded_frames']==len(c['frames'])==n
    assert [f['index'] for f in c['frames']]==list(range(n))
    assert c['unique_render_submissions']+c['dropped_frames']==n
    assert c['unique_displayed_frames']==c['unique_render_submissions'] and not c['headless']
    assert c['queue_peak']<=c['config']['queue_frames']
    assert all(f['output_sha256'] is None and f['decoded_sha256'] is None for f in c['frames'])
    assert not Path(str(folder/'client.json')+'.received.ts').exists(), 'diagnostic capture leaked into timed path'
assert sum(r['submitted'] for r in valid)==11340 and sum(r['drops'] for r in valid)==0
print('PASS: all 3 source/reference hashes; 31 quality rows / 9000 frame scores; 39 valid GUI timing runs / 11340 unique submissions; bytes, EOS, order, bounded queues; timed capture/hashing disabled')
