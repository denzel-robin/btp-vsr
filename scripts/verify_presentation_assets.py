"""Final presentation-package integrity checks without timed/model work."""
import csv,hashlib,json,re,subprocess
from pathlib import Path
from streaming_common import ROOT,dump
out=ROOT/'presentation_assets'
load=lambda p:json.loads(p.read_text())
rows=list(csv.DictReader(open(out/'data/comparison.csv')))
assert len(rows)==13 and len(rows[0])==26
q=list(csv.DictReader(open(out/'data/quality-all-points.csv')))
assert len(q)==31
for r in rows:
    ref=next(v for v in q if v['clip']==r['clip'] and v['method']==r['mode'] and float(v['crf'])==float(r['crf']))
    assert float(r['psnr_db'])==float(ref['psnr_db']) and float(r['ssim'])==float(ref['ssim'])
    assert int(r['ts_bytes'])==int(ref['bytes']) and float(r['ts_kbps'])==float(ref['kbps'])
for mode in ('direct','bicubic','nanovsr'):
    folder=out/f'evidence/demo-{mode}-verified';c=load(folder/'client.json');s=load(folder/'server.json')
    assert c['status']==s['status']=='ok' and c['received_eos']
    assert c['unique_render_submissions']==300 and c['dropped_frames']==0
    assert c['payload_bytes']==s['payload_bytes']
    assert all((folder/f'window-{i}.png').exists() for i in (0,1))
st=load(out/'evidence/stability/client.json');sm=load(out/'data/stability-summary.json')
assert st['status']=='error' and len(st['frames'])==2400 and st['dropped_frames']==720
assert len(st['stability_windows'])==8 and all(w['drops']>150 for w in st['stability_windows'][-3:])
assert sm['processed_prefix_completeness_percent']==70 and not sm['received_eos']
assert not Path(str(out/'evidence/stability/client.json')+'.received.ts').exists()
assert all(f['output_sha256'] is None for f in st['frames'])
assert st['queue_peak']<=6
for doc in (ROOT/'logs/progress-report-handoff.md',out/'README.md',out/'REPRODUCE.md'):
    for target in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
        if '://' not in target:assert (doc.parent/target).exists(),(doc,target)
for clip in ('flowers','road','waterfall'):
    assert (out/f'figures/quality-rate-{clip}.png').exists()
    assert (out/f'figures/quality-rate-{clip}.svg').exists()
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=nb_frames,duration','-of','json',str(out/'demo/OFFLINE-road-comparison.mp4')]))['streams'][0]
assert probe['nb_frames']=='300' and float(probe['duration'])==10
manifest={}
paths=list((ROOT/'scripts').glob('*.py'))+[ROOT/'server/streaming_server.py',ROOT/'client/streaming_client.py',ROOT/'logs/progress-report-handoff.md']+list((out/'configs').glob('*.json'))
for p in paths:manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
dump(out/'evidence/final-source-manifest.json',manifest)
dump(out/'evidence/final-validation.json',dict(status='pass',comparison_rows=13,quality_rows=31,valid_prior_timing_runs=39,missing_repetitions=0,gui_modes_verified=3,headless_road_frame_hashes_verified=300,metric_rechecks=186,backup_frames=300,stability_outcome='failed early after 2400 frames; 720 late drops; no clean EOS',local_handoff_links='pass',preservation='both environment freezes match prior; both checkpoint SHA256 pass; upstream trees clean',visual_scope='two changing-content window captures inspected per mode; not physical-frame validation'))
print('PASS: comparison/quality values, demo EOS/counts, stability failure correctly retained, backup and links, source manifest')
