"""Write the measured comparison section from saved results (no invented values)."""
import json
from pathlib import Path
import numpy as np
from streaming_common import ROOT

out=ROOT/'results/natural-2026-10-15'
quality=json.loads((out/'quality-summary.json').read_text())
runs=json.loads((out/'streaming-summary.json').read_text())
matched=json.loads((ROOT/'configs/natural-matched-crf.json').read_text())
main=[r for r in runs if not r['verification'] and not r['excluded_reason'] and r['buffer_ms']==350]
lines=['## Measured comparison','',
'All quality rows below score every frame. B/C use the same CRF23 bitstream. Direct CRFs were selected from the measured rate curve to approach that bitrate. Streaming columns show three GUI repetitions at the unchanged 350 ms buffer; C uses three full processing warmup chunks. Latency p95 and active unique submission fps are ranges across repetitions. Drops are summed across repetitions. Reconstruction/resize costs are means and exclude decode, GUI and waits; these are reflected in submission rates/latency. GPU allocation is the largest PyTorch allocator peak, not total process/desktop memory.','',
'| Clip | Method / CRF | TS bytes / kbps | RGB PSNR / SSIM | Processing ms/frame | Active submission fps | Latency p95 ms | Drops / frames | Peak PyTorch MiB |',
'|---|---|---:|---:|---:|---:|---:|---:|---:|']
for clip in ('flowers','road','waterfall'):
 for mode,q,warm in [('direct',matched[clip],0),('bicubic',23,0),('nanovsr',23,3)]:
  row=next(v for v in quality if v['clip']==clip and v['method']==mode and v['crf']==q)
  rr=[r for r in main if r['clip']==clip and r['mode']==mode and r['warmup_chunks']==warm]
  assert len(rr)==3,(clip,mode,len(rr))
  def span(key):return f"{min(r[key] for r in rr):.2f}–{max(r[key] for r in rr):.2f}"
  lines.append(f"| {clip} | {mode} / {q} | {row['bytes']:,} / {row['kbps']:.2f} | {row['psnr_db']:.3f} / {row['ssim']:.5f} | {np.mean([r['processing_ms_per_frame'] for r in rr]):.2f} | {span('active_submission_fps')} | {span('latency_p95_ms')} | {sum(r['drops'] for r in rr)} / {sum(r['frames'] for r in rr)} | {max(r['peak_gpu_allocated'] for r in rr)/2**20:.2f} |")
lines+=['','Matched-rate residuals (direct TS bytes relative to B/C):']
for clip in ('flowers','road','waterfall'):
 b=next(q for q in quality if q['clip']==clip and q['method']=='bicubic' and q['crf']==23)
 a=next(q for q in quality if q['clip']==clip and q['method']=='direct' and q['crf']==matched[clip])
 c=next(q for q in quality if q['clip']==clip and q['method']=='nanovsr' and q['crf']==23)
 lines.append(f"- {clip}: {100*(a['bytes']/b['bytes']-1):+.2f}%; direct minus NanoVSR PSNR {a['psnr_db']-c['psnr_db']:+.3f} dB, SSIM {a['ssim']-c['ssim']:+.5f}.")
lines+=['','Cold versus explicitly warmed NanoVSR (three runs of each per clip):']
for clip in ('flowers','road','waterfall'):
 for warm in (0,3):
  rr=[r for r in main if r['clip']==clip and r['mode']=='nanovsr' and r['warmup_chunks']==warm]
  assert len(rr)==3
  lines.append(f"- {clip}, {'warm' if warm else 'cold'}: first chunk {min(r['first_chunk_ms'] for r in rr):.1f}–{max(r['first_chunk_ms'] for r in rr):.1f} ms; first-frame latency {min(r['first_frame_latency_ms'] for r in rr if r['first_frame_latency_ms'] is not None):.1f}–{max(r['first_frame_latency_ms'] for r in rr if r['first_frame_latency_ms'] is not None):.1f} ms; drops {sum(r['drops'] for r in rr)}; model load mean {np.mean([r['model_load_s'] for r in rr]):.3f} s; explicit warmup mean {np.mean([r['warmup_s'] for r in rr]):.3f} s.")
lines+=['','All p50/p95/max latency, queue-age percentiles/maxima, per-frame costs, GPU allocated/reserved, source-duration/session/active submission rates, actual protocol bytes and completeness are in streaming-summary.csv and per-run JSON. All 27 base rate-quality rows and targeted direct extras are in quality-summary.csv; per-frame CSVs retain every metric.','']
(out/'measured-comparison.md').write_text('\n'.join(lines))
print('\n'.join(lines))
