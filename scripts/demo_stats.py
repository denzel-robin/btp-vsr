"""Presentation-only dashboard. Saved quality is never computed in playback."""
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from streaming_common import ROOT


class DemoStats:
    def __init__(self, config, mode):
        self.c, self.mode = config, mode
        self.quality = []
        self.quality_note = 'Direct is near-matched rate, not equal CRF.'
        self.asset = None
        if config.get('encoded_asset'):
            path = ROOT/config['encoded_asset']
            meta = json.loads(path.with_suffix('.json').read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
                raise ValueError('Demo statistics asset hash mismatch')
            self.asset = meta
            if config.get('quality_results'):
                measured = json.loads((ROOT/config['quality_results']).read_text())
                assert measured['playback_size'] == config['playback_size'], 'Quality resolution mismatch'
                selected = next(r for r in measured['summary'] if r['method'] == mode)
                assert selected['asset_sha256'] == meta['sha256'] and selected['frames'] == config['frames'], 'Quality asset mismatch'
                if mode == 'nanovsr':
                    assert config.get('native_output') and measured['chunk_size'] == config['chunk_size']
                    assert hashlib.sha256((ROOT/config['checkpoint']).read_bytes()).hexdigest() == measured['checkpoint_sha256']
                self.quality = measured['summary']
                self.quality_note = 'CRF23 comparison; NOT matched bitrate.'
            # Only associate archived quality with its original exact asset.
            archive = ROOT/'results/natural-2026-10-15'
            if path.parent.parent == archive and config['playback_size'] == [640, 360]:
                rows = list(csv.DictReader(open(archive/'quality-summary.csv')))
                rows = [r for r in rows if r['clip'] == path.parent.name]
                for method in ('direct', 'bicubic', 'nanovsr'):
                    candidates = [r for r in rows if r['method'] == method]
                    if candidates:
                        # Direct uses closest measured rate; LR uses actual CRF23 demo asset.
                        target = next((r for r in rows if r['method']=='bicubic' and float(r['crf'])==23), None)
                        if target:
                            r = min(candidates, key=lambda r: abs(float(r['kbps'])-float(target['kbps'])))
                            self.quality.append(r)

    def draw(self, rgb, report, complete=False):
        # Draw beside the image; never alter reconstruction pixels or output hashes.
        h, w = rgb.shape[:2]
        canvas = np.full((max(h+50, 570), w+460, 3), (24, 28, 35), np.uint8)
        canvas[:h,:w] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        x, y = w+18, 28
        def line(text, color=(225,230,238), scale=.49):
            nonlocal y
            cv2.putText(canvas,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
            y += 23
        line('BTP VSR | '+self.mode.upper(), (110,220,255), .65)
        tx=self.c['playback_size'] if self.mode=='direct' else self.c['transmission_size']
        line(f"TX {tx[0]}x{tx[1]} -> playback {w}x{h}")
        native = self.c.get('native_output', False)
        line(('Native x4 | NO final resize | chunk ' if native else 'Native x4 -> resize | chunk ')+str(self.c['chunk_size']) if self.mode=='nanovsr' else 'Direct decode' if self.mode=='direct' else 'CPU bicubic resizing')
        if self.mode=='nanovsr':line(f"{'Native output' if native else 'Intermediate'}: {tx[0]*4}x{tx[1]*4}")
        line(f"Playout buffer: {self.c['playout_delay_s']*1000:.0f} ms")
        if self.asset:
            pts=self.asset['pts'];duration=(pts[-1]-pts[0])*len(pts)/(len(pts)-1)
            line(f"Asset rate: {self.asset['bytes']*8/duration/1000:.1f} kbps (TS)")
        line('SAVED QUALITY | whole-clip RGB', (110,220,255))
        line('Method       kbps      PSNR dB    SSIM',scale=.46)
        if not self.quality:line('Not measured at this output resolution')
        for r in self.quality:
            line(f"{r['method']:<9} {float(r['kbps']):7.1f}    {float(r['psnr_db']):6.2f}    {float(r['ssim']):.4f}",scale=.44)
        line(self.quality_note if self.quality else '640x360 scores do not apply here.' if native else 'No aligned quality available.',scale=.43)
        line('LIVE SESSION'+(' | COMPLETE' if complete else ' | previous submitted frames'), (110,220,255), .47)
        fs=report['frames'];shown=[f for f in fs if not f['dropped']]
        recent=shown[-60:]
        fps=(len(recent)-1)/(recent[-1]['render_event']-recent[0]['render_event']) if len(recent)>1 else 0
        latency=[f['application_latency_s']*1000 for f in (shown if complete else recent)]
        line(f"Submitted: {report['unique_render_submissions']}/{self.c['frames']} | drops: {report['dropped_frames']}")
        if native:
            processed = report['unique_render_submissions'] + report['dropped_frames']
            if processed:
                line(f"Playback completeness so far: {100*report['unique_render_submissions']/processed:.1f}%")
        line(f'Unique submission rate (last 60): {fps:.1f} fps')
        if latency:line(f'App latency p50/p95: {np.median(latency):.0f}/{np.percentile(latency,95):.0f} ms')
        if native and latency:line(f'App latency max: {max(latency):.0f} ms')
        if report['chunks']:
            chunk=report['chunks'][-1];n=chunk['length']
            line(f"Last chunk: model {chunk['model_ms']/n:.1f} ms/frame")
            line(f"{'Processing (no resize)' if native else 'Processing incl. resize'}: {chunk['processing_s']*1000/n:.1f} ms/frame")
            if native:
                line(f"Processing-only rate: {n/chunk['processing_s']:.1f} fps")
                line('Final resize: disabled (0 ms)')
        if fs:line(f"Queue depth: {fs[-1]['queue_depth']} | age: {fs[-1]['queue_age_s']*1000:.0f} ms")
        line(f"Peak GPU allocation: {report['gpu_peak_allocated_bytes']/2**20:.1f} MiB")
        line(f"Received payload: {report['payload_bytes']/1000:.1f} kB")
        cv2.putText(canvas,'Presentation overlay adds overhead; use runs without --stats for benchmarks.',(12,canvas.shape[0]-32),cv2.FONT_HERSHEY_SIMPLEX,.46,(170,180,195),1,cv2.LINE_AA)
        cv2.putText(canvas,'Application latency is not physical display latency. Saved quality is not a live measurement.',(12,canvas.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,.43,(170,180,195),1,cv2.LINE_AA)
        return canvas
