"""Synchronized presentation view: local reference, live decoder, live NanoVSR."""
import csv,hashlib,json
import cv2
import numpy as np
from streaming_common import ROOT
from demo_stats import DemoStats


class ComparisonView(DemoStats):
    def __init__(self,c,mode):
        super().__init__(c,mode)
        if mode!='nanovsr' or not c.get('native_output'):
            raise ValueError('Three-panel comparison requires native NanoVSR')
        self.reference=np.load(ROOT/c['comparison_reference'],mmap_mode='r')
        if self.reference.shape!=(c['frames'],720,1280,3) or self.reference.dtype!=np.uint8:
            raise ValueError('Reference must be aligned 1280x720 uint8 RGB')
        alignment=list(csv.DictReader(open(ROOT/c['comparison_alignment'])))
        if len(alignment)!=c['frames']:raise ValueError('Reference alignment count mismatch')
        # Entire reference verification is before connection/source scheduling.
        for i,r in enumerate(alignment):
            if int(r['frame'])!=i or abs(float(r['reference_pts'])-i/30)>2e-6 or hashlib.sha256(self.reference[i].tobytes()).hexdigest()!=r['rgb_sha256']:
                raise ValueError(f'Reference alignment/hash mismatch at {i}')
        quality=json.loads((ROOT/c['quality_results']).read_text())
        with open(ROOT/c['source'],'rb') as f:source_hash=hashlib.file_digest(f,'sha256').hexdigest()
        if source_hash!=quality['reference_sha256']:raise ValueError('Reference source identity mismatch')
        self.nano=next(r for r in self.quality if r['method']=='nanovsr')
        self.low=next(r for r in self.quality if r['method']=='bicubic')

    def draw(self,rgb,report,complete=False,decoded_rgb=None,frame_index=None):
        if decoded_rgb is None or frame_index is None:raise ValueError('Live decoded frame and frame index required')
        if rgb.shape!=(720,1280,3) or decoded_rgb.shape!=(180,320,3):raise ValueError('Comparison frame shape mismatch')
        ref=self.reference[frame_index]
        canvas=np.full((820,1536,3),(24,28,35),np.uint8)
        def text(s,x,y,scale=.52,color=(225,230,238)):
            cv2.putText(canvas,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
        cols=[('ORIGINAL / REFERENCE',ref),('TRANSMITTED / DECODED',decoded_rgb),('UPSCALED / LIVE NanoVSR',rgb)]
        for j,(name,a) in enumerate(cols):
            x=j*512
            text(name,x+12,25,.62,(110,220,255))
            text(f'Frame {frame_index} | source time {frame_index/30:.3f} s',x+12,49,.48)
            interp=cv2.INTER_NEAREST if j==1 else cv2.INTER_AREA
            preview=cv2.resize(a,(512,288),interpolation=interp)
            canvas[65:353,x:x+512]=cv2.cvtColor(preview,cv2.COLOR_RGB2BGR)
            text('Preview 512x288: '+('nearest enlargement' if j==1 else 'display-only reduction'),x+12,377,.46)
            # Same reference-space crop. Reference and output stay 1:1 here.
            full=cv2.resize(a,(1280,720),interpolation=cv2.INTER_NEAREST) if j==1 else a
            canvas[414:594,x+96:x+416]=cv2.cvtColor(full[300:480,500:820],cv2.COLOR_RGB2BGR)
            text('Same detail crop | '+('nearest x4 display' if j==1 else 'native pixels 1:1'),x+12,405,.46)
        for i,s in enumerate(['1280x720 | 30 FPS | 300 aligned frames','Local reference; not a second network stream','Original 720p source; FPS normalized 50 -> 30','Quality reference: PSNR/SSIM baseline']):text(s,12,622+i*23,.47)
        for i,s in enumerate(['320x180 | H.264 CRF23 over TCP',f"Actual stream: {self.low['kbps']:.1f} kbps (MPEG-TS)",'Live decoder output before reconstruction','PSNR/SSIM: N/A across different pixel sizes']):text(s,524,622+i*23,.47)
        for i,s in enumerate(['1280x720 | native x4 | no output downscaling',f"Saved whole-clip PSNR {self.nano['psnr_db']:.2f} dB | SSIM {self.nano['ssim']:.4f}",'Chunk3 | lookahead up to 66.7 ms',f"Same {self.nano['kbps']:.1f} kbps stream; no added network bytes"]):text(s,1036,622+i*23,.47)
        shown=[f for f in report['frames'] if not f['dropped']];recent=shown[-60:]
        fps=(len(recent)-1)/(recent[-1]['render_event']-recent[0]['render_event']) if len(recent)>1 else 0
        lat=[f['application_latency_s']*1000 for f in (shown if complete else recent)]
        status='EOS SUMMARY (last processed frame)' if complete else 'LIVE (stats from completed submissions)'
        text(f"{status} | submitted {report['unique_render_submissions']}/{self.c['frames']} | dropped {report['dropped_frames']} | last-60 rate {fps:.1f} FPS",12,732,.52,(110,220,255))
        details=f"Buffer {self.c['playout_delay_s']*1000:.0f} ms | GPU peak {report['gpu_peak_allocated_bytes']/2**20:.1f} MiB"
        if lat:details+=f' | app latency p50/p95 {np.median(lat):.0f}/{np.percentile(lat,95):.0f} ms'
        if report['chunks']:
            ch=report['chunks'][-1];details+=f" | model {ch['model_ms']/ch['length']:.1f} ms/frame"
        text(details,12,758,.50)
        text('Reference and decoder panes follow the SAME reconstructed frame ID; late frames skip in all panes together.',12,786,.47)
        text('View resizing only; native output remains 1280x720. Saved quality is offline; overlay adds cost. No physical-display latency claim.',12,809,.45)
        return canvas
