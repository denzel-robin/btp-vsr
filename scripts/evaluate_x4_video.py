"""Offline x4 evaluation; run with .venv/bin/python on a GPU host.

Rejects existing output directories. Scores lossless RGB frames, never MP4s.
Source dimensions must be 1280x720, or 640x360 for a labelled smoke run.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'external/nanovsr'))
from utils import detect_model_config
from models.nanovsr import NanoVSR
from evaluate import calculate_psnr, calculate_ssim


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype == np.uint8
    return float(calculate_psnr(a, b)), float(calculate_ssim(a, b))


def metric_checks():
    a = np.full((24, 24, 3), 100, np.uint8)
    b = np.full_like(a, 110)
    assert metrics(a, a) == (float('inf'), 1.0)
    p, s = metrics(a, b)
    assert abs(p - 20*math.log10(255/10)) < 1e-12
    expected = (2*100*110 + (0.01*255)**2)/(100**2+110**2+(0.01*255)**2)
    assert abs(s-expected) < 1e-10
    rng = np.random.default_rng(2026)
    a = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    b = np.roll(a, 1, axis=1)
    assert metrics(a, b)[1] < 0.5
    np.testing.assert_allclose(metrics(a, b), metrics(b, a), atol=1e-12)
    # Temporal permutation must not accidentally score as aligned identity.
    seq = np.stack([np.full_like(a, n) for n in (20, 100, 220)])
    assert all(math.isinf(metrics(x, x)[0]) for x in seq)
    assert all(math.isfinite(metrics(x, y)[0]) for x, y in zip(seq, np.roll(seq, 1, axis=0)))
    return 'PASS: identity, analytic constant-error PSNR/SSIM, symmetry, spatial/temporal shift'


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/evaluation-smoke.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = json.loads(args.config.read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg, indent=2))
    commands = []

    def run(cmd, binary=False):
        cmd = [str(x) for x in cmd]
        commands.append(cmd)
        (out/'commands.json').write_text(json.dumps(commands, indent=2))
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.stderr:
            with (out/'ffmpeg-stderr.log').open('ab') as f:
                f.write(result.stderr)
        return result.stdout if binary else result.stdout.decode()

    def probe(path):
        return json.loads(run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                              '-show_streams', '-show_frames', '-show_format', '-of', 'json', path]))

    def decode(path, width, height, count=None):
        cmd = ['ffmpeg', '-v', 'error', '-i', path, '-map', '0:v:0', '-an', '-sn']
        if count is not None:
            cmd += ['-frames:v', count]
        cmd += ['-fps_mode', 'passthrough', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
        raw = run(cmd, True)
        assert len(raw) % (width*height*3) == 0
        return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)

    def save_frames(folder, frames):
        folder.mkdir()
        for i, frame in enumerate(frames):
            assert cv2.imwrite(str(folder/f'{i:06d}.png'), frame[..., ::-1])

    source = ROOT/cfg['source']
    info = probe(source)
    stream = info['streams'][0]
    w, h = stream['width'], stream['height']
    assert (w, h) in ((1280, 720), (640, 360)), 'No upscaled reference allowed'
    assert cfg['source_kind'] in ('synthetic_smoke', 'natural_smoke', 'natural_reference')
    assert (w, h) == (1280, 720) or cfg['source_kind'].endswith('smoke')
    fps = Fraction(stream['avg_frame_rate'])
    n, chunk = cfg['frames'], cfg['chunk_length']
    assert n > chunk >= 1 and cfg['warmup_per_chunk_shape'] >= 1
    pts = np.array([float(f['best_effort_timestamp_time']) for f in info['frames'][:n]])
    assert len(pts) == n
    np.testing.assert_allclose(pts-pts[0], np.arange(n)/float(fps), atol=2e-6, rtol=0)
    hr = decode(source, w, h, n)
    assert len(hr) == n
    save_frames(out/'hr', hr)
    report = {'status': 'RUNNING', 'config': cfg, 'source_sha256': sha(source),
              'source_probe': info, 'metric_checks': metric_checks(),
              'ffmpeg_version': run(['ffmpeg', '-version']), 'torch': torch.__version__,
              'opencv': cv2.__version__, 'numpy': np.__version__,
              'precision': 'float32; no autocast; TF32 disabled',
              'authentic_CDA': 'PENDING; Baidu extraction code zc8e', 'clips': []}
    checkpoint = ROOT/cfg['checkpoint']
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    key = 'root_state_dict'
    for candidate in ('params_ema', 'params', 'model_state_dict'):
        if candidate in state:
            state, key = state[candidate], candidate
            break
    features, blocks = detect_model_config(state)
    model = NanoVSR(num_feat=features or 32, num_blocks=blocks or 8)
    expected = model.state_dict()
    compatibility = {'selected_key': key, 'num_feat': features, 'num_blocks': blocks,
                     'checkpoint_sha256': sha(checkpoint), 'tensors': len(state),
                     'missing': sorted(set(expected)-set(state)),
                     'unexpected': sorted(set(state)-set(expected)),
                     'shape_mismatches': {k: [list(state[k].shape), list(expected[k].shape)]
                                          for k in state.keys() & expected.keys()
                                          if state[k].shape != expected[k].shape}}
    (out/'checkpoint.json').write_text(json.dumps(compatibility, indent=2))
    model.load_state_dict(state, strict=True)
    model.eval()
    assert torch.cuda.is_available(), 'Approved host CUDA access required'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(2026)
    # Check official deployment conversion before using it for the baseline.
    with torch.no_grad():
        sample = torch.rand(1, 3, 3, 8, 8)
        before = model(sample)
        model.switch_to_deploy()
        after = model(sample)
        torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-5)
    report['deploy_max_error'] = (before-after).abs().max().item()
    report['checkpoint'] = compatibility
    model = model.cuda().eval()
    report['hardware'] = run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'])
    all_rows, summary, alignment = [], [], []
    for crf in cfg['crfs']:
        clip_start = time.perf_counter()
        folder = out/f'crf{crf}'
        folder.mkdir()
        encoded = folder/'lr.mp4'
        lw, lh = w//4, h//4
        run(['ffmpeg', '-v', 'error', '-framerate', str(fps), '-start_number', 0,
             '-i', out/'hr/%06d.png', '-frames:v', n, '-an',
             '-vf', f'scale={lw}:{lh}:flags=bicubic:in_range=full:out_range=limited:out_color_matrix=bt709,format=yuv420p',
             '-c:v', 'libx264', '-preset', 'medium', '-crf', crf, '-threads', 1,
             '-g', cfg['gop'], '-keyint_min', cfg['gop'], '-sc_threshold', 0,
             '-bf', 0, '-refs', 1, '-pix_fmt', 'yuv420p', '-color_range', 'tv',
             '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
             '-fps_mode', 'passthrough', encoded])
        encoded_info = probe(encoded)
        es = encoded_info['streams'][0]
        ef = encoded_info['frames']
        assert (es['width'], es['height']) == (lw, lh) and len(ef) == n
        assert es['has_b_frames'] == 0 and es['refs'] == 1
        epts = np.array([float(f['best_effort_timestamp_time']) for f in ef])
        np.testing.assert_allclose(epts-epts[0], pts-pts[0], atol=2e-6, rtol=0)
        lr = decode(encoded, lw, lh)
        assert lr.shape == (n, lh, lw, 3)
        save_frames(folder/'lr', lr)
        duration = float(es['duration'])
        np.testing.assert_allclose(duration, n/float(fps), atol=2e-6)
        packet_bytes = sum(int(f['pkt_size']) for f in ef)
        rows, gpu_ms, warmed = [], 0.0, set()
        sr_list = []
        chunk_records = []
        peak = 0
        with torch.inference_mode():
            for first in range(0, n, chunk):
                last = min(first+chunk, n)
                x = torch.from_numpy(lr[first:last].copy()).permute(0, 3, 1, 2).contiguous().float().div_(255).unsqueeze(0).cuda()
                if last-first not in warmed:
                    for _ in range(cfg['warmup_per_chunk_shape']):
                        warm = model(x)
                        del warm
                    torch.cuda.synchronize()
                    warmed.add(last-first)
                torch.cuda.reset_peak_memory_stats()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                y = model(x)
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end)
                gpu_ms += elapsed
                peak = max(peak, torch.cuda.max_memory_allocated())
                assert y.shape == (1, last-first, 3, h, w) and torch.isfinite(y).all()
                sr_list.extend(y[0].clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
                chunk_records.append({'first': first, 'stop_exclusive': last, 'gpu_ms': elapsed})
                del x, y
        sr = np.stack(sr_list)
        tick = time.perf_counter()
        bicubic = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC) for f in lr])
        bicubic_seconds = time.perf_counter()-tick
        for name, frames in (('bicubic', bicubic), ('nanovsr', sr)):
            save_frames(folder/name, frames)
            for i, frame in enumerate(frames):
                p, s = metrics(frame, hr[i])
                rows.append({'crf': crf, 'method': name, 'frame': i, 'pts_seconds': float(pts[i]-pts[0]), 'psnr_db': p, 'ssim': s})
            subset = rows[-n:]
            summary.append({'crf': crf, 'method': name, 'frames': n,
                            'mean_frame_psnr_db': float(np.mean([r['psnr_db'] for r in subset])),
                            'mean_frame_ssim': float(np.mean([r['ssim'] for r in subset]))})
        for i in range(n):
            alignment.append({'crf': crf, 'frame': i, 'source_pts': float(pts[i]),
                              'lr_pts': float(epts[i]), 'hr_png_sha256': sha(out/'hr'/f'{i:06d}.png'),
                              'lr_png_sha256': sha(folder/'lr'/f'{i:06d}.png')})
        if crf == 23:
            preview = []
            for i in range(n):
                panels = []
                for label, frame in [('Reference', hr[i]), ('Bicubic', bicubic[i]), ('NanoVSR', sr[i])]:
                    panel = cv2.resize(frame, (320, 180))
                    cv2.rectangle(panel, (0, 0), (320, 36), (0, 0, 0), -1)
                    cv2.putText(panel, f'{label} | {cfg["source_kind"]}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1)
                    cv2.putText(panel, f'CRF23 frame {i}', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1)
                    panels.append(panel)
                preview.append(np.concatenate(panels, axis=1))
            save_frames(out/'preview-frames', preview)
            run(['ffmpeg', '-v', 'error', '-framerate', str(fps), '-i', out/'preview-frames/%06d.png',
                 '-c:v', 'libx264', '-crf', 18, '-pix_fmt', 'yuv420p', out/'preview.mp4'])
        all_rows.extend(rows)
        report['clips'].append({'crf': crf, 'encoded_bytes': encoded.stat().st_size,
                                'video_packet_bytes': packet_bytes, 'duration_seconds': duration,
                                'container_bitrate_bps': encoded.stat().st_size*8/duration,
                                'video_payload_bitrate_bps': packet_bytes*8/duration,
                                'probe': encoded_info, 'lr_mp4_sha256': sha(encoded),
                                'model_gpu_ms': gpu_ms, 'timed_frames': n,
                                'peak_allocated_bytes': peak, 'chunks': chunk_records,
                                'bicubic_cpu_seconds': bicubic_seconds,
                                'offline_clip_wall_seconds': time.perf_counter()-clip_start})
        print(f'CRF {crf}: {n} aligned frames verified', flush=True)
    write_csv(out/'per-frame.csv', all_rows)
    write_csv(out/'per-clip.csv', summary)
    write_csv(out/'alignment.csv', alignment)
    report.update(status='PASS', offline_total_wall_seconds=time.perf_counter()-started, summary=summary)
    (out/'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
