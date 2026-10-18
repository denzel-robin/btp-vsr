"""Bounded MPEG-TS decode, optional native-x4 NanoVSR, and paced playback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
import cv2
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from streaming_common import ROOT, config, dump, exact, receive, tcp_options


def load_model(c):
    import torch
    sys.path.insert(0, str(ROOT/'external/nanovsr'))
    from models.nanovsr import NanoVSR
    from utils import detect_model_config
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run with approved host GPU access')
    state = torch.load(ROOT/c['checkpoint'], map_location='cpu', weights_only=True)
    for key in ('params_ema', 'params', 'model_state_dict'):
        if key in state:
            state = state[key]
            break
    features, blocks = detect_model_config(state)
    model = NanoVSR(num_feat=features or 32, num_blocks=blocks or 8)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        model.switch_to_deploy()
    model.cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    return torch, model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/streaming-smoke.json')
    p.add_argument('--mode', choices=['direct', 'bicubic', 'nanovsr'], default='bicubic')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--report', default='results/streaming-client.json')
    p.add_argument('--processing-delay', type=float, default=0, help='test-only seconds per chunk')
    a = p.parse_args()
    c = config(a.config)
    verify = c.get('verification', True)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    report = dict(config=c, mode=a.mode, headless=a.headless, status='error', frames=[], chunks=[],
                  payload_bytes=0, protocol_bytes=0, decoded_frames=0, queue_peak=0,
                  unique_render_submissions=0, unique_displayed_frames=0,
                  displayed_count_scope='Unique OpenCV imshow/waitKey submissions; headless=0; physical presentation unmeasured', dropped_frames=0,
                  gpu_peak_allocated_bytes=0, gpu_peak_reserved_bytes=0, errors=[])
    stop = threading.Event()
    q = queue.Queue(c['queue_frames'])
    proc = sock = None
    threads = []
    torch = model = None
    started = time.monotonic()
    h = hashlib.sha256()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=.1)
                report['queue_peak'] = max(report['queue_peak'], q.qsize())
                return
            except queue.Full:
                pass

    try:
        if a.mode == 'nanovsr':
            load_start = time.monotonic()
            torch, model = load_model(c)
            report['model_load_s'] = time.monotonic() - load_start
            report['warmup_chunks'] = c.get('warmup_chunks', 0)
            warm_start = time.monotonic()
            if c.get('warmup_chunks', 0):
                width, height = c['transmission_size']
                if c.get('warmup_pipeline', False):
                    # Match live NHWC-derived tensor strides, transfers, quantization and resize.
                    dummy = np.zeros((c['chunk_size'], height, width, 3), np.uint8)
                    with torch.inference_mode():
                        for _ in range(c['warmup_chunks']):
                            warm_input = torch.from_numpy(dummy.copy()).permute(0,3,1,2).unsqueeze(0).float().cuda()/255
                            warm_output = model(warm_input)
                            native = warm_output[0].clamp(0,1).mul(255).round().byte().permute(0,2,3,1).cpu().numpy()
                            resized = [cv2.resize(f, tuple(c['playback_size']), interpolation=cv2.INTER_CUBIC) for f in native]
                            del warm_input, warm_output, native, resized
                else:
                    warm_input = torch.zeros(1,c['chunk_size'],3,height,width,device='cuda')
                    with torch.inference_mode():
                        for _ in range(c['warmup_chunks']):
                            warm_output = model(warm_input)
                            del warm_output
                    del warm_input
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            report['warmup_s'] = time.monotonic() - warm_start
            report['checkpoint_sha256'] = hashlib.sha256((ROOT/c['checkpoint']).read_bytes()).hexdigest()
        if not a.headless:
            if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
                raise RuntimeError('No desktop available; use --headless for automated checks')
            cv2.namedWindow('BTP streaming', cv2.WINDOW_AUTOSIZE)
        sock = socket.create_connection(('127.0.0.1', a.port), timeout=30)
        tcp_options(sock)
        stream = sock.makefile('rb')
        kind, data, count = receive(stream)
        assert kind == b'H', 'missing header'
        header = json.loads(data)
        assert header['config'] == c and header['mode'] == a.mode, 'server/client config mismatch'
        report['protocol_bytes'] += count
        report['header'] = header
        w, height = header['size']
        pts = header['source_pts']
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-probesize', '32',
                   '-analyzeduration', '0', '-flags', 'low_delay', '-threads', '1', '-f', 'mpegts', '-i', 'pipe:0',
                   '-map', '0:v:0', '-an', '-fps_mode', 'passthrough', '-threads', '1',
                   '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
        report['decoder_command'] = command
        capture = open(str(a.report)+'.received.ts', 'wb') if verify else None
        err = open(str(a.report)+'.ffmpeg.log', 'wb')
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)

        def feed():
            try:
                while not stop.is_set():
                    kind, data, count = receive(stream)
                    report['protocol_bytes'] += count
                    if kind == b'E':
                        eos = json.loads(data)
                        if eos['returncode'] or eos['frames'] != len(pts):
                            raise RuntimeError(f'server encoder failure: {eos}')
                        report['received_eos'] = True
                        break
                    if kind != b'D':
                        raise ValueError('unexpected protocol record')
                    report['payload_bytes'] += len(data)
                    if verify:
                        capture.write(data)
                        h.update(data)
                    proc.stdin.write(data)
                    proc.stdin.flush()
            except Exception as exc:
                report['errors'].append('receiver: '+repr(exc))
            finally:
                if capture is not None:
                    capture.close()
                try:
                    proc.stdin.close()
                except OSError:
                    pass

        def decode():
            try:
                for i in range(len(pts)):
                    frame = np.frombuffer(exact(proc.stdout, w*height*3), np.uint8).reshape(height, w, 3).copy()
                    decoded = time.monotonic()
                    report['decoded_frames'] += 1
                    put((i, frame, decoded))
                    if stop.is_set():
                        return
                if proc.stdout.read(1):
                    raise ValueError('unexpected extra decoded frames')
                rc = proc.wait(timeout=30)
                if rc:
                    raise RuntimeError(f'decoder exited {rc}')
            except Exception as exc:
                report['errors'].append('decoder: '+repr(exc))
            finally:
                put(None)

        threads = [threading.Thread(target=feed, daemon=True), threading.Thread(target=decode, daemon=True)]
        for t in threads:
            t.start()
        count = c['chunk_size'] if model is not None else 1
        done = False
        while not done:
            batch = []
            for _ in range(count):
                item = q.get(timeout=35)
                if item is None:
                    done = True
                    break
                batch.append(item)
            if not batch:
                break
            begin = time.monotonic()
            model_ms = 0
            if model is not None:
                x = torch.from_numpy(np.stack([b[1] for b in batch])).permute(0,3,1,2).unsqueeze(0).float().cuda()/255
                torch.cuda.synchronize()
                first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with torch.inference_mode():
                    first.record()
                    y = model(x)
                    last.record()
                    last.synchronize()
                    model_ms = first.elapsed_time(last)
                    frames = y.squeeze(0).clamp(0,1).mul(255).round().byte().permute(0,2,3,1).cpu().numpy()
                report['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                report['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                del x, y
            else:
                frames = [b[1] for b in batch]
            resize_start = time.monotonic()
            outputs = [cv2.resize(f, tuple(c['playback_size']), interpolation=cv2.INTER_CUBIC)
                       if (f.shape[1], f.shape[0]) != tuple(c['playback_size']) else f for f in frames]
            resize_s = time.monotonic() - resize_start
            if a.processing_delay:
                time.sleep(a.processing_delay)
            finish = time.monotonic()
            report['chunks'].append(dict(first=batch[0][0], length=len(batch), model_ms=model_ms,
                                         processing_s=finish-begin, resize_s=resize_s))
            for (i, decoded_frame, decoded), frame in zip(batch, outputs):
                scheduled = header['epoch'] + pts[i] - pts[0]
                target = scheduled + c['playout_delay_s']
                remaining = target - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                dropped = time.monotonic() - target > c['drop_late_s']
                if not dropped and not a.headless:
                    cv2.imshow('BTP streaming', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 255 == 27:
                        raise RuntimeError('playback cancelled by Escape')
                rendered = time.monotonic()
                report['dropped_frames'] += int(dropped)
                report['unique_render_submissions'] += int(not dropped)
                report['unique_displayed_frames'] += int(not dropped and not a.headless)
                report['frames'].append(dict(index=i, source_pts=pts[i], source_scheduled=scheduled,
                    decoded=decoded, processing_started=begin, queue_age_s=begin-decoded,
                    decoded_sha256=hashlib.sha256(decoded_frame.tobytes()).hexdigest() if verify else None,
                    queue_depth=q.qsize(), render_event=None if dropped else rendered,
                    dropped=dropped, application_latency_s=None if dropped else rendered-scheduled,
                    output_sha256=hashlib.sha256(frame.tobytes()).hexdigest() if verify else None))
        for t in threads:
            t.join(timeout=35)
        assert not any(t.is_alive() for t in threads), 'worker did not terminate'
        assert not report['errors'], report['errors']
        if verify:
            packet_probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(a.report)+'.received.ts']))
            transport_pts = [float(f['best_effort_timestamp_time']) for f in packet_probe['frames']]
            report['transport_pts'] = transport_pts
            assert len(transport_pts) == len(pts), 'transport timestamp count mismatch'
            assert all(abs((v-transport_pts[0])-(s-pts[0])) < 0.00003 for v,s in zip(transport_pts,pts)), 'transport timestamp mismatch'
        assert report.get('received_eos'), 'missing clean EOS'
        assert [f['index'] for f in report['frames']] == list(range(len(pts))), 'frame count/order mismatch'
        report['status'] = 'ok'
    except Exception as exc:
        report['errors'].append(repr(exc))
        raise
    finally:
        stop.set()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            err.close()
        for t in threads:
            t.join(timeout=2)
        if not a.headless:
            cv2.destroyAllWindows()
        report['payload_sha256'] = h.hexdigest() if verify else None
        report['wall_s_including_model_load'] = time.monotonic()-started
        processing = sum(v['processing_s'] for v in report['chunks'])
        report['processing_throughput_fps'] = len(report['frames'])/processing if processing else 0
        dump(a.report, report)
        print(json.dumps({k: v for k,v in report.items() if k not in ('frames','chunks','config','header')}, indent=2))


if __name__ == '__main__':
    main()
