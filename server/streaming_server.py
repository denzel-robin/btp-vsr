"""One-client, video-only paced localhost MPEG-TS server."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from streaming_common import ROOT, config, dump, send, tcp_options


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/streaming-smoke.json')
    p.add_argument('--mode', choices=['direct', 'bicubic', 'nanovsr'], default='bicubic')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--report', default='results/streaming-server.json')
    a = p.parse_args()
    c = config(a.config)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                 '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(ROOT / c['source'])]
    probe = json.loads(subprocess.check_output(probe_cmd))
    pts = [float(f['best_effort_timestamp_time']) for f in probe['frames']][:c['frames']]
    assert len(pts) == c['frames'] and len(pts) >= 2, 'source too short'
    # ffprobe prints decimal timestamps; a rounded first interval accumulates
    # millisecond-scale drift on long CFR clips. Validate against the full span.
    step = (pts[-1] - pts[0]) / (len(pts) - 1)
    assert step > 0 and all(abs((v-pts[0])-i*step) < 0.0002 for i,v in enumerate(pts)), 'CFR source required'
    size = c['playback_size'] if a.mode == 'direct' else c['transmission_size']
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-re', '-i', str(ROOT/c['source']),
           '-map', '0:v:0', '-an', '-frames:v', str(c['frames']), '-vf', f'scale={size[0]}:{size[1]}:flags=bicubic',
           '-c:v', 'libx264', '-preset', c['preset'], '-tune', 'zerolatency', '-crf', str(c['crf']),
           '-threads', '1', '-pix_fmt', 'yuv420p', '-bf', '0', '-g', '30', '-sc_threshold', '0',
           '-mpegts_copyts', '1', '-muxdelay', '0', '-muxpreload', '0', '-flush_packets', '1', '-f', 'mpegts', 'pipe:1']
    asset_data = asset_meta = None
    if c.get('encoded_asset'):
        asset = ROOT/c['encoded_asset']
        asset_meta = json.loads(asset.with_suffix('.json').read_text())
        asset_data = asset.read_bytes()
        assert hashlib.sha256(asset_data).hexdigest() == asset_meta['sha256']
        assert asset_meta['frames'] == len(pts)
        assert [asset_meta['width'],asset_meta['height']] == size
        assert all(abs((p-asset_meta['pts'][0])-(q-pts[0])) < 0.00003 for p,q in zip(asset_meta['pts'],pts))
        cmd = ['preencoded MPEG-TS replay; no timed encode', str(asset)]
    report = dict(config=c, mode=a.mode, command=cmd, probe_command=probe_cmd, source_pts=pts,
                  source_sha256=hashlib.sha256((ROOT/c['source']).read_bytes()).hexdigest(),
                  payload_bytes=0, protocol_bytes=0, status='error')
    proc = None
    h = hashlib.sha256()
    try:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', a.port))
            listener.listen(1)
            listener.settimeout(60)
            print(f'READY 127.0.0.1:{a.port}', flush=True)
            conn, _ = listener.accept()
            with conn:
                tcp_options(conn)
                epoch = time.monotonic()
                report['source_schedule_epoch'] = epoch
                header = dict(config=c, size=size, source_pts=pts, epoch=epoch, mode=a.mode, step=step)
                report['protocol_bytes'] += send(conn, b'H', json.dumps(header).encode())
                if asset_data is not None:
                    report['asset_sha256'] = asset_meta['sha256']
                    for i in range(len(pts)):
                        wait = epoch + pts[i] - pts[0] - time.monotonic()
                        if wait > 0:
                            time.sleep(wait)
                        data = asset_data[asset_meta['offsets'][i]:asset_meta['offsets'][i+1]]
                        report['payload_bytes'] += len(data)
                        report['protocol_bytes'] += send(conn, b'D', data)
                    rc = 0
                else:
                    with open(str(a.report)+'.ffmpeg.log', 'wb') as err:
                        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
                        while True:
                            data = proc.stdout.read1(1316)
                            if not data:
                                break
                            h.update(data)
                            report['payload_bytes'] += len(data)
                            report['protocol_bytes'] += send(conn, b'D', data)
                        rc = proc.wait(timeout=30)
                report['encoder_returncode'] = rc
                report['protocol_bytes'] += send(conn, b'E', json.dumps(dict(returncode=rc, frames=len(pts))).encode())
                if rc:
                    raise RuntimeError(f'encoder exited {rc}; see ffmpeg log')
                report['status'] = 'ok'
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        report['payload_sha256'] = asset_meta['sha256'] if asset_meta else h.hexdigest()
        report['finished_monotonic'] = time.monotonic()
        report['payload_bitrate_bps'] = report['payload_bytes'] * 8 / (len(pts)*step)
        dump(a.report, report)


if __name__ == '__main__':
    main()
