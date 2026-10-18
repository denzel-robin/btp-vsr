"""Validate saved streaming artifacts, optionally exercise broken TCP peers."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
from streaming_common import ROOT, send


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--failure-checks', action='store_true')
    a = p.parse_args()
    reports = {}
    for mode in ('direct','bicubic','nanovsr'):
        folder = a.directory/f'final-{mode}'
        c = json.loads((folder/'client.json').read_text())
        s = json.loads((folder/'server.json').read_text())
        assert c['status'] == s['status'] == 'ok'
        assert c['payload_sha256'] == s['payload_sha256']
        assert c['payload_bytes'] == s['payload_bytes']
        assert c['protocol_bytes'] == s['protocol_bytes']
        assert c['decoded_frames'] == len(c['frames']) == c['config']['frames']
        assert c['unique_render_submissions'] + c['dropped_frames'] == c['decoded_frames']
        assert c['queue_peak'] <= c['config']['queue_frames']
        assert len(c['transport_pts']) == c['decoded_frames']
        assert [f['index'] for f in c['frames']] == list(range(c['decoded_frames']))
        # Independently decode received MPEG-TS and compare every frame to live decode.
        w,h = c['header']['size']
        raw = subprocess.check_output(['ffmpeg','-v','error','-i',str(folder/'client.json.received.ts'),
                                      '-an','-fps_mode','passthrough','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
        n = w*h*3
        assert len(raw) == n*c['decoded_frames']
        assert [hashlib.sha256(raw[i:i+n]).hexdigest() for i in range(0,len(raw),n)] == [f['decoded_sha256'] for f in c['frames']]
        reports[mode] = c
    b,n = reports['bicubic'],reports['nanovsr']
    assert b['payload_sha256'] == n['payload_sha256'], 'comparison streams differ'
    assert [f['decoded_sha256'] for f in b['frames']] == [f['decoded_sha256'] for f in n['frames']]
    overload = json.loads((a.directory/'overload/client.json').read_text())
    assert overload['status'] == 'ok' and overload['dropped_frames'] > 0
    assert overload['queue_peak'] <= overload['config']['queue_frames']
    if a.failure_checks:
        out = a.directory/'failure-checks'
        out.mkdir(exist_ok=True)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0))
            listener.listen(1)
            port = listener.getsockname()[1]
            def broken_peer():
                conn,_ = listener.accept()
                with conn:
                    send(conn,b'H',json.dumps(b['header']).encode())
                    # A valid initial prefix followed by abrupt close; never a clean EOS.
                    send(conn,b'D',(a.directory/'final-bicubic/client.json.received.ts').read_bytes()[:3000])
            t = threading.Thread(target=broken_peer)
            t.start()
            with open(out/'truncated.log','w') as log:
                rc = subprocess.run([sys.executable,str(ROOT/'client/streaming_client.py'),'--headless',
                    '--port',str(port),'--report',str(out/'truncated.json')], stdout=log,stderr=log,timeout=40).returncode
            t.join(timeout=5)
            assert rc != 0 and not t.is_alive()
        truncated = json.loads((out/'truncated.json').read_text())
        assert truncated['status'] == 'error' and any('receiver:' in e for e in truncated['errors'])
        with open(out/'refused.log','w') as log:
            rc = subprocess.run([sys.executable,str(ROOT/'client/streaming_client.py'),'--headless',
                '--port',str(port),'--report',str(out/'refused.json')],stdout=log,stderr=log,timeout=10).returncode
        assert rc != 0 and 'ConnectionRefusedError' in (out/'refused.json').read_text()
    print('PASS: byte counts, identical B/C inputs, timestamps, all decoded frame hashes/order, bounded overload' +
          (', truncated stream and connection refusal' if a.failure_checks else ''))


if __name__ == '__main__':
    main()
