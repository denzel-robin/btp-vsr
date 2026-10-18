"""Localhost protocol: length-prefixed JSON header, framed MPEG-TS, explicit EOS.
Only trusted localhost peers. TCP wire counters exclude TCP/IP/kernel overhead.
"""
import json
import socket
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def window_title(c, mode):
    if not c.get('demo_labels', False):
        return 'BTP streaming'
    tx = c['playback_size'] if mode == 'direct' else c['transmission_size']
    pw, ph = c['playback_size']
    label = f'{mode} | TX {tx[0]}x{tx[1]} | playback {pw}x{ph}'
    if mode == 'nanovsr':
        step = ' no final resize' if c.get('native_output', False) else ' then resize'
        label += f" | native x4 {tx[0]*4}x{tx[1]*4}{step} | chunk{c['chunk_size']}"
    return label


def config(path):
    c = json.loads(Path(path).read_text())
    for key in ('transmission_size', 'playback_size'):
        assert len(c[key]) == 2 and all(isinstance(v, int) and 0 < v <= 4096 and v % 2 == 0 for v in c[key])
    for key in ('frames', 'chunk_size', 'queue_frames'):
        assert isinstance(c[key], int) and 0 < c[key] <= 18000
    assert c['chunk_size'] <= 64 and c['queue_frames'] <= 256
    assert 0 <= c['crf'] <= 51 and c['playout_delay_s'] >= 0 and c['drop_late_s'] >= 0
    return c


def exact(stream, n):
    out = bytearray()
    while len(out) < n:
        part = stream.read(n - len(out))
        if not part:
            raise EOFError(f'truncated stream: expected {n}, got {len(out)} bytes')
        out.extend(part)
    return bytes(out)


def send(sock, kind, data):
    packet = struct.pack('!cI', kind, len(data)) + data
    sock.sendall(packet)
    return len(packet)


def receive(stream):
    kind, size = struct.unpack('!cI', exact(stream, 5))
    if size > 4 * 1024 * 1024:
        raise ValueError('oversized protocol record')
    return kind, exact(stream, size), size + 5


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + '\n')


def tcp_options(sock):
    sock.settimeout(30)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
