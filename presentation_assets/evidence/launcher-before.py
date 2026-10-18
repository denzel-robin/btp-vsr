"""Launch a localhost server/client pair; visible by default, --headless for checks."""
import argparse
from pathlib import Path
import select
import subprocess
import sys
import time
from streaming_common import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['direct','bicubic','nanovsr'], default='bicubic')
    p.add_argument('--config', default='configs/streaming-smoke.json')
    p.add_argument('--output', required=True)
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--observe', action='store_true', help='capture the playback window twice for visual inspection')
    p.add_argument('--processing-delay', type=float, default=0)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    common = ['--config', a.config, '--mode', a.mode, '--port', str(a.port)]
    server_cmd = [sys.executable, str(ROOT/'server/streaming_server.py'), *common, '--report', str(out/'server.json')]
    client_cmd = [sys.executable, str(ROOT/'client/streaming_client.py'), *common, '--report', str(out/'client.json'),
                  '--processing-delay', str(a.processing_delay)] + (['--headless'] if a.headless else [])
    import json
    (out/'commands.json').write_text(json.dumps([server_cmd,client_cmd], indent=2))
    with open(out/'server.log','w') as sl, open(out/'client.log','w') as cl:
        server = subprocess.Popen(server_cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=sl, text=True)
        try:
            if not select.select([server.stdout],[],[],30)[0]:
                raise RuntimeError('server startup timeout')
            line = server.stdout.readline()
            sl.write(line)
            sl.flush()
            if not line.startswith('READY'):
                raise RuntimeError('server startup failed; see server.log')
            client = subprocess.Popen(client_cmd, cwd=ROOT, stdout=cl, stderr=subprocess.STDOUT)
            try:
                if a.observe:
                    time.sleep(1.5 if a.mode != 'nanovsr' else 4)
                    for i in range(2):
                        subprocess.run(['import', '-window', 'BTP streaming', str(out/f'window-{i}.png')],
                                       stdout=cl, stderr=cl, timeout=5)
                        time.sleep(.6)
                client.wait(timeout=180)
            finally:
                if client.poll() is None:
                    client.kill()
                    client.wait()
            rc = server.wait(timeout=40)
            if client.returncode or rc:
                raise RuntimeError(f'client={client.returncode}, server={rc}; see {out}')
        finally:
            if server.poll() is None:
                server.kill()
            server.wait()
    print(f'Completed {a.mode}: {out}')


if __name__ == '__main__':
    main()
