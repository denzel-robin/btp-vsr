"""Recheck saved evaluation coverage and metrics at every chunk boundary."""
import argparse
import csv
import json

import cv2
import numpy as np

from evaluate_x4_video import metrics


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.output
    cfg = json.loads((root/'config.json').read_text())
    n, chunk = cfg['frames'], cfg['chunk_length']
    indices = sorted({0, n-1} | {i for start in range(chunk, n, chunk) for i in (start-1, start)})
    rows = list(csv.DictReader((root/'per-frame.csv').open()))
    assert len(rows) == n * len(cfg['crfs']) * 2
    count = 0
    for crf in cfg['crfs']:
        for method in ('bicubic', 'nanovsr'):
            chosen = [r for r in rows if int(r['crf']) == crf and r['method'] == method]
            assert [int(r['frame']) for r in chosen] == list(range(n))
            assert len(list((root/f'crf{crf}'/method).glob('*.png'))) == n
            for i in indices:
                actual = cv2.imread(str(root/f'crf{crf}'/method/f'{i:06d}.png'))
                reference = cv2.imread(str(root/'hr'/f'{i:06d}.png'))
                np.testing.assert_allclose(metrics(actual, reference),
                                           [float(chosen[i]['psnr_db']), float(chosen[i]['ssim'])],
                                           atol=1e-12, rtol=0)
                count += 1
    result = {'status': 'PASS', 'metric_rows': len(rows),
              'saved_png_metric_rechecks': count, 'boundary_frames_checked': indices}
    (root/'artifact-checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
