"""Strict CDA checkpoint inspection and synthetic CUDA recurrence smoke test.

Run with .venv-cda/bin/python. No quality or performance measurements.
The package shells below bypass eager training imports; all model/helper
implementations still come directly from the unchanged upstream checkout.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

import mmcv
import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'external/cda-vsr'


def load_architecture():
    if 'basicsr' in sys.modules:
        raise RuntimeError('Run this harness in a fresh Python process')
    if os.environ.get('BASICSR_JIT') == 'True':
        raise RuntimeError('Unset BASICSR_JIT; this harness uses the MMCV binary')
    sys.dont_write_bytecode = True
    for name in ('basicsr', 'basicsr.utils', 'basicsr.archs', 'basicsr.ops'):
        package = types.ModuleType(name)
        package.__path__ = [str(REPO.joinpath(*name.split('.')))]
        sys.modules[name] = package
    logger = importlib.import_module('basicsr.utils.logger')
    sys.modules['basicsr.utils'].get_root_logger = logger.get_root_logger
    return importlib.import_module('basicsr.archs.cdavsr_arch').CDAVSR


def inspect_checkpoint(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    expected = model.state_dict()
    details = {}
    for key in ('params', 'params_ema'):
        if key not in checkpoint:
            details[key] = {'present': False}
            continue
        state = checkpoint[key]
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        mismatched = {k: {'checkpoint': list(state[k].shape),
                          'model': list(expected[k].shape)}
                      for k in set(state) & set(expected)
                      if state[k].shape != expected[k].shape}
        nonfinite = [k for k, v in state.items() if not torch.isfinite(v).all()]
        details[key] = dict(present=True, tensors=len(state),
                            elements=sum(v.numel() for v in state.values()),
                            missing=missing, unexpected=unexpected,
                            shape_mismatches=mismatched, nonfinite=nonfinite,
                            shapes={k: list(v.shape) for k, v in state.items()})
        if not (missing or unexpected or mismatched or nonfinite):
            model.load_state_dict(state, strict=True)
            details[key]['strict_load'] = 'PASS'
    if 'params' in checkpoint and 'params_ema' in checkpoint:
        a, b = checkpoint['params'], checkpoint['params_ema']
        common = [k for k in a.keys() & b.keys() if a[k].shape == b[k].shape]
        details['comparison'] = {
            'different_tensors': sum(not torch.equal(a[k], b[k]) for k in common),
            'max_absolute_difference': max((a[k] - b[k]).abs().max().item() for k in common)}
    # Select exactly the official test YAML's param_key_g; no fallback or renaming.
    return checkpoint, details


def checked(output, state, h, w):
    assert list(output.shape) == [1, 1, 3, 4*h, 4*w], output.shape
    assert isinstance(state, tuple) and len(state) == 3
    assert [list(t.shape) for t in state] == [[1, 64, h, w], [1, 64, h, w], [1, 3, h, w]]
    for tensor in (output, *state):
        assert tensor.is_cuda and not tensor.requires_grad
        assert torch.isfinite(tensor).all(), 'nonfinite output/state'
    return {'output_shape': list(output.shape),
            'state_shapes': [list(t.shape) for t in state], 'finite': True}


@torch.inference_mode()
def smoke(model, h, w):
    assert torch.cuda.is_available(), 'Host CUDA access required'
    torch.manual_seed(2026)
    model = model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    calls = {'I': 0, 'P': 0, 'DCN': 0}
    dcn_shapes = []

    def hook(name):
        def record(module, inputs, output):
            calls[name] += 1
            if name == 'DCN':
                dcn_shapes.append({'inputs': [list(t.shape) for t in inputs],
                                   'output': list(output.shape)})
        return record

    handles = [model.reconstruction_I.register_forward_hook(hook('I')),
               model.reconstruction_P.register_forward_hook(hook('P')),
               model.deform_align.dcn.register_forward_hook(hook('DCN'))]
    frames = torch.rand(1, 3, 3, h, w, device='cuda')
    mv = torch.zeros(1, 3, 2, h, w, device='cuda')
    res = torch.zeros(1, 3, 1, h, w, device='cuda')
    hidden = None
    outputs, records = [], []
    initial_state = None
    for i in range(3):
        out, hidden = model(frames[:, i:i+1], mv[:, i:i+1], res[:, i:i+1],
                            hidden, return_hs=True)
        records.append(checked(out, hidden, h, w))
        outputs.append(out)
        torch.testing.assert_close(hidden[2], frames[:, i], rtol=0, atol=0)
        if i == 0:
            initial_state = tuple(t.clone() for t in hidden)
    assert calls == {'I': 1, 'P': 2, 'DCN': 2}, calls
    # Explicit reset after recurrence: replay first frame and compare every state.
    hidden = None
    reset_out, hidden = model(frames[:, :1], mv[:, :1], res[:, :1], hidden, return_hs=True)
    reset_record = checked(reset_out, hidden, h, w)
    torch.testing.assert_close(reset_out, outputs[0], rtol=0, atol=0)
    for actual, expected in zip(hidden, initial_state):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == {'I': 2, 'P': 2, 'DCN': 2}, calls
    for handle in handles:
        handle.remove()
    # Confirm the public multi-frame forward agrees with caller-managed recurrence.
    sequence, sequence_state = model(frames, mv, res, None, return_hs=True)
    sequential = torch.cat(outputs, dim=1)
    torch.testing.assert_close(sequence, sequential, rtol=1e-4, atol=1e-5)
    for tensor in (sequence, *sequence_state):
        assert torch.isfinite(tensor).all()
    torch.cuda.synchronize()
    return {'status': 'PASS', 'gpu': torch.cuda.get_device_name(0),
            'input_shape': list(frames.shape), 'motion': 'synthetic zeros',
            'residual': 'synthetic zeros; not codec residuals', 'frames': records,
            'reset': reset_record, 'reset_output_max_error': (reset_out-outputs[0]).abs().max().item(),
            'reset_states_exact': True, 'branch_calls_including_reset': calls,
            'dcn_shapes': dcn_shapes, 'sequence_output_shape': list(sequence.shape),
            'sequence_vs_step_max_error': (sequence-sequential).abs().max().item(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'eval_mode': not model.training, 'inference_mode': torch.is_inference_mode_enabled()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=REPO/'pretrained_models/best.pth')
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--height', type=int, default=16)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    assert args.height > 1 and args.width > 1
    versions = {'torch': torch.__version__, 'torchvision': torchvision.__version__,
                'numpy': np.__version__, 'mmcv': mmcv.__version__, 'cuda': torch.version.cuda}
    assert versions == {'torch': '2.1.0+cu121', 'torchvision': '0.16.0+cu121',
                        'numpy': '1.26.4', 'mmcv': '2.2.0', 'cuda': '12.1'}, versions
    model = load_architecture()(num_feat=64, num_frame=25, num_extract_block=3,
                               num_reconstruct_block_I=24, num_reconstruct_block_P=12,
                               center_frame_idx=None, hr_in=False)
    checkpoint, details = inspect_checkpoint(model, args.checkpoint)
    report = {'versions': versions, 'checkpoint': str(args.checkpoint.resolve()),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'selected_key': 'params', 'checkpoint_details': details,
              'authentic_input_inference': 'NOT TESTED', 'synthetic_inference': 'NOT RUN'}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    assert details['params'].get('strict_load') == 'PASS', 'See report for all checkpoint mismatches'
    model.load_state_dict(checkpoint['params'], strict=True)
    if not args.inspect_only:
        report['synthetic_inference'] = smoke(model, args.height, args.width)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'checkpoint_details'}, indent=2))


if __name__ == '__main__':
    main()
