"""CUDA operator milestone only; this does not run CDA-VSR inference."""
import json
import sys

import numpy as np
import torch
import torchvision


def main():
    versions = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    print(json.dumps(versions, indent=2), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: operator verification cannot run")
    import mmcv
    from mmcv.ops import ModulatedDeformConv2d

    assert sys.version_info[:2] == (3, 10), versions
    assert torch.__version__.split('+')[0] == '2.1.0', versions
    assert torchvision.__version__.split('+')[0] == '0.16.0', versions
    assert np.__version__ == '1.26.4', versions
    assert torch.version.cuda == '12.1', versions
    assert mmcv.__version__ == '2.2.0', mmcv.__version__
    print(f"mmcv={mmcv.__version__}; GPU={torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(2026)
    torch.cuda.reset_peak_memory_stats()
    layer = ModulatedDeformConv2d(64, 64, kernel_size=1, deform_groups=4, bias=True).cuda()
    x = torch.randn(1, 64, 16, 16, device='cuda', requires_grad=True)
    offset = (torch.randn(1, 8, 16, 16, device='cuda') * 0.2).requires_grad_()
    mask = torch.rand(1, 4, 16, 16, device='cuda', requires_grad=True)
    y = layer(x, offset, mask)
    assert y.shape == x.shape
    assert torch.isfinite(y).all(), 'nonfinite forward output'
    y.square().mean().backward()
    for name, tensor in [('input', x), ('offset', offset), ('mask', mask),
                         ('weight', layer.weight), ('bias', layer.bias)]:
        assert tensor.grad is not None, f'missing {name} gradient'
        assert torch.isfinite(tensor.grad).all(), f'nonfinite {name} gradient'
        assert tensor.grad.abs().max() > 0, f'zero {name} gradient'
    with torch.no_grad():
        actual = layer(x, torch.zeros_like(offset), torch.ones_like(mask))
        expected = torch.nn.functional.conv2d(x, layer.weight, layer.bias)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        error = (actual - expected).abs().max().item()
    torch.cuda.synchronize()
    print(json.dumps({"status": "PASS", "channels": 64, "kernel_size": 1,
                      "deform_groups": 4, "max_conv_error": error,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "scope": "operator only; full CDA inference untested"}, indent=2))


if __name__ == '__main__':
    main()
