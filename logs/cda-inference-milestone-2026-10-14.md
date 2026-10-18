# CDA checkpoint/model-inference milestone — 2026-10-14

## Outcome

- **Checkpoint loaded successfully:** both `params` and `params_ema` pass
  `load_state_dict(..., strict=True)` against the actual upstream CDAVSR.
  **Selected inference key: `params`**, matching both official test YAMLs.
- **Synthetic recurrent CUDA inference passed** on NVIDIA GeForce RTX 3050
  Laptop GPU: three frames, state reuse, explicit reset, finite outputs/state,
  and x4 output. Evaluation mode and `torch.inference_mode()` were active.
- **Authentic-input inference: NOT TESTED.** No matching processed sample was
  obtained. The official dataset link reaches a Baidu extraction-code page;
  this is not evidence of downloaded or validated dataset contents.
- No training, quality metrics, paper reproduction, or streaming-performance
  claims. Stopped at this milestone.

## Inspected source and actual interface

Local CDA checkout: `5630821e5df4e110878d0caae39bcc53f658e9a9`.
Inspected `basicsr/archs/cdavsr_arch.py`, `arch_util.py`, both test YAMLs,
`basicsr/test.py`, `models/{sr_model,base_model,video_recurrent_model}.py`,
the alternative mix wrapper, dataset readers, package initializers, and README.

`CDAVSR.forward(x, mv, res, hidden_key=None, return_hs=False)` expects:

| Argument/result | Shape and convention |
| --- | --- |
| LR `x` | B,T,3,H,W; float32 RGB, decoded pixel values divided by 255 |
| MV `mv` | B,T,2,H,W; float32, channel order x,y; loader applies no scaling |
| Residual `res` | B,T,1,H,W; codec luma residual map loaded from `.npy`, divided by 255 |
| Output | B,T,3,4H,4W; learned correction plus bilinear x4 LR base; no output clamp |
| State element 0 | B,64,H,W current extracted LR features |
| State element 1 | B,64,H,W reconstructed features after `conv_h` |
| State element 2 | B,3,H,W current LR image; retained but not read by next alignment |

The training loader explicitly transposes MV from 2,H,W to H,W,2 for
augmentation; the test wrapper directly views it as 1,1,2,H,W. Residual files
are H,W in the training loader. Preserve signed residual values if present;
the test path divides by 255 without clipping or subtracting a bias. The
dataset's actual residual range/sign and codec extraction details remain
unverified without a sample/extractor.

From the implemented warp, MV is a sampling displacement from current-grid
coordinates into previous features: `grid + (dx,dy)`, in LR pixel units.
Coarse warp uses nearest sampling, border padding, align_corners=True. The
deformable offsets add `flow.flip(1)` (y,x order) repeated over groups to a
learned tanh-bounded offset correction (magnitude bound 8). Converting raw
codec MV units/reference indices into this dense field is not provided here.

Architecture: 3→64 input convolution; 3 feature-extraction residual blocks;
24 I reconstruction blocks and 12 P blocks. Previous low/high features
concatenate to 128 channels. Alignment condition is 192 features plus 2 MV
channels; offset predictor is 194→16→16→16→12. MMCV DCN is **128→128**,
kernel 1, deform_groups 4, bias=False, with 8 offset and 4 mask channels.
Residual gate is 1→64→64→1, with no terminal sigmoid; aligned features are
multiplied by `(1 + gate)`. Fusion is 192→64; upsampling is 64→48 then
PixelShuffle(4), followed by the bilinear base addition.

First frame with state=None uses the I branch and ignores its MV/residual.
Following frames use P reconstruction and the updated tuple. The module does
not hold hidden state internally. It does not consume a frame-type argument
or automatically reset every 25 frames. The official validation loop sets
state=None at `fi % 25 == 0` and at each clip start; callers must implement
the corresponding reset policy. `num_frame=25` is constructor metadata,
not an enforced input length. A multi-frame call resets only at its first
frame when no initial state is supplied.

## Official path and compatibility decisions

`basicsr/test.py` creates `VideoRecurrentModel`; inherited `SRModel.__init__`
loads the configured `param_key_g`, which is `params` in both test YAMLs.
The active `VideoRecurrentModel.dist_validation` invokes `net_g` with all
three inputs plus state. EMA creation is a training-only path; existence of
`params_ema` in the file does not change the official test selection.

The supplied REDS YAML has an unsupported `spynet_path` constructor argument
and an author-local checkpoint path; the Inter4K checkpoint path is null.
Both contain author-local dataset paths. Their `center_frame_idx: None` is
a YAML string, whereas the harness explicitly passes Python None (the field
is unused by forward). The active validation loop hardcodes 100 frames and
25-frame resets. The separate `test()` helper omits residuals, and the mix
wrapper is also incompatible with the current signature. The harness uses
the actual architecture directly, not these stale orchestration paths.

Added `scripts/check_cda_inference.py`. It creates process-local package
namespaces for basicsr/utils/archs/ops so their eager `__init__` imports do
not load datasets, losses, metrics and training code. It imports the original
logger, registry, arch_util, and CDAVSR implementations from the checkout.
No numerical functions are replaced, copied, mocked, or patched. The legacy
BasicSR DCN helper imports but its absent extension is never invoked; CDAVSR
uses the verified MMCV binary. BASICSR_JIT=True is rejected to prevent an
unnecessary extension build. Upstream bytecode writes are disabled.

**Dependency changes: none.** Existing dependencies suffice, including
torchvision, setuptools/distutils and MMCV. No legacy requirements were
installed. Before/after `.venv-cda` freezes are identical and `pip check`
reports no broken requirements. Torch 2.1.0+cu121, torchvision 0.16.0+cu121,
NumPy 1.26.4, MMCV 2.2.0 and runtime CUDA 12.1 are asserted by the harness.

## Checkpoint details

File: `external/cda-vsr/pretrained_models/best.pth`.
SHA-256: `afc8745b890289ae421c500279d9ccf2a27c92cf3e71133b20840c7816e86d3e`.
Loaded on CPU with `weights_only=True` before CUDA execution.

Both keys contain 179 tensors / 3,110,989 elements. Each has zero missing
keys, zero unexpected keys, zero shape mismatches and zero nonfinite tensors.
Both strictly load. All 179 tensors differ between params and params_ema;
maximum absolute parameter difference is 0.0000959634780883789. After checking
both, the harness strictly reloads **params** for inference. There is no
prefix stripping, fallback key, or strict=False. Full per-tensor shapes and
mismatch lists are in `cda-checkpoint-inspection-2026-10-14.json` and
`cda-inference-results-2026-10-14.json`.

## Exact validation commands and results

From `/home/ibexe/Github/btp-vsr`:

```bash
.venv-cda/bin/python scripts/check_cda_inference.py --inspect-only --report logs/cda-checkpoint-inspection-2026-10-14.json
set -o pipefail
.venv-cda/bin/python scripts/check_cda_inference.py --report logs/cda-inference-results-2026-10-14.json 2>&1 | tee logs/cda-inference-2026-10-14.log
.venv-cda/bin/python -m pip check
```

CPU inspection and approved host GPU execution both exited 0. Synthetic
input was seeded uniform RGB, B=1,T=3,H=16,W=24, with zero MV and zero
residual. Each single-frame output was 1,1,3,64,96; sequence output was
1,3,3,64,96. Every output and recurrent tensor was finite on CUDA without
gradients. Hooks observed I=1/P=2/DCN=2 before reset, and I=2/P=2/DCN=2
after reset, confirming the intended branches and actual DCN execution.
DCN inputs were 1,128,16,24 features, 1,8,16,24 offsets, and 1,4,16,24 masks.

Explicitly resetting state=None after three frames and replaying the first
frame reproduced its output and all three state tensors exactly (max output
error 0.0). A single three-frame forward matched stepwise recurrence with
max output error 0.0 (comparison tolerance rtol=1e-4, atol=1e-5).
Peak PyTorch allocated GPU memory was **27,674,624 bytes** for this small
test, not a full-resolution memory or throughput measurement. Only upstream
TypedStorage/meshgrid deprecation warnings occurred.

## Authentic sample investigation and remaining blockers

The [official README](https://github.com/sspBIT/CDA-VSR#dataset-and-pretrained-models)
advertises processed REDS_CRF18_23_28_new with LR, MV, residual and frame-type
information. Its [Baidu share](https://pan.baidu.com/s/1UHW5piWo2IS17N7ksgi73Q)
uses extraction code `zc8e`. No small standalone matched sample or extraction
script was found in the checkout; the repository's generic LMDB test assets
do not provide a CDA LR/MV/residual triplet. A workspace search found no `.npy`
sample files outside environments. The upstream issue
[How to get MV and Residual Map](https://github.com/sspBIT/CDA-VSR/issues/3)
also asks for the extraction workflow; no answer was visible when inspected.

Browser opening of the Baidu link failed. Approved host command:

```bash
curl -L --max-time 30 -o logs/cda-dataset-download-page.html -w 'HTTP %{http_code}\nFinal URL %{url_effective}\nDownloaded bytes %{size_download}\n' https://pan.baidu.com/s/1UHW5piWo2IS17N7ksgi73Q
```

This exited 0, HTTP 200, 16,012 bytes, redirecting to
`https://pan.baidu.com/share/init?surl=UHW5piWo2IS17N7ksgi73Q`.
The saved page asks for an extraction code. No code submission, file listing
or dataset download was performed; login/captcha/download requirements beyond
this page are unverified. Thus the published dataset route exists, but sample
availability/content has not been validated in this milestone.

For authentic inference, obtain a matched clip starting at its proper reset
boundary: decoded `deLR` RGB frames, corresponding dense 2,H,W `mvs/*.npy`,
and H,W `residual_y/*.npy`, all sorted/aligned by frame ID and at identical
LR dimensions. Use the normalization above and the matching I/P reset policy;
GT at x4 is optional for inference and necessary for any later quality study.
Verify temporal reference direction and MV units from the producer. For new
videos, the missing work is an authenticated codec extraction/preprocessing
pipeline specifying encode/downsample settings, GOP/reference handling,
MV unit conversion/densification, residual decoding and frame alignment.
No residual was replaced by a frame difference. Zero synthetic priors establish
structure only; nonzero codec semantics and reconstruction quality remain
untested.

## Preservation

Before/after original `.venv` freezes in
`cda-inference-original-venv-{before,after}.txt` are identical. CDA dependency
freezes in `cda-inference-deps-{before,after}.txt` are identical. Both checkpoint
hashes match `cda-inference-checkpoints-before.sha256` via `sha256sum -c`.
Both upstream repository statuses remained clean; no upstream source or
existing setup/operator script was edited. No system Python, driver, CUDA,
checkpoint, or existing environment package changes were made. Root Git
cannot assess unrelated changes because the exposed root `.git` is invalid,
as recorded in the preceding milestone.
