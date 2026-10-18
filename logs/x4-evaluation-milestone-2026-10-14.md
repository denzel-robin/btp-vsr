# Reproducible x4 video-evaluation milestone — 2026-10-14

## Outcome and scope

**PASS: synthetic offline evaluation smoke run**, CRF 18/23/28, bicubic and
pretrained NanoVSR, 31 matched frames per method/CRF. This is a synthetic
test-pattern validation, not natural-video quality evidence or a research
improvement. Authentic CDA evaluation remains pending. No training started.

Artifacts: `results/x4-smoke-2026-10-14-r2/` contains `config.json`,
`commands.json` (exact FFmpeg/ffprobe argv), `checkpoint.json`, `results.json`,
`per-frame.csv`, `per-clip.csv`, `alignment.csv`, `artifact-checks.json`,
lossless `hr/*.png`, each CRF's encoded `lr.mp4` and decoded `lr/*.png`,
`bicubic/*.png`, `nanovsr/*.png`, and a labelled `preview.mp4` with PNG inputs.
The preview is 960×180, 31 frames, 1.033333 seconds, CRF23 comparison, ordered
Reference | Bicubic | NanoVSR. It is display-only and never scored.

## Project/source inspection

Read both CDA milestone reports and inspected NanoVSR model, utils loader,
demo and evaluator, all existing baseline/video helpers, and server/client.
NanoVSR revision: `b48e2bad89d01e22226eb8613bee25cb45623fae` (clean).

- `create_low_res.py` creates a **2x** area-downsampled MPEG4 video.
- `create_streaming_baselines.py` independently scales to 640×360 and 320×180
  with CRF23; this is also a 2x relationship, not matched x4 reconstruction.
- `baseline_video.py` simply re-encodes decoded frames; it is not SR.
- The HTTP server serves files. The client downloads the whole file before
  OpenCV validation; its transfer measurement is not a streaming SR latency.
- Upstream evaluation can silently truncate unequal counts and uses a
  non-strict model loader. The new harness rejects count/shape/key mismatches.

Available source `videos/original/test.mp4`: H.264, 640×360, 30 fps,
150 frames/5 seconds. Visual inspection shows a synthetic color/test pattern.
Its creation command is unknown; source bytes SHA-256 and full ffprobe output
are recorded in `results.json`. We use decoded source frames 0–30 as the
reference, with no spatial upscaling. They are already compressed source
imagery, not pristine camera ground truth.

Other local files are derivatives: `hr_640x360`, `test_baseline`, and
`streamed_hr` are 640×360; `test_2x_lr`, `lr_320x180`, and `streamed_lr` are
320×180. `videos/output/nanovsr_720p.mp4` is a generated 1280×720 SR output,
not an authentic HR reference. All inspected videos contain 150 frames.
No suitable natural-video source was found. Required next source: genuine
1280×720 natural footage with documented provenance, producing **320×180 LR
and 1280×720 reconstruction**. A natural 640×360 source can only support a
160×90→640×360 smoke test under this protocol.

## Checkpoint and temporal behavior

Checkpoint: `external/nanovsr/checkpoints/nanovsr_226k.pth`.
SHA-256: `be3df12f2a0d99d9699656aa4a85a51c119663aa84c7040ac6580a0215d0b00e`.
The file is a root state dictionary, 294 tensors, 32 features, 8 blocks in
each forward/backward trunk. **Strict loading passed: missing=[],
unexpected=[], shape_mismatches={}.**

The harness uses upstream config detection and the unchanged NanoVSR class.
It preserves the official loader's key precedence (params_ema, params,
model_state_dict, otherwise root), deployment reparameterization and float32
evaluation behavior, while requiring strict loading. No upstream edits.
Deployment conversion was compared to the original topology on a seeded
1×3×3×8×8 input: max absolute difference **5.960464477539062e-7**, within
rtol=1e-4/atol=1e-5.

Input is B,T,3,H,W RGB float32 /255. Bidirectional feature propagation is
internal to each forward call and initialized to zero in both directions.
There is no state API or state persistence across calls. We use independent,
non-overlapping chunks [0,15), [15,30), [30,31), matching demo chunk length 15.
No temporal padding, repeated frames, overlap, blending, or tail duplication;
the final chunk is length 1. Convolutions retain upstream spatial zero padding.
Backward features are inserted into original temporal order, and output[i]
uses input[i]'s bilinear base. Every input produces one output in order.

A frame can depend on all later frames in its chunk: maximum lookahead
14 frames (14/30 seconds at this source FPS). Both recurrent directions reset
at chunk boundaries. This baseline is not causal and results can depend on
chunk size; no streaming claim follows from its offline timing.

## Reproducible preprocessing and alignment

Added `configs/evaluation-smoke.json` and `scripts/evaluate_x4_video.py`.
The script rejects an existing output directory and unsupported source sizes.
For another source, create a new config with provenance/source_kind; it does
not resize arbitrary footage upward to manufacture a reference.

1. FFprobe source frame timestamps; require constant frame intervals.
2. Decode exactly the first 31 source frames with FFmpeg to RGB24, preserving
   frame rate/order, no audio. Save identical decoded RGB references as PNG.
3. Feed that PNG sequence at 30/1 fps to FFmpeg; downsample 640×360→160×90
   with `scale=160:90:flags=bicubic:in_range=full:out_range=limited:out_color_matrix=bt709,format=yuv420p`.
4. Encode libx264, preset medium, CRF 18/23/28, one encoder thread,
   `-g 30 -keyint_min 30 -sc_threshold 0 -bf 0 -refs 1`, yuv420p, limited
   range, BT.709 matrix/primaries/transfer metadata. No audio or frame-rate
   conversion. FFmpeg version n9.0.1/build is saved verbatim.
5. Decode each MP4 once to RGB24. **Both methods consume that same array**,
   also saved as PNG, not different re-encodes. HR 640×360, LR 160×90,
   and reconstructed 640×360 dimensions are asserted.

Encoded ffprobe verifies 31 frames, no B-frames, one reference frame, and
timestamps matching the normalized source timeline within 2 microseconds.
Duration matches 31/30 seconds. No frame-count truncation is allowed.
Per-frame HR/LR PNG hashes and timestamps are in `alignment.csv`; output
indices are verified to be exactly 0–30 per method/CRF. Saved metrics were
recomputed from PNGs at frames 0,14,15,29,30 (30 checks across chunk boundaries).

| CRF | MP4 bytes | Video packet bytes | Duration (s) | MP4 bitrate (bps) | Video payload bitrate (bps) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 18 | 8,935 | 7,964 | 1.033333 | 69,174.22 | 61,656.79 |
| 23 | 6,742 | 5,772 | 1.033333 | 52,196.15 | 44,686.47 |
| 28 | 5,393 | 4,423 | 1.033333 | 41,752.27 | 34,242.59 |

Bitrate is actual bytes×8/stream duration, not nominal encoder settings.
MP4 bytes include container overhead; payload uses the sum of frame packet
sizes. Encoded hashes and full per-frame probe records are saved.

## Metric protocol and smoke results

Metrics use aligned **RGB uint8 [0,255]**, before any lossy output encoding.
NanoVSR output is clipped to [0,1], multiplied by 255 and rounded with Torch
round (ties-to-even), then converted to uint8, matching the upstream demo.
Bicubic uses OpenCV `INTER_CUBIC` directly on decoded uint8 RGB, with its
uint8 saturation/rounding. HR is the saved decoded RGB reference. Neither
method uses a Y-only conversion. PNG is lossless and scoring does not include
preview labels, resizing or encoding.

PSNR: float64 MSE across all H×W×3 samples, peak 255, 10 log10(255²/MSE),
infinite for identity. No explicit border crop. SSIM uses the inspected
upstream implementation: per-channel float64, Gaussian 11×11 window sigma
1.5, K1=.01, K2=.03, population covariance, valid window centers (exclude
5 pixels each side), averaged over spatial positions and RGB channels.
There is no additional border shave. Clip results are arithmetic means of
per-frame PSNR in dB and per-frame SSIM, not PSNR of a pooled clip MSE.

Checks passed: identity PSNR infinity/SSIM 1; constant 10-level error PSNR
against analytic 20log10(255/10); analytic constant-image SSIM; symmetry;
spatial displacement and temporal permutation rejection. Saved PNG/CSV metric
rechecks passed. Total 186 metric rows = 31×2×3.

**Synthetic test-pattern numbers only:**

| CRF | Bicubic mean PSNR | NanoVSR mean PSNR | Bicubic mean SSIM | NanoVSR mean SSIM |
| --- | ---: | ---: | ---: | ---: |
| 18 | 20.483768 | 20.622133 | 0.834946 | 0.812958 |
| 23 | 20.436096 | 20.551372 | 0.817678 | 0.798027 |
| 28 | 20.363237 | 20.484733 | 0.800080 | 0.783611 |

These numbers exercise the evaluation machinery. They do not support claims
about natural-video quality, paper reproduction, or research improvement.

## Timing scope and actual results

Host GPU execution was explicitly approved. RTX 3050 Laptop GPU, 4096 MiB,
driver 615.71.09; working `.venv` torch 2.14.0+cu130. Float32, eval mode,
inference mode, no autocast; TF32 disabled for CUDA matmul/cuDNN, cuDNN
benchmark disabled. Batch 1, input 160×90, chunks 15/15/1. Warmup: two
untimed forward calls per distinct chunk length per CRF (lengths 15 and 1).

CUDA events surround only model forward, after input transfer and explicit
device synchronization; end event is synchronized before reading elapsed
time. This excludes H2D transfer, output quantization/D2H, decoding, scoring,
disk I/O and preview encoding. Peak allocated memory is measured per timed
forward and maxed across chunks, including resident model/input/output.
Bicubic runs on CPU and has no model GPU timing.

| CRF | Timed frames | Model GPU total ms | Peak allocated bytes | Bicubic CPU seconds | Offline clip wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 18 | 31 | 167.966 | 197,511,168 | 0.028057 | 5.430619 |
| 23 | 31 | 168.749 | 197,511,168 | 0.010626 | 5.546155 |
| 28 | 31 | 172.369 | 197,511,168 | 0.010750 | 5.181288 |

Total offline wall time: **17.336495 seconds**, from entry into main through
preprocessing, loading/conversion, warmup, inference, metrics, PNG/CSV writes
and preview production; excludes Python imports and final JSON serialization.
Per-clip wall time includes that clip's preprocessing and artifacts; CRF23
also includes preview encoding. These are single-run measurements, not a
statistical performance benchmark, live throughput, or streaming latency.

## Commands, correction, and preservation

Successful run and artifact verification, from project root:

```bash
set -o pipefail
.venv/bin/python scripts/evaluate_x4_video.py --config configs/evaluation-smoke.json --output results/x4-smoke-2026-10-14-r2 2>&1 | tee logs/x4-evaluation-smoke-retry-2026-10-14.log
.venv/bin/python scripts/verify_x4_evaluation.py results/x4-smoke-2026-10-14-r2
```

The initial run used output `results/x4-smoke-2026-10-14` and failed during
GPU warmup: deployment conversion inside inference_mode created parameters
without version counters. Corrected the harness to convert under no_grad;
forward measurements still use inference_mode. Initial partial artifacts/log
are retained and are not a successful evaluation. The corrected run exited 0.

No dependency installs/upgrades. `logs/eval-venv-{before,after}.txt` and
`logs/eval-cda-{before,after}.txt` compare identically. Both checkpoint hashes
pass `sha256sum -c logs/eval-checkpoints-before.sha256`. Both external Git
working trees remain clean. Existing source videos, scripts, server/client,
environments and checkpoints were not edited. Root Git is unavailable as
documented in the earlier milestones.

## CDA and next requirements

The checked local CDA README publishes Baidu extraction code **`zc8e`** at
line 106, for `https://pan.baidu.com/s/1UHW5piWo2IS17N7ksgi73Q`.
Authentic CDA evaluation remains pending acquisition/verification of matching
LR frames, motion vectors and codec residuals. No fake residuals were used.
This blocker did not prevent the bicubic/NanoVSR smoke run. Next requirements
are a genuine natural HR source and authentic codec priors for a separate CDA
study. Fine-tuning and streaming evaluation have not begun.
