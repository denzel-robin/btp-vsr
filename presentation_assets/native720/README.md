# Native 1280×720 road comparison

Selected clip: **Narrowing 2+1 road**, LA2, original 1280×720, 50 FPS;
source seconds 10–20, normalized to 30 FPS / 300 frames with no spatial scaling.
Source is already compressed, so this is an aligned decoded reference, not a
pristine camera master. Source URL, license, hashes and conversion details are
in `provenance.json`. Beginning/middle/end contact sheet is display-only.

License: CC BY-SA 3.0, https://creativecommons.org/licenses/by-sa/3.0/.
Source: https://commons.wikimedia.org/wiki/File:Narrowing_2%2B1_road.ogv.
Original file remains `videos/natural-source/road.ogv`. Derived references,
reconstructions, crops and preview images retain this attribution/license.
No new download was required.

## What is compared

- Direct: H.264 1280×720 → decode → 1280×720 playback.
- Bicubic: H.264 320×180 → decode → bicubic 1280×720.
- NanoVSR: the **identical** H.264 320×180 stream → native x4 1280×720.
  No final spatial resizing. Independent chunk3 with up to 66.7 ms lookahead.

All encodes use CRF23, libx264 veryfast/zerolatency, GOP30, no B frames,
BT.709 limited yuv420p. **Equal CRF is not equal bitrate.** This focused single
clip does not establish a matched-rate direct-delivery trade-off. Actual TS
rates are in `comparison.csv` and the stats panel. The two LR methods use
exactly the same bytes; their quality comparison is at identical bitrate.
These are fresh assets made directly from the 720p reference, not the old
360p-reference study's stream or its quality scores.

Every reconstructed frame is scored offline before preview compression using
the existing aligned RGB uint8 PSNR/SSIM protocol. Quality does not depend on
which frames survive playback. `frame-metrics.csv` holds all 900 scores;
`quality.json` includes checkpoint/asset identities and aggregate results.
Raw reference and reconstruction arrays are retained locally in
`results/native720-road-2026-10-18/`, excluded from Git due size.

## Show the reference and demonstrations

From the project root in kitty/fish, run each separately:

```fish
ffplay -autoexit results/native720-road-2026-10-18/reference.nut
.venv/bin/python scripts/run_streaming_demo.py --mode direct --config presentation_assets/native720/direct.json --output results/720-direct-(date +%s) --stats-hold 20
.venv/bin/python scripts/run_streaming_demo.py --mode bicubic --config presentation_assets/native720/bicubic.json --output results/720-bicubic-(date +%s) --stats-hold 20
.venv/bin/python scripts/run_streaming_demo.py --mode nanovsr --config presentation_assets/native720/nanovsr.json --output results/720-nano-(date +%s) --stats-hold 20
```

Stats are enabled by default; `--no-stats` removes the presentation overhead.
The saved native-resolution quality table is labelled separately from live
latency/submission/drop counters. The server paces pre-encoded assets; it does
not encode live in these commands. Existing sustained-playback limitations
remain; this is not a new long-duration performance claim.

## Reproduce

Obtain the same licensed source at its documented path, upstream model revision
and checkpoint, then use the existing working environment:

```fish
.venv/bin/python scripts/compare_native720.py prepare
.venv/bin/python scripts/compare_native720.py score
```

The scorer refuses to overwrite a completed comparison. Reference frames have
an independent index/PTS/hash manifest. `commands.json` records preparation
commands; `comparison-crops.png` uses fixed frame150 and a declared 1:1 crop.
Do not run metric scoring concurrently with timed playback.

## Synchronized original / transmitted / upscaled view

```fish
.venv/bin/python scripts/run_streaming_demo.py --mode nanovsr --config presentation_assets/native720/side-by-side.json --output results/side-by-side-(date +%s) --stats-hold 20
```

All three panes show the same processed frame ID. The original pane reads the
locally retained aligned 720p reference (validated against its frame hashes
before connecting). The middle pane uses the actual live decoded320x180 frame,
and the right pane uses live native1280x720 NanoVSR output—not precomputed SR.
If a frame misses the rendering deadline, all panes skip that frame together.
The EOF summary shows the final processed frame, which may have been late.

Full-frame previews are512x288 to fit a1536-pixel-wide window. The reference and
SR previews are reduced for display only; the LR preview uses nearest-neighbor
enlargement. Below are identical reference-space detail crops at x500,y300,
width320,height180: reference/SR are1:1 pixels; LR is nearest x4 for display.
This viewport resizing does not alter the1280x720 reconstruction or its metrics.

Each panel identifies resolution/source and relevant stats; the shared live
strip shows submissions, drops, FPS, latency, model time and peak allocation.
Native whole-clip PSNR/SSIM are saved offline scores. Raw LR PSNR/SSIM against
720p is marked N/A because pixel dimensions differ. Reference bytes are local
comparison data, not counted as an additional network stream. The composite
view adds CPU/GUI/I/O overhead and is presentation evidence, not a replacement
for prior performance benchmarks. `--no-stats` returns to native video-only.
