# Natural-video streaming trade-off milestone — 2026-10-15

Status: **complete — focused natural-video comparison and repeated GUI streaming tests passed.**

**Result:** NanoVSR improves PSNR/SSIM over bicubic on road and waterfall,
but worsens both on flowers. At the selected near-matched rates, direct 640×360
delivery has higher PSNR/SSIM on all three clips and much lower client
reconstruction cost. The current NanoVSR GUI pipeline does not sustain the
30-fps/latency target consistently despite fitting in GPU memory. Direct road
delivery also passes a three-run 150 ms buffer test. These conclusions apply
to this model, encoder, device and three short clips, not a universal optimum.

## Scope and preservation

Question: does the unchanged pretrained NanoVSR improve whole-stream RGB
quality over bicubic at identical transmitted bytes, at acceptable client cost?
Large upscale factors are not the objective. No training or CDA data work.
Read the previous streaming milestone and inspected server/client, launcher,
configuration, metric implementation and project video assets. No applicable
AGENTS.md; root Git remains unavailable. Only project assets were searched;
all existing video assets were synthetic derivatives. Original streaming source
files were backed up under results/natural-2026-10-15/backups. Prior results,
environments and checkpoints are preserved.

## Sources and references

Three independently sourced, already compressed natural videos, with visible
content checked using beginning/middle/end contact sheets:

- Low motion: *Flowers in park*, Editor, CC BY 3.0, 1920×1080, first 9 s
  (270 frames at the reference rate; source is approximately 9.46 s).
  https://commons.wikimedia.org/wiki/File:Flowers_in_park.webm
- Camera/object motion: *Narrowing 2+1 road*, LA2, CC BY-SA 3.0, 1280×720,
  source seconds 10–20, 300 reference frames.
  https://commons.wikimedia.org/wiki/File:Narrowing_2%2B1_road.ogv
- Fine texture: *Professor Creek Waterfall*, BLMUtah / Bill Stevens, public-domain
  US federal government work, 1920×1080, first 10 s, 300 reference frames.
  Includes textured rock, moving water, people and an original title overlay.
  https://commons.wikimedia.org/wiki/File:Day_2-_Spring_footage_of_Professor_Creek_Waterfall_(20088375152).webm

Full source URLs, license URLs, author credits, downloaded bytes/SHA-256, source
ffprobe metadata and exact conversion commands are in provenance.json and
videos/natural-source/ATTRIBUTION.md. Road-derived imagery remains CC BY-SA 3.0.
Neither existing synthetic video nor unrelated personal directories were used.

Reference conversion: normalize PTS; nearest-frame duplication/drop to 30 fps;
bicubic downsample to 640×360; no aspect-ratio stretch, no spatial upscaling;
remove audio; lossless RGB FFV1/NUT and uint8 RGB .npy arrays. NUT preserves the
rational frame timestamps. Initial FFV1/MKV preparation artifacts are retained
but not used because Matroska's default millisecond timestamps quantize 30 fps.
These are converted references from compressed sources, not pristine ground
truth. Reference arrays and frame-index/PTS/hash CSVs preserve alignment.

## Initial GPU capacity check

scripts/check_natural_capacity.py ran before quality screening. Seeded synthetic
RGB is used only for this shape/cost test: input 1×3×3×180×320; native output
1×3×3×720×1280; final CPU bicubic resize to 640×360. RTX 3050 Laptop, unchanged
float32 model/checkpoint, strict loading, deployment conversion, inference mode.
First chunk: 217.72 ms model, 307.87 ms full processing. After two initial calls,
ten warm measurements averaged 32.52 processed fps, with about 72.7 ms model
and 14.9 ms final resizing per three-frame chunk. Includes a finite-output check
in this capacity test. Peak allocated 376,262,144 bytes; reserved 545,259,520;
device-reported capacity 4,007,591,936 bytes. It fits, but the processing margin
above 30 fps is small and does not prove sustained playback. No input-resolution
reduction or buffer increase was made.

## Offline quality protocol

scripts/natural_quality.py: each reference encoded once at each width/CRF:
640×360 direct; 320×180 low resolution; CRF18/23/28; H.264 libx264 veryfast,
zerolatency, one thread, GOP30, no B frames, one reference, disabled scene cuts,
yuv420p limited-range BT.709, MPEG-TS. Low-resolution bitstream is decoded once
and fed to both bicubic and NanoVSR. Native NanoVSR x4 produces 1280×720;
clamp to [0,1], multiply255, torch round, uint8, CPU bicubic to 640×360. Both
bidirectional recurrent paths reset every independent non-overlapping chunk3;
maximum future lookahead is two frames, 66.7 ms. No model modifications.

All frames are scored, before any lossy preview encoding, using the established
RGB uint8 protocol: float64 PSNR, max255; SSIM averaged across RGB with Gaussian
11×11 sigma1.5 and valid interior (five-pixel border exclusion inherent in SSIM),
no extra crop. Clip metrics are arithmetic means of per-frame PSNR/SSIM. Full
reference and reconstructed .npy arrays, all per-frame CSVs and encoded assets
are retained. Dropped presentation frames do not remove any quality rows.
Equal CRF across resolutions is not equal bitrate. Rate uses actual TS bytes
including mux overhead over reference duration; video packet bytes also saved.

## Timed streaming protocol

The server gained a preencoded-asset path. It validates the asset hash before
scheduling, and sends the unchanged TS bytes over localhost TCP in ffprobe PES
packet-position groups paced to original PTS. No timed re-encoding or remuxing;
thus timing isolates delivery, decode, processing and presentation. The original
live-encode path remains available. Packet boundaries and normalized PTS are
verified; predictive frames are never discarded. Six-frame decoded queue,
finite pipes/socket buffers, backpressure; only late reconstructed frames drop.

Verification mode retains payload capture, decoded/output hashes and EOS
ffprobe checks, and compares every live reconstruction to the offline arrays.
Timing mode disables capture and per-frame/payload diagnostic hashing, and
writes JSON after playback; no PNG or metric scoring occurs in timed playback.
Counts, frame order, EOS, queue metrics and CUDA-event timing remain enabled.

Initial buffer remains 350 ms with the existing 500 ms late-drop tolerance.
This permits up to 850 ms application latency before a drop; absence of drops
alone is not sufficient to call latency good. Cold means fresh client/model
with no explicit warmup; operating-system/GPU caches and power state are not
reset. Final warm trials execute three zero-input chunks through the exact live tensor
layout, H2D transfer, model, native output quantization/D2H and final resizing
before connecting; warmup is separately timed. GUI submission itself is not
warmed and remains in the measured run. Model load
also occurs before connection and is reported separately. No concurrent GPU
jobs are allowed during timed repetitions.

Application latency = return from timed submission/OpenCV imshow+waitKey minus
server's common-monotonic source schedule; not physical display or camera
latency. Processing cost includes H2D, native1280×720 model output, quantization,
D2H and final resize. CUDA-event forward time and resize time are separate.
Unique submissions/source-duration measure completeness; active submission fps
uses (N−1)/(last−first submission), and session fps uses N/(last−server epoch).
Latency percentiles are explicitly over submitted frames, with drops reported
alongside; quality always covers all reconstructed frames. Queue age is decode
completion to chunk processing start, including chunk collection/backpressure.

For screening a lower buffer, a viable point must have zero drops in all three
initial repetitions and p95 latency below 450 ms (350 ms plus 100 ms allowance).
This is a declared local application criterion, not a universal user threshold.

## Verification and retained exceptions

Headless diagnostic replays passed for all three modes on the flower clip:
270/270 source timestamps, actual payload/protocol bytes, all decoded hashes and
all output hashes match the offline arrays exactly. This verifies that the
streaming and offline reconstruction paths agree. Metric self-checks and 186
saved-array metric rechecks passed, including chunk boundaries and final frames.
The extra road rate-match refinement brings the chosen direct rate within 1.3%
of the low-resolution rate; all initial extra assets are retained.

The first direct timing run overlapped the final CPU metric verification job.
It is retained but explicitly excluded via excluded-runs.json and replaced
with a fresh serial run. The verifier ended before the second timing run began.
No offline metric/model job runs concurrently with the valid timed repetitions.

An Escape keypress cancelled waterfall/NanoVSR/cold repetition2 after 186 GUI
submissions (196 decoded). The client records cancellation; closing the socket
produced the server's explicit BrokenPipeError. This incomplete run is retained
as a failed attempt, not counted as a full playback result. The remaining
selection resumes in a new output directory, including that repetition. No
settings were changed to conceal either exception.

The first nine warm trials used contiguous zero-tensor model forwards only.
Their first real chunks still paid tensor-layout/conversion initialization
(~140–147 ms). They are retained as model-only warm trials but excluded from
the final comparison, which repeats only the nine warm trials with the exact
live processing path warmed. Cold trials are unchanged. Both warmup modes
remain reproducible through the explicit warmup_pipeline configuration flag.

## Measured comparison

All quality rows below score every frame. B/C use the same CRF23 bitstream. Direct CRFs were selected from the measured rate curve to approach that bitrate. Streaming columns show three GUI repetitions at the unchanged 350 ms buffer; C uses three full processing warmup chunks. Latency p95 and active unique submission fps are ranges across repetitions. Drops are summed across repetitions. Reconstruction/resize costs are means and exclude decode, GUI and waits; these are reflected in submission rates/latency. GPU allocation is the largest PyTorch allocator peak, not total process/desktop memory.

| Clip | Method / CRF | TS bytes / kbps | RGB PSNR / SSIM | Processing ms/frame | Active submission fps | Latency p95 ms | Drops / frames | Peak PyTorch MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| flowers | direct / 30.1 | 545,388 / 484.79 | 30.687 / 0.90241 | 0.01 | 30.01–30.02 | 353.63–353.70 | 0 / 810 | 0.00 |
| flowers | bicubic / 23 | 545,200 / 484.62 | 29.253 / 0.87952 | 0.49 | 30.01–30.01 | 353.68–353.87 | 0 / 810 | 0.00 |
| flowers | nanovsr / 23 | 545,200 / 484.62 | 29.088 / 0.87352 | 31.15 | 29.33–29.91 | 438.44–588.98 | 0 / 810 | 358.83 |
| road | direct / 30.4 | 565,880 / 452.70 | 32.730 / 0.90796 | 0.01 | 30.02–30.02 | 353.74–353.86 | 0 / 900 | 0.00 |
| road | bicubic / 23 | 559,112 / 447.29 | 30.504 / 0.88714 | 0.50 | 30.01–30.01 | 353.76–353.84 | 0 / 900 | 0.00 |
| road | nanovsr / 23 | 559,112 / 447.29 | 31.743 / 0.89821 | 31.31 | 29.24–29.67 | 507.86–640.60 | 0 / 900 | 358.83 |
| waterfall | direct / 29.8 | 555,728 / 444.58 | 29.568 / 0.86169 | 0.01 | 30.01–30.02 | 353.68–353.86 | 0 / 900 | 0.00 |
| waterfall | bicubic / 23 | 543,884 / 435.11 | 27.759 / 0.80361 | 0.47 | 30.01–30.01 | 353.68–354.79 | 0 / 900 | 0.00 |
| waterfall | nanovsr / 23 | 543,884 / 435.11 | 28.216 / 0.81737 | 31.40 | 29.23–29.44 | 579.00–641.27 | 0 / 900 | 358.83 |

Matched-rate residuals (direct TS bytes relative to B/C):
- flowers: +0.03%; direct minus NanoVSR PSNR +1.599 dB, SSIM +0.02888.
- road: +1.21%; direct minus NanoVSR PSNR +0.986 dB, SSIM +0.00975.
- waterfall: +2.18%; direct minus NanoVSR PSNR +1.352 dB, SSIM +0.04431.

Cold versus explicitly warmed NanoVSR (three runs of each per clip):
- flowers, cold: first chunk 273.5–278.3 ms; first-frame latency 414.8–419.2 ms; drops 0; model load mean 1.432 s; explicit warmup mean 0.000 s.
- flowers, warm: first chunk 91.7–93.3 ms; first-frame latency 356.6–356.9 ms; drops 0; model load mean 1.397 s; explicit warmup mean 0.451 s.
- road, cold: first chunk 272.4–279.4 ms; first-frame latency 412.7–420.5 ms; drops 0; model load mean 1.426 s; explicit warmup mean 0.000 s.
- road, warm: first chunk 91.5–92.8 ms; first-frame latency 356.1–356.9 ms; drops 0; model load mean 1.409 s; explicit warmup mean 0.454 s.
- waterfall, cold: first chunk 275.9–587.2 ms; first-frame latency 415.9–728.6 ms; drops 0; model load mean 1.846 s; explicit warmup mean 0.000 s.
- waterfall, warm: first chunk 92.4–94.1 ms; first-frame latency 355.9–357.0 ms; drops 0; model load mean 1.407 s; explicit warmup mean 0.456 s.

All p50/p95/max latency, queue-age percentiles/maxima, per-frame costs, GPU allocated/reserved, source-duration/session/active submission rates, actual protocol bytes and completeness are in streaming-summary.csv and per-run JSON. All 27 base rate-quality rows and targeted direct extras are in quality-summary.csv; per-frame CSVs retain every metric.

## Interpretation, bottlenecks and lower buffer

At identical LR bytes, NanoVSR minus bicubic at CRF23 is −0.165 dB / −0.00600
SSIM on flowers, +1.239 dB / +0.01107 on road, and +0.458 dB / +0.01377 on
waterfall. The signs hold at CRF18 and CRF28 for each clip. Thus NanoVSR is
**better on two contents and worse on one**, not a universal quality win.
There is no evidence here that perceptual quality or temporal flicker improves;
PSNR/SSIM and the representative crops answer only the measured spatial question.

The direct comparison is not an equal-CRF comparison. The targeted direct
encodes are within 2.2% of B/C's actual TS rate. Direct beats NanoVSR by
0.986–1.599 dB and 0.00975–0.04431 SSIM. Already-overlapping base points tell
the same story: direct CRF28 beats LR+NanoVSR CRF18 on both metrics while using
20.5–37.6% fewer TS bytes across these three clips. The initial road direct
CRF31.1 point also beat NanoVSR CRF23 with 12.1% fewer bytes; it is retained
alongside the refined CRF30.4 match. Direct delivery is preferable at these
measured overlapping rates. This does not establish superiority at every rate,
with every codec/preset, or on other content.

All 39 valid timed GUI runs (36 initial-buffer comparisons plus three lower-buffer
runs) submitted **11,340 unique frames**, with **zero drops** and exact frame
counts/EOS. Completeness is 100%; submissions/source-duration are 30 fps.
Those figures do not imply sustained 30-fps pacing: warmed NanoVSR active
submission rate was only **29.23–29.91 fps**, with growing latency. Its per-run
p50 latency ranged **392.01–510.40 ms**, p95 **438.44–641.27 ms**, and maxima
**445.24–677.85 ms**. The existing 500 ms lateness tolerance allows up to about
850 ms before dropping, so zero drops conceal neither the lateness nor the
missed 450 ms p95 criterion. No resolution or buffer was increased to mask it.

NanoVSR's timed forward cost is **24.29–24.63 ms/frame**, final 1280×720→640×360
CPU resize **4.98–5.41 ms/frame**, and full reconstruction/resize about
**31.2–31.4 ms/frame**. This leaves roughly 2 ms of a 33.3 ms frame budget for
unmeasured-in-that-column decode, GUI submission and scheduling. The measured
active rate and latency drift show that this client consumes that margin.
Inference from these measurements: the synchronous processing/presentation
path is the bottleneck, not VRAM. This is not proof that NanoVSR itself cannot
reach 30 fps with a separately pipelined renderer. No such optimization or
precision change was introduced in this comparison.

Peak PyTorch allocated/reserved were **376,262,144 / 545,259,520 bytes**
(358.83 / 520 MiB). These exclude CUDA context, compositor and other processes;
zero in the direct/bicubic rows means no PyTorch model allocation. Warm NanoVSR
maximum decoded queue age ranged **282.31–310.81 ms**, queue peak at most six.
Per-run CPU/GPU timing, queue p50/p95/max, and GPU before/after clock/temperature/
power snapshots are retained. They do not isolate a causal thermal effect.

Only after all three initial repetitions passed was the road direct point
selected for the lower-buffer test; no NanoVSR road run met the required
three-run p95 criterion. At **150 ms** configured buffer and the same direct
CRF30.4 asset, all three runs submitted 300/300 frames with zero drops. Active
submission rate was **30.014–30.018 fps**; latency p50 **153.15–153.23 ms**,
p95 **153.64–153.69 ms**, maximum **157.90–158.84 ms**. Maximum queue age was
**113.94–114.62 ms**. Quality and bytes are unchanged because the asset is reused.
This is a useful demonstrated point, not a search for the minimum safe buffer.

## Representative visual crops

Inspected lossless crops at reference frame150 (5 s), with identical declared
coordinates across methods, selected from reference imagery rather than metric
extrema. Display enlargement is 2× nearest-neighbor, not SR and not scored.
The reference, B, C and matched-rate A panels are labelled. NanoVSR visibly
sharpens parts of the road/rock examples, while fine flower detail remains
inaccurate; direct retains more reference detail in these examples. These three
crops do not establish temporal/perceptual superiority or represent every frame.

- [Flowers crops](../results/natural-2026-10-15/flowers/comparison-crops.png)
- [Road crops](../results/natural-2026-10-15/road/comparison-crops.png)
- [Waterfall crops](../results/natural-2026-10-15/waterfall/comparison-crops.png)

## Artifacts and reproducibility

All study artifacts are under results/natural-2026-10-15/:

- provenance.json, study-config.json, source-frame-timing.json, ATTRIBUTION.md;
  full original source files remain under videos/natural-source/.
- capacity.json and logs/natural-capacity-2026-10-15.log.
- [All rate-quality points](../results/natural-2026-10-15/quality-summary.csv):
  27 base rows plus four targeted direct rows; **9,000 per-frame scores** in
  clip-level *-frames.csv files. All reference/reconstructed RGB .npy arrays
  and exact encoded TS assets with metadata/hash/packet pacing indices remain.
- [All timing summaries](../results/natural-2026-10-15/streaming-summary.csv),
  per-run JSON/frame/chunk records, commands.json and stderr logs. Includes
  excluded model-only warm and CPU-overlap trials with explicit reasons, plus
  separate verification runs; do not aggregate them blindly. failed-runs.json
  records the retained Escape-cancelled attempt.
- matched-rate-selection.json, matched-rate-adjustment.json, lower-buffer-selection.json,
  excluded-runs.json, crop-selection.json and exact offline-commands.json.

Download URLs and expected source hashes are in provenance.json. After obtaining
the same licensed files, the core offline commands are:

```bash
.venv/bin/python scripts/check_natural_capacity.py
.venv/bin/python scripts/natural_quality.py prepare
.venv/bin/python scripts/natural_quality.py score
.venv/bin/python scripts/natural_quality.py extra --extra-config configs/natural-matched-crf-initial.json
.venv/bin/python scripts/natural_quality.py extra --extra-config configs/natural-matched-crf.json
.venv/bin/python scripts/verify_natural_quality.py
```

The quality script skips completed scored points; preserve its configuration
and output together. Final timed settings can be replayed into a **new** directory:

```bash
.venv/bin/python scripts/run_natural_streams.py configs/natural-final-selection.json --output results/natural-timing-recheck
.venv/bin/python scripts/run_natural_streams.py configs/natural-lower-buffer-selection.json --output results/natural-lower-recheck
```

The final selection uses full-processing warmup. Historical initial,
continuation, model-only warm and verification selections/logs remain for audit.
The summarizers/auditors are scoped to this study's saved output directory.
No automated repeat should be run concurrently with quality scoring. The GUI
accepts Escape to cancel; the client/server record the cancellation explicitly.

Minimal single-playback command from kitty in the project root:

```bash
.venv/bin/python scripts/run_streaming_demo.py --mode nanovsr --config configs/natural-road-nanovsr.json --output results/my-natural-playback
```

## Final validation, limits and stopping point

Final integrity audit passed: all source/reference hashes, 31 quality summary
rows and 9,000 frame scores; 39 valid GUI runs and 11,340 unique submissions;
server/client byte counts, exact EOS/order, bounded queues, and no diagnostic
capture/frame hashing in the timed path. The three headless verification runs
matched every live output to its offline array. Metric self-tests and 186
independent saved-array checks passed. Peak-memory/cost numbers are measured,
not estimates from architecture alone.

Both environment before/after freezes compare identically. Both checkpoints
pass SHA-256 verification against natural-checkpoints-before.sha256. Both
upstream repositories remain clean. Prior milestone results and working
environments are preserved; no drivers, system CUDA, training or CDA data work.

Limitations: three short previously compressed clips; one checkpoint and
float32 implementation; one encoder preset/device; localhost preencoded replay
(excludes live server encode time); no audio, network shaping, power benchmark,
long-duration overload study or physical display timing. GUI submission calls
are counted; no claim that every physical refresh was observed. The only lower
buffer tested is the viable direct road point. No optimal trade-off is claimed.
If extending this work, first test whether bounded overlapping processing and
presentation removes the measured NanoVSR pacing deficit, and evaluate lower
rate overlaps on additional natural footage. That work has not been started.

Stopped after this focused comparison.
