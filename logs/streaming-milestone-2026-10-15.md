# Localhost streaming milestone — 2026-10-15

Implemented and tested: video-only, paced server encoding, TCP delivery, live
client decoding, direct/bicubic/NanoVSR processing and timed playback. Visible
bicubic playback was observed through two window-only desktop captures showing
different moments of the test pattern. This is a functional synthetic test,
not evidence of natural-video quality or an optimal operating point.

## Revised objective and inspection

The project objective is practical streaming trade-offs among bandwidth,
visual quality, application latency and client compute. Fixed x4 resolution
pairs are not requirements. The earlier x4 protocol and its stated next-source
resolution requirements apply only to that model-specific offline baseline.
Read all three 2026-10-14 milestone reports before editing. No applicable
AGENTS.md was found; .agents/.codex are empty. Root .git remains invalid, so
root changes cannot be assessed with Git. Existing HTTP server/full-download
client were inspected and backed up in results/streaming-2026-10-15/legacy_*.py
before replacing them. Existing offline evaluation scripts and upstream model
sources were reused by interface, not modified. No packages, checkpoints,
system drivers, CUDA or training were changed. CDA data acquisition was not
pursued.

## Implementation and limits

- Source: existing synthetic compressed test pattern videos/original/test.mp4,
  640×360, 30 fps. Main tests use its first 90 frames (3 s). Source hash and
  original source timestamps appear in every server report. A separate 91-frame
  run exercises the final chunk. No quality metrics are claimed here.
- A: transmission/playback 320×180. B/C: transmission 160×90, playback 320×180.
  libx264 veryfast, CRF23, single encoder thread, yuv420p, zerolatency,
  no B frames, GOP30, scene-cut disabled. FFmpeg `-re` paces source reading.
  Encoding happens during delivery; there is no pre-encoded-file download.
- MPEG-TS is carried in length-prefixed TCP records: H=JSON metadata,
  D=encoded bytes, E=explicit encoder return code/count. Server/client count
  successful application socket bytes including record overhead separately
  from MPEG-TS payload bytes. TCP/IP headers, retransmissions, and Ethernet
  traffic are **not** included. Server/client payload hashes match. B/C were
  encoded independently with deterministic settings and verified byte-identical,
  including every independently decoded frame hash.
- Live decoder: FFmpeg MPEG-TS→RGB24, one thread, small probe, no frame-rate
  conversion. The source must be CFR and monotonic. Original source PTS are
  carried in the header and assigned in decoder presentation order. B-frames
  are disabled. At EOS an independent probe of the received TS verifies the
  count and normalized timestamps within 30 microseconds; an independent decode
  verifies all live frame hashes/order. Arbitrary VFR/reordered input is not
  supported. Source probing/hashing precede streaming; this is a finite-file
  simulator, not a live-camera ingest implementation.
- Six-frame decoded queue, one active processing chunk (three for NanoVSR,
  one for other modes), bounded pipe/kernel socket buffers and TCP backpressure.
  Receiver never discards encoded predictive frames. When consumers fall
  behind, decode/receive/send eventually block. The decoded queue cannot grow;
  this does **not** guarantee bounded end-to-end age during sustained overload.
  Only fully reconstructed/decoded presentation frames over 500 ms past their
  target are dropped. No adaptive bitrate or upstream catch-up policy exists.
- NanoVSR uses unchanged pretrained architecture, strict checkpoint loading,
  deployment conversion under no_grad, float32 eval/inference mode, no TF32.
  Native output is 640×360, then CPU bicubic resize to 320×180. This is **not**
  native x2 or evidence that x4 plus resize is an efficient final solution.
  Non-overlapping chunks reset both recurrent directions. Each frame may use
  later frames in its chunk: chunk=3 adds up to 2/30 s lookahead, plus queue
  wait and processing. Final chunks are processed at actual length without
  padding, overlap or duplicated frames. No warmup is hidden: cold first-forward
  cost is included, while model loading occurs before connection/scheduling.
- Clean EOS requires explicit E, successful decoder exit and exact frame count.
  Timeout/truncation/encoder errors, config mismatch and cancellation fail the
  run and write error reports. Socket timeout=30 s; launcher enforces startup
  and total timeouts and cleans up children. No automatic reconnection.

## Instrumentation definitions

All scheduling and client events use the same host's `time.monotonic()` clock.
The server records epoch immediately before launching the paced encoder;
source-scheduled(frame)=epoch+(original PTS−first PTS). This is an **intended
source schedule**, not an instrumented camera or actual source-read event.
Encoder startup and pacing deviations therefore contribute to application
latency. Target rendering is source-scheduled+350 ms. Render event is after
headless timed submission, or after OpenCV imshow/waitKey returns. Application
latency is render event minus source-scheduled. It includes configured playout
wait; it is neither photon latency nor camera-to-display latency.

Frame records contain original PTS, decode completion, processing start,
queue age (decode completion to processing start, including chunk collection),
queue depth, drop decision, render event, decoded/output hashes and application
latency. Chunk records separate CUDA-event forward time, CPU final-resize time,
and full processing wall time (H2D, forward, quantization, D2H, resize).
Processing throughput is processed frames / sum(chunk processing seconds),
not network throughput or sustained playback fps; it excludes queue/playout
wait, decoding, model load and diagnostic hash/report work. The artificial
slowdown is included in overload processing time. Reports also include entire
client wall time and PyTorch peak allocated/reserved GPU bytes, not whole-GPU
process memory. Input and output copies and report/capture I/O add overhead.

`unique_render_submissions` counts distinct frame IDs submitted on time;
headless submissions are **not displayed frames**. Early reports leave
`unique_displayed_frames=null`; final code records zero for headless and counts
unique successful OpenCV submissions for GUI runs, with an explicit scope field.
That is an application display counter, not confirmation of each physical refresh.
Window snapshots confirm some visible content, not all 90 physical presentations.

## Measured headless comparison

Artifacts: results/streaming-2026-10-15/final-{direct,bicubic,nanovsr}/.
All decode 90/90 frames in order; all submit 90; zero late drops; queue peak 6.
All stream/decode/order/timestamp/EOS checks pass. These are single smoke runs.

| Mode | TS bytes | TS kbps over 3 s | Processing fps | Median/max application ms | Model total ms | Final resize total ms |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 56,588 | 150.90 | 50,102 | 350.37 / 350.51 | 0 | 1.10 (identity/list overhead) |
| Bicubic | 43,616 | 116.31 | 2,561 | 350.37 / 350.50 | 0 | 34.12 |
| NanoVSR | 43,616 | 116.31 | 87.67 | 350.12 / 425.60 | 808.35 | 119.78 |

NanoVSR peak allocated 95,208,448 bytes (~90.8 MiB), reserved 106,954,752
bytes (~102 MiB), well below the previously measured 4 GiB RTX 3050 capacity.
This conservative starting size was chosen using the previous 160×90 GPU
baseline (~198 MB with chunk15). It is not a predetermined final upscale target.
Initial NanoVSR run: first chunk model forward 588.50 ms, subsequent chunks
about 20.7 ms, max application latency 834.06 ms. Cold-start variation is
material; the final run is not a steady-state performance guarantee.

With 70 ms artificial processing delay per frame, overload test decoded all
90, submitted 14 and dropped 76, queue peak stayed 6. Maximum queue age was
500.50 ms. This proves the bounded queue and presentation-drop path for the
short test, not long-duration low-latency recovery. Broken-peer checks verify
abrupt TCP close without EOS and connection refusal both return nonzero and
write explicit error reports. Independent re-decode verifies all decoded
hashes. Verification script: scripts/verify_streaming.py.

## Launch and evidence

Reproducible commands are in README.md, configurations in configs/streaming-*.json,
and exact per-run launch and FFmpeg argv in results subdirectories. Each run
has server/client JSON, stderr logs and received TS. Initial sandbox attempt
failed with socket PermissionError and is retained as bicubic-initial. Supported
host approvals enabled localhost, GPU and desktop tests; no driver changes.

Observed bicubic playback: visible-bicubic/window-{0,1}.png were captured from
the actual OpenCV desktop window and inspected. Both show the synthetic video
at different animation positions. All 90 frames were submitted, zero dropped;
median/max application latency 353.27/364.93 ms. This observation is limited to
two snapshots, not frame-by-frame physical display verification.

Minimal kitty command (run in project root, choose a new output directory):

```bash
.venv/bin/python scripts/run_streaming_demo.py --mode nanovsr --output results/my-nanovsr-playback
```

## Remaining work and next comparisons

No natural-video quality evidence, optimal trade-off, WAN performance, adaptive
bitrate, audio synchronization, cross-host latency or authentic CDA integration
is established. The dominant measured steady application delay is the configured
350 ms playout buffer; cold GPU startup and noncausal chunk lookahead add risk.
NanoVSR GPU forward and x4 output conversion/resizing cost substantially more
than bicubic; the synthetic bitrate reduction alone cannot justify that cost.

Next: repeat on documented natural 640×360 footage, common 640×360 playback,
compare 160×90 and 320×180 delivery, chunk1/3/5 and CRF18/23/28 with several
repetitions. Measure 320×180 GPU capacity first (its native output is 1280×720,
then resize), and jointly measure quality, deadline misses, compute and bytes.
Try 150/250/350 ms playout budgets with explicit cold-start/warm-start reporting.
Measure longer overload/recovery and controlled network shaping separately.
Stop here: no training or CDA dataset work belongs to this milestone.

## Final visible tail test and preservation

Visible NanoVSR run with configs/streaming-tail.json passed: 91 frames decoded
and uniquely submitted to OpenCV, zero drops, queue peak 6, peak GPU allocated
95,208,448 / reserved 106,954,752 bytes. Last chunk was exactly frame90,
length1 (26.98 ms model / 30.09 ms full processing), with clean EOS. Its two
window captures were inspected and show different moments of visible synthetic
playback. This also exercises the final code's application-level displayed-frame
counter (91); physical display refreshes remain unmeasured.

Main comparison protocol bytes including JSON/record framing: direct 58,359;
bicubic 45,373; NanoVSR 45,373. Payload bytes in the table exclude that framing
but include MPEG-TS overhead. JSON decimal clock lengths and TCP read boundaries
can slightly change framing overhead between repetitions.

Preservation completed: streaming-venv-{before,after}.txt and
streaming-cda-{before,after}.txt compare identically; both checkpoint SHA-256
checks pass against streaming-checkpoints-before.sha256. Both upstream Git
working trees are clean. No environment dependencies were installed or changed.
