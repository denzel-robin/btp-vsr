# Demonstration and reproduction

Run from `/home/ibexe/Github/btp-vsr` in kitty/fish. Each command opens the existing OpenCV player. Close after EOS; Escape intentionally cancels and records an error. Use a fresh output directory on every launch. Run modes serially, on AC power, with no competing evaluation jobs. This demo replays paced pre-encoded assets over TCP; it does not encode live.

```fish
cd /home/ibexe/Github/btp-vsr
.venv/bin/python scripts/run_streaming_demo.py --mode direct --config presentation_assets/configs/direct.json --output results/presentation-direct
.venv/bin/python scripts/run_streaming_demo.py --mode bicubic --config presentation_assets/configs/bicubic.json --output results/presentation-bicubic
.venv/bin/python scripts/run_streaming_demo.py --mode nanovsr --config presentation_assets/configs/nanovsr.json --output results/presentation-nanovsr
```

All show the same ten-second road clip at 640×360 with 350 ms configured playout buffer. Direct uses 640×360 CRF30.4; B/C share the exact 320×180 CRF23 stream. NanoVSR is native x4 (1280×720 intermediate), followed by CPU bicubic resizing to 640×360, independent chunk3, three full-pipeline warmup chunks before connecting. The title identifies the mode and resolutions. NanoVSR is an informative costly comparison, not the reliable lowest-latency choice. Direct is the presentation fallback.

Offline backup (explicitly labelled on every frame):

```fish
ffplay -autoexit presentation_assets/demo/OFFLINE-road-comparison.mp4
```

This is precomputed output encoded into a preview, not a recording of live playback. It must not substantiate latency or frame-rate claims. The road-derived imagery is CC BY-SA 3.0; see ATTRIBUTION.md.

Evidence consolidation (no new model inference):

```fish
.venv/bin/python scripts/verify_natural_artifacts.py
.venv/bin/python scripts/build_presentation_assets.py
python3 scripts/plot_presentation.py
```

Plotting uses existing system Matplotlib, not packages installed into the preserved model environments. The build reads original raw reports and verifies latency percentiles, byte counts, dimensions and shared LR assets. It outputs all 39 valid runs, including cold, warm and the separate direct 150 ms trials.

For an intentional stability rerun, use `scripts/prepare_presentation_demo.py`, then `scripts/run_presentation_stability.py` with a new evidence destination (its default refuses existing output). The launcher accepts `--timeout 360`. Current recorded test config is `configs/stability.json`: 9000 frames, 30 repeated copies of the same 300-frame compressed clip, stream-copy remuxed to continuous PTS. Decoded frame hashes were checked before timing. No quality evidence is added by looping. Model state resets every chunk3; chunk boundaries align with 300-frame loop boundaries; decoder, connection, model and monotonic schedule persist across loops. No hidden buffer increase. Logging is capped by 9000 scalar frame records, 3000 chunk records, 30 progress windows and approximately 61 GPU samples; no reconstructed images saved.

See `../logs/natural-streaming-milestone-2026-10-15.md` for original quality encoding/scoring commands, and `data/provenance.json` for exact source conversion argv and hashes. Do not rerun metric scoring concurrently with timing. Original evidence is in `../results/natural-2026-10-15/`; exclusions and failed attempts are retained there.

## Live results panel (added 2026-10-18)

Add `--stats` to display values beside the video. The final panel stays open for eight seconds (Escape closes it) and is saved as `demo-summary.png` in the new run directory. Use `--stats-hold 20` for more presentation time. The video remains at its configured size; the panel occupies extra window space.

```fish
.venv/bin/python scripts/run_streaming_demo.py --mode direct --config presentation_assets/configs/direct.json --output results/stats-direct-(date +%s) --stats
.venv/bin/python scripts/run_streaming_demo.py --mode bicubic --config presentation_assets/configs/bicubic.json --output results/stats-bicubic-(date +%s) --stats
.venv/bin/python scripts/run_streaming_demo.py --mode nanovsr --config presentation_assets/configs/nanovsr.json --output results/stats-nanovsr-(date +%s) --stats
```

The saved-quality section reads archived whole-clip RGB PSNR/SSIM and actual TS rates, with LR CRF23 and the nearest-rate measured direct encode. These are not live quality measurements. Asset rate is bytes divided by full clip duration, not instantaneous network throughput. Received payload bytes are counted live.

Live values use completed prior render events: unique submission rate and latency use the last 60 submissions during playback; the final panel uses all submitted frames for latency and the last 60 for rate. Processing/model times describe the last completed chunk per frame. Queue age/depth describe the last processed frame; GPU memory is peak PyTorch allocation, not whole-device memory. Final summary drawing/hold happens after EOS and does not add frame submissions; total client wall time includes the hold.

The overlay adds GUI work and is presentation-only. Original benchmark configurations and results remain unchanged. Runs record `presentation_stats_enabled`; use `--no-stats` for comparable performance measurements (the native config enables stats by default). The previous sustained NanoVSR failure remains valid and is not superseded by a short dashboard demo.

## Native NanoVSR output (no final downscaling)

```fish
.venv/bin/python scripts/run_streaming_demo.py --mode nanovsr --config presentation_assets/configs/nanovsr-native.json --output results/native-nanovsr-(date +%s) --stats --stats-hold 20
```

This uses the same 320×180 encoded stream and unchanged chunk3 model, but displays its native 1280×720 output directly. Both warmup and playback bypass final spatial resizing. Clamp/round/uint8 conversion remains necessary to display model RGB output. The panel explicitly omits the old 640×360 PSNR/SSIM comparison, since those scores do not measure native output. This is not a common-playback-size comparison with the earlier A/B modes. Native-output configuration now enables stats by default; pass `--no-stats` to see video alone. Reports record `native_output: true`, `output_size: [1280,720]`, and zero final resize time. Existing comparison settings and results remain unchanged.


Native panel update: live statistics are enabled by default in `nanovsr-native.json`. In addition to the existing counters, it shows submitted/(submitted+dropped) as playback completeness so far, maximum application latency over the same window as p50/p95, last-chunk processing-only FPS, and zero final resizing cost. Processing FPS excludes presentation and waiting and must not be confused with actual submission FPS. Native PSNR/SSIM remain unmeasured. `--stats` and `--no-stats` explicitly override the config; other existing configs keep their prior default.
