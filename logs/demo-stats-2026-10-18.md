# Presentation statistics panel — 2026-10-18

Added optional `--stats` / `--stats-hold` to the existing launcher/client. Dashboard code: `scripts/demo_stats.py`. Original benchmark configs/results and model environments/weights are unchanged.

Saved whole-clip quality and TS rates come from existing natural-study CSVs; current asset hash is checked before connection. The comparison table shows CRF23 LR methods and nearest-rate measured direct delivery. Live completed-frame counters, rolling submission rate/latency, last-chunk model/processing cost, queue metrics, received bytes and allocator peak are separate. Native x4 intermediate resolution is explicit. Overlay does not modify reconstruction pixels. Final PNG is saved after EOS and hold does not add render submissions. Total client wall time includes hold. These runs add presentation overhead and are not benchmark replacements.

Validation: all three modes render from archived reports, with assertions that source RGB and displayed video pixels remain unchanged. Python compilation passes. Actual NanoVSR localhost/GPU/GUI run: `results/demo-stats-playback/`, clean EOS, 300 decoded frames; counts in client.json. Its playback-window screenshot was inspected and shows the new panel beside natural video. Final dashboard is demo-summary.png. No physical refresh claim.

The first `--observe` attempt (`results/demo-stats-visible/`) failed because its fixed startup delay did not find the window before capture timeout; launcher terminated children, leaving evidence. A normal launcher run without --observe completed, and a separate later window capture succeeded. User launch commands do not require --observe. Existing sustained-playback limitations remain; no research/performance sweep was run.

Commands: presentation_assets/REPRODUCE.md, Live results panel section. Use a fresh output folder and omit --stats for benchmark comparisons.
