# Native NanoVSR presentation output

Added presentation_assets/configs/nanovsr-native.json: same 320x180 asset, chunk3, 350ms playout, unchanged pretrained model; native_output=true and playback1280x720. Client validates native dimensions/mode, bypasses final resize in warmup and processing, and records output dimensions and zero resize time. Display conversion remains clamp/round/uint8, with no spatial downscaling.

Dashboard/title identify native output and omit inapplicable archived640x360 quality metrics. Footer is placed below the full720-pixel image so it does not overwrite video pixels. Original comparison config/results are preserved.

Validation: compilation and dashboard pixel-preservation checks pass. Host GPU/localhost/GUI run results/nanovsr-native-demo completed clean EOS,300 decoded, 169 submissions, 131 drops. Every chunk records zero final resize time. Actual window screenshot inspected; video plus panel shows native1280x720. This single presentation run is not a new quality result or sustained-performance claim. Commands in presentation_assets/REPRODUCE.md.


Follow-up: native config now defaults presentation_stats=true, with --no-stats override. Added completeness, latency maximum, processing-only FPS and explicit disabled final-resize status. Compilation and argument parsing pass; an expanded panel rendered from the retained native run was inspected and its video pixels checked unchanged (`results/nanovsr-native-demo/expanded-stats-preview.png`). This preview uses existing measurements, not a new live test. No new native quality scores were computed.
