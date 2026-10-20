# Native 720p comparison — 2026-10-18

Selected the existing licensed original **Narrowing 2+1 road** by LA2, CC BY-SA3.0. Original video is1280x720 at50FPS. Reference uses source seconds10-20, converted to30FPS with no spatial scaling:300 aligned frames. The previously compressed source is not a pristine camera master. Source SHA256 and licensing are in presentation_assets/native720/provenance.json; old results remain untouched.

Fresh320x180 and1280x720 H.264/MPEG-TS CRF23 assets were made from this reference. The LR asset is shared byte-for-byte by bicubic and NanoVSR. NanoVSR retains native x4 architecture/checkpoint and independent chunk3 processing with no final spatial resizing. Both output1280x720. Direct decodes the1280x720 encode. All900 reconstructed frame scores precede any lossy preview encoding, under the established RGB PSNR/Gaussian11x11 SSIM protocol.

| Method | Actual TS kbps | PSNR dB | SSIM |
|---|---:|---:|---:|
| direct | 4046.21 | 39.173 | 0.97953 |
| bicubic | 451.65 | 29.533 | 0.87653 |
| nanovsr | 451.65 | 30.700 | 0.89102 |

NanoVSR improves over bicubic on this one excerpt at identical transmitted bytes. Direct uses much more bitrate; equal CRF is not equal bitrate and this test cannot establish a matched-rate winner. No broad content/generalization, sustained-playback or physical-display claim is made.

Independent checks pass:900 per-frame rows, aggregate arithmetic, all asset hashes/bytes, shared LR stream,12 recomputed boundary-frame metric pairs, and loaded native quality stats for all three modes. The playback panel only loads these results after validating resolution, asset hash/frame count, and (for NanoVSR) checkpoint and chunk size. It explicitly labels this as an unmatched-rate CRF23 comparison. Old640x360 demo results are unchanged.

Reproduce with scripts/compare_native720.py prepare, then score. scripts/verify_native720.py audits offline results. New configs and launch commands: presentation_assets/native720/README.md. Local raw arrays/streams and diagnostic reports: results/native720-road-2026-10-18/. Fixed frame150 crops and all compact metric rows are in presentation_assets/native720/. No new source download, package installation, checkpoint change, training, CDA work or large sweep was performed.

Final headless streaming verification passed all300 live reconstructed frames against saved native output hashes, plus decoded hashes, timestamps, payload/protocol counts and clean EOS. Native output1280x720 and all final resize times zero. Headless submissions=300, drops=0; diagnostic hashing overhead means this is correctness evidence, not a GUI performance benchmark.
