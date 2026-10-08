---
name: ref2va-sample
description: Build one MiniMax-H3 Ref2VA (reference-to-video-and-audio) training sample from a video in inputs/ following PROMPT.md - analyse, pick target and references, export normalized media, write the six-section H3 prompt, assemble dataset.jsonl and verify. Use for any "generate a ref2VA sample / 训练 sample" request in this workspace.
---

# Ref2VA sample workflow

Read `PROMPT.md`, `reference/media-profile.md` and the H3 guides (`reference/minimax-h3/skills/h3-prompt-writing/`) first. Tools live in `scripts/` (see `scripts/README.md`). Media tools use `.venv/bin/python` (FFmpeg comes from imageio-ffmpeg; there is no ffprobe). ASR (automatic speech recognition) and audio tagging use `.venv-asr/bin/python`.

Per video, with `W=outputs/<sample_dir>/work`:

1. **Analyse**: `analyze_video.py <video> $W`, then `asr.py <video> $W/analysis/asr_full.json`, `audio_tags.py` (CLAP, Contrastive Language-Audio Pretraining) and `audio_profile.py` for the sound layers. Look at `shots_overview.jpg` and `uniform_overview.jpg` yourself.
2. **Pick the target first** (5–15 s, a complete beat, hard cuts at both ends if possible), then pick references **outside** it. Check every boundary with `dense_frames.py` at step 1.
3. **Write `$W/export_config.json`** and run `export_sample.py $W/export_config.json $W/stage`, then `inspect_export.py $W/stage $W`. Choose `num_frames = 17k+5`; prefer trimming silent tails over padding (padding of 0.25 s or more fails verification).
4. **Confirm the cuts** visually on `target_shot_boundaries.jpg`. Run `asr.py` on the target WAV to get the timeline of dialogue.
5. **Write `$W/prompt.md`** (the prompt text only) and `$W/record_spec.json`. The spec holds the id, split_group, references, confirmed/rejected/manual cut times, audio-review flags, review_items and manual_checks.
6. **Build and publish** with `build_and_verify.py $W $W/stage outputs/<dir>/pure`. It builds in stage, verifies, moves stage to pure and re-verifies. To rebuild after editing the prompt, first copy `pure/.` back to `$W/stage`.
7. Write `$W/selection.json` covering the target, each reference's role and reason, and any rejected options.

## Pitfalls seen on real inputs (check every time)

- **Single-frame dissolves** at cuts: a blended frame is not a clean start or end. Move the boundary off it.
- **Lighting changes and in-shot glitches** register as "cuts" (lights switched off, UV lamp turning on, caption glitch, flicker). Reject them in `rejected_cut_times`. **Jump cuts** inside one framing are real cuts, so confirm them on source frames.
- **Replays and recurring setups** leak target content into references. Watch for a VHS-rewind recap of target shots, or the same camera setup reused in a cross-cut conversation. `no_near_duplicate_reference_frames` catches some of these, not all, so look.
- **End cards and voice-over**: check the audio envelope near the last cut so that VO (voice-over) from the next shot does not leak in, and so that the first word of a J-cut line is not clipped.
- **Picture frames**: avoid burned-in subtitles, typing cursors and text overlays. Scan neighbouring frames and pick a clean one.
- Letterboxed sources keep their black bars inside the 1344×768 canvas. Describe the letterbox and the subtitles in the bar as visible text.
- Dialogue stays in its original language (e.g. `[Portuguese]`) even when the burned-in subtitles are English.
- No listening capability: CLAP tags, levels and spectral measures are hints only. Flag any music or effects classification you could not confirm in `review_items`, and set `needs_audio_review` when it matters.
