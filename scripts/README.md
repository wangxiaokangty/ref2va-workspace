# Ref2VA scripts

Reusable tools for `PROMPT.md`. Workflow and pitfalls are in [.claude/skills/ref2va-sample/SKILL.md](../.claude/skills/ref2va-sample/SKILL.md).

| Script | Env | Purpose |
| --- | --- | --- |
| `r2v_common.py` | .venv | Shared helpers: FFmpeg (imageio-ffmpeg) path, PTS (presentation timestamp) decode, cut scores, canvas math, contact sheets |
| `analyze_video.py <video> <work>` | .venv | Media parameters, SHA-256, CFR/VFR (constant/variable frame rate) check, candidate shots, audio envelope, overview sheets |
| `dense_frames.py <video> <s> <e> <step> <out.jpg>` | .venv | Dense frame sheet labelled with frame index and PTS |
| `asr.py <media> <out.json> [--start --end --lang]` | .venv-asr | Whisper large-v3-turbo (MLX) transcript with word timestamps |
| `audio_tags.py <media> <s> <e> [--labels a\|b]` | .venv-asr | CLAP (Contrastive Language-Audio Pretraining) zero-shot tags per window |
| `audio_profile.py <media> <s> <e>` | .venv-asr | RMS, spectral centroid/flatness, chroma, tempo estimate |
| `export_sample.py <work>/export_config.json <stage>` | .venv | Exact trims to 24 fps, 17k+5 frames, 1344×768 letterbox canvas, 32 kHz stereo WAV, H.264/AAC; writes provenance.json and export_log.json |
| `inspect_export.py <stage> <work>` | .venv | Measured cuts on the exported target, storyboards, boundary sheet |
| `build_and_verify.py <work> <stage> <pure>` | .venv | Builds dataset.jsonl, runs all checks, publishes, re-verifies, writes verification.json |
