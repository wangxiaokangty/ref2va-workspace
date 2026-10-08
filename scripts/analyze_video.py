"""Step 1: inspect an input video and build overview material for manual shot selection.

Usage:
  .venv/bin/python scripts/analyze_video.py inputs/x.mp4 outputs/<id>/work

Writes into WORK_DIR/analysis/:
  analysis.json      media parameters, SHA-256, PTS statistics, detected shots, audio envelope
  shots_overview.jpg one thumbnail (middle frame) per detected shot, labelled with shot/frame/time
  uniform_overview.jpg evenly spaced frames across the whole video
Cut detection is only a first pass; candidates must be checked with dense frames (dense_frames.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2v_common import (contact_sheet, cut_scores, decode_audio, detect_cuts, extract_frames,  # noqa: E402
                        frame_pts, probe_streams, read_frames_gray, sha256, write_json)


def main(video: str, work: str) -> None:
    video_p, out = Path(video), Path(work) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    info = probe_streams(video_p)
    pts = frame_pts(video_p)
    d = np.diff(pts)
    info.update({
        "path": str(video_p),
        "sha256": sha256(video_p),
        "decoded_frames": len(pts),
        "first_pts": pts[0],
        "last_pts": pts[-1],
        "frame_interval_median": float(np.median(d)),
        "frame_interval_min": float(d.min()),
        "frame_interval_max": float(d.max()),
        "cfr": bool(d.max() - d.min() < 1e-3),
        "measured_fps": float(1.0 / np.median(d)),
    })

    audio, sr = decode_audio(video_p, sr=None, channels=None)
    info["audio_decoded_samples"] = int(audio.shape[0])
    info["audio_decoded_seconds"] = audio.shape[0] / sr if sr else None
    info["audio_channels_decoded"] = int(audio.shape[1])
    if audio.shape[1] == 2:
        diff = float(np.abs(audio[:, 0] - audio[:, 1]).mean())
        info["audio_lr_mean_abs_diff"] = diff
        info["audio_effectively_mono"] = diff < 1e-4
    hop = int(sr * 0.25)
    mono = audio.mean(axis=1)
    env = [float(20 * np.log10(np.sqrt(np.mean(mono[i:i + hop] ** 2)) + 1e-9)) for i in range(0, len(mono), hop)]
    info["audio_rms_db_250ms"] = [round(x, 1) for x in env]

    frames = read_frames_gray(video_p)
    sc = cut_scores(frames)
    cuts = detect_cuts(sc)
    bounds = [0] + cuts + [len(frames)]
    shots = []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        shots.append({"shot": k + 1, "start_frame": s, "end_frame_excl": e,
                      "start_s": round(pts[s], 3), "end_s": round(pts[e] if e < len(pts) else pts[-1] + float(np.median(d)), 3),
                      "cut_score": round(float(sc[s]), 1) if s else None})
    info["shots"] = shots
    info["cut_scores"] = [round(float(x), 1) for x in sc]
    write_json(out / "analysis.json", info)

    mids = [(s["start_frame"] + s["end_frame_excl"] - 1) // 2 for s in shots]
    ims = extract_frames(video_p, mids, w=320)
    contact_sheet(ims, [f"S{s['shot']} f{s['start_frame']}-{s['end_frame_excl']} {s['start_s']:.2f}s" for s in shots],
                  out / "shots_overview.jpg", cols=6)
    n = len(pts)
    uni = list(np.linspace(0, n - 1, min(48, n)).astype(int))
    ims = extract_frames(video_p, uni, w=320)
    contact_sheet(ims, [f"f{i} {pts[i]:.2f}s" for i in uni], out / "uniform_overview.jpg", cols=8, tile_w=240)
    print(f"frames={n} fps~{info['measured_fps']:.3f} cfr={info['cfr']} shots={len(shots)} audio={info.get('audio_codec')}")
    for s in shots:
        print(f"  S{s['shot']:>3} f[{s['start_frame']},{s['end_frame_excl']}) {s['start_s']:.3f}-{s['end_s']:.3f}s score={s['cut_score']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
