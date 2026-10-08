"""Step 2: export reference images, reference videos and the target from a selection config.

Usage:
  .venv/bin/python scripts/export_sample.py <WORK_DIR>/export_config.json <STAGE_DIR>

export_config.json (all frame numbers are zero-based source frames, end exclusive):
{
  "sample_id": "...",
  "source": "inputs/x.mp4",
  "target":  {"start_frame": 396, "end_frame_excl": 649, "num_frames": 192},
  "videos":  [{"name": "video_1", "start_frame": 62, "end_frame_excl": 279, "num_frames": 175,
               "include_audio": true}],
  "audios":  [],   # optional standalone audio refs: {"name": "audio_1", "start_frame", "end_frame_excl"}
  "images":  [{"name": "picture_1", "source_frame": 655, "crop": [x, y, w, h] | null}]
}

Rules implemented (reference/media-profile.md):
* the semantic clip is [pts(start_frame), pts(end_frame_excl)); the exported clip covers
  N/24 seconds from pts(start_frame). If the clip is shorter than N/24 the last selected frame is
  cloned and audio is padded with silence; if longer, the tail is trimmed. Nothing after
  end_frame_excl is ever read.
* video: exact trim on decoded timestamps -> 24 fps CFR (constant frame rate) -> aspect-preserving
  scale + pad to the canvas -> H.264. Audio: trimmed from the source track, resampled to 32 kHz stereo.
* the target WAV is produced directly from the source audio (no AAC round trip).
Writes STAGE_DIR/{inputs,targets,oringin.mp4} and WORK_DIR/provenance.json + export_log.json.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2v_common import (AUDIO_CH, AUDIO_SR, FPS, WORKSPACE, canvas_for, ffmpeg_exe, fit_in_canvas,  # noqa: E402
                        frame_pts, legal_frames, probe_streams, read_json, run, sha256, write_json)

LOG: list[dict] = []


def ff(args: list[str], what: str) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-v", "error", "-y"] + args
    LOG.append({"step": what, "cmd": cmd})
    run(cmd)


def clip_times(pts: list[float], s: int, e: int, n_out: int, dt: float) -> dict:
    t0 = pts[s]
    t_end_sem = pts[e] if e < len(pts) else pts[-1] + dt
    t_out = n_out / FPS
    t1 = min(t0 + t_out, t_end_sem)
    used_frames = sum(1 for p in pts[s:e] if p < t1 - 1e-6)
    return {"t0": t0, "t1": t1, "semantic_end": t_end_sem, "out_duration": t_out,
            "source_duration_used": t1 - t0, "pad_seconds": max(0.0, t_out - (t1 - t0)),
            "trimmed_seconds": max(0.0, (t_end_sem - t0) - t_out), "source_frames_used": used_frames,
            "source_frame_range_used": [s, s + used_frames]}


def vchain(t0: float, t1: float, n: int, sw: int, sh: int, cw: int, ch: int, px: int, py: int) -> str:
    return (f"trim=start={t0:.6f}:end={t1:.6f},setpts=PTS-STARTPTS,fps={FPS}:round=near,"
            f"scale={sw}:{sh}:flags=lanczos,setsar=1,pad={cw}:{ch}:{px}:{py}:black,"
            f"tpad=stop_mode=clone:stop={n},trim=end_frame={n},format=yuv420p")


def achain(t0: float, t1: float, n_samples: int) -> str:
    return (f"atrim=start={t0:.6f}:end={t1:.6f},asetpts=PTS-STARTPTS,"
            f"aresample={AUDIO_SR}:filter_size=64:phase_shift=10:cutoff=0.97,"
            f"aformat=sample_fmts=s16:channel_layouts=stereo,apad=whole_len={n_samples},"
            f"atrim=end_sample={n_samples}")


def export_wav(src: str, t: dict, n_samples: int, out: Path) -> None:
    ff(["-i", src, "-map", "0:a:0", "-af", achain(t["t0"], t["t1"], n_samples), "-ac", str(AUDIO_CH),
        "-c:a", "pcm_s16le", str(out)], f"wav {out.name}")
    data, sr = sf.read(out, dtype="int16")
    assert sr == AUDIO_SR and data.shape == (n_samples, AUDIO_CH), (sr, data.shape, n_samples)


def export_mp4(src: str, t: dict, n: int, geo: tuple, out: Path, wav: Path | None) -> None:
    sw, sh, cw, ch, px, py = geo
    args = ["-i", src]
    if wav is not None:
        args += ["-i", str(wav)]
    args += ["-filter_complex", f"[0:v:0]{vchain(t['t0'], t['t1'], n, sw, sh, cw, ch, px, py)}[v]", "-map", "[v]"]
    if wav is not None:
        args += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_SR), "-ac", str(AUDIO_CH)]
    else:
        args += ["-an"]
    args += ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-r", str(FPS), "-fps_mode", "cfr",
             "-frames:v", str(n), "-movflags", "+faststart", str(out)]
    ff(args, f"mp4 {out.name}")


def main(cfg_path: str, stage: str) -> None:
    cfg = read_json(cfg_path)
    work = Path(cfg_path).parent
    stage_p = Path(stage)
    if stage_p.exists():
        shutil.rmtree(stage_p)
    (stage_p / "inputs").mkdir(parents=True)
    (stage_p / "targets").mkdir(parents=True)
    src = str(WORKSPACE / cfg["source"])
    info = probe_streams(src)
    pts = frame_pts(src)
    dt = float(np.median(np.diff(pts)))
    cw, ch = canvas_for(info["width"], info["height"])
    sw, sh, px, py = fit_in_canvas(info["width"], info["height"], cw, ch)
    geo = (sw, sh, cw, ch, px, py)
    prov: dict = {
        "source_path_from_workspace": cfg["source"], "source_sha256": sha256(src),
        "source_width": info["width"], "source_height": info["height"],
        "source_fps_measured": 1 / dt, "source_cfr": bool(np.ptp(np.diff(pts)) < 1e-3),
        "source_decoded_frames": len(pts), "source_audio": {k: info.get(k) for k in ("audio_codec", "audio_sample_rate", "audio_layout")},
        "frame_numbering": "zero-based, end-exclusive; times are decoded PTS (presentation timestamp) in seconds",
        "spatial_transform": f"{info['width']}x{info['height']} -> {sw}x{sh} (lanczos, aspect preserved), pad to {cw}x{ch} at ({px},{py}) with black; no crop",
        "temporal_transform": f"exact trim on decoded PTS, resample to {FPS} fps CFR (fps filter, nearest frame); clone last selected frame if short; audio resampled to {AUDIO_SR} Hz stereo, padded with silence if short",
        "media_path_base": "../pure (relative to this provenance.json)",
        "images": [], "reference_videos": [], "reference_audios": [],
    }

    for im in cfg.get("images", []):
        out = stage_p / "inputs" / f"{im['name']}.png"
        vf = f"select='eq(n\\,{im['source_frame']})'"
        if im.get("crop"):
            x, y, w, h = im["crop"]
            vf += f",crop={w}:{h}:{x}:{y}"
        ff(["-i", src, "-map", "0:v:0", "-vf", vf, "-vsync", "0", "-frames:v", "1", str(out)], f"image {out.name}")
        prov["images"].append({"path": f"inputs/{out.name}", "source_frame": im["source_frame"],
                               "source_seconds": pts[im["source_frame"]],
                               "transform": f"crop {im['crop']} (x,y,w,h) at source resolution" if im.get("crop") else "full frame at source resolution; no synthesis or cropping",
                               "sha256": sha256(out)})

    for v in cfg.get("videos", []):
        n = v["num_frames"]
        assert legal_frames(n), n
        t = clip_times(pts, v["start_frame"], v["end_frame_excl"], n, dt)
        out = stage_p / "inputs" / f"{v['name']}.mp4"
        wav = None
        if v.get("include_audio", False):
            wav = work / "intermediate" / f"{v['name']}_audio.wav"
            wav.parent.mkdir(exist_ok=True)
            export_wav(src, t, round(n / FPS * AUDIO_SR), wav)
        export_mp4(src, t, n, geo, out, wav)
        prov["reference_videos"].append({
            "path": f"inputs/{out.name}", "source_frame_start": v["start_frame"], "source_frame_end_exclusive": v["end_frame_excl"],
            "source_start_seconds": t["t0"], "source_end_seconds": t["semantic_end"],
            "source_seconds_used": [t["t0"], t["t1"]], "source_frame_range_used": t["source_frame_range_used"],
            "num_frames": n, "duration_seconds": n / FPS, "fps": FPS, "width": cw, "height": ch,
            "audio_track": wav is not None, "include_audio": bool(v.get("include_audio", False)),
            "audio_sample_rate": AUDIO_SR if wav else None, "audio_channels": AUDIO_CH if wav else None,
            "time_padding_seconds": t["pad_seconds"], "tail_trimmed_seconds": t["trimmed_seconds"],
            "sha256": sha256(out)})

    for a in cfg.get("audios", []):
        t0, t1 = pts[a["start_frame"]], pts[a["end_frame_excl"]] if a["end_frame_excl"] < len(pts) else pts[-1] + dt
        ns = round((t1 - t0) * AUDIO_SR)
        out = stage_p / "inputs" / f"{a['name']}.wav"
        export_wav(src, {"t0": t0, "t1": t1}, ns, out)
        prov["reference_audios"].append({"path": f"inputs/{out.name}", "source_seconds": [t0, t1],
                                         "samples_per_channel": ns, "sha256": sha256(out)})

    tg = cfg["target"]
    n = tg["num_frames"]
    assert legal_frames(n), n
    t = clip_times(pts, tg["start_frame"], tg["end_frame_excl"], n, dt)
    ns = round(n / FPS * AUDIO_SR)
    wav = stage_p / "targets" / "target_audio.wav"
    export_wav(src, t, ns, wav)
    mp4 = stage_p / "targets" / "target.mp4"
    export_mp4(src, t, n, geo, mp4, wav)
    prov["target_video"] = {
        "path": "targets/target.mp4", "audio_path": "targets/target_audio.wav",
        "source_frame_start": tg["start_frame"], "source_frame_end_exclusive": tg["end_frame_excl"],
        "source_start_seconds": t["t0"], "source_end_seconds": t["semantic_end"],
        "source_seconds_used": [t["t0"], t["t1"]], "source_frame_range_used": t["source_frame_range_used"],
        "num_frames": n, "duration_seconds": n / FPS, "fps": FPS, "width": cw, "height": ch,
        "audio_sample_rate": AUDIO_SR, "audio_channels": AUDIO_CH, "waveform_samples_per_channel": ns,
        "audio_mono_to_stereo": "mono" in (info.get("audio_layout") or ""),
        "time_padding_seconds": t["pad_seconds"], "tail_trimmed_seconds": t["trimmed_seconds"],
        "sha256": sha256(mp4), "audio_sha256": sha256(wav)}

    shutil.copy2(src, stage_p / "oringin.mp4")
    prov["oringin_sha256"] = sha256(stage_p / "oringin.mp4")
    prov["oringin_matches_source"] = prov["oringin_sha256"] == prov["source_sha256"]
    write_json(work / "provenance.json", prov)
    write_json(work / "export_log.json", LOG)
    print(f"target: src {t['t0']:.3f}-{t['t1']:.3f}s (semantic end {t['semantic_end']:.3f}) N={n} "
          f"pad={t['pad_seconds']:.3f}s trim={t['trimmed_seconds']:.3f}s")
    for r in prov["reference_videos"]:
        print(f"{r['path']}: src {r['source_seconds_used']} N={r['num_frames']} pad={r['time_padding_seconds']:.3f} trim={r['tail_trimmed_seconds']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
