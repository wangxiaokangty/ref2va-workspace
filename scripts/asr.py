"""ASR (automatic speech recognition) helper. Run with the ASR environment:

  .venv-asr/bin/python scripts/asr.py <media> <out.json> [--start S --end E] [--lang en]

Uses mlx-whisper (whisper-large-v3-turbo, cached locally) with word timestamps. With --start/--end the
audio is cut first and timestamps are relative to the clip. ASR output is evidence, not ground truth.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path



def ffmpeg() -> str:
    try:
        import imageio_ffmpeg as iff

        return iff.get_ffmpeg_exe()
    except Exception:  # pragma: no cover
        ws = Path(__file__).resolve().parent.parent
        cands = list((ws / ".venv").glob("lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"))
        return str(cands[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("out")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--lang")
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
    a = ap.parse_args()
    import mlx_whisper

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.f32"
        cmd = [ffmpeg(), "-v", "error", "-y", "-i", a.media]
        if a.start is not None:
            cmd += ["-ss", f"{a.start:.4f}"]
        if a.end is not None:
            cmd += ["-to", f"{a.end:.4f}"]
        cmd += ["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-acodec", "pcm_f32le", str(wav)]
        subprocess.run(cmd, check=True)
        import numpy as np

        audio = np.fromfile(wav, dtype=np.float32)
        kw = dict(path_or_hf_repo=a.model, word_timestamps=True, condition_on_previous_text=False,
                  temperature=0.0)
        if a.lang:
            kw["language"] = a.lang
        r = mlx_whisper.transcribe(audio, **kw)
    segs = []
    for s in r["segments"]:
        segs.append({"start": round(s["start"], 3), "end": round(s["end"], 3), "text": s["text"].strip(),
                     "avg_logprob": round(s.get("avg_logprob", 0), 3),
                     "no_speech_prob": round(s.get("no_speech_prob", 0), 3),
                     "words": [{"w": w["word"], "s": round(w["start"], 3), "e": round(w["end"], 3),
                                "p": round(w.get("probability", 0), 3)} for w in s.get("words", [])]})
    res = {"media": a.media, "clip_start": a.start, "clip_end": a.end, "model": a.model,
           "language": r.get("language"), "segments": segs,
           "note": "ASR evidence only; timestamps relative to clip start when clipped."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for s in segs:
        print(f"[{s['start']:7.2f}-{s['end']:7.2f}] lp={s['avg_logprob']:.2f} ns={s['no_speech_prob']:.2f} {s['text']}")


if __name__ == "__main__":
    main()
