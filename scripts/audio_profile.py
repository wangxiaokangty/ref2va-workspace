"""Describe audio layers in a time window for annotation evidence (run with .venv-asr: needs librosa).

Usage: .venv-asr/bin/python scripts/audio_profile.py <media> <start> <end> [--hop 0.5]
Prints per-window RMS (dB), spectral centroid, flatness, dominant pitch classes and the strongest
spectral peaks, plus onset rate and a global tempo estimate. These are measurements to support
(not replace) listening; tonal stable peaks over silence suggest a music bed / drone.
"""
import argparse
import subprocess
from pathlib import Path

import librosa
import numpy as np


def load(media, start, end, sr=22050):
    ws = Path(__file__).resolve().parent.parent
    ff = sorted((ws / ".venv").glob("lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"))[0]
    raw = subprocess.run([str(ff), "-v", "error", "-ss", str(start), "-to", str(end), "-i", media, "-vn", "-ac", "1",
                          "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media"); ap.add_argument("start", type=float); ap.add_argument("end", type=float)
    ap.add_argument("--hop", type=float, default=0.5)
    a = ap.parse_args()
    y, sr = load(a.media, a.start, a.end)
    names = "C C# D D# E F F# G G# A A# B".split()
    h = int(a.hop * sr)
    for i in range(0, len(y) - h // 2, h):
        seg = y[i:i + h]
        rms = 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-9)
        S = np.abs(librosa.stft(seg, n_fft=4096))
        spec = S.mean(1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        cen = float((spec * freqs).sum() / (spec.sum() + 1e-9))
        flat = float(librosa.feature.spectral_flatness(S=S).mean())
        pk = np.argsort(spec[5:800])[-4:][::-1] + 5
        chroma = librosa.feature.chroma_stft(S=S ** 2, sr=sr).mean(1)
        top = [names[k] for k in np.argsort(chroma)[-3:][::-1]]
        print(f"{a.start + i / sr:7.2f}s rms={rms:6.1f}dB centroid={cen:6.0f}Hz flat={flat:.3f} "
              f"chroma={top} peaks={[int(freqs[k]) for k in pk]}")
    on = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
    print(f"onsets/s={len(on) / (len(y) / sr):.2f} tempo_est={float(np.atleast_1d(tempo)[0]):.1f} BPM")


if __name__ == "__main__":
    main()
