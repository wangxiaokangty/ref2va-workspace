"""Zero-shot audio tagging with CLAP (Contrastive Language-Audio Pretraining), as annotation evidence.

Usage: .venv-asr/bin/python scripts/audio_tags.py <media> <start> <end> [--win 2.0] [--labels "a|b|c"]
Uses the locally cached laion/clap-htsat-unfused model. Scores are relative (softmax over the label set);
treat them as hints to cross-check with ASR, waveform and visuals, never as ground truth.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

DEFAULT = ("speech|female speech|male speech|whispering|crying or sobbing|music|background music score|"
           "muffled music through a wall|piano|strings|synthesizer drone|drums and beat|electronic dance music|"
           "pop song with vocals|rain|thunder|wind|room tone silence|telephone line noise|phone ringing|"
           "beep|footsteps|door|crowd chatter|cat meow|crunching snack|sizzling frying oil|whoosh sound effect|"
           "cartoon sound effect|applause|laughter|typing on keyboard|pen writing on paper|paper rustling|birds chirping|traffic")


def load(media, start, end, sr=48000):
    ws = Path(__file__).resolve().parent.parent
    ff = sorted((ws / ".venv").glob("lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"))[0]
    raw = subprocess.run([str(ff), "-v", "error", "-ss", str(start), "-to", str(end), "-i", media, "-vn", "-ac", "1",
                          "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy(), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media"); ap.add_argument("start", type=float); ap.add_argument("end", type=float)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--labels", default=DEFAULT)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    labels = a.labels.split("|")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    y, sr = load(a.media, a.start, a.end)
    w = int(a.win * sr)
    with torch.no_grad():
        t = proc(text=[f"the sound of {l}" for l in labels], return_tensors="pt", padding=True)
        te = model.get_text_features(**t)
        te = te if torch.is_tensor(te) else te.pooler_output
        te = te / te.norm(dim=-1, keepdim=True)
        for i in range(0, max(1, len(y) - w // 3), w):
            seg = y[i:i + w]
            inp = proc(audio=[seg], sampling_rate=sr, return_tensors="pt")
            ae = model.get_audio_features(**inp)
            ae = ae if torch.is_tensor(ae) else ae.pooler_output
            ae = ae / ae.norm(dim=-1, keepdim=True)
            p = (100 * ae @ te.T).softmax(-1)[0].numpy()
            top = np.argsort(p)[::-1][:a.top]
            print(f"{a.start + i / sr:7.2f}-{a.start + (i + len(seg)) / sr:7.2f}s " +
                  ", ".join(f"{labels[k]}={p[k]:.2f}" for k in top))


if __name__ == "__main__":
    main()
