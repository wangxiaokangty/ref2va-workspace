"""Shared helpers for Ref2VA (reference-to-video-and-audio) sample construction.

All tools only depend on the workspace `.venv` (numpy, Pillow, soundfile, imageio-ffmpeg).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent

# Local media profile (reference/media-profile.md)
FPS = 24
AUDIO_SR = 32000
AUDIO_CH = 2
AREA_BUDGET = 768 * 1344
SHORT_SIDE = 768
MIN_SEC, MAX_SEC = 5.0, 15.0          # target range (local profile)
REF_MIN_SEC, REF_MAX_SEC = 2.0, 15.0  # reference video/audio range (H3 docs)


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run(cmd: list[str], capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=capture, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-4000:]}")
    return p


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def legal_frames(n: int) -> bool:
    return n >= 5 and (n - 5) % 17 == 0


def legal_below(n: int) -> int:
    return 17 * ((n - 5) // 17) + 5


def legal_above(n: int) -> int:
    return 17 * math.ceil((n - 5) / 17) + 5


def canvas_for(width: int, height: int) -> tuple[int, int]:
    """Short side 768, area budget 768*1344, both sides multiples of 32."""
    landscape = width >= height
    a = width / height if landscape else height / width
    long_max = (AREA_BUDGET // SHORT_SIDE) // 32 * 32
    long_side = min(int(round(SHORT_SIDE * a / 32)) * 32, long_max)
    return (long_side, SHORT_SIDE) if landscape else (SHORT_SIDE, long_side)


def fit_in_canvas(width: int, height: int, cw: int, ch: int) -> tuple[int, int, int, int]:
    """Scale preserving aspect ratio (even dims), returns (sw, sh, pad_x, pad_y)."""
    s = min(cw / width, ch / height)
    sw = int(round(width * s / 2)) * 2
    sh = int(round(height * s / 2)) * 2
    return sw, sh, (cw - sw) // 2, (ch - sh) // 2


def probe_streams(path: Path | str) -> dict:
    """Parse `ffmpeg -i` stream header (ffprobe is not available in this workspace)."""
    p = run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], check=False)
    txt = p.stderr
    info: dict = {"raw_header": [l.strip() for l in txt.splitlines() if "Stream #" in l or "Duration" in l]}
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", txt)
    if m:
        info["container_duration"] = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    v = re.search(r"Stream #\S+.*Video: (\w+).*?, (\d{2,5})x(\d{2,5})", txt)
    if v:
        info["video_codec"], info["width"], info["height"] = v[1], int(v[2]), int(v[3])
    fr = re.search(r"Video:.*?([\d.]+) fps", txt)
    if fr:
        info["nominal_fps"] = float(fr[1])
    a = re.search(r"Audio: (\w+).*?, (\d+) Hz, ([\w.()]+)", txt)
    if a:
        info["audio_codec"], info["audio_sample_rate"], info["audio_layout"] = a[1], int(a[2]), a[3].rstrip(",")
    info["has_audio"] = a is not None
    return info


def frame_pts(path: Path | str) -> list[float]:
    """Decode every video frame and return its presentation timestamp (seconds)."""
    p = run([ffmpeg_exe(), "-hide_banner", "-nostats", "-i", str(path), "-map", "0:v:0",
             "-vf", "showinfo", "-f", "null", "-"], check=True)
    pts = [float(m[1]) for m in re.finditer(r"pts_time:([-\d.e]+)", p.stderr)]
    return pts


def decode_audio(path: Path | str, sr: int | None = None, channels: int | None = None,
                 start: float | None = None, end: float | None = None) -> tuple[np.ndarray, int]:
    """Decode audio to float32 [samples, channels]."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostats", "-v", "error"]
    cmd += ["-i", str(path), "-map", "0:a:0"]
    af = []
    if start is not None or end is not None:
        parts = []
        if start is not None:
            parts.append(f"start={start:.6f}")
        if end is not None:
            parts.append(f"end={end:.6f}")
        af.append("atrim=" + ":".join(parts))
        af.append("asetpts=PTS-STARTPTS")
    if af:
        cmd += ["-af", ",".join(af)]
    if sr:
        cmd += ["-ar", str(sr)]
    if channels:
        cmd += ["-ac", str(channels)]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-2000:])
    ch = channels or 2
    if sr is None:
        sr = probe_streams(path).get("audio_sample_rate", 0)
    data = np.frombuffer(p.stdout, dtype=np.float32)
    if channels is None:
        ch = 1 if "mono" in probe_streams(path).get("audio_layout", "") else 2
    return data.reshape(-1, ch), sr


def read_frames_gray(path: Path | str, w: int = 160, h: int = 90) -> np.ndarray:
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-v", "error", "-i", str(path), "-map", "0:v:0",
                        "-vf", f"scale={w}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-2000:])
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)


def cut_scores(frames: np.ndarray) -> np.ndarray:
    """Per-frame cut score between frame i-1 and i (index i); combines pixel and histogram diffs."""
    f = frames.astype(np.float32)
    pix = np.zeros(len(f))
    pix[1:] = np.abs(f[1:] - f[:-1]).mean(axis=(1, 2, 3))
    hist = np.zeros(len(f))
    prev = None
    for i, fr in enumerate(frames):
        hsts = [np.histogram(fr[..., c], bins=32, range=(0, 256))[0] for c in range(3)]
        hv = np.concatenate(hsts).astype(np.float32)
        hv /= hv.sum() + 1e-9
        if prev is not None:
            hist[i] = 0.5 * np.abs(hv - prev).sum() * 100
        prev = hv
    return 0.5 * pix + 0.5 * hist


def detect_cuts(scores: np.ndarray, thresh: float = 8.0, min_gap: int = 4) -> list[int]:
    """Cut at index i means a new shot starts at frame i. Uses an absolute threshold plus a
    local-ratio test so that fast motion does not trigger every frame."""
    cuts: list[int] = []
    n = len(scores)
    for i in range(1, n):
        s = scores[i]
        lo, hi = max(1, i - 6), min(n, i + 7)
        neigh = np.concatenate([scores[lo:i], scores[i + 1:hi]])
        local = float(np.median(neigh)) if len(neigh) else 0.0
        if s >= thresh and s >= 3.0 * local + 3:
            if cuts and i - cuts[-1] < min_gap:
                if s > scores[cuts[-1]]:
                    cuts[-1] = i
                continue
            cuts.append(i)
    return cuts


def write_json(path: Path | str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


def read_json(path: Path | str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def contact_sheet(images: list, labels: list[str], out: Path | str, cols: int = 6, tile_w: int = 320) -> None:
    from PIL import Image, ImageDraw

    if not images:
        return
    ims = [im if isinstance(im, Image.Image) else Image.fromarray(im) for im in images]
    tw = tile_w
    th = int(round(ims[0].height * tw / ims[0].width))
    rows = math.ceil(len(ims) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * (th + 18)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for k, (im, lab) in enumerate(zip(ims, labels)):
        r, c = divmod(k, cols)
        sheet.paste(im.resize((tw, th)), (c * tw, r * (th + 18) + 18))
        d.text((c * tw + 4, r * (th + 18) + 3), lab, fill=(255, 255, 0))
    sheet.save(out, quality=88)


def extract_frames(path: Path | str, indices: list[int], w: int | None = None) -> list:
    """Extract exact decoded frames by zero-based index (as PIL images)."""
    from PIL import Image

    if not indices:
        return []
    sel = "+".join(f"eq(n\\,{i})" for i in sorted(set(indices)))
    vf = f"select='{sel}'"
    if w:
        vf += f",scale={w}:-2"
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-v", "error", "-i", str(path), "-map", "0:v:0",
                        "-vf", vf, "-vsync", "0", "-f", "image2pipe", "-vcodec", "png", "-"],
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-2000:])
    out, data = [], p.stdout
    sig = b"\x89PNG\r\n\x1a\n"
    starts = [m.start() for m in re.finditer(re.escape(sig), data)]
    import io

    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(data)
        out.append(Image.open(io.BytesIO(data[s:e])).convert("RGB"))
    by_idx = dict(zip(sorted(set(indices)), out))
    return [by_idx[i] for i in indices]
