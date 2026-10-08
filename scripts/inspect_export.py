"""Step 3: inspect exported clips: measured cut times on the exported timeline + storyboards.

Usage:
  .venv/bin/python scripts/inspect_export.py <STAGE_DIR> <WORK_DIR>

Writes WORK_DIR/target_cuts_measured.json, target_storyboard.jpg, target_shot_boundaries.jpg and
<video_N>_storyboard.jpg for each reference video. Cut times are frame_index / 24 on the exported clip,
the values to use for `[Shot N] At MM:SS.mmm` after visual confirmation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2v_common import FPS, contact_sheet, cut_scores, detect_cuts, extract_frames, read_frames_gray, write_json  # noqa: E402


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:06.3f}"


def storyboard(mp4: Path, out: Path, step_s: float = 0.5) -> list[int]:
    frames = read_frames_gray(mp4)
    n = len(frames)
    cuts = detect_cuts(cut_scores(frames))
    idx = list(range(0, n, int(step_s * FPS)))
    ims = extract_frames(mp4, idx, w=320)
    contact_sheet(ims, [f"#{i} {i / FPS:.3f}s" for i in idx], out, cols=6)
    return cuts, n


def main(stage: str, work: str) -> None:
    st, wk = Path(stage), Path(work)
    tgt = st / "targets" / "target.mp4"
    cuts, n = storyboard(tgt, wk / "target_storyboard.jpg")
    bidx, labels = [], []
    for c in cuts:
        for k in (c - 1, c):
            bidx.append(k)
            labels.append(f"{'last' if k < c else 'first'} #{k} {k / FPS:.3f}s")
    bidx += [n - 1]
    labels += [f"final #{n - 1}"]
    contact_sheet(extract_frames(tgt, bidx, w=320), labels, wk / "target_shot_boundaries.jpg", cols=6)
    write_json(wk / "target_cuts_measured.json", {
        "num_frames": n, "fps": FPS, "detected_cut_frames": cuts,
        "detected_cut_times": [round(c / FPS, 3) for c in cuts],
        "detected_cut_mmss": [mmss(c / FPS) for c in cuts],
        "note": "automatic detection on the exported target; confirm visually with target_shot_boundaries.jpg"})
    print("target frames", n, "cuts", [(c, mmss(c / FPS)) for c in cuts])
    for v in sorted((st / "inputs").glob("video_*.mp4")):
        c, nv = storyboard(v, wk / f"reference_{v.stem}_storyboard.jpg")
        print(v.name, "frames", nv, "cuts", [(x, mmss(x / FPS)) for x in c])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
