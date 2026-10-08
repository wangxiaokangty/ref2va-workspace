"""Dense frame sheet for checking a candidate interval or shot boundary.

Usage:
  .venv/bin/python scripts/dense_frames.py <video> <start_frame> <end_frame_excl> <step> <out.jpg> [cols] [tile_w]

Labels show the zero-based source frame index and its PTS (presentation timestamp).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2v_common import contact_sheet, extract_frames, frame_pts  # noqa: E402


def main() -> None:
    video, s, e, step, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    cols = int(sys.argv[6]) if len(sys.argv) > 6 else 6
    tw = int(sys.argv[7]) if len(sys.argv) > 7 else 320
    pts = frame_pts(video)
    idx = list(range(s, min(e, len(pts)), step))
    ims = extract_frames(video, idx, w=tw)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    contact_sheet(ims, [f"f{i} {pts[i]:.3f}s" for i in idx], out, cols=cols, tile_w=tw)
    print(out, len(idx), "frames")


if __name__ == "__main__":
    main()
