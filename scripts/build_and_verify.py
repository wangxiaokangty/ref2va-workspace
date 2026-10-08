"""Step 4: assemble dataset.jsonl in the staging dir, verify it, publish to OUTPUT_DIR, verify again.

Usage:
  .venv/bin/python scripts/build_and_verify.py <WORK_DIR> <STAGE_DIR> <OUTPUT_DIR> [--verify-only]

Inputs in WORK_DIR:
  prompt.md        final H3 six-section prompt (the file content is the prompt string, verbatim)
  record_spec.json {"id", "split_group", "references": [...], "needs_audio_review": bool,
                    "uncertain_audio_spans_seconds": [[s, e], ...],
                    "manual_checks": [{"check", "status", "detail"}]}   # human/agent semantic review
  provenance.json, export_config.json (from export_sample.py)
  target_cuts_measured.json (from inspect_export.py)
Writes WORK_DIR/verification.json with status completed | needs_review | failed.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2v_common import (AUDIO_CH, AUDIO_SR, FPS, MAX_SEC, MIN_SEC, REF_MAX_SEC, REF_MIN_SEC, WORKSPACE,  # noqa: E402
                        decode_audio, extract_frames, frame_pts, legal_frames, probe_streams, read_json,
                        sha256, write_json)

SECTIONS = ["subject_definitions", "summary", "retention_analysis", "detailed_description",
            "overall_soundscape", "non_diegetic_music"]
VIS_MARKS = {"fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"}
AUD_MARKS = {"fully_copy", "partially_copy", "reference", "weak_reference"}
TASK_TYPES = {"keyframe completion", "reference generation", "video editing", "video continuation",
              "audio reuse", "audio reference"}
TARGET_KEYS = ["video", "audio", "width", "height", "num_frames", "fps", "audio_sample_rate",
               "audio_channels", "waveform_samples_per_channel"]


class Checks:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, name: str, ok: bool | None, detail="") -> None:
        st = "not_checked" if ok is None else ("passed" if ok else "failed")
        self.items.append({"check": name, "status": st, "detail": str(detail)})

    @property
    def failed(self) -> list[dict]:
        return [c for c in self.items if c["status"] == "failed"]


def build(work: Path, stage: Path) -> dict:
    spec = read_json(work / "record_spec.json")
    prov = read_json(work / "provenance.json")
    prompt = (work / "prompt.md").read_text(encoding="utf-8").strip("\n")
    tv = prov["target_video"]
    rec = {
        "id": spec["id"], "split_group": spec["split_group"],
        "path_base": "directory containing dataset.jsonl", "prompt": prompt,
        "references": spec["references"],
        "target": {"video": "targets/target.mp4", "audio": "targets/target_audio.wav",
                   "width": tv["width"], "height": tv["height"], "num_frames": tv["num_frames"], "fps": FPS,
                   "audio_sample_rate": AUDIO_SR, "audio_channels": AUDIO_CH,
                   "waveform_samples_per_channel": tv["waveform_samples_per_channel"]},
        "needs_audio_review": bool(spec["needs_audio_review"]),
        "uncertain_audio_spans_seconds": spec["uncertain_audio_spans_seconds"],
    }
    with open(stage / "dataset.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def split_sections(prompt: str) -> dict[str, str] | None:
    pat = re.compile(r"^(" + "|".join(SECTIONS) + r"):\s*$", re.M)
    hits = list(pat.finditer(prompt))
    if [h[1] for h in hits] != SECTIONS or hits[0].start() != 0:
        return None
    out = {}
    for k, h in enumerate(hits):
        end = hits[k + 1].start() if k + 1 < len(hits) else len(prompt)
        out[h[1]] = prompt[h.end():end].strip()
    return out


def ssim_like(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute difference (0-255) on 64x36 grayscale."""
    x = np.asarray(a.convert("L").resize((64, 36)), np.float32)
    y = np.asarray(b.convert("L").resize((64, 36)), np.float32)
    return float(np.abs(x - y).mean())


def verify(root: Path, work: Path, c: Checks) -> dict:
    prov = read_json(work / "provenance.json")
    lines = (root / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    c.add("jsonl_single_line", len(lines) == 1, f"{len(lines)} line(s)")
    rec = json.loads(lines[0])
    types = {"id": str, "split_group": str, "path_base": str, "prompt": str, "references": list, "target": dict,
             "needs_audio_review": bool, "uncertain_audio_spans_seconds": list}
    c.add("jsonl_fields_types", all(isinstance(rec.get(k), t) for k, t in types.items()) and set(rec) == set(types),
          {k: type(rec.get(k)).__name__ for k in types})
    c.add("path_base_value", rec["path_base"] == "directory containing dataset.jsonl")
    c.add("target_fields", all(k in rec["target"] for k in TARGET_KEYS), sorted(rec["target"]))
    prompt = rec["prompt"]
    c.add("prompt_not_placeholder", len(prompt) > 1500 and not re.search(r"TODO|TBD|\.\.\.\s*$|<placeholder>", prompt),
          f"{len(prompt)} chars")

    # paths
    refs = rec["references"]
    paths = [r["path"] for r in refs] + [rec["target"]["video"], rec["target"]["audio"]]
    c.add("relative_paths_resolve", all(not Path(p).is_absolute() and (root / p).is_file() for p in paths), paths)
    c.add("references_only_inputs", all(r["path"].startswith("inputs/") for r in refs))
    on_disk = sorted(str(p.relative_to(root)) for p in (root / "inputs").iterdir())
    c.add("no_unused_or_missing_inputs", on_disk == sorted(r["path"] for r in refs), on_disk)
    c.add("target_not_in_references", not any(p.startswith("targets/") or "oringin" in p for p in [r["path"] for r in refs]))
    c.add("oringin_hash_matches_source", sha256(root / "oringin.mp4") == prov["source_sha256"])

    # images
    n_img = n_vid = n_aud = 0
    vid_total = aud_total = 0.0
    for r in refs:
        p = root / r["path"]
        if r["type"] == "image":
            n_img += 1
            try:
                im = Image.open(p)
                im.load()
                c.add(f"image_readable:{r['path']}", True, f"{im.size} {im.mode}")
            except Exception as e:  # noqa: BLE001
                c.add(f"image_readable:{r['path']}", False, e)
        elif r["type"] == "video":
            n_vid += 1
            info = probe_streams(p)
            pts = frame_pts(p)
            n = len(pts)
            dur = n / FPS
            vid_total += dur
            d = np.diff(pts)
            c.add(f"ref_video_params:{r['path']}", legal_frames(n) and info.get("width") and np.allclose(d, 1 / FPS, atol=1e-3),
                  f"{n} frames, {dur:.4f}s, {info.get('width')}x{info.get('height')}")
            c.add(f"ref_video_duration_range:{r['path']}", REF_MIN_SEC <= dur <= REF_MAX_SEC, f"{dur:.3f}s")
            inc = bool(r.get("include_audio"))
            c.add(f"ref_video_audio_flag:{r['path']}", inc == info["has_audio"] and (("audio_label" in r) == inc),
                  f"include_audio={inc} stream={info['has_audio']}")
            if inc:
                n_aud_sync = 1
                a, _ = decode_audio(p, sr=AUDIO_SR, channels=2)
                c.add(f"ref_video_audio_decode:{r['path']}", abs(a.shape[0] / AUDIO_SR - dur) < 0.05,
                      f"{a.shape[0]} samples/ch vs {dur:.4f}s")
        elif r["type"] == "audio":
            n_aud += 1
            data, sr = sf.read(p)
            aud_total += len(data) / sr
            c.add(f"ref_audio_range:{r['path']}", REF_MIN_SEC <= len(data) / sr <= REF_MAX_SEC, f"{len(data) / sr:.3f}s")
    c.add("reference_limits", n_img <= 9 and n_vid <= 3 and vid_total <= 15 + 1e-6 and n_aud <= 3 and aud_total <= 15
          and len(refs) <= 12 and (n_img + n_vid) > 0,
          f"images={n_img} videos={n_vid} video_total={vid_total:.3f}s audio_files={n_aud} audio_total={aud_total:.3f}s files={len(refs)}")

    # target video
    tg = rec["target"]
    tp = root / tg["video"]
    info = probe_streams(tp)
    pts = frame_pts(tp)
    n = len(pts)
    T = n / FPS
    c.add("target_video_decode_params", n == tg["num_frames"] and info["width"] == tg["width"] and info["height"] == tg["height"]
          and tg["fps"] == FPS and np.allclose(np.diff(pts), 1 / FPS, atol=1e-3),
          f"decoded {n} frames {info['width']}x{info['height']}; record {tg['num_frames']} {tg['width']}x{tg['height']}")
    c.add("target_frames_legal_17k5", legal_frames(n), n)
    c.add("target_duration_range", MIN_SEC <= T <= MAX_SEC, f"{T:.4f}s")
    c.add("canvas_multiple_of_32", tg["width"] % 32 == 0 and tg["height"] % 32 == 0, f"{tg['width']}x{tg['height']}")
    c.add("target_mp4_has_audio", info["has_audio"], info.get("raw_header"))

    # WAV
    wp = root / tg["audio"]
    sfi = sf.info(wp)
    exp = round(n / FPS * AUDIO_SR)
    c.add("target_wav_format", sfi.samplerate == AUDIO_SR == tg["audio_sample_rate"] and sfi.channels == AUDIO_CH == tg["audio_channels"]
          and sfi.subtype == "PCM_16" and sfi.format == "WAV", f"{sfi.samplerate} Hz {sfi.channels} ch {sfi.format}/{sfi.subtype}")
    c.add("target_wav_samples", sfi.frames == exp == tg["waveform_samples_per_channel"],
          f"wav={sfi.frames} expected={exp} record={tg['waveform_samples_per_channel']}")
    c.add("target_wav_video_duration_within_1_sample", abs(sfi.frames / AUDIO_SR - T) <= 1 / AUDIO_SR,
          f"|{sfi.frames / AUDIO_SR:.6f} - {T:.6f}|")
    wav, _ = sf.read(wp, dtype="float32")
    c.add("target_wav_not_silent", float(np.sqrt((wav ** 2).mean())) > 1e-3, f"rms={float(np.sqrt((wav ** 2).mean())):.4f}")
    # MP4 audio vs WAV alignment
    ma, _ = decode_audio(tp, sr=AUDIO_SR, channels=2)
    m = min(len(ma), len(wav))
    x, y = ma[:m].mean(1), wav[:m].mean(1)
    lags = range(-1600, 1601, 8)
    best = max(lags, key=lambda L: float(np.dot(x[max(0, L):m + min(0, L)], y[max(0, -L):m - max(0, L)])))
    corr = float(np.corrcoef(x[max(0, best):m + min(0, best)], y[max(0, -best):m - max(0, best)])[0, 1])
    c.add("mp4_audio_matches_wav", abs(best) <= 16 and corr > 0.95,
          f"best lag {best} samples ({best / AUDIO_SR * 1000:.2f} ms), corr={corr:.4f}, mp4 samples={len(ma)} (AAC padding {len(ma) - len(wav)}), wav={len(wav)}")

    # provenance / source interval checks
    tv = prov["target_video"]
    t0, t1 = tv["source_seconds_used"]
    sem_end = tv["source_end_seconds"]
    ok = True
    det = []
    for r in prov["reference_videos"] + prov["reference_audios"]:
        a0, a1 = r.get("source_seconds_used") or r["source_seconds"]
        inter = min(a1, sem_end) - max(a0, t0)
        det.append(f"{r['path']} [{a0:.3f},{a1:.3f}) vs target [{t0:.3f},{sem_end:.3f}) overlap={max(0, inter):.3f}s")
        ok &= inter <= 1e-6
    for im in prov["images"]:
        inside = t0 - 1e-6 <= im["source_seconds"] < sem_end
        det.append(f"{im['path']} @ {im['source_seconds']:.3f}s inside_target={inside}")
        ok &= not inside
    c.add("source_intervals_disjoint", ok, det)
    c.add("no_padding_beyond_tolerance", tv["time_padding_seconds"] <= 0.25,
          f"target pad {tv['time_padding_seconds']:.3f}s, trimmed {tv['tail_trimmed_seconds']:.3f}s")

    src = WORKSPACE / prov["source_path_from_workspace"]
    s0, s1 = tv["source_frame_range_used"]
    srcf = extract_frames(src, [s0, s1 - 1])
    tf = extract_frames(tp, [0, n - 1 - round(tv["time_padding_seconds"] * FPS)])
    d_first, d_last = ssim_like(srcf[0], tf[0]), ssim_like(srcf[1], tf[1])
    c.add("target_boundary_frames_match_source", d_first < 6 and d_last < 6,
          f"first frame vs src f{s0}: {d_first:.2f}; last real frame vs src f{s1 - 1}: {d_last:.2f} (mean abs diff 0-255)")
    if s0 > 0:
        prevf = extract_frames(src, [s0 - 1])[0]
        c.add("target_first_frame_not_previous_shot", True,
              f"diff to src f{s0 - 1} (previous frame) = {ssim_like(prevf, tf[0]):.2f}, to f{s0} = {d_first:.2f}")
    sa, _ = decode_audio(src, sr=AUDIO_SR, channels=2, start=t0, end=t1)
    mm = min(len(sa), len(wav))
    cc = float(np.corrcoef(sa[:mm].mean(1), wav[:mm].mean(1))[0, 1])
    c.add("target_wav_matches_source_interval", cc > 0.98, f"corr with source [{t0:.3f},{t1:.3f}) = {cc:.4f}")

    # near-duplicate check: reference media vs target frames
    tsamp = extract_frames(tp, list(range(0, n, 6)))
    dups = []
    for r in refs:
        p = root / r["path"]
        if r["type"] == "image":
            cand = [Image.open(p).convert("RGB")]
        elif r["type"] == "video":
            nv = len(frame_pts(p))
            cand = extract_frames(p, list(range(0, nv, 6)))
        else:
            continue
        best_d = min(ssim_like(a, b) for a in cand for b in tsamp)
        dups.append(f"{r['path']}: min frame diff to target {best_d:.2f}")
        if best_d < 3.0:
            ok_dup = False
            dups[-1] += " NEAR-DUPLICATE"
    c.add("no_near_duplicate_reference_frames", not any("NEAR-DUPLICATE" in d for d in dups), dups)

    # prompt structure
    secs = split_sections(prompt)
    c.add("prompt_six_sections_in_order", secs is not None)
    if secs:
        labels = set(re.findall(r"<(Subject|Picture|Video|Audio) (\d+)>", prompt))
        by = {k: sorted(int(n_) for t, n_ in labels if t == k) for k in ("Subject", "Picture", "Video", "Audio")}
        c.add("labels_numbered_consecutively", all(v == list(range(1, len(v) + 1)) for v in by.values()), by)
        ref_labels = {r["label"] for r in refs} | {r["audio_label"] for r in refs if r.get("audio_label")}
        pic_vid_aud = {f"<{t} {n_}>" for t, n_ in labels if t != "Subject"}
        c.add("media_labels_map_to_references", pic_vid_aud == ref_labels, f"prompt={sorted(pic_vid_aud)} refs={sorted(ref_labels)}")
        defs = secs["subject_definitions"]
        defined = set(re.findall(r"^<(?:Subject|Video|Audio|Picture) \d+>", defs, re.M))
        subj = {f"<Subject {n_}>" for t, n_ in labels if t == "Subject"}
        c.add("subjects_defined", subj <= defined, f"used={sorted(subj)} defined={sorted(defined)}")
        aud_used = {f"<Audio {n_}>" for t, n_ in labels if t == "Audio"}
        c.add("audio_labels_defined", aud_used <= defined, sorted(aud_used))
        m_ = re.match(r"\[([a-z +]+)\]", secs["summary"])
        tt = [x.strip() for x in m_[1].split("+")] if m_ else []
        c.add("summary_task_types_valid", bool(tt) and set(tt) <= TASK_TYPES and len(tt) == len(set(tt))
              and (("audio reference" in tt or "audio reuse" in tt) == bool(aud_used)), tt)
        ra_ok, ra_det = True, []
        for line in secs["retention_analysis"].splitlines():
            mm_ = re.match(r"(<(Subject|Picture|Video|Audio) \d+>)[^:]*:\s*([a-z_]+)\s+-", line)
            if not mm_:
                ra_ok = False
                ra_det.append(f"unparsed: {line[:60]}")
                continue
            marks = AUD_MARKS if mm_[2] == "Audio" else VIS_MARKS
            ra_ok &= mm_[3] in marks
            ra_det.append(f"{mm_[1]}={mm_[3]}")
        defined_lines = set(re.findall(r"^(<(?:Subject|Video|Audio|Picture) \d+>)", defs, re.M))
        ra_labels = set(re.findall(r"^(<(?:Subject|Video|Audio|Picture) \d+>)", secs["retention_analysis"], re.M))
        c.add("retention_markers_valid_and_complete", ra_ok and ra_labels == defined_lines, ra_det)
        c.add("no_speaker_ids_in_retention", not re.search(r"\(S\d", secs["retention_analysis"]))
        dd = secs["detailed_description"]
        shots = re.findall(r"\[Shot (\d+)\](?: At (\d\d):(\d\d\.\d{3}),)?", dd)
        nums = [int(s[0]) for s in shots]
        times = [int(s[1]) * 60 + float(s[2]) for s in shots[1:] if s[1]]
        c.add("shot_format", nums == list(range(1, len(nums) + 1)) and shots and not shots[0][1]
              and len(times) == len(shots) - 1 and all(a < b for a, b in zip(times, times[1:]))
              and all(0 < t < T for t in times), f"shots={nums} times={times}")
        c.add("style_sentence_before_shot_1", not dd.lstrip().startswith("[Shot"))
        meas = read_json(work / "target_cuts_measured.json")
        spec = read_json(work / "record_spec.json")
        det = meas["detected_cut_times"]
        mt = spec.get("confirmed_cut_times", det)
        unsupported = [t for t in mt if not any(abs(t - d_) <= 1 / FPS + 1e-6 for d_ in det)
                       and t not in spec.get("manual_cut_times", [])]
        c.add("confirmed_cuts_supported_by_detection", not unsupported,
              f"confirmed={mt} detected={det} rejected={spec.get('rejected_cut_times', [])} unsupported={unsupported}")
        c.add("shot_times_match_measured_cuts", len(mt) == len(times) and all(abs(a - b) <= 1 / FPS + 1e-6 for a, b in zip(mt, times)),
              f"prompt={times} confirmed={mt}")
        d_blocks = re.findall(r"<d>(.*?)</d>", dd, re.S)
        c.add("dialogue_tags_wellformed", prompt.count("<d>") == prompt.count("</d>") == len(d_blocks)
              and all(re.match(r"\[[A-Z][a-zA-Z ]+\] \S", b) for b in d_blocks), d_blocks)
        c.add("dialogue_only_in_detailed_description", "<d>" not in secs["overall_soundscape"] + secs["non_diegetic_music"])
        spk = re.findall(r"\(S(\d+)(?:,S\d+)*\)", dd)
        first_order = list(dict.fromkeys(int(x) for x in spk))
        c.add("speaker_ids_in_first_use_order", first_order == list(range(1, len(first_order) + 1)), first_order)
        words = len(re.findall(r"[A-Za-z']+", dd))
        c.add("detailed_description_length", 250 <= words <= 650, f"{words} words (guide 350-500; accuracy first)")
    unclear = "[unclear]" in prompt
    spans = rec["uncertain_audio_spans_seconds"]
    c.add("audio_review_flags_consistent", (not unclear or rec["needs_audio_review"]) and (bool(spans) <= rec["needs_audio_review"]),
          f"[unclear]={unclear} needs_audio_review={rec['needs_audio_review']} spans={spans}")
    c.add("uncertain_spans_within_target", all(0 <= s < e <= T for s, e in spans), spans)
    pm = (work / "prompt.md").read_text(encoding="utf-8").strip("\n")
    c.add("prompt_md_matches_record", pm == prompt)
    return rec


def main() -> None:
    work, stage, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    verify_only = "--verify-only" in sys.argv
    spec = read_json(work / "record_spec.json")
    if not verify_only:
        build(work, stage)
        c = Checks()
        verify(stage, work, c)
        if c.failed:
            write_json(work / "verification_stage.json", {"status": "failed", "checks": c.items})
            for f in c.failed:
                print("FAILED", f["check"], f["detail"])
            sys.exit(1)
        if out.exists():
            old = out / "dataset.jsonl"
            if old.exists():
                o = json.loads(old.read_text(encoding="utf-8").splitlines()[0])
                if o["id"] != spec["id"] or sha256(out / "oringin.mp4") != read_json(work / "provenance.json")["source_sha256"]:
                    sys.exit(f"refusing to overwrite {out}: different sample/source; choose a new OUTPUT_DIR")
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stage), str(out))
    c = Checks()
    verify(out, work, c)
    for m in spec.get("manual_checks", []):
        c.items.append({"check": "manual:" + m["check"], "status": m["status"], "detail": m.get("detail", "")})
    review = list(spec.get("review_items", []))
    if c.failed:
        status = "failed"
    elif spec["needs_audio_review"] or review or any(m["status"] == "not_checked" and m.get("required") for m in spec.get("manual_checks", [])):
        status = "needs_review"
    else:
        status = "completed"
    write_json(work / "verification.json", {
        "status": status, "sample_id": spec["id"], "output_dir": str(out),
        "review_items": review, "checks": c.items,
        "not_done": ["model inference with H3", "training", "human listening test (agent used ASR + waveform/visual inspection only)"]})
    (work / "verification_stage.json").unlink(missing_ok=True)
    n_p = sum(x["status"] == "passed" for x in c.items)
    print(f"status={status} passed={n_p} failed={len(c.failed)} not_checked={sum(x['status'] == 'not_checked' for x in c.items)}")
    for f in c.failed:
        print("FAILED", f["check"], f["detail"])


if __name__ == "__main__":
    main()
