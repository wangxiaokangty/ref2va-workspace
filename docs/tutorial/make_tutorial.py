"""Render the Ref2VA pipeline tutorial video (Chinese narration + slides).

Usage (from the workspace root, macOS only because narration uses `say -v Tingting`):
  .venv/bin/python docs/tutorial/make_tutorial.py
Output: docs/tutorial/ref2va_pipeline_tutorial.mp4 (1920x1080, 24 fps, H.264 + AAC).
Illustrations are taken from the real samples in outputs/ and inputs/.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "scripts"))
from r2v_common import extract_frames, ffmpeg_exe  # noqa: E402

W, H = 1920, 1080
BG = (18, 20, 27)
FG = (236, 238, 243)
DIM = (150, 156, 170)
ACC = (255, 176, 59)
ACC2 = (94, 196, 255)
OK = (120, 210, 140)
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"
OUT = WS / "docs" / "tutorial" / "ref2va_pipeline_tutorial.mp4"
VOICE, RATE = "Tingting", "205"


def font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(MONO if mono else FONT, size)


def wrap(draw: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=f) > width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def fit(im: Image.Image, bw: int, bh: int) -> Image.Image:
    s = min(bw / im.width, bh / im.height)
    return im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)


def labelled_row(frames: list[Image.Image], labels: list[str], tile_w: int = 600, gap: int = 16) -> Image.Image:
    tiles = [fit(f, tile_w, 2000) for f in frames]
    th = max(t.height for t in tiles)
    row = Image.new("RGB", (len(tiles) * tile_w + (len(tiles) - 1) * gap, th + 52), BG)
    d = ImageDraw.Draw(row)
    for k, (t, lab) in enumerate(zip(tiles, labels)):
        x = k * (tile_w + gap)
        row.paste(t, (x, 52))
        d.text((x + 4, 6), lab, font=font(30), fill=ACC)
    return row


def stack(rows: list[Image.Image], gap: int = 24) -> Image.Image:
    w = max(r.width for r in rows)
    out = Image.new("RGB", (w, sum(r.height for r in rows) + gap * (len(rows) - 1)), BG)
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + gap
    return out


# ---------------------------------------------------------------- illustrations
def illustrations() -> dict[str, Image.Image]:
    il: dict[str, Image.Image] = {}
    he = WS / "outputs/homeexchange_4cded0ce_001"
    pic = Image.open(he / "pure/inputs/picture_1.png").convert("RGB")
    vid = extract_frames(he / "pure/inputs/video_1.mp4", [20, 100, 170])
    tgt = extract_frames(he / "pure/targets/target.mp4", [30, 80, 140, 220])
    il["concept_refs"] = stack([labelled_row([pic], ["<Picture 1>  人物身份"], 520),
                                labelled_row(vid, ["<Video 1> 平行读信场景", "", ""], 250, 10)])
    il["concept_tgt"] = labelled_row(tgt[:2], ["目标 target.mp4（原片真实片段）", ""], 330, 10)
    il["concept_tgt2"] = labelled_row(tgt[2:], ["", ""], 330, 10)

    il["overview"] = Image.open(WS / "outputs/embed_cheetos_9181878e_001/work/analysis/shots_overview.jpg").convert("RGB")
    ch = WS / "outputs/embed_cheetos_9181878e_001"
    il["select"] = stack([
        labelled_row(extract_frames(WS / "inputs/embed-9181878ecd263a88.mp4", [180, 135]), ["参考图 1：母亲（目标外）", "参考图 2：孩子（目标外）"], 420),
        labelled_row(extract_frames(ch / "pure/inputs/video_1.mp4", [40, 140]), ["参考视频：检测仪 + 父亲音色", ""], 420),
        labelled_row(extract_frames(ch / "pure/targets/target.mp4", [60, 130, 220]), ["目标：黑光揭晓", "", ""], 276, 12)])

    hl = WS / "inputs/fde007ab7e7439e3.mp4"
    dis = extract_frames(hl, [395, 396, 397])
    lights = extract_frames(ch / "pure/targets/target.mp4", [23, 28, 31])
    lights[1] = ImageOps.autocontrast(lights[1], cutoff=1)
    dor = WS / "inputs/embed-b36ed8ad1f7919d5.mp4"
    jump = extract_frames(dor, [698, 699])
    il["bounds"] = stack([
        labelled_row(dis, ["f395 上一镜头", "f396 叠化混合帧 ×", "f397 干净新镜头"], 380, 12),
        labelled_row(lights, ["灯亮", "关灯（增强显示）", "紫外灯亮：同一机位 ≠ 切镜"], 380, 12),
        labelled_row(jump, ["f698", "f699 真正的跳切 √"], 380, 12)])
    leak = extract_frames(dor, [905, 745])
    same = [extract_frames(hl, [250])[0], Image.open(WS / "outputs/helpline_180_fde007ab_001/work/dense/t_full_80.png").convert("RGB")]
    il["leak"] = stack([
        labelled_row(leak, ["参考候选：倒带回放 ×", "目标里的同一动作"], 560),
        labelled_row(same, ["参考候选：窗边机位 ×", "目标第 2 镜头：同一机位"], 560)])
    il["shots"] = Image.open(he / "work/target_shot_boundaries.jpg").convert("RGB")
    return il


# ---------------------------------------------------------------- slide rendering
def base(title: str, step: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 8], fill=ACC)
    if step:
        d.text((80, 52), step, font=font(30), fill=ACC)
    d.text((80, 92 if step else 70), title, font=font(60), fill=FG)
    return im, d


def bullets(d, items: list[str], x: int, y: int, width: int, size: int = 36, gap: int = 18) -> int:
    f = font(size)
    for it in items:
        sub = it.startswith("  ")
        txt = it.strip()
        col = DIM if sub else FG
        xx = x + (40 if sub else 0)
        d.text((xx, y), "–" if sub else "•", font=f, fill=ACC2 if not sub else DIM)
        for k, line in enumerate(wrap(d, txt, f, width - (xx - x) - 40)):
            d.text((xx + 40, y), line, font=f, fill=col)
            y += size + 12
        y += gap
    return y


def paste_fit(im: Image.Image, pic: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    p = fit(pic, x1 - x0, y1 - y0)
    im.paste(p, (x0 + (x1 - x0 - p.width) // 2, y0 + (y1 - y0 - p.height) // 2))


def box(d, xy, text, fill=(34, 38, 50), outline=ACC2, size=34, col=FG, sub=None):
    d.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=3)
    x0, y0, x1, y1 = xy
    f = font(size)
    lines = text.split("\n")
    total = len(lines) * (size + 10) + (30 if sub else 0)
    y = (y0 + y1 - total) // 2
    for ln in lines:
        d.text(((x0 + x1 - d.textlength(ln, font=f)) / 2, y), ln, font=f, fill=col)
        y += size + 10
    if sub:
        fs = font(24, mono=all(ord(c) < 128 for c in sub))
        d.text(((x0 + x1 - d.textlength(sub, font=fs)) / 2, y + 4), sub, font=fs, fill=DIM)


def arrow(d, x0, y0, x1, y1, col=ACC):
    d.line([x0, y0, x1, y1], fill=col, width=5)
    import math
    a = math.atan2(y1 - y0, x1 - x0)
    for s in (2.6, -2.6):
        d.line([x1, y1, x1 + 22 * math.cos(a + s), y1 + 22 * math.sin(a + s)], fill=col, width=5)


def slides(il: dict) -> list[tuple[Image.Image, list[str]]]:
    S: list[tuple[Image.Image, list[str]]] = []

    im, d = base("Ref2VA 训练数据合成 Pipeline")
    d.text((80, 200), "MiniMax-H3 · 从单个真实视频构造参考生成样本", font=font(44), fill=ACC2)
    bullets(d, ["输入：inputs/ 下的一条视频", "输出：outputs/<样本>/pure（数据）+ work（过程与证据）",
                "工具：scripts/ 下 8 个脚本 + ref2va-sample skill"], 80, 360, 1700, 40)
    d.text((80, 960), "本视频由 docs/tutorial/make_tutorial.py 生成，示例画面均来自本次真实样本", font=font(28), fill=DIM)
    S.append((im, ["这个视频介绍我们现在使用的 Ref2VA 训练数据合成流程。",
                   "目标是从一条真实视频中，构造一条可以用于 MiniMax H3 参考生成训练的样本。"]))

    im, d = base("一条样本是什么", "任务定义")
    paste_fit(im, il["concept_refs"], (80, 220, 880, 740))
    d.text((905, 380), "+ H3 提示词", font=font(38), fill=ACC)
    arrow(d, 905, 460, 1100, 460)
    paste_fit(im, il["concept_tgt"], (1140, 220, 1860, 470))
    paste_fit(im, il["concept_tgt2"], (1140, 480, 1860, 730))
    bullets(d, ["目标必须从原视频精确切出，不调用模型生成",
                "参考图片、参考视频的源区间与目标不相交",
                "原片 oringin.mp4 和目标本身永远不作为条件输入"], 80, 790, 1760, 34, 8)
    S.append((im, ["每条样本的形式是：参考素材，加上 H3 格式的提示词，对应原视频里真实存在的一段目标音视频。",
                   "目标不是模型生成的，而是从输入视频里精确切出来的。",
                   "参考图片和参考视频必须来自目标区间之外，原片和目标本身永远不会作为条件输入。"]))

    im, d = base("整体流程：七步", "总览")
    steps = [("1 分析", "analyze_video.py"), ("2 选段", "人工判断"), ("3 边界核查", "dense_frames.py"),
             ("4 导出", "export_sample.py"), ("5 声音证据", "asr / audio_tags"), ("6 写提示词", "prompt.md"),
             ("7 组装验证发布", "build_and_verify.py")]
    xs = [80, 600, 1120, 1120, 600, 80, 600]
    ys = [260, 260, 260, 540, 540, 540, 800]
    for (t, s), x, y in zip(steps, xs, ys):
        manual = "人工" in s or "prompt" in s
        box(d, (x, y, x + 440, y + 180), t, outline=ACC if manual else ACC2, sub=s)
    arrow(d, 520, 350, 600, 350); arrow(d, 1040, 350, 1120, 350); arrow(d, 1340, 440, 1340, 540)
    arrow(d, 1120, 630, 1040, 630); arrow(d, 600, 630, 520, 630); arrow(d, 300, 720, 600, 860)
    d.text((1200, 820), "蓝框：脚本自动完成", font=font(32), fill=ACC2)
    d.text((1200, 880), "橙框：需要看画面后判断", font=font(32), fill=ACC)
    S.append((im, ["整个流程分为七步。",
                   "脚本负责可重复的机械工作，包括解码、检测、导出和验证；选段、确认切镜和写提示词，则需要实际看画面、看证据后再判断。",
                   "下面逐步说明。"]))

    im, d = base("分析视频", "第 1 步 · analyze_video.py")
    bullets(d, ["完整解码，记录分辨率、每帧真实时间戳、是否恒定帧率",
                "音轨参数、文件 SHA-256",
                "镜头切换初筛，每个镜头取中间帧拼成概览图",
                "全片 ASR（Whisper）转录"], 80, 220, 640, 34)
    paste_fit(im, il["overview"], (760, 220, 1860, 900))
    d.text((760, 920), "Cheetos 广告的镜头概览 shots_overview.jpg", font=font(28), fill=DIM)
    S.append((im, ["第一步，用 analyze video 检查视频。",
                   "它会完整解码，记录分辨率、每一帧的真实时间戳、是否恒定帧率、音轨参数和文件哈希。",
                   "然后做一次镜头切换初筛，把每个候选镜头的中间帧拼成概览图，同时对全片跑一遍语音识别。",
                   "右边是 Cheetos 广告的镜头概览。"]))

    im, d = base("先选目标，再找参考", "第 2 步 · 选段")
    bullets(d, ["目标：5–15 秒、语义完整的一段（一次对话、一个笑点、一组步骤）",
                "参考图片：人物 / 产品 / 场景身份",
                "参考视频：叙事结构、剪辑风格；启用声音后提供音色或配乐参考",
                "没有作用的参考一律删掉"], 80, 220, 760, 34)
    paste_fit(im, il["select"], (900, 200, 1860, 1000))
    S.append((im, ["第二步是选段。",
                   "先选目标：五到十五秒、语义完整的一段，比如一次完整的对话，或一个笑点。",
                   "再在目标区间之外找参考：图片提供人物、产品或场景的身份特征，参考视频提供叙事结构和剪辑风格，启用声音后还能提供音色和配乐参考。",
                   "没有作用的参考一律删掉。"]))

    im, d = base("逐帧核查边界", "第 3 步 · dense_frames.py")
    paste_fit(im, il["bounds"], (80, 200, 1300, 1000))
    bullets(d, ["切镜检测只是初筛", "叠化混合帧不能当起止帧", "灯光变化、字幕特效 ≠ 切镜",
                "真正的跳切要保留"], 1340, 260, 520, 34)
    S.append((im, ["第三步，用 dense frames 逐帧核查每一个边界，自动切镜检测只是初筛。",
                   "比如求助热线片里，镜头之间有单帧叠化，边界要避开这一帧混合画面。",
                   "Cheetos 片里，关灯和紫外灯亮起被误判成切镜，其实是同一个机位。",
                   "反过来，Doritos 片里的跳切是真正的切镜，需要保留。"]))

    im, d = base("防止目标内容泄漏到参考", "第 3 步 · 参考检查")
    paste_fit(im, il["leak"], (80, 200, 1300, 1000))
    bullets(d, ["回放、倒带重放目标镜头", "同一机位反复出现", "片尾旁白漏进目标", "参考图上的字幕、打字光标",
                "  自动近重复检测抓不全，必须人工看帧"], 1340, 260, 540, 32)
    S.append((im, ["选参考时，还要防止目标内容泄漏。",
                   "Doritos 片里有一段倒带回放，重放的正是目标镜头，所以参考视频改为从油炸镜头开始。",
                   "求助热线片里，最初的参考片段和目标用了同一个机位，也被替换掉了。",
                   "片尾旁白、字幕和打字光标同样要逐帧排除。自动近重复检测抓不全，必须人工看。"]))

    im, d = base("按配置精确导出", "第 4 步 · export_sample.py")
    cfg = ('{"source": "inputs/x.mp4",\n'
           ' "target": {"start_frame": 2674,\n            "end_frame_excl": 2992,\n            "num_frames": 243},\n'
           ' "videos": [{"name": "video_1",\n   "start_frame": 2992, "end_frame_excl": 3233,\n   "num_frames": 192, "include_audio": true}],\n'
           ' "images": [{"name": "picture_1",\n   "source_frame": 2470, "crop": null}]}')
    d.rounded_rectangle((80, 220, 900, 760), radius=16, fill=(28, 31, 42))
    d.multiline_text((110, 250), cfg, font=font(26, mono=True), fill=(200, 220, 255), spacing=10)
    rows = [("帧率", "24 FPS 恒定帧率"), ("帧数", "N = 17k + 5"), ("画面", "等比缩放到 1344×768，补边"),
            ("目标声音", "原片直接提取 → 32 kHz 双声道 16-bit WAV"), ("补齐", "只复制末帧 + 静音，绝不读入下一镜头"),
            ("记录", "provenance.json / export_log.json")]
    y = 230
    for k, v in rows:
        d.text((960, y), k, font=font(34), fill=ACC)
        d.text((1160, y), v, font=font(34), fill=FG)
        y += 90
    S.append((im, ["第四步，把选择写进 export config，运行 export sample。",
                   "它按真实时间戳精确裁切，转成每秒二十四帧的恒定帧率，帧数取十七 k 加五。",
                   "画面保持比例缩放到一三四四乘七六八，并补边。",
                   "目标音频直接从原片对应区间提取，重采样成三十二千赫双声道 WAV。",
                   "帧数不够时只能重复末帧、补静音，永远不读入下一个镜头；能裁掉静默尾巴时优先裁掉。"]))

    im, d = base("在导出结果上确认切镜", "第 4 步 · inspect_export.py")
    paste_fit(im, il["shots"], (80, 200, 1860, 760))
    bullets(d, ["在导出后的时间线上重新测量切镜，生成分镜图和切点前后对比图",
                "人工确认后写入 record_spec.json 的 confirmed_cut_times，提示词镜头时间以此为准"], 80, 800, 1760, 32, 8)
    S.append((im, ["导出之后运行 inspect export，在导出后的时间线上重新测量切镜，生成分镜图和每个切点前后的对比图。",
                   "人工确认后，把真实的切镜时间写进 record spec，提示词里的镜头时间就以这里为准。"]))

    im, d = base("收集声音证据", "第 5 步 · asr.py / audio_tags.py / audio_profile.py")
    bullets(d, ["asr.py：Whisper large-v3-turbo，词级时间戳，台词保留原语言",
                "audio_tags.py：CLAP 零样本标注（音乐、音效、环境声）",
                "audio_profile.py：电平、频谱平坦度、音高、节奏",
                "波形包络：确认台词起止、旁白是否漏进目标",
                "无法试听 → 证据不足的地方标记 needs_audio_review"], 80, 230, 1760, 38, 26)
    S.append((im, ["第五步收集声音证据。",
                   "asr 用 Whisper 给出带词级时间戳的转录；audio tags 用 CLAP 做零样本声音标注；audio profile 测量电平、频谱和节奏。",
                   "再结合波形包络，确认台词的起止，例如确认片尾旁白不会漏进目标。",
                   "由于没有试听能力，这些都只是证据，不能确定的地方要标记为待复核。"]))

    im, d = base("写 H3 六段式提示词", "第 6 步 · prompt.md")
    secs = [("subject_definitions", "主体定义：<Subject N> 与 <Picture / Video / Audio N> 映射"),
            ("summary", "摘要：[reference generation + audio reference] …"),
            ("retention_analysis", "保留分析：fully_preserved / reference 等固定标记"),
            ("detailed_description", "逐镜头描述：[Shot N] At MM:SS.mmm，对白 <d>[语言] 原文</d>"),
            ("overall_soundscape", "环境声与物理声音"),
            ("non_diegetic_music", "非剧情配乐，没有就写 N/A")]
    y = 230
    for k, v in secs:
        d.text((80, y), k + ":", font=font(32, mono=True), fill=ACC2)
        d.text((640, y), v, font=font(32), fill=FG)
        y += 100
    S.append((im, ["第六步写 H3 六段式提示词，顺序固定：主体定义、摘要、保留分析、逐镜头描述、整体声音，以及非剧情配乐。",
                   "标签要能对应到实际输入，镜头时间必须与导出视频一致，对白保留原语言，屏幕文字原样引用。"]))

    im, d = base("组装、验证、发布", "第 7 步 · build_and_verify.py")
    checks = ["JSONL 字段与类型、相对路径", "帧数 17k+5、24 FPS、1344×768", "WAV 采样率 / 声道 / 采样数精确",
              "MP4 音轨与 WAV 对齐（互相关）", "目标首尾帧、波形与源区间一致", "参考与目标源区间不相交",
              "近重复检测", "标签映射、六段顺序", "镜头时间 = 确认切镜（±1 帧）", "原片哈希、提示词副本一致"]
    for k, c in enumerate(checks):
        x, y = (80 if k < 5 else 900), 230 + (k % 5) * 80
        d.text((x, y), "√", font=font(36), fill=OK)
        d.text((x + 56, y), c, font=font(34), fill=FG)
    d.text((80, 680), "stage/ 全部通过 → 移到 pure/ → 再验证一次 → verification.json", font=font(36), fill=ACC)
    for k, (s, desc) in enumerate([("completed", "无已知待复核"), ("needs_review", "有声音 / 语义不确定"),
                                   ("skipped", "无有效参考或无音轨"), ("blocked", "输入或能力缺失")]):
        box(d, (80 + k * 450, 780, 500 + k * 450, 960), s, size=34, sub=desc)
    S.append((im, ["最后运行 build and verify。",
                   "它先在暂存目录组装 dataset JSON L，再执行五十多项检查：字段类型、路径、帧数、采样数、音画对齐、源区间不相交、近重复、标签映射和镜头时间等。",
                   "全部通过才发布到 pure 目录，并再验证一次。",
                   "结果状态分为四种：completed、needs review、skipped 和 blocked。"]))

    im, d = base("输出目录与本次结果", "交付")
    tree = ("outputs/<样本>/\n├── pure/\n│   ├── dataset.jsonl\n│   ├── oringin.mp4\n│   ├── inputs/  picture_N.png  video_N.mp4\n"
            "│   └── targets/ target.mp4  target_audio.wav\n└── work/\n    ├── selection.json   选段理由\n    ├── provenance.json  来源与变换\n"
            "    ├── prompt.md / record_spec.json\n    ├── verification.json\n    └── analysis/ dense/ asr/ 分镜图")
    d.multiline_text((80, 220), tree, font=font(28), fill=(200, 220, 255), spacing=12)
    res = [("求助热线（葡语）", "10.1 s", "needs_review"), ("Cheetos 黑光灯", "12.3 s", "needs_review"),
           ("Doritos 菜谱", "13.7 s", "needs_review"), ("HomeExchange 读信", "10.1 s", "completed")]
    y = 240
    for name, dur, st in res:
        d.text((1080, y), name, font=font(34), fill=FG)
        d.text((1460, y), dur, font=font(34), fill=DIM)
        d.text((1600, y), st, font=font(30), fill=OK if st == "completed" else ACC)
        y += 80
    d.text((1080, y + 20), "结构与媒体检查全部通过；needs_review 都是声音待试听", font=font(28), fill=DIM)
    S.append((im, ["每条样本输出 pure 和 work 两个目录。",
                   "pure 是数据本身；work 保存选段理由、来源记录、提示词副本、验证结果和全部过程材料。",
                   "这次四个视频各生成了一条样本，结构和媒体检查全部通过，其中三条因为声音判断需要人工试听，标记为待复核。"]))

    im, d = base("处理新视频", "复用")
    bullets(d, ["把视频放进 inputs/，按 .claude/skills/ref2va-sample 的步骤执行",
                "必须人工判断：选段、切镜确认、提示词、record_spec",
                "已知限制",
                "  非 MP4 输入还不能正确生成 oringin.mp4",
                "  无音轨视频不会自动标记 skipped",
                "  检测阈值基于这四个视频调整，新类型的片子要多看几眼"], 80, 240, 1760, 38, 22)
    S.append((im, ["处理新视频时，把文件放进 inputs，按 ref2va sample 这个 skill 的步骤执行即可。",
                   "需要注意：目前还不支持非 MP4 输入，也不会自动跳过没有音轨的视频，检测阈值也是基于这四个视频调整的，遇到新类型的片子请多看几眼。",
                   "谢谢观看。"]))
    return S


def caption(im: Image.Image, text: str) -> Image.Image:
    im = im.copy()
    d = ImageDraw.Draw(im, "RGBA")
    f = font(34)
    lines = wrap(d, text, f, 1700)
    h = len(lines) * 48 + 30
    d.rectangle([0, H - h - 10, W, H], fill=(0, 0, 0, 200))
    y = H - h + 5
    for ln in lines:
        d.text(((W - d.textlength(ln, font=f)) / 2, y), ln, font=f, fill=(255, 255, 255))
        y += 48
    return im


SPOKEN = {"Ref2VA": "Ref to V A", "JSONL": "JSON L", "CLAP": "clap", "ASR": "A S R"}


def main() -> None:
    import math

    import numpy as np
    import soundfile as sf

    ff = ffmpeg_exe()
    il = illustrations()
    S = slides(il)
    sr = 48000
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        concat, audio = [], []
        n = 0
        for si, (im, sentences) in enumerate(S):
            for k, sent in enumerate(sentences):
                spoken = sent
                for a, b in SPOKEN.items():
                    spoken = spoken.replace(a, b)
                aiff = tdp / f"{n:03d}.aiff"
                wav = tdp / f"{n:03d}.wav"
                subprocess.run(["say", "-v", VOICE, "-r", RATE, "-o", str(aiff), spoken], check=True)
                subprocess.run([ff, "-v", "error", "-y", "-i", str(aiff), "-ar", str(sr), "-ac", "1", str(wav)], check=True)
                x, _ = sf.read(wav, dtype="float32")
                tail = 0.9 if k == len(sentences) - 1 else 0.35
                frames = math.ceil((len(x) / sr + tail) * 24)  # whole video frames -> exact A/V alignment
                seg = np.zeros(round(frames / 24 * sr), np.float32)
                seg[:len(x)] = x
                audio.append(seg)
                png = tdp / f"{n:03d}.png"
                caption(im, sent).save(png)
                concat.append(f"file '{png}'\nduration {frames / 24:.6f}\n")
                n += 1
            print(f"slide {si + 1}/{len(S)} done")
        concat.append(f"file '{tdp / f'{n - 1:03d}.png'}'\n")  # concat demuxer needs the last file repeated
        (tdp / "list.txt").write_text("".join(concat))
        sf.write(tdp / "narration.wav", np.concatenate(audio), sr, subtype="PCM_16")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([ff, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(tdp / "list.txt"),
                        "-i", str(tdp / "narration.wav"), "-map", "0:v", "-map", "1:a",
                        "-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-tune", "stillimage",
                        "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-shortest", "-movflags", "+faststart", str(OUT)],
                       check=True)
        prev = OUT.parent / "slides"
        if prev.exists():
            shutil.rmtree(prev)
        prev.mkdir()
        for si, (im, _) in enumerate(S):
            im.save(prev / f"slide_{si + 1:02d}.png")
    print(OUT)


if __name__ == "__main__":
    main()
