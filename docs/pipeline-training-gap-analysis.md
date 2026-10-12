# Ref2VA 数据 pipeline 接入训练的差距分析

本文对比 `outputs/*/pure/` 中现有 5 条样本与 `format/` 下三个训练框架的数据格式，说明 pipeline 需要怎样修改，产出的数据才能真正送进训练。

分析依据：

- 本地产出：`outputs/*/pure/dataset.jsonl`、`inputs/`、`targets/`，以及 `outputs/*/work/` 中的 `provenance.json`、`verification.json`。媒体参数用 workspace `.venv` 中的 FFmpeg 实际探测。
- 训练格式：[format/DiffSynth-Studio](../format/DiffSynth-Studio/README.md)、[format/MiniMax-H3-Trainer](../format/MiniMax-H3-Trainer/README.md)、[format/verl-omni](../format/verl-omni/README.md)，以及这些 README 中引用的上游固定版本源码结论。
- 本次只做了静态分析，没有运行任何框架的预处理或训练。

---

## 1. 结论

现有 `pure/` 是**逐样本的本地中间格式**，单条记录本身质量较高：六段式提示词、参考与目标不交叠、24 FPS（frames per second，每秒帧数）、`17k+5` 帧数、32 kHz 双声道都已满足。但它离“可训练”还差三类问题：

| 等级 | 问题 | 影响 |
| --- | --- | --- |
| **阻塞** | 目标长度和分辨率与训练配置不一致。三个框架的样例配置都是 832×480×124 帧（5.17 秒）；现有目标为 1344×768、243–328 帧（10.1–13.7 秒），且 5 条提示词都有落在 5.17 秒之后的切镜时间 | 目标被截断或缩放后，提示词与监督信号对不上 |
| **阻塞** | 没有面向训练框架的打包层：没有汇总清单、没有框架字段映射、没有按参考布局分组、没有按 `split_group` 划分训练/验证 | 任何框架都无法直接读取 `pure/` |
| **阻塞** | 质量标记没有门控：5 条中 4 条是 `needs_review`，而三个框架都不读取 `needs_audio_review` / `uncertain_audio_spans_seconds` | 待复核的对白和声音描述会被当作真值训练 |
| 需修 | 1344×768 画布中有上下各 6 像素黑边；参考视频音轨开关的语义没有映射到各框架；`oringin.mp4` 混在纯数据目录里；过程记录的 schema（模式）不统一 | 学到黑边、误用参考音轨、潜在泄漏与溯源断链 |

修改思路是：**保留 `pure/` 作为唯一事实来源（source of truth），在其上方加一层“训练配置驱动的导出”和一层“框架打包器”**，而不是把 `pure/` 直接改成某个框架的格式。

---

## 2. 现有产出概况

| 样本 | 状态 | 参考布局 | 目标帧数 / 时长 | 最后一个切镜时间 | 待复核区间 |
| --- | --- | --- | --- | --- | --- |
| `a_theatre_of_memories_001` | needs_review | 2 图 + 1 带音轨视频 | 243 / 10.125 s | 00:07.542 | `[6.2, 8.5]`、`[7.542, 10.125]`（两段重叠，未合并） |
| `embed_cheetos_9181878e_001` | needs_review | 2 图 + 1 带音轨视频 | 294 / 12.25 s | 00:10.208 | `[10.6, 11.0]` |
| `embed_doritos_b36ed8ad_001` | needs_review | 1 图 + 1 带音轨视频 | 328 / 13.667 s | 00:12.000 | 5 段 |
| `helpline_180_fde007ab_001` | needs_review | 1 图 + 1 带音轨视频 | 243 / 10.125 s | 00:08.042 | 2 段，覆盖 3.04 s 之后约一半时长 |
| `homeexchange_4cded0ce_001` | completed | 1 图 + 1 带音轨视频 | 243 / 10.125 s | 00:07.083 | 无 |

共同媒体参数（实测）：

- 目标 `target.mp4`：H.264 1344×768，24 FPS，内含 AAC（Advanced Audio Coding）32 kHz 立体声；`target_audio.wav`：16-bit PCM（pulse-code modulation）32 kHz 立体声。
- 参考视频 `video_1.mp4`：1344×768，24 FPS，均带 AAC 音轨，且记录中都是 `include_audio: true`；时长 6.58–10.125 秒。
- 参考图片：均为 1280×720 原始整帧 PNG。
- 画面处理：1280×720 → 1344×756 等比缩放，上下各补 6 像素黑边。
- 提示词长度：715–1042 个英文单词。

---

## 3. 三个框架的格式要点

| 维度 | DiffSynth-Studio（监督微调） | MiniMax-H3-Trainer（社区，监督微调） | verl-omni（FlowGRPO / DiffusionNFT 奖励训练） |
| --- | --- | --- | --- |
| 清单 | 一个 JSON 数组 `metadata.json` | `.json` 数组 / `.jsonl` / `.csv`，推荐 JSON 数组 | `train.jsonl` + `test.jsonl`，再转 Parquet |
| 是否需要目标 | 需要：`video` + `input_audio` | 需要：`video`，可选 `audio` 覆盖目标音频 | **不需要**，模型在线生成后由奖励函数评分 |
| 提示词字段 | `prompt` | `caption`（别名 `prompt`、`text`） | `prompt`；转换器自动在前面加 `<image>…<video>…<audio>` |
| 参考写法 | `references: [{type, image/video/audio}]`；带音轨视频必须写 `type: video_audio` 并同时给 `video` 和 `audio` | `reference_image` / `reference_video` / `reference_audio` 三个数组；`reference_video` 会自动提取视频内音轨 | `images` / `videos`（可带 `start_time_seconds`）/ `audios` |
| 目标音频 | `input_audio` 必须是带音轨的视频文件（算子先读视频帧数再截取音频），不能直接给 WAV | `audio` 可指向独立 WAV | 无 |
| 相对路径基准 | `--dataset_base_path` | **运行时工作目录**，不是清单所在目录 | `--input_dir`；Parquet 内写绝对路径，换机器需重转 |
| 尺寸 / 帧数 | 全局 `--height 480 --width 832 --num_frames 124`，帧数须 `17n+5` | 全局 `--resolution-bucket 832x480x124` | 由启动器 `NUM_FRAMES` 等决定，不在清单里 |
| 参考布局约束 | 每条可不同 | 配置里的 reference 条目与缓存参考**按顺序严格一一匹配**，同一缓存根目录只能放同一种布局 | 每条可不同；需保证 `MAX_PROMPT_EMBEDS` 够用 |
| 训练/验证划分 | 无内置验证集 | `val_split_every` 按序号抽取，不感知来源分组 | 显式的 `train.jsonl` / `test.jsonl` |
| 真正进训练的文件 | 预处理生成的 `cache/*.pth` | 预处理生成的 `.precomputed/` 下的 safetensors 缓存与 `index.json` | `train.parquet` / `test.parquet` |

三者共同点：参考的编号顺序都是“图片 → 视频 → 独立音频”，与本地 `<Picture N>` / `<Video N>` / `<Audio N>` 的编号规则一致；参考数量上限也与 H3 规格一致（图 ≤ 9、视频 ≤ 3、音频 ≤ 3、合计 ≤ 12）。这部分现有数据不需要改。

---

## 4. 必须修改（阻塞项）

### 4.1 目标长度与分辨率要由训练配置决定

**现状。** pipeline 只保证 `17k+5` 帧、5–15 秒、1344×768。三个框架的样例配置都使用单一的 832×480×124：

- DiffSynth 用全局 `--num_frames`，目标音频再按读取到的视频帧数截取。
- H3-Trainer 用全局 `--resolution-bucket`。
- verl-omni 的 `NUM_FRAMES` 决定在线生成的长度，奖励函数又拿完整提示词当 `ground_truth`。

现有 5 条提示词描述的都是 10–13.7 秒的内容，切镜时间最晚到 `00:12.000`。按 124 帧训练时，目标只剩前 5.17 秒，而提示词仍在描述后面的镜头、对白和音乐，监督信号和文本条件不一致。具体是截断还是报错，要在上游代码里再确认一次，但两种结果都不能直接用。

**修改。**

1. 在 `PROMPT.md` §1 增加训练配置参数，例如：

   ```text
   TRAIN_PROFILE=h3_832x480x124   # 或 h3_1344x768x243 等
   ```

   每个 profile 固定 `width`、`height`、`num_frames`。目标帧数从“任意合法 `17k+5`”改为**正好等于 profile 的帧数**。如果要多档时长，就按帧数分桶，每桶单独打包、单独预处理。
2. `scripts/r2v_common.py` 中的画布和帧数常量改为从 profile 读取；`build_and_verify.py` 增加检查：`target.num_frames == profile.num_frames`，并且提示词中所有 `At MM:SS.mmm` 都小于 `num_frames / 24`。
3. **现有 5 条不能靠截断复用。** 如果训练用 124 帧，需要重新选 5.17 秒的目标区间并重写提示词。如果确实要在 243 帧上训练，3 条 243 帧样本可以留在 243 帧桶，cheetos（294）和 doritos（328）要重做，或各自单独成桶。同时要评估 1344×768×243 的显存与 `max_seq_tokens`。
4. `PROMPT.md` §4 的表格中，“默认 1344×768”改成“按 `TRAIN_PROFILE`”，并在 `reference/media-profile.md` 记录各 profile 的来源依据。

### 4.2 增加框架打包层

**现状。** `pure/` 每个样本单独一个目录、一个单行 `dataset.jsonl`，路径相对于该目录。三个框架都要求**一个汇总清单 + 统一根目录**，字段名也都不同。

**修改。** 新增 `scripts/pack_for_training.py`。它读取若干 `outputs/*/pure/dataset.jsonl`，按目标框架写出一个独立的训练根目录，并且不改动 `pure/`：

```text
training_packs/<framework>/<pack_name>/
├── media/<sample_id>/...          # 从 pure/ 硬链接或复制，不含 oringin.mp4
├── metadata.json | samples/*.json | train.jsonl + test.jsonl
├── configs/*.yaml                 # H3-Trainer：每种参考布局一份
├── prepare.sh
└── pack_manifest.json             # 样本列表、来源哈希、过滤原因、split 结果
```

以 `homeexchange_4cded0ce_001` 为例，三种映射如下（路径相对各自根目录）。

**DiffSynth-Studio** `metadata.json` 中的一项：

```json
{
  "video": "media/homeexchange_4cded0ce_001/targets/target.mp4",
  "prompt": "subject_definitions:\n...",
  "input_audio": "media/homeexchange_4cded0ce_001/targets/target.mp4",
  "references": [
    {"type": "image", "image": "media/homeexchange_4cded0ce_001/inputs/picture_1.png"},
    {"type": "video_audio",
     "video": "media/homeexchange_4cded0ce_001/inputs/video_1.mp4",
     "audio": "media/homeexchange_4cded0ce_001/inputs/video_1.mp4"}
  ]
}
```

- `include_audio: true` 映射为 `type: video_audio`，`video` 和 `audio` 都要写，读取器不会自动补齐。`include_audio: false` 映射为 `type: video`。独立参考音频映射为 `type: audio`。
- `input_audio` 必须指向带音轨的 `target.mp4`，不能用 `target_audio.wav`（见 §5.1）。
- 启动参数必须同时有 `--data_file_keys "video,input_audio,references"` 和 `--extra_inputs "input_audio,references"`。

**MiniMax-H3-Trainer** 清单中的一项：

```json
{
  "id": "homeexchange_4cded0ce_ref2va_001",
  "video": "media/homeexchange_4cded0ce_001/targets/target.mp4",
  "audio": "media/homeexchange_4cded0ce_001/targets/target_audio.wav",
  "caption": "subject_definitions:\n...",
  "reference_image": ["media/homeexchange_4cded0ce_001/inputs/picture_1.png"],
  "reference_video": ["media/homeexchange_4cded0ce_001/inputs/video_1.mp4"]
}
```

- `id` 会成为缓存文件名，必须全局唯一。现有 `id` 满足。
- `audio` 可以直接用无损 WAV，这一点比 DiffSynth 更贴合现有产出。
- 路径相对于**运行时工作目录**，所以 `prepare.sh` 要先 `cd` 到打包根目录。

**verl-omni** `train.jsonl` 中的一行：

```json
{"prompt": "subject_definitions:\n...", "images": ["media/homeexchange_4cded0ce_001/inputs/picture_1.png"], "videos": [{"path": "media/homeexchange_4cded0ce_001/inputs/video_1.mp4", "start_time_seconds": 0.0}]}
```

- 目标媒体不进入清单；`prompt` 原文会成为奖励函数的 `ground_truth`。
- 不要自己在 `prompt` 前加 `<image>` 等占位符，转换器会加。
- 同一模态不要同时写单数和复数字段。
- 上传到训练机后要在那里重新运行 `prepare_ref2va_data.py`，因为 Parquet 里存的是绝对路径。

### 4.3 H3-Trainer：按参考布局分组

H3-Trainer 用 `zip(..., strict=True)` 把配置中的 `conditions` 与缓存中的参考逐个对齐。现有 5 条有两种布局：

| 布局 | 样本 | 需要的 `training_strategy.video.conditions` |
| --- | --- | --- |
| `image×2 + video×1`（带音轨） | theatre、cheetos | `image`、`image`、`video` |
| `image×1 + video×1`（带音轨） | doritos、helpline、homeexchange | `image`、`video` |

打包器需要按“图片数 / 视频数 / 音频数 / 视频是否带音轨”生成布局键，每个布局键对应一份清单、一个 `.precomputed/<layout>` 和一份 YAML（YAML Ain't Markup Language）配置。`probability` 保持 `1.0`；上游说明参考 dropout（随机丢弃）尚未切换到无参考的文本嵌入。

pipeline 侧还可以把“参考布局”作为选材时的软约束，让同一批数据尽量落在少数几种布局上，否则每个桶的样本会很少。

### 4.4 用 `split_group` 真正划分训练集和验证集

`split_group` 已经写了，但没有任何环节使用它：

- verl-omni 需要显式的 `train.jsonl` / `test.jsonl`。
- H3-Trainer 的 `val_split_every` 按序号抽样，会把同源片段拆到训练和验证两边。
- DiffSynth 没有内置验证集。

**修改。** 打包器按 `split_group` 做确定性划分，例如对 `split_group` 取哈希再按比例分桶。同一组整组进入同一侧，划分结果写进 `pack_manifest.json`。H3-Trainer 设 `val_split_every: 0`，验证集单独打一个包或写进 `validation.samples`。

另外，同一作品的不同上传版本目前按文件哈希分组（`src:<文件名>:sha256-…`），重复来源不会被合并。pipeline 应在 `analyze_video.py` 阶段加入感知哈希或音频指纹去重，把重复来源归到同一个 `split_group`。

### 4.5 质量门控：`needs_review` 不能直接进训练

三个框架都不读取 `needs_audio_review` 和 `uncertain_audio_spans_seconds`，也没有按时间区间屏蔽 loss（损失）的机制。现在 5 条中 4 条处于 `needs_review`，helpline 的待复核区间约占目标时长的一半。

**修改。**

1. 打包器读取 `work/verification.json` 的 `status`，默认只收 `completed`，用 `--allow-needs-review` 显式放开。被过滤的样本和原因写进 `pack_manifest.json`。
2. pipeline 增加人工复核回写流程：复核后更新提示词和 `verification.json`，状态改为 `completed`，并记录复核人和时间。
3. `build_and_verify.py` 对 `uncertain_audio_spans_seconds` 做排序、合并重叠区间（theatre 现有 `[6.2, 8.5]` 和 `[7.542, 10.125]` 重叠），并增加阈值：待复核总时长超过目标时长的一定比例时，直接判为不可训练。

---

## 5. 需要修改（非阻塞，但会影响训练效果或可维护性）

### 5.1 目标音频的两种载体

- DiffSynth 只能从视频容器读目标音频，实际训练会用 `target.mp4` 里的 AAC，`target_audio.wav` 用不上。现有 AAC 码率约 160–200 kb/s，通常可以接受。如果要求无损，可以额外导出一份 PCM 音轨的 MOV/MKV 作为 `input_audio`，但要先确认上游算子能解码该容器。
- H3-Trainer 可以用 WAV，应优先用 WAV。
- pipeline 的验证已经检查了 MP4 与 WAV 的同步。建议把“MP4 内音轨解码后与 WAV 的相关性 / 偏移”也写进 `verification.json`，因为 DiffSynth 实际训练的是 MP4 那一份。

### 5.2 去掉黑边，按训练画布导出

现在 1344×768 画布中有 1344×756 的有效画面和上下 6 像素黑边。训练时还会再缩放或裁切到 832×480（宽高比约 1.733，与 16:9 的 1.778 不同）。最终可能保留一部分黑边，让模型学到边框。

**修改。** 在 `TRAIN_PROFILE` 中加入 `fit_mode`，可选 `pad` 或 `center_crop`。默认改为按训练画布等比缩放后居中裁切，裁掉量写进 `provenance.json`。如果裁切会切掉主体，改选片段或改用其他画幅的 profile，不要补边。参考视频也按同一规则处理；参考图片保留原始整帧即可，因为各框架都会自行处理图片。

### 5.3 参考视频音轨开关要落到媒体文件上

`include_audio` 目前只是记录里的标记。各框架的行为不同：

- H3-Trainer 的 `reference_video` 只要文件里有音轨就会提取。
- DiffSynth 靠 `type` 区分。
- verl-omni 只传路径，由生成后端决定。

**修改。** `include_audio: false` 时，`export_sample.py` 直接导出无音轨的 `video_N.mp4`，让文件本身和标记一致，三个框架就不会有歧义。现有 5 条都是 `true`，暂不受影响，但应补上这条规则和对应检查：`include_audio == 文件是否有音轨`。

### 5.4 `oringin.mp4` 不进入训练包

`oringin.mp4` 包含目标区间，单个 4–26 MB。PROMPT.md 已规定它只用于溯源，但它和训练媒体放在同一个 `pure/` 目录里。任何“扫描目录”式的加载器或手工打包都可能误带它，造成目标泄漏。

**修改。** 打包器使用白名单，只复制 `dataset.jsonl` 中 `references[].path` 和 `target.*` 引用的文件。长期可以考虑把原片移到 `work/`，并保留哈希引用。目录结构的变化需要同步更新 `PROMPT.md` §6 和 `examples/`。

### 5.5 统一过程记录的 schema，修复溯源

- `a_theatre_of_memories_001` 的 `provenance.json` 字段结构与其他 4 条不同（例如 `source_fps` 与 `source_fps_measured`），`verification.json` 的顶层键也不同，而且缺少 `record_spec.json`。
- 它记录的来源是 `inputs/embed-c4ff36186d837bc5.mp4`，这个文件已经不在 `inputs/` 中，溯源链断了。

**修改。** 为 `dataset.jsonl`、`provenance.json`、`verification.json` 定义 JSON Schema 并加 `schema_version` 字段，由 `build_and_verify.py` 强制校验。打包器拒收 schema 不合格或来源文件哈希无法复核的样本。theatre 样本要么补回源视频并按新 schema 重建，要么在打包时排除。

### 5.6 提示词长度与文本编码器上限

现有提示词有 715–1042 个英文单词。H3 指南只要求 `detailed_description` 写 350–500 词，六段合计会明显更长。需要在每个框架的文本编码器和 `processor` 上实测 token（词元）数，确认没有截断。H3-Trainer 的 `max_seq_tokens: 70000` 是总序列上限，不等于文本上限。如果会截断，`non_diegetic_music`、`overall_soundscape` 这些靠后的段落会最先丢失。建议在打包器里调用对应 tokenizer（分词器）统计并报警。

### 5.7 verl-omni 的特殊性

verl-omni 不用目标媒体，pipeline 中最昂贵的目标导出和对齐对它没有直接价值。它真正需要的是：

- 提示词描述的时长与 `NUM_FRAMES` 一致（同 §4.1）。
- 奖励模型能理解六段式提示词，否则 `ground_truth` 用原文会让奖励信号变弱。必要时由打包器另外生成一份简短的 `ground_truth` 版本，但这需要修改上游转换器或奖励配置，属于训练侧的决策。
- 样本量足够：上游默认批大小可能大于现有 5 条。

---

## 6. 修改后的整体流程

```text
inputs/*.mp4
   │  PROMPT.md + TRAIN_PROFILE（新增）
   ▼
outputs/<sample_id>/pure/      ← 唯一事实来源，帧数 / 画布由 profile 决定
outputs/<sample_id>/work/      ← schema 化的 provenance / verification
   │  scripts/pack_for_training.py（新增）
   │    1. 质量门控（status、待复核比例、schema、来源哈希）
   │    2. 按 split_group 划分 train / val
   │    3. 按参考布局分组（H3-Trainer）
   │    4. 字段映射 + 白名单复制媒体（不含 oringin.mp4）
   │    5. tokenizer 长度检查
   ▼
training_packs/
├── diffsynth/<pack>/metadata.json + prepare.sh
├── h3_trainer/<pack>/samples/<layout>.json + configs/<layout>.yaml + prepare.sh
└── verl_omni/<pack>/train.jsonl + test.jsonl  →（训练机上）parquet/
   │  各框架官方预处理
   ▼
.pth / .precomputed safetensors / .parquet  ← 真正进入训练的文件
```

---

## 7. 修改清单

| 序号 | 修改位置 | 内容 | 优先级 |
| --- | --- | --- | --- |
| 1 | `PROMPT.md` §1、§4；`reference/media-profile.md` | 新增 `TRAIN_PROFILE`；目标帧数与画布改由 profile 决定 | P0 |
| 2 | `scripts/r2v_common.py`、`export_sample.py` | 常量改为读取 profile；新增 `fit_mode`，默认裁切不补边；`include_audio: false` 时导出无声参考视频 | P0 |
| 3 | `scripts/build_and_verify.py` | 检查帧数等于 profile、切镜时间小于目标时长、待复核区间合并与比例阈值、`include_audio` 与文件一致、JSON Schema 校验 | P0 |
| 4 | 新增 `scripts/pack_for_training.py` | 三个框架的打包器：门控、划分、布局分组、字段映射、白名单复制、tokenizer 检查 | P0 |
| 5 | 现有 `outputs/` | 按选定 profile 重做目标与提示词；完成 4 条 `needs_review` 的人工复核；修复或排除 theatre 样本 | P0 |
| 6 | `scripts/analyze_video.py` | 跨视频重复来源检测，归并 `split_group` | P1 |
| 7 | `PROMPT.md` §6、`examples/` | 评估是否把 `oringin.mp4` 移到 `work/` | P2 |

---

## 8. 需要在上游代码或训练环境中确认的事项

本文没有运行任何框架，以下几点只能由实际预处理确认：

1. DiffSynth 在 `--num_frames` 小于视频帧数时是截取前段、均匀抽帧，还是报错；H3-Trainer 的 `--resolution-bucket` 对更长视频的处理方式也要确认。结论影响 §4.1 中“多档时长分桶”的具体实现。
2. 两个监督微调框架在 832×480 下对 16:9 素材是缩放后裁切还是拉伸，决定 §5.2 的裁切策略。
3. 各文本编码器的最大 token 长度（§5.6）。
4. 在 DiffSynth 中，`video_audio` 参考的音频与提示词中的 `<Audio 1>` 标签是否按出现顺序对应。H3-Trainer 和 verl-omni 也需确认带音轨视频的音频在条件序列中的位置，与提示词的标签编号是否一致。
5. 预处理后检查 H3-Trainer `index.json` 的 `num_trainable` 和 `has_audio`，并用 `--decode 1` 的回放核对音画对齐。DiffSynth 检查 `.pth` 的数量是否等于清单条数。
