# scripts 脚本说明与 PROMPT.md 对应关系

本文按当前源码说明 `scripts/` 下 9 个 Python 文件的用途、输入输出、调用顺序及其承担的提示词要求。仓库实际文件名是 [PROMPT.md](../PROMPT.md)，没有 `PROMPTS.md`；下文均指这个文件。

`PROMPT.md` 规定完整的数据构造任务；脚本负责抽帧、声音分析、媒体导出、数据组装与程序检查。选哪段视频、参考素材是否有效、画面和声音应如何描述，仍由执行任务的 agent 根据实际素材判断。脚本不会读取 `PROMPT.md` 后自动完成全部任务，也不会调用 H3 生成视频或训练模型。

## 1. 与提示词各章节的对应关系

| PROMPT.md 章节 | 对应脚本 | 脚本提供的能力 | 仍需 agent 完成的工作 |
| --- | --- | --- | --- |
| §1 本次参数 | 各入口脚本的参数、`export_config.json` | 接收源视频、过程目录、暂存目录及导出配置 | 解析用户覆盖参数，确定样本编号与目录；多样本逐条组织 |
| §2 先读取依据，区分两种格式 | 没有专门脚本；`r2v_common.py` 保存部分本地媒体常量，`build_and_verify.py` 实现本地记录结构 | 固化部分媒体和数据格式约定 | 阅读样例、H3 指南和媒体规格，区分本地数据格式与模型提示词格式 |
| §3.1 检查输入与内容 | `analyze_video.py`、`dense_frames.py`；声音按需使用 `asr.py`、`audio_tags.py`、`audio_profile.py` | 解码、参数与时间戳分析、候选切镜、预览图、转录及声音特征 | 实际查看画面、核对声音，判断镜头与动作内容 |
| §3.2 先选目标，再找参考 | 上述分析工具；`export_sample.py` 接收最终选择 | 提供选段证据，将选择落实为媒体 | 决定目标和参考，说明每份参考的用途，编写选择与导出配置 |
| §3.3 检查参考与目标的关系 | `build_and_verify.py` | 检查记录中的来源区间、参考路径、抽样近重复画面 | 判断身份、动作、叙事和声音关系；归并同源作品的 `split_group` |
| §4 导出媒体并对齐时间 | `export_sample.py`、`r2v_common.py`、`inspect_export.py` | 精确裁切、规范化、记录溯源，在导出目标上重新检测切镜 | 选择自然边界与合法帧数，确认裁切、补齐和切镜是否合理 |
| §5 编写与目标一致的 H3 提示词 | `inspect_export.py`、三个声音工具提供证据；`build_and_verify.py` 检查格式 | 提供目标相对时间和声音线索，检查六段结构、标签、对白及时间格式 | 编写完整英文提示词，核实语义、对白、参考关系与待复核区间 |
| §6 输出目录与数据记录 | `export_sample.py`、`build_and_verify.py` | 生成媒体、溯源记录、单行数据记录和验证报告 | 准备 `selection.json`、`prompt.md`、`record_spec.json` 等材料 |
| §7 验收后再报告完成 | `build_and_verify.py`，结合预览和声音证据 | 暂存检查、发布、发布后复查，汇总手工检查状态 | 补齐语义核验、处理异常或跳过状态，用中文报告结果 |

## 2. 运行约定与环境

命令从 **workspace 根目录**执行。多数入口的相对路径按当前工作目录解析；`export_config.json` 内的 `source` 则由脚本按 workspace 根目录解析。统一从根目录运行可以避免二者不一致。

| 环境 | 使用脚本 | 主要依赖 |
| --- | --- | --- |
| `.venv/bin/python` | 通用模块、视频分析、抽帧、导出、导出检查、构建验证 | `numpy`、`Pillow`、`soundfile`、`imageio-ffmpeg` |
| `.venv-asr/bin/python` | `asr.py` | `mlx-whisper`、`numpy`，默认模型为 `mlx-community/whisper-large-v3-turbo` |
| `.venv-asr/bin/python` | `audio_tags.py` | `torch`、`transformers`、`numpy`，模型为 `laion/clap-htsat-unfused` |
| `.venv-asr/bin/python` | `audio_profile.py` | `librosa`、`numpy` |

这里的环境名是现有脚本的使用约定，不表示脚本会自动安装依赖或模型。模型未缓存时，模型加载可能需要下载。`audio_tags.py` 与 `audio_profile.py` 会直接寻找本 workspace `.venv` 中由 `imageio-ffmpeg` 提供的 FFmpeg；`asr.py` 优先从自身环境获取，失败后再查找 `.venv`。

下文命令中的 `<...>` 都是需要替换的参数，不可原样执行。`WORK_DIR` 通常为 `outputs/<sample_id>/work`，`OUTPUT_DIR` 为 `outputs/<sample_id>/pure`，`STAGE_DIR` 可用专门的 `outputs/<sample_id>/work/stage`。

时间和帧号有三种口径，使用时需要区分：

- 源帧号从 0 开始；`start_frame` 包含起点，`end_frame_excl` 不包含终点。
- 源时间使用解码得到的 PTS（presentation timestamp，显示时间戳）；可变帧率源视频不能统一用“帧号 ÷ 平均帧率”换算。
- 导出视频为 24 FPS（frames per second，每秒帧数）的 CFR（constant frame rate，恒定帧率）视频；提示词中的目标切镜时间从导出目标的 0 秒开始，以“导出帧号 ÷ 24”计算。

## 3. 逐个脚本说明

### 3.1 r2v_common.py：公共媒体处理与配置

[查看源码](r2v_common.py)。这是供其他脚本导入的公共模块，没有独立命令行任务。

它集中提供：

- 本地媒体常量：24 FPS、32000 Hz、双声道、目标 5–15 秒、参考音视频 2–15 秒，以及画布面积预算。
- 合法帧数判断与上下取整：`N = 17 × k + 5`。这些函数不自动保证片段时长或语义完整。
- FFmpeg 路径获取、命令执行、SHA-256（Secure Hash Algorithm 256-bit）校验值计算、JSON（JavaScript Object Notation）读写。
- 媒体头信息解析、逐帧 PTS 解码、音频解码、指定帧提取与拼图。
- 相邻画面的像素和直方图差异、候选切镜检测、等比缩放和补边尺寸计算。

**对应要求：** §3.1 的媒体检查、§4 的本地规范化规则，以及 §7 的检查基础。它不会动态解析规格文档；更换媒体配置时，需要同步核对这些常量及调用代码。规格依据见 [media-profile.md](../reference/media-profile.md)。

### 3.2 analyze_video.py：全片分析与概览

[查看源码](analyze_video.py)。用于开始处理一个视频时建立全片概览，帮助寻找候选目标和参考。

```bash
.venv/bin/python scripts/analyze_video.py <INPUT_VIDEO> <WORK_DIR>
```

**输入：** 原视频路径、过程目录。

**输出：**

| WORK_DIR 下的文件 | 内容 |
| --- | --- |
| `analysis/analysis.json` | 输入哈希、媒体头参数、解码帧数、PTS 间隔统计、帧率模式初判、音频参数和每 250 毫秒的音量统计、候选镜头及切镜分数 |
| `analysis/shots_overview.jpg` | 每个自动检测镜头的中间帧，带镜头号、源帧号和源时间 |
| `analysis/uniform_overview.jpg` | 全片均匀抽取最多 48 帧的概览 |

**对应要求：** §3.1“检查输入与内容”，为 §3.2“先选目标，再找参考”提供线索。

**边界：** 它不理解剧情、不自动选出最佳目标。切镜检测是初筛；音量与左右声道差异也不能证明具体声音来源。当前流程会尝试解码音轨，无音轨时可能直接报错，不会自行写出 `skipped` 状态。

### 3.3 dense_frames.py：候选区间密集抽帧

[查看源码](dense_frames.py)。用于查看动作过程、检查切镜前后帧、寻找清晰参考图片。

```bash
.venv/bin/python scripts/dense_frames.py \
  <INPUT_VIDEO> <start_frame> <end_frame_excl> <step> <out.jpg> [cols] [tile_w]
```

`step` 是抽帧步长，单位为**源帧**，不是秒；例如 `step=1` 表示逐帧抽取。`cols` 默认 6 列，`tile_w` 默认每格宽 320 像素。输出文件建议放在 `WORK_DIR`。

**输出：** 一张带零起始源帧号和实际 PTS 的图片拼图。

**对应要求：** §3.1 的候选区间密集查看，§3.2 的参考图片和自然边界选择，以及 §3.3 的视觉关系核对。

**边界：** 它只生成供查看的图，不保存选择理由，也不导出最终参考素材。

### 3.4 asr.py：对白和歌词的转录证据

[查看源码](asr.py)。ASR（automatic speech recognition，自动语音识别）工具，使用 Whisper 生成分段和逐词转录。

```bash
.venv-asr/bin/python scripts/asr.py <media> <out.json> \
  [--start S] [--end E] [--lang en] [--model MODEL]
```

`media` 可以是原片、导出目标或参考媒体；`S`、`E` 是该输入媒体上的开始和结束秒数。脚本将所选声音转换为 16000 Hz 单声道供识别，不会改变交付媒体的声音规格。

**输出：** JSON 转录文件，包含输入路径、裁切参数、模型、语言、分段文本、词级时间及模型置信信息；终端也输出分段摘要。

**对应要求：** §3.1 的声音核对；§5 的原语言对白、`<d>` 标签内容、截断和听不清区间判断；§6 的转录证据保存。

**时间口径：** 使用裁切参数后，转录时间相对于裁出片段的起点。转录原片局部时，不能直接把它当成目标相对时间。需要目标时间线证据时，可以对导出后的 `targets/target.mp4` 重新转录。

**边界：** 不自动生成 `(S1)` 等稳定说话人身份，不保证歌词或对白正确，也不设置数据记录的 `needs_audio_review`。这些都需要结合素材判断。

### 3.5 audio_tags.py：候选声音类别匹配

[查看源码](audio_tags.py)。使用 CLAP（Contrastive Language-Audio Pretraining，对比语言音频预训练）模型，比较每个声音窗口与候选文本标签的匹配程度。

```bash
.venv-asr/bin/python scripts/audio_tags.py <media> <start> <end> \
  [--win 2.0] [--labels "speech|music|footsteps"] [--top 5]
```

开始和结束位置单位为秒，默认窗口长 2 秒。内部使用 48000 Hz 单声道进行分析。标签可自行提供；默认集合包含对白、音乐、乐器、环境声和动作音效等。

**输出：** 终端逐窗口输出排名靠前的标签及相对分数，时间对应输入媒体的时间位置；没有内置输出文件参数，使用过的结果需重定向保存到 `WORK_DIR`。

**对应要求：** 为 §5 的同期声音、`overall_soundscape` 和 `non_diegetic_music` 提供辅助证据。

**边界：** 分数来自当前候选标签集合上的归一化，不能当作声音客观存在的概率。它不能证明说话人身份、具体乐器或参考波形被复制，也不应独立决定 `fully_copy`、`partially_copy` 等关系。

### 3.6 audio_profile.py：音量、频谱与节奏分析

[查看源码](audio_profile.py)。不依赖语义分类模型，计算一个声音区间的信号特征。

```bash
.venv-asr/bin/python scripts/audio_profile.py <media> <start> <end> [--hop 0.5]
```

开始和结束位置单位为秒，`--hop` 默认每 0.5 秒分析一窗。内部使用 22050 Hz 单声道。

**输出：** 终端打印 RMS（root mean square，均方根）音量、频谱重心、频谱平坦度、主要音高类别、频谱峰值，以及全区间的起音密度和 BPM（beats per minute，每分钟拍数）估计。需自行重定向保存分析结果。

**对应要求：** §3.1 的声音分析和 §5 的声景、配乐描述依据；可辅助观察静音、持续音和节奏变化。

**边界：** 信号特征不直接等于“钢琴”“雨声”等语义，也不证明参考音频复用；节拍估计对非音乐或复杂混音可能不可靠。

### 3.7 export_sample.py：根据已选区间导出媒体

[查看源码](export_sample.py)。将 agent 确认的素材选择转成实际交付媒体，是 §4 的主要执行脚本。

```bash
.venv/bin/python scripts/export_sample.py <WORK_DIR>/export_config.json <STAGE_DIR>
```

**输入：** agent 事先编写的 `export_config.json`。此文件不是脚本自动选段后生成的。

| 配置字段 | 含义 |
| --- | --- |
| `source` | 原片路径，相对 workspace 根目录解析 |
| `target` | 包含 `start_frame`、`end_frame_excl` 和导出 `num_frames` |
| `images` | 图片列表；每项有 `name`、`source_frame`，可选 `crop: [x, y, w, h]`，裁剪框使用源分辨率像素坐标 |
| `videos` | 参考视频列表；每项有 `name`、源帧区间、`num_frames`、`include_audio` |
| `audios` | 可选独立参考音频列表；每项有 `name`、`start_frame`、`end_frame_excl` |
| `sample_id` | 可记录样本标识，但当前导出逻辑不使用它；最终记录的 `id` 来自 `record_spec.json` |

图片、参考视频和独立音频列表可按实际需要省略或置空。名称应按最终约定设置为 `picture_1`、`video_1`、`audio_1` 等，脚本不会自动重新编号。`num_frames` 是导出后的帧数，不一定等于源区间帧数。

**实际处理：**

1. 按源帧的实际 PTS 确定裁切区间，精确裁切后重新编码。
2. 输出 24 FPS、`17k+5` 帧的视频，等比缩放并补边。画布按源宽高比、短边 768 和面积预算计算，常见横屏为 1344 × 768，并非所有画幅都固定为这一尺寸。
3. 所选区间较短时重复末帧并补静音；较长时裁去尾部。补齐量和截尾量写入溯源记录。
4. 直接从原片提取目标 WAV（Waveform Audio File Format），转换为 32000 Hz、双声道、16-bit PCM（pulse-code modulation，脉冲编码调制），每声道采样数为 `round(num_frames / 24 × 32000)`。
5. 将目标画面编码为 H.264，并将上述 WAV 编为 AAC（Advanced Audio Coding，高级音频编码）音轨写入目标 MP4（MPEG-4 Part 14）。
6. 参考视频仅在 `include_audio: true` 时导出音轨；为此生成的中间 WAV 保存在 `WORK_DIR/intermediate/`。独立参考音频按其源区间导出，不套用视频的 `17k+5` 帧规则。

**输出：**

- `STAGE_DIR/inputs/`：配置中选择的图片、参考视频及独立音频。
- `STAGE_DIR/targets/target.mp4`、`target_audio.wav`：目标视频和波形。
- `STAGE_DIR/oringin.mp4`：源文件副本，保留样例拼写。
- 配置文件所在目录下的 `provenance.json`、`export_log.json`：源哈希、区间、空间与时间变换、补齐、输出哈希和实际 FFmpeg 命令。因此应将配置放在 `WORK_DIR`。

**对应要求：** §3.2 的素材选择落地、§4 的媒体导出、§6 的媒体目录与溯源材料。

**当前边界：** 不会生成提示词、`selection.json` 或 `dataset.jsonl`；不会在导出前完成参考关系和全部时长限制检查。现有暂存目录会被删除后重建，必须使用专用 `STAGE_DIR`。原片目前通过直接复制生成 `oringin.mp4`，尚未实现 §6 对非 MP4 输入要求的真实封装转换流程。

### 3.8 inspect_export.py：以导出目标重新确认时间线

[查看源码](inspect_export.py)。在规范化之后运行，用于避免将原片绝对切镜时间直接写进提示词。

```bash
.venv/bin/python scripts/inspect_export.py <STAGE_DIR> <WORK_DIR>
```

**输入：** 暂存目录中的 `targets/target.mp4` 和 `inputs/video_*.mp4`。

**输出到 WORK_DIR：**

| 文件 | 用途 |
| --- | --- |
| `target_cuts_measured.json` | 导出目标帧数、检测到的切镜帧号、秒数和 `MM:SS.mmm` 时间 |
| `target_storyboard.jpg` | 目标每 0.5 秒抽帧的分镜预览 |
| `target_shot_boundaries.jpg` | 每处候选切镜的前后帧，以及目标末帧 |
| `reference_video_N_storyboard.jpg` | 各参考视频的分镜预览 |

**对应要求：** §4“规范化完成后重新确定目标切镜时间”、§5 的 `[Shot N] At MM:SS.mmm`，以及 §6 的预览材料。

**边界：** 仍是自动检测结果，需要实际查看边界图。确认、排除或补充切镜后，将结果写入 `record_spec.json`；本脚本不自动编写镜头描述，也不导出对白。

### 3.9 build_and_verify.py：组装记录、检查与发布

[查看源码](build_and_verify.py)。将准备好的提示词、媒体与元数据组装成一条 JSONL（JSON Lines）记录，完成程序检查后发布。

```bash
.venv/bin/python scripts/build_and_verify.py <WORK_DIR> <STAGE_DIR> <OUTPUT_DIR>
```

**实际读取的过程文件：**

| WORK_DIR 下的文件 | 谁准备 | 用途 |
| --- | --- | --- |
| `prompt.md` | agent | 最终英文六段式提示词；读取时会去掉文件首尾换行 |
| `record_spec.json` | agent | 样本标识、参考映射、声音质量字段、切镜确认和语义核验记录 |
| `provenance.json` | `export_sample.py` | 组装目标参数并检查来源、区间与导出结果 |
| `target_cuts_measured.json` | `inspect_export.py` | 对照提示词中的切镜时间 |

`selection.json` 和 `export_config.json` 应按流程保留，但当前构建验证代码不直接读取它们。

`record_spec.json` 的必需字段为 `id`、`split_group`、`references`、`needs_audio_review`、`uncertain_audio_spans_seconds`。`references` 使用 `PROMPT.md` §6 规定的对象结构，并与导出媒体一致。可选补充字段如下：

| 字段 | 作用 |
| --- | --- |
| `confirmed_cut_times` | 确认后的目标相对切镜秒数；未提供时使用自动检测结果 |
| `manual_cut_times` | 人工／agent 查看后补充的切镜，用来支持自动检测未覆盖的边界 |
| `rejected_cut_times` | 记录被排除的检测结果；当前用于检查详情展示，不会自动从确认列表中删除时间 |
| `manual_checks` | 语义检查列表；每项包含 `check`、`status`、可选 `detail`，以及可选 `required` |
| `review_items` | 仍需复核的问题列表；非空时影响最终状态 |

**主要程序检查：** 数据字段和路径、输入文件是否全部被引用、原片哈希、图片可读性、视频解码与帧率帧数、参考数量和时长、目标画布对齐、WAV 规格与采样数、目标 MP4 与 WAV 的声音对应、来源区间分离、目标首尾帧和源音频对应、参考画面的抽样近重复、提示词格式和声音质量字段一致性。

目标元数据先从 `provenance.json` 组装，再用实际媒体解码结果核对。发布流程是：暂存目录写入 `dataset.jsonl` → 暂存程序检查 → 将暂存目录移动到输出目录 → 再次检查输出目录 → 合并 `manual_checks` 并写报告。

**输出与状态：**

- 暂存程序检查失败：写 `WORK_DIR/verification_stage.json`，状态为 `failed`，停止发布。
- 发布或复查后：写 `WORK_DIR/verification.json`，当前脚本可给出 `completed`、`needs_review` 或 `failed`。
- 程序检查或录入的手工检查失败时为 `failed`；有声音待复核标记、`review_items`，或 `required: true` 的检查为 `not_checked` 时为 `needs_review`；其余情况为 `completed`。

只复查已发布结果时使用：

```bash
.venv/bin/python scripts/build_and_verify.py <WORK_DIR> <STAGE_DIR> <OUTPUT_DIR> --verify-only
```

此模式仍要求传入三个位置参数，但不读取暂存媒体、不重新构建或移动数据；检查 `OUTPUT_DIR` 后更新验证报告。

**对应要求：** §5 的格式检查、§6 的数据记录组装、§7 的程序验收与发布。

**当前边界：** 验证不等于完成语义理解。`manual_checks` 是录入已有核验结果，不是脚本实际观看或试听；它们在发布后才合并。现有输出的覆盖保护只核对已有记录的 `id` 和原片哈希，没有完整比较配置；满足条件时仍会替换输出，缺少 `dataset.jsonl` 的已有输出目录也会被替换。

## 4. 六段式提示词具体由谁完成

六段内容都由 agent 写入 `WORK_DIR/prompt.md`，没有脚本自动生成。`build_and_verify.py` 负责复制到数据记录及检查可程序化的结构。

| 提示词段落 | 主要依据 | 当前程序检查 |
| --- | --- | --- |
| `subject_definitions` | 实际参考图片、视频和启用的声音；全片与密集预览 | 标签连续编号、媒体标签映射、主体与声音标签定义 |
| `summary` | 目标实际情节、目标与参考的真实关系 | 任务类型合法性及其与声音标签的对应 |
| `retention_analysis` | 逐份比较参考与目标，明确保留或改变的属性 | 视觉／声音关系标记、条目与定义标签的对应 |
| `detailed_description` | 目标分镜、确认的切镜时间、动作与声音证据 | 镜头编号、时间格式与范围、切镜时间一致性、对白标签、说话人编号首次出现顺序、字数范围 |
| `overall_soundscape` | 实际环境与物理声音；转录、标签和信号分析辅助 | 六段顺序；不在本段重复写 `<d>` 对白 |
| `non_diegetic_music` | 实际配乐证据；没有配乐时按要求写 `N/A` | 六段顺序；不在本段重复写 `<d>` 对白 |

程序无法仅通过这些格式检查判断人物、衣着、动作、乐器或参考关系是否正确，也不会自动确认 `N/A` 是否符合实际声音。

## 5. 一条样本的实际执行顺序

1. 阅读 `PROMPT.md` §2 要求的依据，确定输入、样本标识和目录。
2. 运行 `analyze_video.py`，查看全片概览；用 `dense_frames.py` 细看候选区间。
3. 按需运行三个声音工具，先确定目标，再选择目标外的有效参考；保存 `selection.json`，编写 `export_config.json`。
4. 运行 `export_sample.py`，向专用暂存目录导出媒体并生成溯源、命令记录。
5. 运行 `inspect_export.py`，查看目标和参考预览，确认导出后的切镜；必要时对导出媒体重新转录、分析声音。
6. 根据实际目标编写 `prompt.md` 和 `record_spec.json`，录入已做的语义核验与待复核事项。
7. 运行 `build_and_verify.py`。发现问题后修复相应配置、媒体或文本，并重跑受影响步骤；仅复查已发布文件时使用 `--verify-only`。
8. 查看最终报告，按 `PROMPT.md` §7 返回输出路径、状态、目标区间、参考用途和待复核事项。

所有过程图、转录、配置、日志与报告放入 `work/`；最终 `pure/` 保存数据记录、原片副本和交付媒体。`examples/` 与 `reference/` 保持只读。

## 6. 当前实现与完整要求的差异

以下差异决定了为什么不能把“脚本正常退出”直接视为完成 `PROMPT.md`：

| 事项 | 当前实现与使用含义 |
| --- | --- |
| 自动选段与内容标注 | 脚本提供分析材料；`selection.json`、导出配置、最终提示词和语义核验仍需 agent 准备 |
| `skipped`／`blocked` | 构建脚本未实现这两个任务状态；无音轨、不适合构造或能力缺失时，需按提示词另外记录原因与状态 |
| 非 MP4 来源 | 导出脚本只复制为 `oringin.mp4`；非 MP4 输入不能依靠改名满足封装转换要求 |
| 输出复用 | 尚未实现“同输入且同配置”的完整复用判断，执行发布前仍需由 agent 判断目录是否适合复用 |
| 语义验收完整性 | 未提供 `manual_checks` 时，程序不会自动补齐必需的语义检查；报告中的 `completed` 必须结合实际完成的核验理解 |
| 重复画面与边界 | 近重复检测每隔 6 帧抽样，用低分辨率灰度平均差作启发式比较；自动首尾画面和源音频对应检查主要针对目标，不能替代对全部参考边界及重复内容的查看 |
| 验证阈值 | 当前要求提示词超过 1500 字符、详细描述为 250–650 个英文词、目标整体音量超过固定阈值、目标补齐不超过 0.25 秒。这些是脚本检查条件；提示词指南通常建议详细描述 350–500 词，并允许真实的局部安静片段 |
| 检查报告与退出码 | 只有暂存检查的已收集失败会显式以非零码退出；发布后复查可能写 `failed` 后正常结束，异常也可能直接中断。因此需读取报告，不能仅看退出码 |
| 未做项目的声明 | 报告的 `not_done` 当前固定列出未做 H3 推理、训练和人工试听，并非动态探测结果；应核对其是否符合本次实际工作 |

进一步的本地操作记录可参考 [ref2va-sample 工作流说明](../.claude/skills/ref2va-sample/SKILL.md)；任务要求和本次用户参数仍以 [PROMPT.md](../PROMPT.md) 及用户后续指令为准。
