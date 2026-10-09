# DiffSynth-Studio：Ref2VA 监督微调样例

核对版本：`ee631d4907d8d6fc3d39063a2a65c00a954c9bf5`，2026-10-09 获取。

本目录的 [metadata.json](metadata.json) 是 5 条样本组成的 JSON 数组，可作为官方 Ref2VA 数据预处理入口的输入。图片、视频和音频均在 `media/` 中。

## 字段与角色

| 字段 | 类型 | 含义 |
|---|---|---|
| `video` | 字符串路径 | 目标视频，作为监督信号 |
| `prompt` | 字符串 | 对目标画面、动作、声音的描述 |
| `input_audio` | 字符串路径 | **目标音轨**。本例与 `video` 指向同一个带音轨 MP4 |
| `references` | 对象数组 | 输入条件，支持以下四类参考块 |

```json
[
  {
    "video": "media/targets/01_single_image.mp4",
    "prompt": "The red circle from the reference image moves from left to right over a pale background. Static camera. A soft 440 Hz electronic tone plays; no speech.",
    "input_audio": "media/targets/01_single_image.mp4",
    "references": [
      {"type": "image", "image": "media/images/red_circle.png"}
    ]
  }
]
```

参考块的准确字段如下；这些路径都相对数据根目录解析：

| `type` | 必须提供的媒体字段 | 本目录覆盖 |
|---|---|---|
| `image` | `image` | 01、02、03、04 |
| `video` | `video` | 04，无声视频 |
| `audio` | `audio` | 03，独立参考音频 |
| `video_audio` | **同时提供 `video`、`audio`** | 05，两项都指向同一个带音轨 MP4 |

`references` 读取器也接受单个对象，但本例统一使用数组。`video_audio.audio` 不会因为已有 `video` 而自动补齐。参考图保持原始分辨率交给管线；参考视频在加载时裁切到训练画布、按 24 fps 采样，参考音频由读取器重采样到 32 kHz。[依据：MiniMaxH3ReferenceLoader](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/diffsynth/utils/data/minimax_h3.py)

## 必须配套的加载参数

```bash
--dataset_base_path /absolute/path/to/format/DiffSynth-Studio
--dataset_metadata_path /absolute/path/to/format/DiffSynth-Studio/metadata.json
--data_file_keys "video,input_audio,references"
--extra_inputs "input_audio,references"
--height 480 --width 832 --num_frames 124
```

`references` 必须出现在两个参数中：`data_file_keys` 控制文件解码，`extra_inputs` 控制传入模型。`input_audio` 同理。仅把字段放进 JSON 不等于模型会收到该字段。[依据：train.py](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/examples/minimax_h3/model_training/train.py)

该版本的目标音频算子 `LoadAudioWithTorchaudio` 会先尝试读取媒体的视频帧数，再按时长截取或填充音频。因此样例遵循官方做法，`input_audio` 使用带音轨 MP4，不能把参考音频字段与它互换，也不要假设独立 WAV 可直接替代这个算子的输入。需要独立目标 WAV 时，先合入视频或改用适配的音频读取算子。`references[].audio` 则走另一条读取路径，可以用 WAV。[依据：音频算子与帧采样器](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/diffsynth/core/data/operators.py#L289)

本例不写 `frame_rate` 字段：这个训练入口将帧率固定为 24；在元数据里写成 30 不会改变加载行为。`num_frames` 必须符合 `17n + 5`，不能直接照搬其他视频模型的 121 帧设置。[依据：训练参数检查](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/examples/minimax_h3/model_training/train.py#L157)

## 如何接入训练

先安装官方框架和它要求的音视频解码依赖，再在对应环境执行。以下命令会加载/按需获取模型并生成缓存，本次交付没有执行这些模型计算：

```bash
export DIFFSYNTH_REPO=/absolute/path/to/DiffSynth-Studio-upstream
bash /absolute/path/to/format/DiffSynth-Studio/prepare.sh
```

[prepare.sh](prepare.sh) 按官方 LoRA（Low-Rank Adaptation，低秩适配）脚本第一阶段配置，以 `sft:data_process` 写出 `cache/<进程编号>/<样本编号>.pth`。第二阶段使用官方脚本的 `sft:train` 部分，将 `--dataset_base_path` 指向这里的 `cache`，不再传 `--dataset_metadata_path`；加载 Ref2VA 的 DiT（Diffusion Transformer，扩散变换器）权重。全量训练使用同一份原始数据格式，但采用官方 full 配置。[官方 LoRA 两阶段脚本](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA.sh)、[全量训练脚本](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA.sh)

原始 JSON 和真实媒体已验证；`.pth` 必须由对应模型预处理生成，不能用这里的 JSON 重命名替代。
