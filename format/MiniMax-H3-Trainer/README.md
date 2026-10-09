# MiniMax-H3-Trainer：Ref2VA 原始清单与缓存格式

本次核对的是社区项目 [AviadDahan/MiniMax-H3-Trainer](https://github.com/AviadDahan/MiniMax-H3-Trainer)，版本 `7a1563a267a17da898df3396acbd6a9d24fcb142`。它与 MiniMax 官方仓库是不同项目。

## 本目录的五份样例

| 清单 | 参考布局 | 对应配置 |
|---|---|---|
| [01_single_image.json](samples/01_single_image.json) | 1 张图 | [01_single_image.yaml](configs/01_single_image.yaml) |
| [02_two_images.json](samples/02_two_images.json) | 2 张图 | [02_two_images.yaml](configs/02_two_images.yaml) |
| [03_image_audio.json](samples/03_image_audio.json) | 1 张图 + 1 段独立音频 | [03_image_audio.yaml](configs/03_image_audio.yaml) |
| [04_image_video.json](samples/04_image_video.json) | 1 张图 + 1 段无声视频 | [04_image_video.yaml](configs/04_image_video.yaml) |
| [05_video_audio.json](samples/05_video_audio.json) | 1 段带音轨视频 | [05_video_audio.yaml](configs/05_video_audio.yaml) |

**每份清单与同名配置单独使用。** 源码用 `zip(..., strict=True)` 将配置的 reference 条目与缓存中的参考几何逐个匹配，所以这五种布局不能直接混在一个缓存根目录并套用同一份固定参考数量配置。这里分开输出到 `.precomputed/<样例名>`；多个相同布局的真实样本可放进同一清单。[依据：FlexibleStrategy._collect_references](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/src/h3_trainer/training_strategies/flexible.py#L135)

## 原始清单字段

支持 `.json` 数组、`.jsonl` 逐行对象、`.csv` 表格。本例使用 JSON 数组以保留多个参考的列表类型。

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | 字符串 | 缓存文件名，建议显式且唯一；省略时取目标视频文件名主干 |
| `video` | 字符串路径 | 监督目标视频 |
| `caption` | 字符串 | 目标画面和声音描述 |
| `audio` | 可选字符串路径 | **目标音频**覆盖项；省略时从目标视频提取 |
| `reference_image` | 字符串或字符串数组 | 参考图；复数别名 `reference_images` 也被读取 |
| `reference_video` | 字符串或字符串数组 | 参考视频；复数别名 `reference_videos` 也被读取 |
| `reference_audio` | 字符串或字符串数组 | 独立参考音频；复数别名 `reference_audios` 也被读取 |

例如第 03 份样例明确区分了目标音频和参考音频：

```json
[
  {
    "id": "03_image_audio",
    "video": "media/targets/03_image_audio.mp4",
    "caption": "The red circle from the image reference moves up and down over a pale background. Use the soft 660 Hz electronic tone from the audio reference. Static camera; no speech.",
    "reference_image": "media/images/red_circle.png",
    "reference_audio": "media/references/tone_660.wav",
    "audio": "media/targets/03_image_audio.wav"
  }
]
```

`video` 别名包括 `target_video`、`media_path`、`file`；`caption` 别名包括 `prompt`、`text`。读取器不会把 CSV 中形如 `"[... ]"` 的字符串自动解析成数组，多参考请优先使用 JSON。路径原样传给媒体函数，**相对路径基于运行时工作目录，不是清单所在目录**。[依据：read_manifest、COLUMN_ALIASES、MediaPass](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/src/h3_trainer/preprocessing/builder.py)

参考按“图片 → 视频 → 独立音频”的顺序编码，列表内部保留原顺序。`reference_video` 会尝试提取视频内音轨，不使用 DiffSynth 那样的 `type: video_audio` 字段；想只提供视觉运动时，应使用无声视频。第 04 份用无声 MP4，第 05 份用带音轨 MP4。[依据：_encode_references](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/src/h3_trainer/preprocessing/builder.py#L254)

## 预处理与训练入口

先按上游的安装方式准备训练环境。模型路径需要包含 `transformer_ref/`、`text_encoder/`、`processor/`、`tokenizer/`、`vae/`、`audio_vae/` 等的 Diffusers 布局；不是直接把 `Ref2VA/` 原始打包目录传进来。[依据：model_loader.py](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/src/h3_trainer/model_loader.py)

```bash
export H3_TRAINER_REPO=/absolute/path/to/MiniMax-H3-Trainer-upstream
export H3_MODEL_PATH=/absolute/path/to/MiniMax-H3-diffusers-root
export H3_SAMPLE_ROOT=/absolute/path/to/format/MiniMax-H3-Trainer

# 可换成表中的任一样例名；只处理这一份清单。
bash "$H3_SAMPLE_ROOT/prepare.sh" 03_image_audio

# 模型与分布式训练环境就绪后使用同名配置；以下未在本次调研中执行。
deepspeed --num_gpus 8 "$H3_TRAINER_REPO/scripts/train.py" \
  "$H3_SAMPLE_ROOT/configs/03_image_audio.yaml"
```

`prepare.sh` 会切换到正确工作目录，传 `--references --resolution-bucket 832x480x124 --reference-canvas native --decode 1`。**该版本 `process_dataset.py` 的参数解析器没有 `--variant`**，虽然部分说明示例写了这个参数；本例以源码为准，不传它。训练模型分支通过配置里的 `model.variant: ref2va` 选择。[依据：实际命令行解析器](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/scripts/process_dataset.py)

配置使用 `probability: 1.0`，保证参考始终存在；源码指出参考 dropout（随机丢弃）尚未同步切换为不带参考的文本嵌入。样例只有一条，故设置 `val_split_every: 0`，防止唯一样本被分到验证集；这些 10 步配置只用于流程演示。[依据：参考丢弃说明](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/src/h3_trainer/training_strategies/flexible.py#L145)、[数据配置](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/src/h3_trainer/config.py#L342)

## 真正送入训练的数据

原始清单经过 VAE（Variational Autoencoder，变分自编码器）和文本编码器两次处理后，生成以下结构；**这些缓存没有在本目录伪造或预置**：

```text
.precomputed/03_image_audio/
  index.json
  latents/03_image_audio.safetensors
  audio_latents/03_image_audio.safetensors
  conditions/03_image_audio.safetensors
  reference_latents/03_image_audio.safetensors
  decoded_videos/03_image_audio.mp4
```

| 缓存 | 实际内容 |
|---|---|
| `index.json` | 顶层 `version`、`geometry`、`reference_canvas`、`num_samples`、`num_trainable`、`samples`；每条记录有 `id`、宽高/帧数、各模态行数、`has_audio`、`caption`、`source`、`reference_geometry` 等 |
| `latents/*.safetensors` | `latents`：目标视频潜变量行 |
| `audio_latents/*.safetensors` | `latents`：目标音频潜变量行 |
| `conditions/*.safetensors` | `prompt_embeds`、`text_token_tags` |
| `reference_latents/*.safetensors` | `geometry`，以及按模态存在的 `latents`、`audio_latents` |

训练端 `data.preprocessed_data_root` 指向以上缓存根目录，不指向原始 JSON。缺少文本嵌入的样本会跳过；缺少可用目标音轨的样本不会等价于正常音视频联合监督。预处理后应检查 `index.json` 的 `num_trainable` 和 `--decode` 的音画回放。[依据：缓存写出逻辑](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/src/h3_trainer/preprocessing/builder.py#L450)、[缓存读取逻辑](https://github.com/AviadDahan/MiniMax-H3-Trainer/blob/7a1563a267a17da898df3396acbd6a9d24fcb142/src/h3_trainer/datasets.py)
