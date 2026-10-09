# verl-omni：Ref2VA 奖励训练数据样例

核对版本：`54c557ed572d484f3d79e705a044254af879edff`。本目录对应 MiniMax-H3 的 FlowGRPO / DiffusionNFT 在线奖励训练路径，不能把它理解为“目标视频 + caption”的监督微调格式。[官方 Ref2VA 训练说明](https://github.com/verl-project/verl-omni/blob/54c557ed572d484f3d79e705a044254af879edff/examples/diffusionnft_trainer/minimax_h3/README.md#ref2va-multi-reference-training)

已提供 [train.jsonl](train.jsonl) 中的 5 条训练输入、[test.jsonl](test.jsonl) 中的 1 条验证输入，以及实际转换得到的 [train.parquet](parquet/train.parquet)、[test.parquet](parquet/test.parquet)。`test` 是上游脚本采用的文件名，对应训练启动器的验证数据。

## 原始 JSONL 字段

每行是一个完整 JSON 对象，行间没有逗号，文件最外层没有数组括号：

```json
{"prompt":"The red circle from the reference image moves from left to right over a pale background. Static camera. A soft 440 Hz electronic tone plays; no speech.","images":["media/images/red_circle.png"]}
{"prompt":"The red circle from the image reference follows the left-to-right motion in the silent video reference. Pale background, static camera. A soft 770 Hz electronic tone plays; no speech.","images":["media/images/red_circle.png"],"videos":[{"path":"media/references/red_motion_silent.mp4","start_time_seconds":0.0}]}
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `prompt` | 非空字符串 | 期望生成的音画内容 |
| `images` | 字符串数组 | 参考图片路径 |
| `videos` | 字符串/对象数组 | 参考视频路径；对象支持 `path`、`start_time_seconds` |
| `audios` | 字符串数组 | 独立参考音频路径 |

也支持单数 `image`、`video`、`audio`，但不要同时提供同一模态的单数和复数：转换器优先使用复数，不会自动合并。视频对象的路径键也接受 `video`、`video_path`；`start_time_seconds` 是非负秒数，省略时写入 `null`。本例显式的 `0.0` 展示的是从头使用参考片段。[依据：prepare_ref2va_data.py](https://github.com/verl-project/verl-omni/blob/54c557ed572d484f3d79e705a044254af879edff/examples/diffusionnft_trainer/minimax_h3/prepare_ref2va_data.py)

该转换器要求至少有一张图或一段视频；仅音频参考会被拒绝。数量上限：图 ≤ 9、视频 ≤ 3、独立音频 ≤ 3、三个列表长度合计 ≤ 12。相对路径相对 `--input_dir` 解析，文件必须存在。不要自行往原始 `prompt` 前添加 `<image>` 等占位符，转换器会添加。

第 05 条把带音轨 MP4 放在 `videos` 中，不使用 DiffSynth 的 `references` / `type: video_audio` 写法。JSONL 转换阶段只传递视频路径，不解码音轨；实际参考的音画处理交给训练的生成后端。上游管线测试包含内部 `video_audio` 参考几何，但本次只验证数据与媒体，未运行后端生成。[依据：原始媒体传递](https://github.com/verl-project/verl-omni/blob/54c557ed572d484f3d79e705a044254af879edff/verl_omni/pipelines/minimax_h3_diffusion_nft/agent_loop.py#L37)、[音画参考管线测试](https://github.com/verl-project/verl-omni/blob/54c557ed572d484f3d79e705a044254af879edff/tests/pipelines/test_minimax_h3_ref2va_nft_on_cpu.py)

## 实际 Parquet 列

Parquet 是列式二进制数据文件，不是将 JSONL 改后缀。本目录的转换器为上游原样副本，校验时已与固定版本逐字节比对。

| 列 | 实际结构/作用 |
|---|---|
| `data_source` | `"minimax_h3_ref2va"` |
| `prompt` | 消息数组：`[{"role":"user","content":"<image>...<video>...<audio>...原提示词"}]` |
| `ability` | `"reference_to_audio_video"` |
| `images` | `[{"bytes": 图片文件的原始二进制字节}]`；不是文件名、Base64 字符串或像素数组 |
| `videos` | `[{"video":"绝对路径","start_time_seconds":0.0或null}]` |
| `audios` | 独立音频文件的绝对路径数组 |
| `reward_model` | `{"style":"model","ground_truth":"原提示词"}` |
| `extra_info` | `split`、`index`、`source_images`、`source_videos`、`source_audios` |

`ground_truth` 在这里是给奖励函数使用的提示词，不是目标视频，也不是预先标注好的数值奖励。模型在线生成视频/音频，再由奖励函数评分；本目录因此没有监督目标视频列。训练脚本仍需要奖励模型或合适的奖励函数配置。[依据：转换器输出](https://github.com/verl-project/verl-omni/blob/54c557ed572d484f3d79e705a044254af879edff/examples/diffusionnft_trainer/minimax_h3/prepare_ref2va_data.py#L120)、[FlowGRPO 启动脚本的奖励配置](https://github.com/verl-project/verl-omni/blob/54c557ed572d484f3d79e705a044254af879edff/examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_ref2va_lora_v1.sh)

[parquet_preview.json](parquet_preview.json) 包含实际 Arrow 类型和逐行预览，图片字节替换成了长度与 SHA-256（Secure Hash Algorithm 256-bit，256 位安全散列算法）摘要。该预览**不是训练文件**。

## 重新转换与接入

移动文件夹或上传到训练服务器后，重新执行转换，更新绝对路径。在已安装 `pandas`、`pyarrow` 的 Python 环境运行：

```bash
sample_root=/absolute/path/to/format/verl-omni
python "$sample_root/prepare_ref2va_data.py" \
  --input_dir "$sample_root" \
  --output_dir "$sample_root/parquet"
```

也可以调用上游仓库中的同一份脚本：

```bash
python examples/diffusionnft_trainer/minimax_h3/prepare_ref2va_data.py \
  --input_dir /absolute/path/to/format/verl-omni \
  --output_dir /absolute/path/to/format/verl-omni/parquet
```

注意所核对版本的部分 FlowGRPO README 将转换器写在 `examples/flowgrpo_trainer/...`，该路径下实际没有这个文件；上面给的是代码中存在的 `diffusionnft_trainer` 路径，两种算法共享该数据格式。

训练时让 `DATA_DIR` 指向 `parquet/`，启动器会读取 `train.parquet`、`test.parquet`。可选用 [FlowGRPO v1 启动器](https://github.com/verl-project/verl-omni/blob/54c557ed572d484f3d79e705a044254af879edff/examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_ref2va_lora_v1.sh)或 [DiffusionNFT 启动器](https://github.com/verl-project/verl-omni/blob/54c557ed572d484f3d79e705a044254af879edff/examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_ref2va_lora.sh)。前者使用当前默认的 v1 训练架构；本次不将此结论外推为所有算法启动器都已迁移。

上游默认批大小可能大于本例的 5 条训练数据，正式启动前要扩充数据或调整批大小及相应策略更新参数。输出尺寸/帧数属于生成管线配置，不写在这份数据清单里；本例素材的 124 帧不自动改变启动器的 `NUM_FRAMES`。不同参考布局还需要足够的 `MAX_PROMPT_EMBEDS`，避免参考条件超出传输上限。[依据：Ref2VA 说明与启动器](https://github.com/verl-project/verl-omni/blob/54c557ed572d484f3d79e705a044254af879edff/examples/diffusionnft_trainer/minimax_h3/README.md#ref2va-multi-reference-training)
