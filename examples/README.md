# 样例：Can't Touch This

[input.mp4](input.mp4) 是完整源视频，[pure/dataset.jsonl](pure/dataset.jsonl) 是一条配对记录。[pure/](pure/) 的文件与原样例逐字节一致；这里单独补充可在当前 workspace 内阅读的说明。

配对关系为：人物与产品图片、办公室片段的叙事与声音参考，加上目标描述，对应搬沙发和车库卧推的真实目标音视频。目标来自原片，没有运行视频生成模型。

## 素材对应关系

源帧号从 0 开始，区间右端不包含。源视频帧率为 `24000/1001` FPS（frames per second，每秒帧数）。

| 文件 | 源位置 | 用途 |
| --- | --- | --- |
| [picture_1.png](pure/inputs/picture_1.png) | 第 409 帧，约 17.059 秒 | 主角、沾粉手指及零食包装 |
| [picture_2.png](pure/inputs/picture_2.png) | 第 558 帧，约 23.273 秒 | 戴墨镜人物的外观 |
| [video_1.mp4](pure/inputs/video_1.mp4) | 帧区间 `[62, 200)`，约 2.586–8.342 秒 | 办公室求助、拒绝、人物出现的叙事结构及声音风格 |
| [target.mp4](pure/targets/target.mp4) | 帧区间 `[200, 374)`，约 8.342–15.599 秒 | 搬沙发和车库卧推两个情节，共六个镜头 |
| [target_audio.wav](pure/targets/target_audio.wav) | 同目标区间 | 从原片提取的目标声音监督 |
| [oringin.mp4](pure/oringin.mp4) | 完整源视频 | 溯源副本，不作为模型参考输入 |

图片与参考视频均不来自目标源帧区间。同源参考包含冗余人物信息，这条样例不证明参考能力已被严格解耦。`<Audio 1>` 是 `video_1.mp4` 中启用的同步音轨，不是另一份独立文件。

目标规范化后为 1344 × 768、24 FPS、175 帧，即约 7.292 秒；参考视频为 141 帧，即 5.875 秒。目标波形为 32000 Hz 双声道、16-bit PCM（pulse-code modulation），每声道 233333 个采样点。处理方法见 [媒体配置](../reference/media-profile.md)。

## 预览与来源记录

- [全片概览](evidence/overview.jpg)
- [参考片段预览](evidence/reference_storyboard.jpg)
- [目标六镜头预览](evidence/target_storyboard.jpg)
- [来源与变换记录](evidence/provenance.json)
- [历史自动语音识别结果](evidence/transcript_clips_asr.json)
- [本次复制完整性检查](evidence/package-verification.json)

`provenance.json` 和转录文件中的媒体相对路径，以各自注明的 `../pure` 为基准；`source_path_from_workspace` 从 workspace 根目录解析。重新打包只调整了这些路径说明，未改变原始来源区间和媒体哈希。

## 已知限制

目标约 1.2–2.6 秒的搬运工短句尚待声音复核，提示词保留 `[unclear]`，记录中 `needs_audio_review` 为 `true`。历史转录中出现的低置信度猜测不能当作准确台词。

原样例做过媒体及结构检查，本次打包检查的是文件复制完整性、路径和记录一致性；没有新增人工试听、模型推理或训练验证。后续 agent 应按当前任务重新检查选段和标注，不能照抄历史结果作为新任务的验收结果。
