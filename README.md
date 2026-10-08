# Ref2VA 数据构造 workspace

这个目录包含执行任务需要的提示词、样例输入与输出、H3 提示词指南及媒体规格依据。可以整体移动、复制，或直接作为 agent 的 workspace 打开；文件路径均在目录内解析。

## 开始使用

将当前目录设为 workspace，向 agent 发送：

```text
请阅读并执行 PROMPT.md，使用默认样例输入，完成数据生成与验证。
```

处理自己的视频时，把视频放入 `inputs/`，例如 `inputs/my_video.mp4`，然后发送：

```text
请阅读并执行 PROMPT.md。
本次 INPUT_VIDEO=inputs/my_video.mp4
OUTPUT_DIR=outputs/my_video/pure
WORK_DIR=outputs/my_video/work
```

所有参数中的相对路径均相对于这个 workspace 根目录。默认输入是 [examples/input.mp4](examples/input.mp4)，已有标准输出位于 [examples/pure/](examples/pure/)，新结果写入 `outputs/`。

## 目录

```text
.
├── AGENTS.md                 # Agent 的目录约定
├── PROMPT.md                 # 完整执行提示词
├── README.md                 # 使用入口
├── examples/
│   ├── README.md             # 样例来源、配对关系和已知限制
│   ├── input.mp4             # 原始样例视频
│   ├── pure/                 # 样例 dataset.jsonl 与全部媒体
│   └── evidence/             # 来源记录、预览、转录和复制检查
├── reference/
│   ├── minimax-h3/           # H3 规格摘录和完整提示词技能
│   ├── media-profile.md      # 样例媒体配置及实现差异
│   └── SOURCES.json          # 资料来源与文件校验值
├── inputs/                   # 放入后续待处理视频
└── outputs/                  # 新数据及对应的过程材料
```

数据文件采用本地 JSONL（JSON Lines）结构，模型提示词遵循 H3 六段式 Ref2VA（reference-to-video-and-audio）规范，两者的区别见 [PROMPT.md](PROMPT.md)。

样例的纯数据目录按原样复制，包括 `oringin.mp4` 这一文件名。原视频在 `examples/input.mp4` 和 `examples/pure/oringin.mp4` 各有一份，便于独立读取输入和原样保留输出结构。

## 执行环境

这里自包含的是任务资料和示例数据。执行媒体处理仍需要 Python、FFmpeg 或等效工具，以及 agent 的图像／视频理解能力；对白核对还需要可用的音频理解或 ASR（automatic speech recognition，自动语音识别）能力。agent 应使用当前环境或在本目录建立独立环境，不依赖原下载工程的虚拟环境。

媒体导出规则见 [reference/media-profile.md](reference/media-profile.md)。H3 格式指南为本地版本快照，出处见 [reference/SOURCES.json](reference/SOURCES.json)。

样例保留了声音待复核标记。文件复制检查通过不表示声音已完成复核，具体见 [examples/README.md](examples/README.md)。
