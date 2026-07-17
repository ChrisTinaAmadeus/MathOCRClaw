# Bench30：小规模 Baseline（本周交付）

用于初版智能体对照评测的 **30 张高质量试卷页** + **千问 3-VL baseline Markdown**。  
流程可迁移到 MathDoc 全量（约 3600+ 张）。

数据来源：MathDoc / [winnk123/papers](https://github.com/winnk123/papers) 相关测试集；本目录只收录筛选后的小基准，不含全量 raw。

## 目录

```text
benchmark/bench30/
├── README.md                 # 本说明
├── BASELINE_30_GUIDE.md      # 从零到评估的完整流程记录
├── manifest.tsv              # id ↔ 原图路径 ↔ 官方 GT 路径 ↔ 难度备注
├── images/                   # 30 张原图（01_*.jpg …）
├── baseline/                 # 千问3-VL 提取的 30 份 Markdown（本周「我」的交付）
├── gt/                       # 官方 Result_GT 拷贝（做法 A，便于对齐评测）
├── prompts/
│   ├── extract_v1.txt        # 固定提取 Prompt（全量应复用/版本升级）
│   └── README.md
├── reports/                  # MathDoc 评估器跑通后的汇总（GT vs baseline）
└── scripts/                  # 提取与评测准备脚本（可参考迁移）
```

## 选取原则（高质量）

- **纳入**：跨页、手写分栏、选择题涂改等真实难点；印刷清晰、手写可辨。
- **不纳入**：选择题仅圈选/打钩不写字（视为不合格）；学生草图不识别、不管。
- **配比**：尽量覆盖清晰 / 中等噪声 / 困难样本；10 个原始分组尽量分散。

## Baseline 生成

- 模型：`qwen3-vl-plus`（DashScope OpenAI 兼容接口）
- Prompt：`prompts/extract_v1.txt`（题号、分区标题、`【答案：】`、无法识别标记等与 MathDoc 评估器对齐）
- 脚本参考：`scripts/extract_baseline.py`

## 与智能体对接建议

1. 对 `images/` 跑初版智能体，输出与 `baseline/` **同名或同 id** 的 Markdown。
2. 用 MathDoc 评估流水线，或仓库内后续评测脚本，对比 `gt/` / `baseline/`。
3. 迭代预处理（扫描化、去红笔、放大单题框等）与拒答策略（证据不足标 `U` / `[无法识别]`），再扩全量。

## 本周评测快照（官方 GT vs 本 baseline）

详见 `reports/summary_report.txt`。摘要：

- 流程：30 对全部跑通，对齐失败 0
- 题干相似度约 **0.77**；答案准确率约 **0.65**
- 图片相似度约 **0**（baseline 插图多为占位，未对齐官方裁图）
- 拒答 F1 偏低（GT 标无法识别时，模型仍常输出内容）

## 注意

- **不要**提交 API Key / `.env` / HF token。
- 全量原始数据请自行从 MathDoc / HuggingFace 拉取，勿塞进本仓库。
