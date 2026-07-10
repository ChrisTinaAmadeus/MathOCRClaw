# MathOCRClaw

[简体中文](#简体中文) · [English](#english)

## 简体中文

MathOCRClaw 是一个面向真实试卷照片的数学 OCR 智能体。它识别印刷题干与学生手写答案，通过裁图证据复核识别结果，并在证据不足时主动输出 `U`，而不是猜测。

```text
试卷照片
  → 去阴影、去红笔
  → API 生成整页题干 Markdown
  → 本地题目检测、版面分析与题号对齐
  → 逐题提取并验证手写答案
  → 题干和答案一一对应的最终结果
```

### 快速开始

需要 Windows PowerShell、Conda、可用的 DashScope/OpenAI 兼容多模态 API，以及放在仓库根目录的 `checkpoint_best_total.pth`。

```powershell
conda env create --prefix .\.conda\messtoclean -f environment.yml
```

创建不会被 Git 跟踪的 `.env.local`：

```dotenv
DASHSCOPE_API_KEY=your_api_key
MTC_VLM_MODEL=qwen3.7-plus
```

图片可以位于任意本地路径；推荐放在同样不会被 Git 跟踪的 `input/` 中。运行完整工作流：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_agent.ps1 -Image .\input\page_0001.jpg -Full
```

复用已有本地检测和匹配结果时加 `-SkipLayout`。不加 `-Full` 时会关闭题号读取、文本修复和题图关系检查，以减少 API 请求。

### 输出结构

所有用户可见的运行产物只分为四类（内部缓存位于 `.cache/`）：

```text
workflow/
├─ preprocessed/                  # 扫描化图片及预处理统计 JSON
├─ api_markdown/                  # API 原始题干 Markdown 与响应 JSON
├─ code_outputs/                  # 纯代码/本地模型阶段
│  ├─ rfdetr/                     # 检测 JSONL、裁图和可视化
│  ├─ doclayout/                  # 版面 JSON 和可视化
│  └─ match/                      # 阅读顺序、匹配 JSON 和题目裁图
└─ agent_outputs/<page_name>/     # 智能体最终结果
   ├─ result.md                   # 每道题后紧跟其手写答案
   ├─ result.json                 # 同结构的机器可读结果
   └─ verification.json           # 题干对齐与证据校验详情
```

`result.md` 不再先列全部题目、再列全部答案，而是按照“题目 16 → 手写答案 16 → 题目 17 → 手写答案 17”的顺序输出。只对成功对齐到题干的主问题裁图提取答案，避免把页面边缘或其他试卷的手写内容混入最终结果。

### 代码结构与检查

```text
agent/workflow.py       唯一端到端入口与四类输出管理
match/                  本地检测、版面分析、阅读顺序和题图匹配
proofread/              题干修复、证据验证和主动拒绝
scripts/run_agent.ps1   用户入口
```

```powershell
.\.conda\messtoclean\python.exe -m unittest discover -s tests -v
.\.conda\messtoclean\python.exe -m agent.workflow --help
```

`workflow/`、`input/`、本地环境、API 密钥、模型权重和 `Reference/` 均被 Git 忽略；`Reference/` 只用于本地研究，不上传 GitHub。

---

## English

MathOCRClaw is a math-OCR agent for real-world exam photos. It recognizes printed questions and student handwriting, verifies predictions against aligned crops, and returns `U` instead of guessing when evidence is insufficient.

```text
exam photo
  → shadow normalization and red-ink removal
  → API-generated whole-page question Markdown
  → local detection, layout analysis, and question alignment
  → per-question handwriting extraction and verification
  → paired question-and-answer results
```

### Quick start

Requirements: Windows PowerShell, Conda, a DashScope/OpenAI-compatible multimodal API, and `checkpoint_best_total.pth` in the repository root.

```powershell
conda env create --prefix .\.conda\messtoclean -f environment.yml
```

Create a Git-ignored `.env.local`:

```dotenv
DASHSCOPE_API_KEY=your_api_key
MTC_VLM_MODEL=qwen3.7-plus
```

The input image may be anywhere locally; `input/` is a convenient Git-ignored location. Run the full workflow with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_agent.ps1 -Image .\input\page_0001.jpg -Full
```

Add `-SkipLayout` to reuse local detection and matching outputs. Without `-Full`, question-number reading, text patching, and question/figure checks are disabled to reduce API calls.

### Output layout

User-facing runtime artifacts are grouped into exactly four categories (internal caches live under `.cache/`):

```text
workflow/
├─ preprocessed/                  # normalized scans and preprocessing JSON
├─ api_markdown/                  # raw API question Markdown and response JSON
├─ code_outputs/                  # local code/model stages
│  ├─ rfdetr/                     # detection JSONL, crops, and visualizations
│  ├─ doclayout/                  # layout JSON and visualizations
│  └─ match/                      # reading order, matching JSON, and question crops
└─ agent_outputs/<page_name>/     # final agent output
   ├─ result.md                   # each question followed by its handwriting
   ├─ result.json                 # machine-readable paired structure
   └─ verification.json           # alignment and evidence details
```

`result.md` now follows “question 16 → answer 16 → question 17 → answer 17,” rather than listing every question before every answer. Handwriting extraction only runs on main crops aligned to verified questions, preventing marginal or neighboring-page handwriting from entering the final result.

### Code layout and checks

```text
agent/workflow.py       single end-to-end entrypoint and output manager
match/                  local detection, layout, reading order, and matching
proofread/              question repair, evidence verification, and abstention
scripts/run_agent.ps1   user entrypoint
```

```powershell
.\.conda\messtoclean\python.exe -m unittest discover -s tests -v
.\.conda\messtoclean\python.exe -m agent.workflow --help
```

`workflow/`, `input/`, local environments, API secrets, model weights, and `Reference/` are Git-ignored. `Reference/` is local research material and is never uploaded to GitHub.
