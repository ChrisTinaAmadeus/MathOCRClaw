# MathOCRClaw

简体中文 · [English](README.md)

MathOCRClaw 是一个面向真实试卷照片的数学 OCR 智能体。它通过两次全局 API 调用识别印刷题干与学生手写答案：第一次生成整页初稿，第二次结合本地检测与裁图上下文进行统一校对并直接产生最终结果。

```text
试卷照片
  → 去阴影并保留彩色笔迹
  → API #1 生成整页题干与手写初稿 Markdown
  → 本地题目检测、版面分析与上下文包构建
  → API #2 综合初稿、整页图、题目裁图和版面信息全局校对
  → 直接生成题干和答案一一对应的最终结果
```

### 快速开始

需要 Linux、Bash、Conda、可用的 DashScope/OpenAI 兼容多模态 API，以及放在仓库根目录的 `checkpoint_best_total.pth`。GPU 为可选项，版面检测默认使用 CPU。

创建独立的 Linux 环境。安装脚本固定使用 `.conda/mathocrclaw`，不会复用从其他操作系统复制来的环境。

```bash
bash scripts/setup_env.sh
```

创建不会被 Git 跟踪的 `.env.local`，然后填入 API 密钥：

```bash
cp --no-clobber .env.example .env.local
```

图片可以位于任意本地路径；推荐放在同样不会被 Git 跟踪的 `input/` 中。运行完整工作流：

```bash
bash scripts/run_agent.sh --image input/page_0001.jpg --full
```

复用已有本地检测和匹配结果时加 `--skip-layout`。未命中缓存时，工作流固定进行两次 API 调用；`--full` 只会在第二次请求中为每题增加一张手写细节图，不会增加调用次数。需要在当前终端中交互使用该环境时，执行 `source scripts/activate_env.sh`。

### 基准数据

智能体评测集包含 **32 张高质量试卷图片和Qwen3.7-plus baseline Markdown**。图片及其对应的 baseline 输出在 [`benchmark/images/`](benchmark/images/) 和 [`benchmark/baseline/`](benchmark/baseline/) 中统一使用 `page01` 至 `page32` 的配对名称。全量 MathDoc 数据不纳入本仓库。

### 输出结构

用户可见的运行产物包含原图与各处理阶段（内部缓存位于 `.cache/`）：

```text
workflow/
└─ <page_name>/                   # 每张图片只有这一层页名目录
   ├─ image/                      # 未经处理的原始输入图片（按原格式保留）
   ├─ preprocessed/               # 扫描化图片及预处理统计 JSON
   ├─ api_markdown/               # 第一次 API 的整页初稿 Markdown 与响应 JSON
   ├─ code_outputs/               # 纯代码/本地模型阶段
   │  ├─ rfdetr/                  # 检测 JSONL、裁图和可视化
   │  ├─ doclayout/               # 版面 JSON 和可视化
   │  └─ match/                   # 阅读顺序、题目上下文包和裁图
   └─ agent_outputs/              # 不再重复创建 <page_name>/
      ├─ result.md                # 每道题后紧跟其手写答案
      ├─ result.json              # 同结构的机器可读结果
      └─ verification.json        # 第二次全局校对的原始响应
```

`<page_name>/api_markdown/<page>.md` 是第一次 API 直接按照 `benchmark/prompts/extract_v2.txt` 生成的整页初稿；Python 代码不维护另一份提示词副本。随后的本地阶段解析该 Markdown 并生成 `<page_name>/code_outputs/match/question_contexts.json` 及题干/作答区/细节视图，不调用 API。第二次 API 在同一个请求中读取初稿 Markdown、整页图、所有题目上下文图与版面摘要，直接返回最终题干、分问答案和来源上下文 ID。`result.md` 严格采用与金标准一致的逐题格式；被划去的作废内容不会进入初稿、终稿或不确定片段。预处理只校正阴影和不均匀光照，保留红笔在内的彩色标注；框线叠加图仅作为几何诊断产物。

### 代码结构与检查

```text
agent/workflow.py       唯一端到端入口与输出管理
match/                  本地检测、版面分析、阅读顺序和题图匹配
proofread/              图像、缓存与历史校验工具
scripts/setup_env.sh    创建或更新 Linux Conda 环境
scripts/run_agent.sh    用户入口
```

```bash
bash scripts/check_env.sh
.conda/mathocrclaw/bin/python -m unittest discover -s tests -v
.conda/mathocrclaw/bin/python -m agent.workflow --help
```

仓库通过 `.gitattributes` 将所有文本文件固定为 LF，避免 CRLF 转换导致 Git 把整份文件误判为改动。`workflow/`、`input/`、本地环境、API 密钥、模型权重和 `Reference/` 均被 Git 忽略；`Reference/` 只用于本地研究，不上传 GitHub。
