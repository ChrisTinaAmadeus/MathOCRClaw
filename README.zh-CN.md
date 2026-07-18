# MathOCRClaw

简体中文 · [English](README.md)

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

### 小基准 Bench30

初版智能体评测使用 **30 张高质量试卷图片和千问 3-VL baseline Markdown**。详见 [`benchmark/bench30/`](benchmark/bench30/README.md)，其中包含 `manifest.tsv`、官方 GT 副本、baseline 输出和评测摘要。全量 MathDoc 数据不纳入本仓库。

### 输出结构

用户可见的运行产物包含原图与各处理阶段（内部缓存位于 `.cache/`）：

```text
workflow/
├─ image/                         # 未经处理的原始输入图片（按原格式保留）
├─ preprocessed/                  # 扫描化图片及预处理统计 JSON
├─ api_markdown/                  # API 原始题干 Markdown 与响应 JSON
├─ code_outputs/                  # 纯代码/本地模型阶段
│  ├─ rfdetr/<page_name>/         # 按页保存检测 JSONL、裁图和可视化
│  ├─ doclayout/<page_name>/      # 按页保存版面 JSON 和可视化
│  └─ match/                      # 阅读顺序、匹配 JSON 和题目裁图
└─ agent_outputs/<page_name>/     # 智能体最终结果
   ├─ result.md                   # 每道题后紧跟其手写答案
   ├─ result.json                 # 同结构的机器可读结果
   └─ verification.json           # 题干对齐与证据校验详情
```

`result.md` 只呈现题目与手写识别结果，不包含校验状态或证据；题框坐标、选框评分、裁图路径、识别原始响应和证据校验详情全部保存在 `result.json`。手写阶段会综合检测类别和题干结构区分选择题、填空题、简答题：选择题根据题干字符数、行数、选项数和邻题边界自适应扩大，以覆盖圈选外溢与涂改；填空题保留适度填空边距；简答题答案框从题干框底边开始并延伸到同栏下一题，保证不包含题干。预处理使用 HSV、Lab 色彩和通道差联合检测，并通过邻域传播与图像修复清除暗淡红笔及抗锯齿残迹。`code_outputs/match/<page>/viz/*_handwriting_overlay.png` 会在同一页以绿色显示题干框、洋红色显示手写答案框，对应坐标及题型判定保存在 `handwriting_regions.json`。

### 代码结构与检查

```text
agent/workflow.py       唯一端到端入口与输出管理
match/                  本地检测、版面分析、阅读顺序和题图匹配
proofread/              题干修复、证据验证和主动拒绝
scripts/run_agent.ps1   用户入口
```

```powershell
.\.conda\messtoclean\python.exe -m unittest discover -s tests -v
.\.conda\messtoclean\python.exe -m agent.workflow --help
```

`workflow/`、`input/`、本地环境、API 密钥、模型权重和 `Reference/` 均被 Git 忽略；`Reference/` 只用于本地研究，不上传 GitHub。
