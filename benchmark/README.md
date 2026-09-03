# Benchmark

这个 benchmark 用同一套金标准分别评分两阶段输出：

1. API1 按 `prompts/extract_v2.txt` 直接生成一次 `api_markdown/pageXX.md`，得到当前模型基准分；
2. API2 对同一份 API1 初稿独立执行三次，结果保存到 `agent_outputs/api2_runs/run_XX/result.json`；
3. 三次 API2 分数的算术平均值作为最终工作流分，报告同时保留每次分数，并给出相对 API1 的差值以及结构、题干、手写答案、状态、漏识别和幻觉等子指标。

默认运行全部 32 页，每页工作目录彼此隔离：

```bash
bash benchmark/run_benchmark.sh --no-cache
```

先跑单页烟雾测试：

```bash
bash benchmark/run_benchmark.sh --pages page01 --no-cache --fail-fast
```

复用已有产物只重新打分：

```bash
bash benchmark/run_benchmark.sh \
  --pages page01 page02 \
  --work-root benchmark/runs/20260815-120000/workflow \
  --output-dir benchmark/runs/rescore \
  --score-only
```

输出目录包含 `report.json`、`report.md`、逐页报告和三份 API2 结果。标准网络评测每页执行 1 次 API1 和 3 次 API2；三次 API2 即使启用缓存也会作为独立样本实际调用。模型、超时、token 上限、布局设备、缓存和估算单价均可通过 `--help` 配置。API 请求的实际次数、token 和延迟由工作流记录；若服务端不返回 usage，对应 token 字段为 `null`。

API1 与 API2 的上传/读取超时默认均为 360 秒，分别可用 `--baseline-timeout` 和 `--review-timeout` 覆盖。例如对超大图片可使用 `--baseline-timeout 600 --review-timeout 600`。

默认打分器遵循 [EVALUATION.md](EVALUATION.md) 的确定性聚合框架，但不额外调用裁判模型：语义和公式等价使用本地可复现的近似算法，裁判调用数固定为 0。这样不会把额外裁判成本混入 OCR 工作流成本；将来接入固定裁判模型时可以保留相同的报告结构。

第一次 API 的提示词不在工作流代码中复制维护，而是直接读取 [`prompts/extract_v2.txt`](prompts/extract_v2.txt)。其 Markdown 响应由本地解析器生成同目录 JSON 元数据，解析过程不调用模型。该提示词依据 32 页金标准统一了题号、`### 手写答案`、LaTeX、`<插图>` 和拒识标记，并要求彻底忽略所有被划去的作废内容。

两阶段使用同一份金标准语义：选择题只保留最终选项，填空题只保留最终值/公式和必要单位，解答题保留按小问组织的完整正式过程。选择、填空的推算草稿、排除痕迹和图上批注不会进入候选答案。

API2 不再自由重写整页，而是返回 `api2_patch_v4` 审查记录。题干和解答题只允许对 API1 字段做精确子串替换；选择题与填空题可返回结构化最终答案，但必须同时提交基于 `symbol_profile_v3` 的对比式逐字形观察。profile 只限制候选字符空间、单选/多选基数并列出易混淆笔画，不含标准答案。对 API1 中的非规范高中数学符号，profile 还生成带唯一原文锚点的 `symbol_audit` 目标；API2 必须结合彩色局部图、`answer_ink` 和同一书写者的无标签对比字形逐处判定。模型自写的符号替换文本不会被信任：本地合并器校验 `audit_id`、可视笔画和候选符号后，直接从 profile 的唯一 `draft_fragment` 和候选替换表生成原子补丁；该通道不允许修改数字、答案值或推导。其他 API1 已有合法答案的符号修改仍必须引用同页清晰字形作为独立参照，否则回退 `keep`。

对 API1 中每个已有 `<插图>`，API2 还必须提交 `figure_audit_v1`：分别判断来源（印刷图、学生作答图、印刷与手写混合图、作废草图、误标文字/公式或无法确定）和内容角色。判断时会同时发送彩色题干上下文、题干视图和可用的 detector 图块；彩色上下文用于区分印刷、蓝色学生笔迹与批改笔迹，detector 只负责定位，不能单独作为归属证据。印刷/混合图必须列出排版文字、规则虚线、网格或均匀线宽等印刷锚点；学生图/草图必须列出笔色、自由手线条和手写标签；只有出现明确划除、擦除或覆盖且 `visible_cancellation=true` 时，才接受作废草图分类，仅仅孤立、自由手绘、尺寸小或位于答题区都不够。学生绘制但参与正式推导的图必须标为保留的解题图。当前审计版本仅记录证据：本地门控无条件保留所有已有 `<插图>`，缺失、矛盾或低证据分类一律回退 `uncertain`，不会授权模型删图。`keep` 在结果 JSON 和 Markdown 中保持 API1 答案，不再自动增加 LaTeX 定界符。所有小问均为 `no_answer` 时按一个无支持答案状态评分，不得因重复占位符产生幻觉。每次 API2 的评分候选都来自对应的门控后 `api2_runs/run_XX/result.json`。
