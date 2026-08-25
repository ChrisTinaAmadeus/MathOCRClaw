# Benchmark

这个 benchmark 用同一套金标准分别评分两阶段输出：

1. API1 按 `prompts/extract_v2.txt` 直接生成 `api_markdown/pageXX.md`，得到当前模型基准分；
2. API2 综合初稿、原图与本地证据后的 `agent_outputs/result.json`，得到工作流分；
3. 报告给出两者差值，以及结构、题干、手写答案、状态、漏识别和幻觉等子指标。

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

输出目录包含 `report.json`、`report.md` 和逐页报告。模型、超时、token 上限、布局设备、缓存和估算单价均可通过 `--help` 配置。API 请求的实际次数、token 和延迟由工作流记录；若服务端不返回 usage，对应 token 字段为 `null`。

API1 与 API2 的上传/读取超时默认均为 360 秒，分别可用 `--baseline-timeout` 和 `--review-timeout` 覆盖。例如对超大图片可使用 `--baseline-timeout 600 --review-timeout 600`。

默认打分器遵循 [EVALUATION.md](EVALUATION.md) 的确定性聚合框架，但不额外调用裁判模型：语义和公式等价使用本地可复现的近似算法，裁判调用数固定为 0。这样不会把评测成本混入“两次 OCR API”的工作流成本；将来接入固定裁判模型时可以保留相同的报告结构。

第一次 API 的提示词不在工作流代码中复制维护，而是直接读取 [`prompts/extract_v2.txt`](prompts/extract_v2.txt)。其 Markdown 响应由本地解析器生成同目录 JSON 元数据，解析过程不调用模型。该提示词依据 32 页金标准统一了题号、`### 手写答案`、LaTeX、`<插图>` 和拒识标记，并要求彻底忽略所有被划去的作废内容。

两阶段使用同一份金标准语义：选择题只保留最终选项，填空题只保留最终值/公式和必要单位，解答题保留按小问组织的完整正式过程。选择、填空的推算草稿、排除痕迹和图上批注不会进入候选答案。API2 优先使用 `answer_parts[].final_answer` 生成评分候选，同时通过 `question_type` 和 `section_heading_before` 固定题型与章节边界；最终展示的 Markdown 也由这些结构化字段确定性渲染，以保证评分内容和用户实际看到的内容一致。
