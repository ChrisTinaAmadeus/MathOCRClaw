# MessToClean 论文实验流程还原

本文档用于记录论文中的完整实验流程，重点解释“原始 Markdown / structured output 是从哪里来的”，以及它如何进入当前仓库中的 Stage 3。

## 1. 核心结论

当前仓库没有提供“整页图片 -> 原始 Markdown”的生成代码。

论文实验中，这一步应由被评测的 MLLM backbone 完成。也就是说，GPT、Qwen、Gemini、GLM 等模型先根据试卷图片生成一个 page-level candidate，论文中记作 `Mpage`，也可以理解为原始结构化文本或 Markdown 初稿。

MessToClean 随后使用检测框、阅读顺序和题目裁剪图作为 pixel-aligned evidence，对这个初稿进行验证、拒识和最小修补。

因此，论文实验中的流程不是：

```text
图片 -> RF-DETR -> match.json -> 自动变成 Markdown
```

而更接近：

```text
图片 -> MLLM 直接生成候选 Markdown
图片 -> RF-DETR/版面检测 -> 题目框、图框、阅读顺序、题图匹配
候选 Markdown + 题目级视觉证据 -> GVP 校验/修补 -> 最终结构化 Markdown
```

## 2. 论文明确描述的三段流程

论文把 MessToClean 描述为一个 evidence-driven structured reconstruction pipeline，主要包含三步。

### 2.1 Stage 1: Evidence Extraction

输入：

```text
Real-World Degraded Exam Image
```

处理：

```text
Two-stage fine-tuned RF-DETR
```

目标：

- 检测 question text blocks。
- 检测 figure regions。
- 在遮挡、手写、噪声较强的真实试卷照片中得到 pixel-aligned bounding boxes。

输出：

```text
question boxes
figure boxes
pixel-aligned evidence boxes
```

这一步对应当前仓库中的：

```text
rfdetr_infer.py
doclayout_infer.py
workflow/stage1_rfdetr/
workflow/stage1_doclayout/
```

### 2.2 Stage 2: Layout and Reading Order Reconstruction

输入：

```text
Stage 1 检测出的题目框和图框
```

处理：

- 判断单页/跨页布局。
- 恢复页面层级结构。
- 推断稳定的 global reading order。
- 对 question boxes 和 figure boxes 做全局匹配。

输出：

```text
Questionsordered
read_index
question-figure bindings
Iiq: question-level pixel-aligned evidence binding
```

其中 `Iiq` 表示第 `i` 道题对应的视觉证据，通常包括题目裁剪图和相关图像区域。

这一步对应当前仓库中的：

```text
match/
workflow/stage2_match/<page>/match.json
workflow/stage2_match/<page>/questions/.../question.png
```

### 2.3 Stage 3: Evidence-Constrained GVP Loop

论文中的 GVP 指：

```text
Generator
Verifier
Patcher
```

输入：

```text
page-level candidate Mpage
question-level evidence bindings Iiq
```

处理：

1. Generator 产生 page-level candidate `Mpage`。
2. Verifier 将每一道题的文本与对应裁剪图证据 `Iiq` 对照，输出判定 `v`。
3. Patcher 只在需要时触发，进行 whitelist-constrained minimal fixes。
4. 修补后再验证。

输出：

```text
structured Markdown
auditable edit logs
```

这一步对应当前仓库中的：

```text
proofread/
workflow/stage3_page_md/<page>.md
workflow/stage3_out/<page>/<page>_proofread.md
workflow/stage3_out/<page>/<page>_report.json
```

## 3. 原始 Markdown 在论文实验中的位置

论文图 2 中有如下关系：

```text
Multimodal Large Language Models (MLLMs)
        -> Generator
        -> Structed.md File
        -> Verifier / Patcher
```

同时论文第 3.3 节描述：

```text
Generator produces a page-level candidate Mpage.
Verifier checks each question against its cropped evidence Iiq.
Patcher is triggered only when needed.
```

因此，`Mpage` 就是 Stage 3 的待校正文稿。

在当前仓库中，它表现为：

```text
workflow/stage3_page_md/page_0001.md
```

但当前仓库没有提供自动生成该文件的上游脚本，所以本地第一次迷你实验中的 `page_0001.md` 是手写示例，不是论文实验中的真实 MLLM generator 输出。

## 4. 论文实验的批量运行方式

论文评测了 12 个 MLLM backbones，包括开源模型和闭源模型，例如：

```text
Qwen3-VL
GLM-4.5V / GLM-4.6V
GPT-5 / GPT-4o / GPT-4o-mini
Gemini-2.5 Pro / Flash
```

论文说使用 unified protocol，并且没有 backbone-specific prompt customization。

据此，完整批量实验可以精确拆成以下逻辑：

```text
for backbone in MLLM_BACKBONES:
  for page in test_pages:
    1. Direct one-shot:
       page image -> backbone -> candidate structured output / Mpage

    2. Evidence extraction:
       page image -> RF-DETR / layout detector -> question boxes + figure boxes

    3. Structure recovery:
       boxes -> reading order + question-figure alignment + crops

    4. MessToClean GVP:
       Mpage + question crops + figure bindings
       -> verifier
       -> patcher if needed
       -> final structured Markdown + audit log

    5. Evaluation:
       final prediction vs ground truth
       -> StemSim
       -> ImgSim
       -> refusal precision / recall / F1
```

## 5. Direct Baseline 与 MessToClean 的关系

论文中的 Direct one-shot baseline 指：

```text
整页图片 -> MLLM -> 结构化输出
```

这个输出本身可以直接参与评测，作为 baseline。

MessToClean 则不是替代这个初始生成步骤，而是在其后增加 evidence grounding、verification 和 patching：

```text
Direct output / Mpage
        +
question-level pixel evidence
        -> MessToClean
        -> repaired / rejected / audited output
```

所以从实验资源复用角度看，同一个 backbone 的 Direct 输出很可能同时扮演两个角色：

- Direct baseline 的预测结果。
- MessToClean GVP 的初始候选稿。

## 6. WholePage Ablation 的含义

论文还比较了 WholePage baseline。

WholePage 与 Ours 的区别不是 agent chain 不同，而是视觉证据粒度不同：

```text
Ours:
  每道题使用 question-level crop 和 alignment 证据

WholePage:
  每道题仍使用完整页面图作为证据
  不提供 question-level crops 或 alignment
```

论文结论是：即使使用相同 backbone 和相同 agent chain，WholePage 也明显弱于 question-level evidence。

这说明 MessToClean 的关键收益来自：

```text
题目级视觉证据
明确阅读顺序
题图绑定
可审计的局部校验与修补
```

而不仅仅是“多调用了几次模型”。

## 7. 当前仓库缺失的实验组件

当前仓库已经包含：

```text
Stage 1 检测推理
Stage 2 阅读顺序与题图匹配
Stage 3 对已有 Markdown 做证据校验和修补
```

当前仓库缺失：

```text
1. 批量调用 MLLM 生成 Mpage / 原始 Markdown 的脚本
2. 12 个 backbone 的批量调度配置
3. 论文实验使用的统一 prompt
4. Direct baseline 输出目录
5. FullAuditLog / JSONL 风格实验日志的完整实现
6. Ground truth 数据集与评测脚本
7. StemSim / ImgSim / refusal F1 的完整复现实验入口
```

因此，这个仓库更像是论文方法的核心 pipeline 子集，而不是完整论文实验复现包。

## 8. 如果要在当前仓库补齐论文式流程

建议新增一个 Stage 2.5：

```text
Stage 2.5: image_to_md
```

输入：

```text
workflow/images/<page>.jpg
```

输出：

```text
workflow/stage3_page_md/<page>.md
```

然后整体工作流变为：

```text
run_stage1.ps1
run_stage2.ps1
run_ocr_md.ps1
run_stage3.ps1
```

其中 `run_ocr_md.ps1` 调用 MLLM，把每张整页图生成候选 Markdown。

这样最符合论文结构，也最符合当前代码边界：Stage 3 继续只做校验和修补，不混入原始 OCR 生成逻辑。

## 9. 对本地 page_0001 实验的解释

当前本地文件：

```text
workflow/stage3_page_md/page_0001.md
```

不是论文式 `Generator` 产生的 `Mpage`，而是手写的流程测试稿。

因此第一次迷你实验只能说明：

```text
Stage 1 能检测题目框
Stage 2 能生成 match.json 和题目 crop
Stage 3 能读取 Markdown、调用 VLM、输出 proofread.md/report.json
```

不能说明：

```text
当前仓库已经具备完整整页 OCR 能力
```

下一步真正关键的是补齐：

```text
page image -> Mpage / candidate Markdown
```

也就是论文里的 Generator 初稿生成环节。
