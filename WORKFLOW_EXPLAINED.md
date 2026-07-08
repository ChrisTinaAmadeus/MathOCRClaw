# MessToClean 工作流说明：以 `page_0001.jpg` 为例

这份文档用当前这张图片解释整个流程。目标是让不熟悉 OCR 的人也能知道：每个阶段输入什么、做什么、输出什么、哪里会调用外部模型，以及这次结果暴露了哪些问题。

## 一句话总览

MessToClean 不是一个“直接把图片 OCR 成 Markdown”的单一步骤工具。它更像一个三段式流水线：

```text
图片
  -> Stage 1：用本地检测模型找出题目框、版面框、图片块
  -> Stage 2：用纯代码把题目框排序，并把题目和图片块匹配起来
  -> Stage 3：拿“外部已有的整页 Markdown”与题目框截图做证据校验和修正
  -> 校对后的 Markdown
```

你的理解基本正确：前两个阶段是本地代码和本地模型推理，不调用 DashScope API。第三阶段才会调用 `qwen3.7-plus`。

需要特别注意：Stage 3 不是从零 OCR 整页图片。它需要你先提供一个整页 Markdown，例如：

```text
workflow\stage3_page_md\page_0001.md
```

Stage 3 的工作是检查这个 Markdown 是否能被图片证据支持，并尽量修正或屏蔽不可靠内容。

## 输入图片

本次图片是：

```text
workflow\images\page_0001.jpg
```

这是一张真实拍摄的试卷页照片，存在几个典型困难：

- 不是扫描件，而是手机拍摄照片。
- 画面里有多张纸叠放、遮挡。
- 左右两边都有题目，且页面不是严格平整。
- 有大量手写痕迹，会干扰文字识别。
- 右侧还露出旁边试卷的边缘内容。

这些问题会影响题目框检测、阅读顺序恢复和 Markdown 校验。

## Stage 1：本地检测

运行命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage1.ps1
```

Stage 1 实际包含两件事。

### 1.1 RF-DETR 题目框检测

脚本：

```text
rfdetr_infer.py
```

输入：

```text
workflow\images\page_0001.jpg
checkpoint_best_total.pth
```

输出：

```text
workflow\stage1_rfdetr\rfdetr_infer_results.jsonl
workflow\stage1_rfdetr\overlay\page_0001_overlay.jpg
workflow\stage1_rfdetr\crops\page_0001\*.png
```

它做的事情：

- 在原图里找“题目区域”的矩形框。
- 给每个框一个置信度分数。
- 把每个题目框裁剪成单独的小图。
- 保存一张带框的可视化图，方便你肉眼检查检测是否准。

本次结果：

```text
检测到 6 个题目相关框
其中 4 个主要题目框：problem_solving_question
其中 2 个残缺/边缘题目框：partial_question
```

这里不调用外部 API。

### 1.2 PP-DocLayout 版面检测

脚本：

```text
doclayout_infer.py
```

输入：

```text
workflow\images\page_0001.jpg
```

输出：

```text
workflow\stage1_doclayout\json\page_0001.json
workflow\stage1_doclayout\vis_img\...
```

它做的事情：

- 检测页面中的版面元素。
- 尝试找出图片、图表、插图等区域。
- 输出版面检测 JSON 和可视化图。

本次结果：

```text
没有得到可用于后续匹配的 figure/image 区域
Stage 2 中 F=0
```

这不一定是错误，因为这张图主要是文字和手写推导，没有明显独立印刷图。

这里也不调用外部 API。

## Stage 2：排序与匹配

运行命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage2.ps1
```

脚本：

```text
python -m match.match
```

输入：

```text
workflow\stage1_rfdetr\rfdetr_infer_results.jsonl
workflow\stage1_doclayout\json\page_0001.json
workflow\images\page_0001.jpg
```

输出：

```text
workflow\stage2_match\page_0001\match.json
workflow\stage2_match\page_0001\questions\...\question.png
workflow\stage2_match\page_0001\viz\...
```

它做的事情：

- 读取 Stage 1 找到的题目框。
- 根据题目框位置判断阅读顺序。
- 如果页面有图，会把图片块匹配到对应题目。
- 为 Stage 3 准备结构化的 `match.json` 和题目截图。

本次结果：

```text
Q=6
F=0
```

意思是：

- `Q=6`：Stage 2 接收并排序了 6 个题目相关框。
- `F=0`：没有可匹配的图片块。

`match.json` 里有一个很重要的字段：

```text
reading_order: [2, 3, 4, 1, 0, 5]
```

它表示程序认为题目框的阅读顺序是：

```text
先读左页上方 -> 左页下方 -> 右页上方 -> 右页中部 -> 右页下部 -> 最右边残缺内容
```

这里仍然不调用外部 API。

## Stage 3：证据校验与修正

运行快速验证模式：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage3.ps1
```

运行完整模式：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage3.ps1 -Full
```

输入：

```text
workflow\stage2_match\page_0001\match.json
workflow\stage2_match\page_0001\questions\...\question.png
workflow\stage3_page_md\page_0001.md
```

输出：

```text
workflow\stage3_out\page_0001\page_0001_proofread.md
workflow\stage3_out\page_0001\page_0001_report.json
workflow\stage3_out\_cache.json
```

它做的事情：

- 读取你提供的整页 Markdown。
- 把 Markdown 按题号切成题目块。
- 把 Markdown 题目块和 Stage 2 的题目截图对齐。
- 调用 `qwen3.7-plus` 判断：这段 Markdown 是否能在对应截图中找到证据。
- 如果证据不足，就输出 `[HALLUCINATION]` 或 `[UNREADABLE]`。
- 如果启用完整模式，还会做题号识别、图片筛选、文本修复等更慢的步骤。

这里会调用外部 API。

本次快速模式中，每个 VLM 请求耗时大约 5 到 9 秒。之前 Stage 3 很慢，是因为完整配置会产生更多 VLM 请求，并且每个请求都有网络等待和重试成本。

## 快速模式和完整模式的区别

快速模式用于确认链路能跑通：

```text
关闭 crop-qno
关闭 figure 检查
关闭文本修复器
保留基本证据校验
```

完整模式用于正式实验：

```text
开启 crop-qno：让 VLM 从裁剪图中读题号
开启文本修复器：尝试修正 Markdown
开启 figure 检查：筛选题图关系
更多 API 请求，更慢
```

所以：

```text
快速模式：更快，适合冒烟测试
完整模式：更慢，适合正式结果
```

## 这次输出为什么只保留了题目 16

当前输出是：

```text
16. 已知集合 ...

17. [HALLUCINATION]

18. [HALLUCINATION]

19. [HALLUCINATION]
```

这暴露了几个问题。

### 问题 1：我写的 `page_0001.md` 不是模型 OCR 结果

这次的 `workflow\stage3_page_md\page_0001.md` 是为了验证流程，由我根据图片内容手动写出的示例 Markdown。

它不是当前 pipeline 自动 OCR 出来的。

所以不能把它理解成“模型已经完整识别出了 Markdown”。更准确地说：

```text
这份 Markdown 是人为准备的 Stage 3 输入，用来测试 Stage 3 能否跑通。
```

真实实验中，你需要把上游 OCR/MLLM 生成的整页 Markdown 放到这个目录。

### 问题 2：Stage 3 不会凭空补出 Markdown 中缺失的题

Stage 3 是校对器，不是完整 OCR 入口。

如果输入 Markdown 缺题，它通常不会自动把整页所有题都重新 OCR 出来。它只会处理 Markdown 里已经存在的题目块。

所以如果上游 Markdown 只有“题目 1”或只有部分题目，最终输出也很可能只有那部分题目，或者把不可靠内容 mask 掉。

### 问题 3：第一题块曾经被当成普通文本块

报告里显示：

```text
n_blocks = 4
n_md_qblocks = 3
```

这说明 Markdown 被切成了 4 个块，但只有 3 个被识别为“题目块”。

也就是说，题目 16 当时没有进入正式题目校验流程，而是被当成普通文本保留了。

原因很可能是 Markdown 文件第一行带了 UTF-8 BOM 或题号解析规则对第一行不够鲁棒。这个问题已经在代码里加了处理：

```text
clean_page_md 会先去掉开头的 BOM
```

### 问题 4：快速模式没有读取截图中的真实题号

快速模式关闭了：

```text
--use-crop-qno
```

所以 Stage 3 对齐 Markdown 和题目截图时，主要依赖 Stage 2 的阅读顺序，而不是“截图里实际写着第几题”。

这张图是复杂实拍图，左右页、遮挡、残缺内容都存在。只靠阅读顺序对齐容易错位。

报告里可以看到：

```text
crop_qno: null
mode: offset_search
```

意思是：它没有从截图中读到题号，只能用位置顺序猜。

### 问题 5：Markdown 内容和裁剪图证据不完全匹配

报告中 17、18、19 的校验结果是：

```text
v_strict: N
v_lenient: N
action: mask_n
```

意思是：VLM 看了对应的题目截图后，认为 Markdown 里的文字不能被图片证据支持。

这可能来自两个原因：

- Markdown 是我手动写的简化文本，和图片中真实题干不完全一致。
- Stage 3 快速模式下对齐错位，拿错截图去验证了某一道题的 Markdown。

因此输出 `[HALLUCINATION]` 是符合当前逻辑的：证据不足，就不要冒险保留。

## 这次结果暴露出的核心问题

### 1. 当前 pipeline 缺少真正的“整页 OCR/MLLM 生成 Markdown”步骤

README 里也写了 Stage 3 的输入是：

```text
整页 Markdown：由上游 OCR/MLLM 流程生成
```

但当前仓库主要提供的是：

```text
Stage 1 检测
Stage 2 匹配
Stage 3 校对
```

它没有把“整页照片直接转 Markdown”的完整 OCR 模块打包进来。

所以真正实验前，需要先确定整页 Markdown 从哪里来。

### 2. 上游 Markdown 质量决定最终输出上限

Stage 3 可以修补、校验、屏蔽，但它不是万能重建器。

如果输入 Markdown：

- 缺题
- 题号错
- 内容和图片不一致
- 只识别出第一题

那么最终输出也会受影响。

### 3. 复杂拍摄图会让阅读顺序和题号对齐变难

这张图不是干净扫描件，存在遮挡和左右页混杂。Stage 2 能检测到 6 个框，但这些框不一定天然对应 Markdown 里的 16、17、18、19。

如果要更稳，需要完整模式的 crop-qno：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage3.ps1 -Full
```

完整模式会更慢，但能让 VLM 直接读每个题目截图上的题号，从而减少错位。

### 4. 当前验证 Markdown 太简化

我写的验证 Markdown 是为了跑通链路，不是严谨 OCR 结果。比如题目 18、19 的原图中有更多细节、公式和条件，示例 Markdown 只写了摘要式内容。

证据校验器看到摘要文本和截图不一致，就会倾向于判定为不可靠。

## 推荐的正式实验流程

正式实验建议这样跑：

1. 放入图片：

```text
workflow\images\page_0001.jpg
```

2. 跑 Stage 1：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage1.ps1
```

3. 跑 Stage 2：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage2.ps1
```

4. 用你的 OCR/MLLM 生成整页 Markdown：

```text
workflow\stage3_page_md\page_0001.md
```

5. 快速检查链路：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage3.ps1
```

6. 正式完整校对：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage3.ps1 -Full
```

## 怎么判断结果是不是好

重点看两个文件：

```text
workflow\stage3_out\page_0001\page_0001_proofread.md
workflow\stage3_out\page_0001\page_0001_report.json
```

如果 Markdown 输出大量：

```text
[HALLUCINATION]
[UNREADABLE]
```

通常说明：

- 上游 Markdown 和图片不匹配；
- 题目框和 Markdown 对齐错位；
- 图片太模糊或被遮挡；
- 快速模式关闭了题号识别，导致错位；
- 或者 OCR 本身漏题。

如果 report 里有大量：

```text
crop_qno: null
v_strict: N
v_lenient: N
```

说明模型没有足够证据保留对应文本。

## 当前结论

这次验证证明：

```text
Stage 1 能检测题目框
Stage 2 能生成 match.json
Stage 3 能调用 qwen3.7-plus 并输出 proofread.md 和 report.json
```

也暴露出：

```text
当前缺少稳定的整页 OCR/MLLM Markdown 生成步骤
快速模式下题号对齐不够稳
示例 Markdown 不是图片的严格 OCR 结果
复杂实拍图对检测、排序和校验都有挑战
```

因此，下一步最关键的不是再调 Stage 1/2，而是补齐或接入“整页图片 -> Markdown”的上游 OCR/MLLM 步骤，并让它生成和图片尽量一致的 `page_0001.md`。
