# MessToClean 四条启动命令

默认会一次性处理 `workflow\images` 里的所有图片。你现在只放一张图，就只跑这一张；以后放多张图，就会批量全跑。

Stage 3 需要整页 Markdown，文件名要和图片名一致：

```text
workflow\images\page_0001.jpg
workflow\stage3_page_md\page_0001.md
```

## 1. 完整运行 Stage 1

RF-DETR 题目框检测 + PP-DocLayout 版面检测：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage1.ps1
```

## 2. 完整运行 Stage 2

读取 Stage 1 结果，完成阅读顺序恢复和题图匹配：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage2.ps1
```

## 3. 完整运行 Stage 3

读取 Stage 2 结果和 `workflow\stage3_page_md` 里的 Markdown，调用 `.env.local` 中配置的模型校对：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage3.ps1
```

默认 Stage 3 使用快速验证模式：关闭 crop-qno、figure 检查和文本修复器，用于快速确认链路跑通。需要完整模式时运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage3.ps1 -Full
```

## 4. 完整运行 Stage 1-3

一条命令跑完整流程：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all.ps1
```

## 默认输入输出

```text
workflow\images             输入图片
workflow\stage1_rfdetr      Stage 1 RF-DETR 输出
workflow\stage1_doclayout   Stage 1 DocLayout 输出
workflow\stage2_match       Stage 2 输出
workflow\stage3_page_md     Stage 3 输入 Markdown
workflow\stage3_out         Stage 3 输出
```

API key 和模型配置在 `.env.local`，当前默认模型是：

```text
MTC_VLM_MODEL=qwen3.7-plus
```
