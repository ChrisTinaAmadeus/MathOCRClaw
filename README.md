# MathOCRClaw

MathOCRClaw is a minimal exam-OCR agent built on top of the MessToClean-style reconstruction pipeline. It is designed for real-world exam page photos where printed questions and student handwriting appear together.

The current goal is not to be a perfect handwriting OCR system yet. The first working version provides a complete, auditable workflow:

```text
exam image
  -> whole-page VLM baseline
  -> question-region and layout evidence collection
  -> printed-question verification
  -> handwritten-answer evidence extraction
  -> verified JSON / Markdown result
```

## What It Does

- Calls a multimodal API on the whole page to produce a baseline structured OCR result.
- Reuses local RF-DETR question detection to locate question regions.
- Reuses PP-DocLayout and the existing matching logic to recover reading order and question crops.
- Verifies printed question text against cropped image evidence before keeping it.
- Extracts visible student handwriting from each question crop.
- Verifies the extracted handwriting against image evidence.
- Writes final auditable outputs for downstream grading, review, or dataset building.

## Current Status

This repository currently contains the minimal working agent loop:

```text
agent/simple_agent.py
scripts/run_agent.ps1
```

The answer-recognition branch is intentionally simple: it uses the multimodal API on each question crop, then asks the model to verify whether the extracted handwritten answer is visually supported. Future versions can replace this branch with a dedicated handwriting OCR model or a trained answer-area detector.

## Repository Layout

```text
MathOCRClaw/
├─ agent/
│  └─ simple_agent.py          # Minimal end-to-end agent workflow
├─ proofread/                  # Evidence-grounded question verification pipeline
├─ match/                      # Reading order and question/figure matching
├─ scripts/
│  ├─ run_agent.ps1            # One-command agent entrypoint
│  ├─ run_stage1.ps1           # RF-DETR + DocLayout inference
│  ├─ run_stage2.ps1           # Matching and crop generation
│  └─ run_stage3.ps1           # Question proofread / verification
├─ rfdetr_infer.py             # Local question detector inference
├─ doclayout_infer.py          # Local layout detector inference
├─ proofread_page_v6_6.py      # Existing proofread CLI wrapper
├─ environment.yml             # Conda environment definition
└─ requirements-local.txt      # Local Python dependency snapshot
```

Generated files are written under `workflow/` and are ignored by git.

## Setup

Create the Conda environment:

```powershell
conda env create -f environment.yml
```

Or use the existing local environment if it has already been created:

```powershell
.\.conda\messtoclean\python.exe -m agent.simple_agent --help
```

Configure your multimodal API key in `.env.local` or in the shell environment:

```text
DASHSCOPE_API_KEY=your_api_key
MTC_VLM_MODEL=qwen3.7-plus
```

The default API base is DashScope OpenAI-compatible mode:

```text
https://dashscope.aliyuncs.com/compatible-mode/v1
```

You can override it with `--api-base` and `--model`.

## Model Weights

The local question detector expects an RF-DETR checkpoint such as:

```text
checkpoint_best_total.pth
```

This file is intentionally ignored by git because it is larger than GitHub's normal 100MB file limit. Keep it locally, download it separately, or manage it with Git LFS / release assets.

## Quick Start

Run the complete minimal agent on one page image:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_agent.ps1 -Image .\workflow\images\page_0001.jpg
```

If Stage 1 and Stage 2 outputs already exist and you want to reuse them:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_agent.ps1 -Image .\workflow\images\page_0001.jpg -SkipLayout
```

For a slower but more complete verification pass:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_agent.ps1 -Image .\workflow\images\page_0001.jpg -Full
```

## Outputs

By default, agent outputs are saved under:

```text
workflow/agent_out/<page_name>/
```

Important files:

```text
baseline.json          # Whole-page VLM baseline
baseline.md            # Baseline converted to Markdown for question verification
answer_evidence.json   # Handwritten-answer extraction and verification records
final_result.json      # Combined machine-readable result
final_result.md        # Human-readable summary
```

The existing proofread outputs are written under:

```text
workflow/stage3_out/<page_name>/
```

## Minimal Agent Contract

The final JSON currently contains:

```json
{
  "page": "page_0001",
  "image": "workflow/images/page_0001.jpg",
  "baseline": {},
  "verified_question_markdown": "...",
  "question_verification_report": {},
  "answer_evidence": [],
  "outputs": {}
}
```

Each `answer_evidence` item records the crop path, extracted handwriting, extraction status, visual evidence note, and a verification verdict:

```text
Y = supported by visible handwriting
N = not supported or contradicted by the crop
U = uncertain / unclear
```

## Development Checks

Compile-check the Python files:

```powershell
$files = rg --files -g "*.py" -g "!.conda/**" -g "!.cache/**" -g "!workflow/**" -g "!__pycache__/**"
.\.conda\messtoclean\python.exe -m py_compile $files
```

Check the agent CLI:

```powershell
.\.conda\messtoclean\python.exe -m agent.simple_agent --help
```

## GitHub Notes

Do not commit:

- `.env.local`
- `.conda/`
- `.cache/`
- `.paddlex/`
- `workflow/`
- `checkpoint_best_total.pth`
- other `*.pth`, `*.pdiparams`, `*.onnx`, or large model artifacts

Use Git LFS or GitHub Releases if model artifacts need to be distributed with the project.

## Roadmap

- Add a dedicated student-answer area detector.
- Add handwriting-specific OCR and math-expression recognition.
- Bind handwritten answers to question numbers more robustly.
- Add confidence calibration and human-review queues.
- Add batch processing and evaluation scripts.
