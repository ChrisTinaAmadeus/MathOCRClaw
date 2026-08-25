# MathOCRClaw Benchmark Report

- Run ID: `20260825-172850`
- Model: `qwen3.7-plus`
- Pages: 1/1
- Baseline (API1): **84.36**
- Workflow (API2): **84.36**
- Gain: **+0.00**

| Page | API1 baseline | API2 workflow | Gain | Omission Δ | Hallucination Δ |
|---|---:|---:|---:|---:|---:|
| page10 | 84.36 | 84.36 | +0.00 | +0.000 | +0.000 |

## Format diagnostics

| Page | Gold LaTeX spans | API1 LaTeX spans | API2 LaTeX spans |
|---|---:|---:|---:|
| page10 | 22 | 25 | 25 |

## API statistics

- Logical workflow calls: 2
- Actual network requests: 2
- Successful / failed requests: 2 / 0
- Prompt / completion / total tokens: 7049 / 10561 / 17610
- API latency: 198.55s
- Estimated cost: not configured

## Evaluator

The default scorer is deterministic and local: question alignment, CER/bigram/critical-span text scoring, normalized LaTeX soft alignment, answer-type scoring, state Macro-F1, and hallucination penalty. It makes no additional judge-model calls.
