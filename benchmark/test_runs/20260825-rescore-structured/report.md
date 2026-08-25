# MathOCRClaw Benchmark Report

- Run ID: `20260825-172440`
- Model: `qwen3.7-flash`
- Pages: 2/2
- Baseline (API1): **90.57**
- Workflow (API2): **92.18**
- Gain: **+1.61**

| Page | API1 baseline | API2 workflow | Gain | Omission Δ | Hallucination Δ |
|---|---:|---:|---:|---:|---:|
| page03 | 95.86 | 95.86 | +0.00 | +0.000 | +0.000 |
| page27 | 85.28 | 88.49 | +3.21 | +0.000 | +0.000 |

## Format diagnostics

| Page | Gold LaTeX spans | API1 LaTeX spans | API2 LaTeX spans |
|---|---:|---:|---:|
| page03 | 50 | 49 | 49 |
| page27 | 86 | 87 | 87 |

## API statistics

- Logical workflow calls: 4
- Actual network requests: 4
- Successful / failed requests: 4 / 0
- Prompt / completion / total tokens: 37606 / 37812 / 75418
- API latency: 693.67s
- Estimated cost: not configured

## Evaluator

The default scorer is deterministic and local: question alignment, CER/bigram/critical-span text scoring, normalized LaTeX soft alignment, answer-type scoring, state Macro-F1, and hallucination penalty. It makes no additional judge-model calls.
