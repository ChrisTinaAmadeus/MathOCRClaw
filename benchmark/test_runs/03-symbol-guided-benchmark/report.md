# MathOCRClaw Benchmark Report

- Run ID: `20260827-163923`
- Model: `qwen3.7-plus`
- API1 timeout: 360s
- API2 timeout: 360s
- Pages: 31/32
- Baseline (API1): **84.87**
- Workflow (API2): **84.64**
- Gain: **-0.23**

| Page | API1 baseline | API2 workflow | Gain | Omission Δ | Hallucination Δ |
|---|---:|---:|---:|---:|---:|
| page01 | 81.60 | 81.29 | -0.31 | +0.000 | +0.000 |
| page02 | 81.29 | 77.45 | -3.84 | +0.000 | +0.000 |
| page03 | 97.20 | 97.20 | +0.00 | +0.000 | +0.000 |
| page04 | 92.91 | 92.91 | +0.00 | +0.000 | +0.000 |
| page05 | 85.48 | 85.48 | +0.00 | +0.000 | +0.000 |
| page06 | 85.58 | 85.58 | +0.00 | +0.000 | +0.000 |
| page07 | 76.53 | 76.53 | +0.00 | +0.000 | +0.000 |
| page08 | 81.78 | 79.17 | -2.62 | +0.000 | +0.000 |
| page09 | 85.57 | 85.57 | +0.00 | +0.000 | +0.000 |
| page10 | 85.07 | 85.08 | +0.01 | +0.000 | +0.000 |
| page11 | 73.43 | 80.69 | +7.26 | +0.000 | +0.000 |
| page12 | 58.30 | 58.30 | +0.00 | +0.000 | +0.000 |
| page13 | ERROR | ERROR | — | — | — |
| page14 | 86.94 | 86.99 | +0.05 | +0.000 | +0.000 |
| page15 | 83.89 | 83.89 | +0.00 | +0.000 | +0.000 |
| page16 | 91.62 | 91.62 | +0.00 | +0.000 | +0.000 |
| page17 | 92.73 | 91.03 | -1.70 | +0.000 | +0.000 |
| page18 | 95.07 | 95.07 | +0.00 | +0.000 | +0.000 |
| page19 | 87.82 | 87.85 | +0.03 | +0.000 | +0.000 |
| page20 | 66.35 | 65.12 | -1.23 | +0.000 | +0.000 |
| page21 | 64.34 | 63.52 | -0.81 | +0.000 | +0.000 |
| page22 | 68.52 | 68.52 | +0.00 | +0.000 | +0.000 |
| page23 | 99.02 | 99.04 | +0.01 | +0.000 | +0.000 |
| page24 | 88.75 | 88.50 | -0.26 | +0.000 | +0.000 |
| page25 | 93.06 | 93.06 | +0.00 | +0.000 | +0.000 |
| page26 | 89.14 | 89.14 | +0.00 | +0.000 | +0.000 |
| page27 | 84.78 | 82.18 | -2.59 | +0.000 | +0.000 |
| page28 | 94.88 | 93.71 | -1.18 | +0.000 | +0.000 |
| page29 | 83.77 | 83.77 | +0.00 | +0.000 | +0.000 |
| page30 | 94.11 | 94.11 | +0.00 | +0.000 | +0.000 |
| page31 | 85.26 | 85.26 | +0.00 | +0.000 | +0.000 |
| page32 | 96.15 | 96.15 | +0.00 | +0.000 | +0.000 |

## Format diagnostics

| Page | Gold LaTeX spans | API1 LaTeX spans | API2 LaTeX spans |
|---|---:|---:|---:|
| page01 | 45 | 35 | 35 |
| page02 | 56 | 50 | 51 |
| page03 | 50 | 44 | 44 |
| page04 | 43 | 43 | 43 |
| page05 | 30 | 24 | 24 |
| page06 | 41 | 36 | 36 |
| page07 | 45 | 65 | 65 |
| page08 | 83 | 77 | 77 |
| page09 | 48 | 43 | 43 |
| page10 | 22 | 25 | 25 |
| page11 | 59 | 58 | 58 |
| page12 | 70 | 54 | 54 |
| page14 | 67 | 59 | 59 |
| page15 | 23 | 19 | 19 |
| page16 | 97 | 99 | 99 |
| page17 | 89 | 90 | 90 |
| page18 | 47 | 59 | 59 |
| page19 | 50 | 51 | 51 |
| page20 | 124 | 100 | 100 |
| page21 | 59 | 66 | 66 |
| page22 | 64 | 67 | 67 |
| page23 | 62 | 65 | 65 |
| page24 | 104 | 101 | 101 |
| page25 | 80 | 77 | 77 |
| page26 | 50 | 56 | 56 |
| page27 | 86 | 87 | 88 |
| page28 | 104 | 102 | 102 |
| page29 | 53 | 34 | 34 |
| page30 | 36 | 36 | 36 |
| page31 | 25 | 27 | 27 |
| page32 | 39 | 37 | 37 |

## API statistics

- Logical workflow calls: 62
- Actual network requests: 62
- Successful / failed requests: 62 / 0
- Prompt / completion / total tokens: 523075 / 423726 / 946801
- API latency: 8150.26s
- Estimated cost: not configured

## Evaluator

The default scorer is deterministic and local: question alignment, CER/bigram/critical-span text scoring, normalized LaTeX soft alignment, answer-type scoring, state Macro-F1, and hallucination penalty. It makes no additional judge-model calls.

## Failures

- `page13`: FileNotFoundError: [Errno 2] No such file or directory: '/home/puppeteer/MathOCRClaw/benchmark/test_runs/20260827-163923/workflow/page13/agent_outputs/result.json'
