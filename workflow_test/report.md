# MathOCRClaw Benchmark Report

- Run ID: `20260820-162708`
- Model: `qwen3.7-plus`
- Pages: 32/32
- Baseline (API1): **83.06**
- Workflow (API2): **81.30**
- Gain: **-1.76**

| Page | API1 baseline | API2 workflow | Gain | Omission Δ | Hallucination Δ |
|---|---:|---:|---:|---:|---:|
| page01 | 81.87 | 81.99 | +0.12 | +0.000 | +0.000 |
| page02 | 82.67 | 82.67 | +0.00 | +0.000 | +0.000 |
| page03 | 95.86 | 82.97 | -12.89 | +0.000 | +0.000 |
| page04 | 89.96 | 94.71 | +4.75 | +0.000 | +0.000 |
| page05 | 87.58 | 80.98 | -6.60 | +0.000 | +0.000 |
| page06 | 85.77 | 85.77 | +0.00 | +0.000 | +0.000 |
| page07 | 74.02 | 76.23 | +2.20 | +0.000 | +0.000 |
| page08 | 78.46 | 66.54 | -11.92 | +0.000 | +0.000 |
| page09 | 58.76 | 70.26 | +11.50 | +0.000 | +0.000 |
| page10 | 85.59 | 62.74 | -22.84 | +0.000 | +0.000 |
| page11 | 73.39 | 71.69 | -1.71 | +0.000 | +0.000 |
| page12 | 83.46 | 84.34 | +0.88 | +0.000 | +0.000 |
| page13 | 71.43 | 71.54 | +0.11 | +0.000 | +0.000 |
| page14 | 88.22 | 88.22 | +0.00 | +0.000 | +0.000 |
| page15 | 83.66 | 83.89 | +0.23 | +0.000 | +0.000 |
| page16 | 88.68 | 88.75 | +0.07 | +0.000 | +0.000 |
| page17 | 94.47 | 94.47 | +0.00 | +0.000 | +0.000 |
| page18 | 98.96 | 82.71 | -16.25 | +0.000 | +0.000 |
| page19 | 71.91 | 85.77 | +13.87 | +0.000 | -0.200 |
| page20 | 65.83 | 67.56 | +1.73 | +0.000 | +0.000 |
| page21 | 63.03 | 65.81 | +2.78 | +0.000 | +0.000 |
| page22 | 72.70 | 65.06 | -7.64 | +0.000 | +0.000 |
| page23 | 83.36 | 81.18 | -2.18 | +0.000 | +0.000 |
| page24 | 84.79 | 87.82 | +3.03 | +0.000 | +0.000 |
| page25 | 90.73 | 91.09 | +0.36 | +0.000 | +0.000 |
| page26 | 88.89 | 82.54 | -6.35 | +0.000 | +0.000 |
| page27 | 85.28 | 73.17 | -12.11 | +0.000 | +0.000 |
| page28 | 93.25 | 93.29 | +0.03 | +0.000 | +0.000 |
| page29 | 83.71 | 86.09 | +2.38 | +0.000 | +0.000 |
| page30 | 97.70 | 97.70 | +0.00 | +0.000 | +0.000 |
| page31 | 83.41 | 83.41 | +0.00 | +0.000 | +0.000 |
| page32 | 90.60 | 90.60 | +0.00 | +0.000 | +0.000 |

## Format diagnostics

| Page | Gold LaTeX spans | API1 LaTeX spans | API2 LaTeX spans |
|---|---:|---:|---:|
| page01 | 45 | 36 | 37 |
| page02 | 56 | 53 | 53 |
| page03 | 50 | 49 | 49 |
| page04 | 43 | 43 | 43 |
| page05 | 30 | 27 | 27 |
| page06 | 41 | 41 | 41 |
| page07 | 45 | 60 | 61 |
| page08 | 83 | 80 | 81 |
| page09 | 48 | 48 | 48 |
| page10 | 22 | 25 | 25 |
| page11 | 59 | 55 | 53 |
| page12 | 70 | 52 | 53 |
| page13 | 83 | 75 | 75 |
| page14 | 67 | 64 | 64 |
| page15 | 23 | 19 | 19 |
| page16 | 97 | 92 | 92 |
| page17 | 89 | 89 | 89 |
| page18 | 47 | 48 | 62 |
| page19 | 50 | 47 | 50 |
| page20 | 124 | 126 | 134 |
| page21 | 59 | 69 | 69 |
| page22 | 64 | 78 | 90 |
| page23 | 62 | 77 | 77 |
| page24 | 104 | 117 | 105 |
| page25 | 80 | 80 | 80 |
| page26 | 50 | 45 | 48 |
| page27 | 86 | 87 | 84 |
| page28 | 104 | 102 | 102 |
| page29 | 53 | 35 | 52 |
| page30 | 36 | 36 | 36 |
| page31 | 25 | 28 | 28 |
| page32 | 39 | 37 | 37 |

## API statistics

- Logical workflow calls: 64
- Actual network requests: 62
- Successful / failed requests: 62 / 0
- Prompt / completion / total tokens: 427294 / 482624 / 909918
- API latency: 8855.98s
- Estimated cost: not configured

## Evaluator

The default scorer is deterministic and local: question alignment, CER/bigram/critical-span text scoring, normalized LaTeX soft alignment, answer-type scoring, state Macro-F1, and hallucination penalty. It makes no additional judge-model calls.
