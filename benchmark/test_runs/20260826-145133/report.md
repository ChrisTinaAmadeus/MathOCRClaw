# MathOCRClaw Benchmark Report

- Run ID: `20260826-145133`
- Model: `qwen3.7-plus`
- API1 timeout: 360s
- API2 timeout: 360s
- Pages: 32/32
- Baseline (API1): **85.65**
- Workflow (API2): **83.99**
- Gain: **-1.67**

| Page | API1 baseline | API2 workflow | Gain | Omission Δ | Hallucination Δ |
|---|---:|---:|---:|---:|---:|
| page01 | 85.17 | 82.03 | -3.14 | +0.000 | +0.000 |
| page02 | 77.65 | 81.11 | +3.47 | +0.000 | +0.000 |
| page03 | 96.37 | 96.37 | +0.00 | +0.000 | +0.000 |
| page04 | 92.30 | 92.30 | +0.00 | +0.000 | +0.000 |
| page05 | 83.52 | 82.72 | -0.80 | +0.000 | +0.000 |
| page06 | 85.50 | 85.50 | +0.00 | +0.000 | +0.000 |
| page07 | 73.55 | 80.42 | +6.88 | -0.111 | +0.000 |
| page08 | 80.82 | 80.82 | +0.00 | +0.000 | +0.000 |
| page09 | 87.09 | 87.09 | +0.00 | +0.000 | +0.000 |
| page10 | 84.76 | 84.76 | +0.00 | +0.000 | +0.000 |
| page11 | 86.97 | 79.72 | -7.26 | +0.000 | +0.000 |
| page12 | 78.75 | 70.70 | -8.05 | +0.000 | +0.000 |
| page13 | 75.99 | 71.91 | -4.08 | +0.000 | +0.000 |
| page14 | 87.08 | 87.77 | +0.68 | +0.000 | +0.000 |
| page15 | 83.66 | 83.66 | +0.00 | +0.000 | +0.000 |
| page16 | 83.48 | 91.94 | +8.46 | -0.071 | +0.000 |
| page17 | 94.63 | 94.78 | +0.15 | +0.000 | +0.000 |
| page18 | 96.07 | 96.26 | +0.19 | +0.000 | +0.000 |
| page19 | 87.92 | 60.91 | -27.01 | -0.200 | +0.400 |
| page20 | 67.98 | 67.28 | -0.70 | +0.000 | +0.000 |
| page21 | 65.54 | 65.54 | +0.00 | +0.000 | +0.000 |
| page22 | 78.39 | 78.39 | +0.00 | +0.000 | +0.000 |
| page23 | 99.68 | 99.68 | +0.00 | +0.000 | +0.000 |
| page24 | 84.96 | 85.30 | +0.34 | +0.000 | +0.000 |
| page25 | 94.18 | 94.18 | +0.00 | +0.000 | +0.000 |
| page26 | 88.89 | 88.89 | +0.00 | +0.000 | +0.000 |
| page27 | 83.76 | 89.45 | +5.69 | +0.000 | +0.000 |
| page28 | 92.01 | 85.34 | -6.66 | +0.000 | +0.000 |
| page29 | 86.66 | 83.70 | -2.95 | +0.000 | +0.000 |
| page30 | 97.46 | 97.46 | +0.00 | +0.000 | +0.000 |
| page31 | 85.57 | 83.81 | -1.77 | +0.000 | +0.000 |
| page32 | 94.58 | 77.74 | -16.84 | +0.000 | +0.000 |

## Format diagnostics

| Page | Gold LaTeX spans | API1 LaTeX spans | API2 LaTeX spans |
|---|---:|---:|---:|
| page01 | 45 | 36 | 37 |
| page02 | 56 | 52 | 52 |
| page03 | 50 | 41 | 41 |
| page04 | 43 | 43 | 43 |
| page05 | 30 | 23 | 23 |
| page06 | 41 | 36 | 36 |
| page07 | 45 | 61 | 61 |
| page08 | 83 | 82 | 82 |
| page09 | 48 | 50 | 50 |
| page10 | 22 | 25 | 25 |
| page11 | 59 | 61 | 61 |
| page12 | 70 | 41 | 41 |
| page13 | 83 | 70 | 74 |
| page14 | 67 | 60 | 61 |
| page15 | 23 | 19 | 19 |
| page16 | 97 | 98 | 99 |
| page17 | 89 | 90 | 91 |
| page18 | 47 | 54 | 55 |
| page19 | 50 | 48 | 51 |
| page20 | 124 | 108 | 113 |
| page21 | 59 | 62 | 62 |
| page22 | 64 | 67 | 67 |
| page23 | 62 | 63 | 63 |
| page24 | 104 | 104 | 104 |
| page25 | 80 | 80 | 80 |
| page26 | 50 | 45 | 45 |
| page27 | 86 | 87 | 87 |
| page28 | 104 | 105 | 98 |
| page29 | 53 | 37 | 38 |
| page30 | 36 | 36 | 36 |
| page31 | 25 | 28 | 28 |
| page32 | 39 | 36 | 20 |

## API statistics

- Logical workflow calls: 64
- Actual network requests: 64
- Successful / failed requests: 64 / 0
- Prompt / completion / total tokens: 451352 / 489127 / 940479
- API latency: 9377.98s
- Estimated cost: not configured

## Evaluator

The default scorer is deterministic and local: question alignment, CER/bigram/critical-span text scoring, normalized LaTeX soft alignment, answer-type scoring, state Macro-F1, and hallucination penalty. It makes no additional judge-model calls.
