# MathOCRClaw

[简体中文](README.zh-CN.md) · English

MathOCRClaw is a math-OCR agent for real-world exam photos. It recognizes printed questions and student handwriting, verifies predictions against aligned crops, and returns `U` instead of guessing when evidence is insufficient.

```text
exam photo
  → shadow normalization and red-ink removal
  → API-generated whole-page question Markdown
  → local detection, layout analysis, and question alignment
  → per-question handwriting extraction and verification
  → paired question-and-answer results
```

### Quick start

Requirements: Windows PowerShell, Conda, a DashScope/OpenAI-compatible multimodal API, and `checkpoint_best_total.pth` in the repository root.

```powershell
conda env create --prefix .\.conda\messtoclean -f environment.yml
```

Create a Git-ignored `.env.local`:

```dotenv
DASHSCOPE_API_KEY=your_api_key
MTC_VLM_MODEL=qwen3.7-plus
```

The input image may be anywhere locally; `input/` is a convenient Git-ignored location. Run the full workflow with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_agent.ps1 -Image .\input\page_0001.jpg -Full
```

Add `-SkipLayout` to reuse local detection and matching outputs. Without `-Full`, question-number reading, text patching, and question/figure checks are disabled to reduce API calls.

### Bench30 benchmark

The initial agent evaluation set contains **30 high-quality exam images plus Qwen3-VL baseline Markdown**. See [`benchmark/bench30/`](benchmark/bench30/README.md) for the manifest, official GT copies, baseline outputs, and evaluation summary. The full MathDoc dataset is not included in this repository.

### Output layout

User-facing runtime artifacts retain the original image and every processing stage (internal caches live under `.cache/`):

```text
workflow/
├─ image/                         # untouched original input images
├─ preprocessed/                  # normalized scans and preprocessing JSON
├─ api_markdown/                  # raw API question Markdown and response JSON
├─ code_outputs/                  # local code/model stages
│  ├─ rfdetr/<page_name>/         # page-scoped detection JSONL, crops, and visualizations
│  ├─ doclayout/<page_name>/      # page-scoped layout JSON and visualizations
│  └─ match/                      # reading order, matching JSON, and question crops
└─ agent_outputs/<page_name>/     # final agent output
   ├─ result.md                   # each question followed by its handwriting
   ├─ result.json                 # machine-readable paired structure
   └─ verification.json           # alignment and evidence details
```

`result.md` contains only questions and recognized handwriting. Verification status and evidence are stored in `result.json`, together with frame coordinates, selection scores, crop paths, and raw recognition details. Choice frames expand adaptively from text length, line count, option count, and neighboring-question boundaries so corrections outside the detector box stay with the question. Short-answer regions begin at the stem box's bottom edge. Preprocessing combines HSV, Lab chroma, channel excess, weak-edge propagation, and inpainting to remove dark or faded red ink. `code_outputs/match/<page>/viz/*_handwriting_overlay.png` shows green stem boxes and magenta handwriting regions on one page; `handwriting_regions.json` stores their coordinates, scores, and type decisions.

### Code layout and checks

```text
agent/workflow.py       single end-to-end entrypoint and output manager
match/                  local detection, layout, reading order, and matching
proofread/              question repair, evidence verification, and abstention
scripts/run_agent.ps1   user entrypoint
```

```powershell
.\.conda\messtoclean\python.exe -m unittest discover -s tests -v
.\.conda\messtoclean\python.exe -m agent.workflow --help
```

`workflow/`, `input/`, local environments, API secrets, model weights, and `Reference/` are Git-ignored. `Reference/` is local research material and is never uploaded to GitHub.
