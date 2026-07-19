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

Requirements: Linux, Bash, Conda, a DashScope/OpenAI-compatible multimodal API, and `checkpoint_best_total.pth` in the repository root. A CUDA-capable GPU is optional; CPU is the default for layout detection.

Create the isolated Linux environment. The setup script always uses `.conda/mathocrclaw`, so an environment copied from another operating system is never reused.

```bash
bash scripts/setup_env.sh
```

Create a Git-ignored `.env.local` and add your API key:

```bash
cp --no-clobber .env.example .env.local
```

The input image may be anywhere locally; `input/` is a convenient Git-ignored location. Run the full workflow with:

```bash
bash scripts/run_agent.sh --image input/page_0001.jpg --full
```

Add `--skip-layout` to reuse local detection and matching outputs. Without `--full`, question-number reading, text patching, and question/figure checks are disabled to reduce API calls. For an interactive shell in the same environment, run `source scripts/activate_env.sh`.

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
scripts/setup_env.sh    create or update the Linux Conda environment
scripts/run_agent.sh    user entrypoint
```

```bash
bash scripts/check_env.sh
.conda/mathocrclaw/bin/python -m unittest discover -s tests -v
.conda/mathocrclaw/bin/python -m agent.workflow --help
```

All repository text files are pinned to LF through `.gitattributes`, preventing CRLF conversions from appearing as full-file Git changes. `workflow/`, `input/`, local environments, API secrets, model weights, and `Reference/` are Git-ignored. `Reference/` is local research material and is never uploaded to GitHub.
