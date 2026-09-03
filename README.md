# MathOCRClaw

[简体中文](README.zh-CN.md) · English

MathOCRClaw is a math-OCR agent for real-world exam photos. It uses two global API calls: the first produces a whole-page draft, and the second combines that draft with local layout/context evidence to propose minimal visual corrections. A deterministic local gate applies only safe, evidenced patches to produce the final result.

```text
exam photo
  → shadow normalization with colored ink preserved
  → API #1: whole-page question and handwriting draft Markdown
  → local detection, layout analysis, and context construction
  → API #2: one global review returning exact, evidence-linked patches
  → deterministic validation and merge against the API #1 draft
  → final paired question-and-answer results
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

Add `--skip-layout` to reuse local detection and matching outputs. On a cache miss, the workflow always makes exactly two API calls. `--full` only adds one magnified answer view per question to the second request; it does not add calls. For an interactive shell in the same environment, run `source scripts/activate_env.sh`.

### Benchmark

The agent evaluation set contains **32 high-quality exam images plus Qwen3.7-plus baseline Markdown**. Images and their corresponding baseline outputs use matching names from `page01` through `page32` under [`benchmark/images/`](benchmark/images/) and [`benchmark/baseline/`](benchmark/baseline/). The full MathDoc dataset is not included in this repository.

### Output layout

User-facing runtime artifacts retain the original image and every processing stage (internal caches live under `.cache/`):

```text
workflow/
└─ <page_name>/                   # the only page-name directory for this image
   ├─ image/                      # untouched original input image
   ├─ preprocessed/               # normalized scan and preprocessing JSON
   ├─ api_markdown/               # API call 1 whole-page draft and response JSON
   ├─ code_outputs/               # local code/model stages
   │  ├─ rfdetr/                  # detection JSONL, crops, and visualizations
   │  ├─ doclayout/               # layout JSON and visualizations
   │  └─ match/                   # reading order, linked contexts, and crops
   └─ agent_outputs/              # no repeated <page_name>/ directory
      ├─ result.md                # each question followed by its handwriting
      ├─ result.json              # machine-readable paired structure
      └─ verification.json        # raw response from the global second pass
```

API call 1 reads `benchmark/prompts/extract_v2.txt` directly and writes `<page_name>/api_markdown/<page>.md`; the Python workflow does not maintain a second prompt copy. The local middle stage parses that Markdown and creates stem/answer views, a grayscale local-contrast handwriting companion, and a per-question `symbol_profile_v1`. The profile exposes only question-conditioned candidate families and discriminating strokes (for example B/D, `<`/`\le`, interval brackets, or `\varnothing`/0/phi), never a correct answer; same-page contexts sharing a tag also form unlabelled writer-style comparison groups. API call 2 returns `api2_patch_v4` records: unchanged fields use `keep`; stem and solution corrections are exact substring patches; choice/fill corrections additionally require a context-linked contrastive symbol observation. Every existing `<插图>` also receives a `figure_audit_v1` record that independently classifies its source (`printed_figure`, `student_diagram`, `mixed_figure`, `scratch_sketch`, `not_a_figure`, or `uncertain`) and retained-content role. API2 receives the full-color context, stem view, and available detector crops for this audit; detector routing alone is never treated as visual evidence. Local validation requires compatible source/role pairs and concrete printed or handwriting features, but conservatively preserves every existing marker even when the model calls it scratch work. `result.md` uses the same per-question format as the gold Markdown. Color context remains authoritative for separating student and marker ink; the grayscale view is only companion evidence for faint strokes.

### Code layout and checks

```text
agent/workflow.py       single end-to-end entrypoint and output manager
match/                  local detection, layout, reading order, and matching
proofread/              image, cache, and legacy verification utilities
scripts/setup_env.sh    create or update the Linux Conda environment
scripts/run_agent.sh    user entrypoint
```

```bash
bash scripts/check_env.sh
.conda/mathocrclaw/bin/python -m unittest discover -s tests -v
.conda/mathocrclaw/bin/python -m agent.workflow --help
```

All repository text files are pinned to LF through `.gitattributes`, preventing CRLF conversions from appearing as full-file Git changes. `workflow/`, `input/`, local environments, API secrets, model weights, and `Reference/` are Git-ignored. `Reference/` is local research material and is never uploaded to GitHub.
