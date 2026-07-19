#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_stage2.sh [options]

Options:
  --image-dir PATH
  --rfdetr-jsonl PATH
  --doclayout-json-dir PATH
  --out-dir PATH
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image_dir="workflow/preprocessed"
rfdetr_jsonl="workflow/code_outputs/rfdetr/rfdetr_infer_results.jsonl"
doclayout_json_dir="workflow/code_outputs/doclayout/json"
out_dir="workflow/code_outputs/match"

while (($#)); do
  case "$1" in
    --image-dir) image_dir="${2:?--image-dir requires a value}"; shift 2 ;;
    --rfdetr-jsonl) rfdetr_jsonl="${2:?--rfdetr-jsonl requires a value}"; shift 2 ;;
    --doclayout-json-dir) doclayout_json_dir="${2:?--doclayout-json-dir requires a value}"; shift 2 ;;
    --out-dir) out_dir="${2:?--out-dir requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

source "${SCRIPT_DIR}/runtime_env.sh"
cd -- "${MTC_ROOT}"

mkdir -p -- "${out_dir}"
exec "${MTC_PYTHON}" -m match.match \
  --rfdetr-jsonl "${rfdetr_jsonl}" \
  --doclayout-json-dir "${doclayout_json_dir}" \
  --pages-root "${image_dir}" \
  --output-dir "${out_dir}" \
  --match-algo v2 \
  --match-backend flow \
  --ro-overlap-mode keep_large \
  --match-overlap-mode keep_large \
  --save-viz \
  --draw-edges \
  --max-edges-per-question 2 \
  --max-fig-per-question 2 \
  --q-pad-ratio 0.02
