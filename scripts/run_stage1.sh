#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_stage1.sh --image PATH [options]

Options:
  --rfdetr-out PATH
  --doclayout-out PATH
  --checkpoint PATH
  --doclayout-device DEVICE
  --doclayout-model-dir PATH
  --optimize-rfdetr
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image=""
rfdetr_out="workflow/code_outputs/rfdetr"
doclayout_out="workflow/code_outputs/doclayout"
checkpoint="checkpoint_best_total.pth"
doclayout_device="cpu"
doclayout_model_dir="${MTC_DOCLAYOUT_MODEL_DIR:-}"
optimize_rfdetr=0

while (($#)); do
  case "$1" in
    --image) image="${2:?--image requires a value}"; shift 2 ;;
    --rfdetr-out) rfdetr_out="${2:?--rfdetr-out requires a value}"; shift 2 ;;
    --doclayout-out) doclayout_out="${2:?--doclayout-out requires a value}"; shift 2 ;;
    --checkpoint) checkpoint="${2:?--checkpoint requires a value}"; shift 2 ;;
    --doclayout-device) doclayout_device="${2:?--doclayout-device requires a value}"; shift 2 ;;
    --doclayout-model-dir) doclayout_model_dir="${2:?--doclayout-model-dir requires a value}"; shift 2 ;;
    --optimize-rfdetr) optimize_rfdetr=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

source "${SCRIPT_DIR}/runtime_env.sh"
cd -- "${MTC_ROOT}"

[[ -n "${image}" ]] || { echo "--image is required" >&2; usage >&2; exit 2; }
[[ -f "${image}" ]] || { echo "Image not found: ${image}" >&2; exit 1; }
[[ -f "${checkpoint}" ]] || { echo "Checkpoint not found: ${checkpoint}" >&2; exit 1; }

for output_dir in "${rfdetr_out}" "${doclayout_out}"; do
  [[ -n "${output_dir}" && "${output_dir}" != "/" && "${output_dir}" != "${MTC_ROOT}" ]] || {
    echo "Refusing unsafe output directory: ${output_dir}" >&2
    exit 2
  }
  rm -rf -- "${output_dir}"
  mkdir -p -- "${output_dir}"
done

rfdetr_args=(
  -m match.rfdetr_infer
  --image-path "${image}"
  --checkpoint "${checkpoint}"
  --output-dir "${rfdetr_out}"
  --overwrite-jsonl
  --clean-output
  --num-classes 4
)
((optimize_rfdetr == 0)) || rfdetr_args+=(--optimize-for-inference)
"${MTC_PYTHON}" "${rfdetr_args[@]}"

doclayout_args=(
  -m match.doclayout_infer
  --image-path "${image}"
  --output-dir "${doclayout_out}"
  --model-name PP-DocLayout_plus-L
  --device "${doclayout_device}"
)
[[ -z "${doclayout_model_dir}" ]] || doclayout_args+=(--model-dir "${doclayout_model_dir}")
"${MTC_PYTHON}" "${doclayout_args[@]}"
