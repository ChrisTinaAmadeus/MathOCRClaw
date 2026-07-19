#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_agent.sh --image PATH [options]

Options:
  --full                    Enable question-number reading, patching, and figure checks.
  --skip-layout             Reuse existing local detection and matching output.
  --work-root PATH          Workflow output root (default: workflow).
  --doclayout-device DEVICE Paddle layout device (default: cpu).
  -h, --help                Show this help.

Other options are passed through to python -m agent.workflow.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image=""
work_root="workflow"
doclayout_device="cpu"
skip_layout=0
full=0
passthrough=()

while (($#)); do
  case "$1" in
    --image)
      [[ $# -ge 2 ]] || { echo "--image requires a value" >&2; exit 2; }
      image="$2"
      shift 2
      ;;
    --work-root)
      [[ $# -ge 2 ]] || { echo "--work-root requires a value" >&2; exit 2; }
      work_root="$2"
      shift 2
      ;;
    --doclayout-device)
      [[ $# -ge 2 ]] || { echo "--doclayout-device requires a value" >&2; exit 2; }
      doclayout_device="$2"
      shift 2
      ;;
    --skip-layout)
      skip_layout=1
      shift
      ;;
    --full)
      full=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      passthrough+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${image}" ]]; then
  usage >&2
  exit 2
fi

source "${SCRIPT_DIR}/runtime_env.sh"
cd -- "${MTC_ROOT}"

args=(
  -m agent.workflow
  --image "${image}"
  --work-root "${work_root}"
  --doclayout-device "${doclayout_device}"
)
((skip_layout == 0)) || args+=(--skip-layout)
((full == 0)) || args+=(--with-patcher --with-fig --use-crop-qno)
args+=("${passthrough[@]}")

exec "${MTC_PYTHON}" "${args[@]}"
