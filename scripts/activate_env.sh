#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file so activation affects the current shell:" >&2
  echo "  source scripts/activate_env.sh" >&2
  exit 2
fi

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/runtime_env.sh" || return

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found in PATH." >&2
  return 1
fi

eval "$(conda shell.bash hook)"
conda activate "${MTC_ENV_DIR}"
cd -- "${MTC_ROOT}"
echo "Activated MathOCRClaw: ${MTC_ENV_DIR}"
python --version
