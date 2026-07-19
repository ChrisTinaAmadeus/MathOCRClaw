#!/usr/bin/env bash

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "MathOCRClaw supports Linux only." >&2
  return 1 2>/dev/null || exit 1
fi

MTC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MTC_ENV_DIR="${MTC_ENV_DIR:-${MTC_ROOT}/.conda/mathocrclaw}"
MTC_PYTHON="${MTC_ENV_DIR}/bin/python"

export MTC_ROOT MTC_ENV_DIR MTC_PYTHON
export PATH="${MTC_ENV_DIR}/bin:${PATH}"
export HF_HOME="${MTC_ROOT}/.cache/huggingface"
export MODELSCOPE_CACHE="${MTC_ROOT}/.cache/modelscope"
export PADDLE_HOME="${MTC_ROOT}/.cache/paddle"
export PADDLEOCR_HOME="${MTC_ROOT}/.cache/paddleocr"
export XDG_CACHE_HOME="${MTC_ROOT}/.cache"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"

mkdir -p -- \
  "${HF_HOME}" \
  "${MODELSCOPE_CACHE}" \
  "${PADDLE_HOME}" \
  "${PADDLEOCR_HOME}"

if [[ ! -x "${MTC_PYTHON}" ]]; then
  echo "Linux environment not found: ${MTC_ENV_DIR}" >&2
  echo "Run: bash scripts/setup_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi
