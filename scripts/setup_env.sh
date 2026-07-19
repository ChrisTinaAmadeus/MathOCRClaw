#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${MTC_ENV_DIR:-${ROOT}/.conda/mathocrclaw}"

export XDG_CACHE_HOME="${ROOT}/.cache"
export CONDA_PKGS_DIRS="${ROOT}/.cache/conda/pkgs"
mkdir -p -- "${XDG_CACHE_HOME}" "${CONDA_PKGS_DIRS}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "MathOCRClaw supports Linux only." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found. Install Miniforge or Miniconda, then retry." >&2
  exit 1
fi

cd -- "${ROOT}"
if [[ -x "${ENV_DIR}/bin/python" ]]; then
  echo "Updating Linux environment: ${ENV_DIR}"
  conda env update --prefix "${ENV_DIR}" --file environment.yml --prune
else
  if [[ -e "${ENV_DIR}" ]]; then
    echo "The target exists but is not a valid Linux Conda environment: ${ENV_DIR}" >&2
    echo "Move or remove that directory, then retry." >&2
    exit 1
  fi
  echo "Creating Linux environment: ${ENV_DIR}"
  conda env create --prefix "${ENV_DIR}" --file environment.yml
fi

bash scripts/check_env.sh
