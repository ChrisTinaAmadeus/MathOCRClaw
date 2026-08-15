#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_python="${repo_root}/.conda/mathocrclaw/bin/python"

if [[ ! -x "${runtime_python}" ]]; then
  runtime_python="python3"
fi

cd "${repo_root}"
exec "${runtime_python}" -m benchmark.run_benchmark "$@"
