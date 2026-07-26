#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 --verify|--headline|--full [--no-resume] [--evidence-dir PATH]" >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-python}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
"$python_bin" scripts/reproduce_phase3.py "$@"
