#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-python3}"
export QTYCHE_EXECUTION_PLATFORM="${QTYCHE_EXECUTION_PLATFORM:-qbraid_lab}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
"$python_bin" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit("qBraid reproduction requires Python 3.11 or newer")
print(f"using Python {sys.version.split()[0]}")
if sys.version_info[:2] != (3, 12):
    print("note: Python 3.12 is preferred for the final clean-room validation")
PY

if ! "$python_bin" -m pip --version >/dev/null 2>&1; then
  echo "pip is absent; bootstrapping it with ensurepip"
  "$python_bin" -m ensurepip --upgrade
fi
"$python_bin" -m pip install --upgrade pip
"$python_bin" -m pip install -r requirements-qbraid.txt
"$python_bin" -m pip install --no-deps -e .
"$python_bin" -m qtyche_qrc.cli verify-qbraid
echo "setup complete; next run: python scripts/reproduce_phase3.py --verify"
