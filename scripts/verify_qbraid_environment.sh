#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-python3}"
export QTYCHE_EXECUTION_PLATFORM="${QTYCHE_EXECUTION_PLATFORM:-qbraid_lab}"
"$python_bin" -m qtyche_qrc.cli verify-qbraid
