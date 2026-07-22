#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-python3}"
export QTYCHE_EXECUTION_PLATFORM="qbraid_lab"

"$python_bin" scripts/reproduce_phase3.py \
  --stage smoke \
  --summary results/qbraid/qbraid_smoke_summary.json

"$python_bin" - <<'PY'
import json
from pathlib import Path

path = Path("results/qbraid/qbraid_smoke_summary.json")
summary = json.loads(path.read_text(encoding="utf-8"))
if summary["status"] != "success":
    raise SystemExit("qBraid smoke summary records a failure")
if summary["execution_platform"] != "qbraid_lab":
    raise SystemExit("qBraid smoke did not record qbraid_lab execution")
if summary["qrc_backend"] != "numpy_density_matrix_exact" or not summary["exact_noiseless"]:
    raise SystemExit("qBraid smoke did not record the exact noiseless QRC backend")
if not summary["verified_output_checksums"]:
    raise SystemExit("qBraid smoke did not verify output checksums")
print(path)
PY
