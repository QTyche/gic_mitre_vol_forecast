#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-python3}"
export QTYCHE_EXECUTION_PLATFORM="qbraid_lab"

usage() {
  echo "usage: $0 [--seed 2026|2027|2028 | --all-seeds]" >&2
}

arguments=(--stage public-pilot --summary results/qbraid/qbraid_public_pilot_summary.json)
if [[ $# -eq 0 ]]; then
  arguments+=(--seed 2026)
elif [[ $# -eq 1 && "$1" == "--all-seeds" ]]; then
  arguments+=(--all-seeds)
elif [[ $# -eq 2 && "$1" == "--seed" ]]; then
  case "$2" in
    2026|2027|2028) arguments+=(--seed "$2") ;;
    *) usage; exit 2 ;;
  esac
else
  usage
  exit 2
fi

# The stage verifier accepts only the already-present immutable snapshot and
# checksum-verified processed files. This script never downloads market data.
"$python_bin" scripts/reproduce_phase3.py "${arguments[@]}"

"$python_bin" - <<'PY'
import json
from pathlib import Path

path = Path("results/qbraid/qbraid_public_pilot_summary.json")
summary = json.loads(path.read_text(encoding="utf-8"))
if summary["status"] != "success" or summary["public_data_provenance"] is None:
    raise SystemExit("qBraid public pilot did not complete verified public-data execution")
print(path)
PY
