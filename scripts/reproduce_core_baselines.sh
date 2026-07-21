#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_dir/.uv-cache}"

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

uv run python -c '
from pathlib import Path
from qtyche_qrc.models.dataset import load_model_dataset
data = load_model_dataset(Path("data/processed/public_market"))
if data.is_synthetic or data.data_source_type != "public_market":
    raise SystemExit("refusing core baselines: expected a non-synthetic public_market manifest")
'

experiment_dirs=()
run_experiment() {
  local command_name="$1"
  local config_path="$2"
  local experiment_dir
  experiment_dir="$(uv run python -m qtyche_qrc.cli "$command_name" --config "$config_path")"
  experiment_dirs+=("$experiment_dir")
  uv run python -m qtyche_qrc.cli evaluate-experiment --experiment-dir "$experiment_dir" >/dev/null
}

run_experiment train-baseline configs/models/public_market/majority_classifier.yaml
run_experiment train-baseline configs/models/public_market/regime_persistence.yaml
run_experiment search-baseline configs/models/public_market/logistic_regression.yaml
run_experiment search-baseline configs/models/public_market/esn_classifier_search.yaml
run_experiment train-baseline configs/models/public_market/rv_persistence.yaml
./scripts/diagnose_esn_regression.sh
run_experiment search-baseline configs/models/public_market/esn_regressor_search.yaml

uv run python -m qtyche_qrc.cli compare-public-baselines \
  --results-dir results/public_market \
  --output-dir results/tables

uv run python -c '
import json
import sys
from pathlib import Path
directories = [Path(value) for value in sys.argv[1:]]
for directory in directories:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "success" or manifest["is_synthetic"]:
        raise SystemExit(f"invalid public result manifest: {directory}")
' "${experiment_dirs[@]}"
