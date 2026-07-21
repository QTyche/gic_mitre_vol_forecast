#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_dir/.uv-cache}"
uv run python -m qtyche_qrc.cli create-fixture-data --config configs/data.yaml
uv run python -m qtyche_qrc.cli prepare-data --config configs/data.yaml
uv run python -m qtyche_qrc.cli audit-data --processed-dir data/processed
uv run python -m qtyche_qrc.cli inspect-targets --processed-dir data/processed

experiment_dirs=()
run_smoke_experiment() {
  local command_name="$1"
  local config_path="$2"
  local experiment_dir
  experiment_dir="$(uv run python -m qtyche_qrc.cli "$command_name" \
    --config "$config_path" --allow-synthetic-results)"
  experiment_dirs+=("$experiment_dir")
  uv run python -m qtyche_qrc.cli evaluate-experiment --experiment-dir "$experiment_dir"
}

run_smoke_experiment train-baseline configs/models/majority_classifier.yaml
run_smoke_experiment train-baseline configs/models/regime_persistence.yaml
run_smoke_experiment search-baseline configs/models/logistic_regression.yaml
run_smoke_experiment train-baseline configs/models/rv_persistence.yaml
run_smoke_experiment search-baseline configs/models/esn_classifier_smoke.yaml
run_smoke_experiment search-baseline configs/models/esn_regressor_smoke.yaml

uv run python -m qtyche_qrc.cli compare-baselines \
  --results-dir results \
  --output-dir results/tables \
  --latest-per-model
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
