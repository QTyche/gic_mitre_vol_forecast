#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_dir/.uv-cache}"

allow_flag=""
if [[ "${1:-}" == "--allow-synthetic-results" ]]; then
  allow_flag="--allow-synthetic-results"
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--allow-synthetic-results]" >&2
  exit 2
fi

is_synthetic="$(uv run python -c \
  'from pathlib import Path; from qtyche_qrc.models.dataset import load_model_dataset; print(str(load_model_dataset(Path("data/processed")).is_synthetic).lower())')"
if [[ "$is_synthetic" == "true" && -z "$allow_flag" ]]; then
  echo "refusing core baselines: processed data are synthetic fixtures; use --allow-synthetic-results only for marked integration output" >&2
  exit 2
fi

uv run python -m qtyche_qrc.cli train-baseline \
  --config configs/models/majority_classifier.yaml $allow_flag
uv run python -m qtyche_qrc.cli train-baseline \
  --config configs/models/regime_persistence.yaml $allow_flag
uv run python -m qtyche_qrc.cli search-baseline \
  --config configs/models/logistic_regression.yaml $allow_flag
uv run python -m qtyche_qrc.cli train-baseline \
  --config configs/models/rv_persistence.yaml $allow_flag
uv run python -m qtyche_qrc.cli search-baseline \
  --config configs/models/esn_classifier_search.yaml $allow_flag
uv run python -m qtyche_qrc.cli search-baseline \
  --config configs/models/esn_regressor_search.yaml $allow_flag
