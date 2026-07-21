#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_dir/.uv-cache}"

uv run python - <<'PY'
from pathlib import Path
from qtyche_qrc.models.dataset import load_model_dataset

data = load_model_dataset(Path("data/processed/public_market"))
if data.is_synthetic or data.data_source_type != "public_market":
    raise SystemExit("public QRC pilot requires verified non-synthetic public-market data")
PY

for seed in 2026 2027 2028; do
  uv run python -m qtyche_qrc.cli generate-qrc-features \
    --config configs/models/qrc_classifier_pilot.yaml \
    --reservoir-seed "$seed"
  uv run python -m qtyche_qrc.cli train-qrc \
    --config configs/models/qrc_classifier_pilot.yaml \
    --reservoir-seed "$seed"
  uv run python -m qtyche_qrc.cli train-qrc \
    --config configs/models/qrc_regressor_pilot.yaml \
    --reservoir-seed "$seed"
done

uv run python -m qtyche_qrc.cli compare-qrc-seeds \
  --results-dir results/qrc_public_pilot \
  --output-dir results/tables

echo "QRC public pilot complete; outputs are correctness and stability evidence, not superiority claims."
