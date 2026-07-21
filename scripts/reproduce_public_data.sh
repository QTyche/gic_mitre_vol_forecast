#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_dir/.uv-cache}"

uv run python -m qtyche_qrc.cli download-public-data \
  --config configs/data_public_market.yaml
uv run python -m qtyche_qrc.cli prepare-data \
  --config configs/data_public_market.yaml \
  --cached
uv run python -m qtyche_qrc.cli audit-data \
  --processed-dir data/processed/public_market
uv run python -m qtyche_qrc.cli describe-data \
  --processed-dir data/processed/public_market
uv run python -c '
from pathlib import Path
from qtyche_qrc.data.config import load_data_config
from qtyche_qrc.data.download import verify_public_snapshot
from qtyche_qrc.models.dataset import load_model_dataset
config = load_data_config(Path("configs/data_public_market.yaml"))
verify_public_snapshot(config)
data = load_model_dataset(config.processed_path)
if data.is_synthetic or data.data_source_type != "public_market":
    raise SystemExit("processed public dataset has invalid source metadata")
print("public snapshot and processed checksums verified")
'
