#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_dir/.uv-cache}"

uv run python -m qtyche_qrc.cli diagnose-esn-regression \
  --config configs/models/public_market/esn_regressor_search.yaml \
  --output-dir results/diagnostics/esn_regression
