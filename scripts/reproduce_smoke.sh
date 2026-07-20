#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_dir/.uv-cache}"
uv run python -m qtyche_qrc.cli create-fixture-data --config configs/data.yaml
uv run python -m qtyche_qrc.cli prepare-data --config configs/data.yaml
uv run python -m qtyche_qrc.cli audit-data --processed-dir data/processed
uv run python -m qtyche_qrc.cli inspect-targets --processed-dir data/processed
uv run pytest
