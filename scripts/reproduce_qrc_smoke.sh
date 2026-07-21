#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_dir/.uv-cache}"

uv run python -m qtyche_qrc.cli generate-qrc-features \
  --config configs/models/qrc_classifier_smoke.yaml \
  --allow-synthetic-results

classifier_dir="$(uv run python -m qtyche_qrc.cli train-qrc \
  --config configs/models/qrc_classifier_smoke.yaml \
  --allow-synthetic-results)"
regressor_dir="$(uv run python -m qtyche_qrc.cli train-qrc \
  --config configs/models/qrc_regressor_smoke.yaml \
  --allow-synthetic-results)"

for experiment_dir in "$classifier_dir" "$regressor_dir"; do
  test -f "$experiment_dir/manifest.json"
  test -f "$experiment_dir/test_metrics.json"
  test -f "$experiment_dir/model/qrc_hamiltonian.npz"
  test -f "$experiment_dir/model/input_projection.npy"
  test -f "$experiment_dir/model/observables.json"
  test -f "$experiment_dir/model/readout.npz"
  test -f "$experiment_dir/qrc_backend_metadata.json"
  test -f "$experiment_dir/qrc_numerical_diagnostics.json"
  test -f "$experiment_dir/qrc_feature_metadata.json"
  uv run python -m qtyche_qrc.cli inspect-qrc --experiment-dir "$experiment_dir" >/dev/null
done

uv run python - "$classifier_dir" "$regressor_dir" <<'PY'
import json
import pathlib
import sys

for value in sys.argv[1:]:
    manifest = json.loads((pathlib.Path(value) / "manifest.json").read_text())
    assert manifest["is_synthetic"] is True
    assert manifest["qrc_features_generated_without_labels"] is True
    assert manifest["test_evaluated_after_readout_freeze"] is True
print("QRC fixture smoke passed: synthetic, offline, label-free reservoir features")
PY
