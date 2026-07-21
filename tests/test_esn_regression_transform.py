from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qtyche_qrc.experiments.esn_regression_diagnostics import select_esn_regression_head
from qtyche_qrc.models.baselines.esn import (
    ESNConfig,
    ESNRegressor,
    inverse_variance_targets,
    transform_variance_targets,
)
from tests.data_helpers import prepared_model_dataset


def _parameters(transform: str = "log_variance") -> dict[str, Any]:
    return {
        "reservoir_size": 8,
        "spectral_radius": 0.9,
        "input_scaling": 0.5,
        "leaking_rate": 0.4,
        "sparsity": 0.5,
        "washout": 2,
        "ridge_alpha": 0.1,
        "state_policy": "carry_inputs",
        "target_transform": transform,
        "transform_epsilon": 1e-12,
    }


def test_log_variance_transform_round_trip_and_positive_inverse() -> None:
    values = np.array([1e-8, 1e-5, 0.001, 0.05], dtype=float)

    transformed = transform_variance_targets(values, "log_variance", 1e-12)
    restored = inverse_variance_targets(transformed, "log_variance", 1e-12)

    np.testing.assert_allclose(restored, values, rtol=1e-12, atol=1e-15)
    assert np.all(restored > 0)


def test_log_variance_serialization_preserves_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(41)
    features = rng.normal(size=(60, 3))
    targets = np.exp(rng.normal(-6.0, 0.7, size=60))
    config = ESNConfig(seed=9, **_parameters())
    model = ESNRegressor(("a", "b", "c"), config)
    model.fit(features, targets)
    expected = model.predict(features[:10])
    model.save(tmp_path)

    loaded = ESNRegressor.load(tmp_path)

    np.testing.assert_allclose(loaded.predict(features[:10]), expected)
    assert np.all(expected > 0)


def test_regression_head_selection_has_no_test_interface_and_reports_direct_instability(
    tmp_path: Path,
) -> None:
    dataset = prepared_model_dataset(tmp_path)
    selection = dataset.for_selection()

    selected, candidates, diagnostics = select_esn_regression_head(
        selection,
        _parameters("log_variance"),
        [0.001, 0.1],
        ["direct_variance", "log_variance"],
        seed=2026,
        variance_floor=1e-12,
    )

    assert selected["target_transform"] in {"direct_variance", "log_variance"}
    assert all("negative_prediction_count" in row for row in candidates)
    assert all("floored_prediction_count" in row for row in candidates)
    assert diagnostics["selection_dataset"] == "training labels and validation metrics only"
    with pytest.raises(AttributeError, match="unavailable"):
        _ = selection.test_metrics
