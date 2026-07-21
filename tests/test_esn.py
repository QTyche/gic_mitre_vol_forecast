from pathlib import Path

import numpy as np
import pytest

from qtyche_qrc.models.baselines.esn import (
    ESNClassifier,
    ESNConfig,
    ESNReservoir,
    split_reservoir_states,
)


def _config(seed: int = 7) -> ESNConfig:
    return ESNConfig(
        reservoir_size=9,
        spectral_radius=0.8,
        input_scaling=0.3,
        leaking_rate=0.4,
        sparsity=0.5,
        washout=1,
        ridge_alpha=1e-3,
        seed=seed,
        state_policy="carry_inputs",
    )


def test_esn_seed_controls_matrices_and_spectral_radius() -> None:
    first = ESNReservoir(3, _config(7))
    second = ESNReservoir(3, _config(7))
    different = ESNReservoir(3, _config(8))

    np.testing.assert_array_equal(first.W_in, second.W_in)
    np.testing.assert_array_equal(first.W_res, second.W_res)
    assert not np.array_equal(first.W_res, different.W_res)
    assert first.measured_spectral_radius == pytest.approx(0.8, abs=1e-10)


def test_esn_states_have_expected_dimensions() -> None:
    reservoir = ESNReservoir(3, _config())
    states = reservoir.transform_sequence(np.ones((12, 3)), reset=True)

    assert states.shape == (12, 9)
    assert reservoir.get_state().shape == (9,)


def test_esn_serialization_preserves_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(10)
    features = rng.normal(size=(50, 3))
    targets = np.arange(50) % 3
    model = ESNClassifier(("a", "b", "c"), _config())
    model.fit(features, targets)
    expected = model.predict_proba(features[:8])
    model.save(tmp_path)

    loaded = ESNClassifier.load(tmp_path)

    np.testing.assert_allclose(loaded.predict_proba(features[:8]), expected)


def test_validation_and_test_labels_cannot_change_reservoir_states() -> None:
    rng = np.random.default_rng(12)
    X_train = rng.normal(size=(20, 3))
    X_validation = rng.normal(size=(8, 3))
    X_test = rng.normal(size=(7, 3))
    labels_a = (rng.integers(0, 3, size=8), rng.integers(0, 3, size=7))
    labels_b = (np.zeros(8, dtype=int), np.full(7, 2, dtype=int))

    first = ESNReservoir(3, _config())
    _, validation_a, test_a = split_reservoir_states(
        first, X_train, X_validation, X_test, "carry_inputs"
    )
    second = ESNReservoir(3, _config())
    _, validation_b, test_b = split_reservoir_states(
        second, X_train, X_validation, X_test, "carry_inputs"
    )

    assert not np.array_equal(labels_a[0], labels_b[0])
    assert not np.array_equal(labels_a[1], labels_b[1])
    np.testing.assert_array_equal(validation_a, validation_b)
    assert test_a is not None and test_b is not None
    np.testing.assert_array_equal(test_a, test_b)
