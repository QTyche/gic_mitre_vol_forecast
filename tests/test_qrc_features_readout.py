import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from qtyche_qrc.experiments.qrc_capacity import (
    effective_rank_from_singular_values,
    squared_correlation_capacity,
)
from qtyche_qrc.models.qrc.features import (
    FeatureCacheIntegrityError,
    generate_or_load_features,
    load_feature_cache,
    make_feature_cache_key,
)
from qtyche_qrc.models.qrc.readout import QRCClassifier, QRCReadoutConfig, QRCRegressor
from qtyche_qrc.models.qrc.reservoir import QRCConfig, QuantumReservoir, split_qrc_features


def _config(
    *,
    virtual_nodes: int = 1,
    input_scaling: float = 0.5,
    state_policy: str = "carry_inputs",
) -> QRCConfig:
    return QRCConfig(
        virtual_nodes=virtual_nodes,
        input_scaling=input_scaling,
        state_policy=state_policy,
    )


def test_changing_target_labels_does_not_change_qrc_features() -> None:
    inputs = np.random.default_rng(1).normal(size=(8, 3))
    labels = np.arange(8) % 3
    changed_labels = labels[::-1]
    first = QuantumReservoir(3, _config()).transform(inputs, reset=True)
    second = QuantumReservoir(3, _config()).transform(inputs, reset=True)
    assert not np.array_equal(labels, changed_labels)
    assert np.array_equal(first, second)


def test_carry_inputs_and_reset_are_distinct_documented_policies() -> None:
    rng = np.random.default_rng(4)
    train = rng.normal(size=(5, 2))
    validation = rng.normal(size=(4, 2))
    carry = QuantumReservoir(2, _config(state_policy="carry_inputs"))
    _, carry_validation, _ = split_qrc_features(carry, train, validation, None, "carry_inputs")
    reset = QuantumReservoir(2, _config(state_policy="reset"))
    _, reset_validation, _ = split_qrc_features(reset, train, validation, None, "reset")
    assert not np.allclose(carry_validation, reset_validation)


def test_feature_cache_key_changes_when_dynamical_configuration_changes() -> None:
    base = _config()
    changed = replace(base, tau=2.0)
    first = make_feature_cache_key(
        processed_data_manifest_checksum="data", feature_names=("a",), config=base
    )
    second = make_feature_cache_key(
        processed_data_manifest_checksum="data", feature_names=("a",), config=changed
    )
    assert first.checksum != second.checksum


def test_corrupted_feature_cache_checksum_fails_clearly(tmp_path: Path) -> None:
    config = _config()
    key = make_feature_cache_key(
        processed_data_manifest_checksum="data", feature_names=("a",), config=config
    )
    values = np.linspace(-1.0, 1.0, 6)[:, None]
    bundle = generate_or_load_features(
        cache_root=tmp_path,
        key=key,
        feature_names=("a",),
        config=config,
        X_train=values[:2],
        X_validation=values[2:4],
        X_test=values[4:],
    )
    metadata_path = bundle.cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["array_checksums"]["train"] = "corrupted"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(FeatureCacheIntegrityError, match="checksum mismatch"):
        load_feature_cache(tmp_path, key)


def test_classifier_probabilities_are_finite_and_sum_to_one() -> None:
    rng = np.random.default_rng(2)
    features = rng.normal(size=(18, 6))
    model = QRCClassifier(tuple(f"f{i}" for i in range(6)), QRCReadoutConfig())
    model.fit(features, np.arange(18) % 3)
    probabilities = model.predict_proba(features)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_log_variance_inverse_predictions_are_positive() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(20, 5))
    targets = np.exp(rng.normal(-5.0, 0.2, size=20))
    model = QRCRegressor(tuple(f"f{i}" for i in range(5)), QRCReadoutConfig())
    model.fit(features, targets)
    assert np.all(model.predict(features) > 0)


def test_qrc_classifier_serialization_and_reload_preserve_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    features = rng.normal(size=(15, 4))
    model = QRCClassifier(("a", "b", "c", "d"), QRCReadoutConfig(0.01))
    model.fit(features, np.arange(15) % 3)
    expected = model.predict_proba(features)
    model.save(tmp_path)
    assert np.array_equal(QRCClassifier.load(tmp_path).predict_proba(features), expected)


def test_qrc_regressor_serialization_and_reload_preserve_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(size=(15, 4))
    targets = np.exp(rng.normal(-4.0, 0.1, size=15))
    model = QRCRegressor(("a", "b", "c", "d"), QRCReadoutConfig(0.01))
    model.fit(features, targets)
    expected = model.predict(features)
    model.save(tmp_path)
    assert np.array_equal(QRCRegressor.load(tmp_path).predict(features), expected)


def test_memory_capacity_matches_perfect_controlled_example() -> None:
    target = np.linspace(-1.0, 1.0, 20)
    assert squared_correlation_capacity(target, 3.0 * target + 2.0) == pytest.approx(1.0)


def test_effective_rank_matches_equal_controlled_spectrum() -> None:
    result = effective_rank_from_singular_values(np.ones(4))
    assert result["effective_rank"] == pytest.approx(4.0)
    assert result["numerical_rank"] == 4


def test_angles_are_recorded_without_clipping_or_wrapping() -> None:
    reservoir = QuantumReservoir(1, _config(input_scaling=10.0))
    reservoir.transform(np.asarray([[2.0], [-2.0]]), reset=True)
    diagnostics = reservoir.angle_diagnostics()
    assert diagnostics["fraction_absolute_greater_than_2pi"] == 1.0
    assert diagnostics["maximum"] == pytest.approx(20.0) or diagnostics["minimum"] == pytest.approx(
        -20.0
    )


def test_connected_correlations_are_diagnostics_not_readout_features() -> None:
    reservoir = QuantumReservoir(2, _config(virtual_nodes=2))
    features = reservoir.transform(np.zeros((3, 2)), reset=True)
    assert features.shape[1] == 2 * (3 + 3)
    assert reservoir.observables.metadata()["connected_correlations_in_readout"] is False


def test_same_cache_is_reused_without_regenerating_features(tmp_path: Path) -> None:
    config = _config()
    key = make_feature_cache_key(
        processed_data_manifest_checksum="data", feature_names=("a",), config=config
    )
    values = np.arange(6, dtype=float)[:, None]
    first = generate_or_load_features(
        cache_root=tmp_path,
        key=key,
        feature_names=("a",),
        config=config,
        X_train=values[:2],
        X_validation=values[2:4],
        X_test=values[4:],
    )
    second = generate_or_load_features(
        cache_root=tmp_path,
        key=key,
        feature_names=("a",),
        config=config,
        X_train=values[:2],
        X_validation=values[2:4],
        X_test=values[4:],
    )
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert np.array_equal(first.test, second.test)
