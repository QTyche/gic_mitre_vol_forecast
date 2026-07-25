import gzip
import json
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import qtyche_qrc.experiments.qrc_mnist as mnist_experiment
from qtyche_qrc.data.mnist import (
    COLUMN_BANDS,
    DIGITS,
    MNISTBenchmarkData,
    MNISTOfficialPartitions,
    MNISTSelectedSplit,
    build_mnist_benchmark_data,
    class_counts,
    compress_image_rows,
    deterministic_stratified_indices,
    read_idx_images,
    read_idx_labels,
)
from qtyche_qrc.experiments.qrc_mnist import (
    ESN_FEATURE_DIMENSION,
    FULL_COUNTS,
    IMAGE_FEATURE_DIMENSION,
    READOUT_C_GRID,
    ROW_FEATURE_DIMENSION,
    SMOKE_COUNTS,
    _condition_from_config,
    digit_classification_metrics,
    fit_validation_selected_readout,
    generate_or_load_mnist_qrc_features,
    load_mnist_study_config,
    qrc_config_for_seed,
    reservoir_image_row_features,
    run_mnist_classifier,
    temporal_summary_features,
)
from qtyche_qrc.models.qrc.reservoir import QuantumReservoir


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_path() -> Path:
    return _root() / "configs/qrc_mnist_benchmark.yaml"


def _labels(per_digit: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return np.repeat(np.arange(10, dtype=np.uint8), per_digit)


def _fake_source_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": "MNIST",
        "official_partitions_preserved": True,
        "files": {
            key: {"sha256": key * 8}
            for key in ("train_images", "train_labels", "test_images", "test_labels")
        },
    }


def _fake_split(name: str, source: str, labels: np.ndarray[Any, Any]) -> MNISTSelectedSplit:
    count = len(labels)
    sequences = np.random.default_rng(count).random((count, 28, 5))
    return MNISTSelectedSplit(
        name=name,
        source_partition=source,
        official_indices=np.arange(count, dtype=np.int64),
        images=np.zeros((count, 28, 28), dtype=np.uint8),
        labels=np.asarray(labels, dtype=np.int64),
        sequences=np.asarray(sequences, dtype=float),
    )


def _fake_benchmark_data(per_class: int = 1) -> MNISTBenchmarkData:
    labels = np.repeat(np.arange(10, dtype=np.int64), per_class)
    train = _fake_split("train", "official_train", np.tile(labels, 2))
    validation = _fake_split("validation", "official_train", labels)
    test = _fake_split("test", "official_test", labels)
    return MNISTBenchmarkData(
        train=train,
        validation=validation,
        test=test,
        source_manifest=_fake_source_manifest(),
        index_manifest={"checksum": "indices"},
        preprocessing_manifest={"checksum": "preprocessing"},
        subset_checksum="subset",
    )


def test_official_mnist_idx_partition_headers_are_handled_strictly(tmp_path: Path) -> None:
    images = np.arange(10 * 28 * 28, dtype=np.uint8).reshape(10, 28, 28)
    labels = np.arange(10, dtype=np.uint8)
    image_path = tmp_path / "images.gz"
    label_path = tmp_path / "labels.gz"
    with gzip.open(image_path, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, 10, 28, 28))
        handle.write(images.tobytes())
    with gzip.open(label_path, "wb") as handle:
        handle.write(struct.pack(">II", 2049, 10))
        handle.write(labels.tobytes())

    assert np.array_equal(read_idx_images(image_path, expected_count=10), images)
    assert np.array_equal(read_idx_labels(label_path, expected_count=10), labels)
    with pytest.raises(ValueError, match="count changed"):
        read_idx_images(image_path, expected_count=60_000)


def test_deterministic_stratified_selection_is_balanced_and_disjoint() -> None:
    train_labels = _labels(800)
    test_labels = _labels(200)
    first = deterministic_stratified_indices(
        train_labels,
        test_labels,
        train_per_digit=600,
        validation_per_digit=100,
        test_per_digit=100,
        seed=2026,
    )
    repeated = deterministic_stratified_indices(
        train_labels,
        test_labels,
        train_per_digit=600,
        validation_per_digit=100,
        test_per_digit=100,
        seed=2026,
    )

    assert all(np.array_equal(first[key], repeated[key]) for key in first)
    assert len(first["train"]) == 6000
    assert len(first["validation"]) == 1000
    assert len(first["test"]) == 1000
    assert np.intersect1d(first["train"], first["validation"]).size == 0
    assert class_counts(train_labels[first["train"]]) == {str(digit): 600 for digit in DIGITS}
    assert class_counts(train_labels[first["validation"]]) == {str(digit): 100 for digit in DIGITS}
    assert class_counts(test_labels[first["test"]]) == {str(digit): 100 for digit in DIGITS}
    namespaced_train = {("official_train", int(value)) for value in first["train"]}
    namespaced_validation = {("official_train", int(value)) for value in first["validation"]}
    namespaced_test = {("official_test", int(value)) for value in first["test"]}
    assert namespaced_train.isdisjoint(namespaced_validation)
    assert namespaced_train.isdisjoint(namespaced_test)
    assert namespaced_validation.isdisjoint(namespaced_test)


def test_five_band_row_compression_has_explicit_boundaries_range_and_length() -> None:
    image = np.zeros((1, 28, 28), dtype=np.uint8)
    for band, (start, end) in enumerate(COLUMN_BANDS):
        image[:, :, start:end] = np.uint8(band * 50)

    compressed = compress_image_rows(image)

    assert COLUMN_BANDS == ((0, 6), (6, 12), (12, 18), (18, 23), (23, 28))
    assert compressed.shape == (1, 28, 5)
    assert np.allclose(compressed[0, 0], np.arange(5) * 50 / 255.0)
    assert float(compressed.min()) >= 0.0
    assert float(compressed.max()) <= 1.0


def test_selected_subset_manifest_uses_official_training_and_test_partitions() -> None:
    train_labels = _labels(4)
    test_labels = _labels(2)
    official = MNISTOfficialPartitions(
        train_images=np.zeros((40, 28, 28), dtype=np.uint8),
        train_labels=train_labels,
        test_images=np.zeros((20, 28, 28), dtype=np.uint8),
        test_labels=test_labels,
        source_manifest=_fake_source_manifest(),
    )
    data = build_mnist_benchmark_data(
        official,
        train_per_digit=2,
        validation_per_digit=1,
        test_per_digit=1,
        seed=2026,
    )

    assert data.train.source_partition == "official_train"
    assert data.validation.source_partition == "official_train"
    assert data.test.source_partition == "official_test"
    assert data.is_synthetic is False
    assert data.preprocessing_manifest["learned_parameters"] is False


def test_reservoir_resets_between_images_but_carries_within_image() -> None:
    sequence = np.random.default_rng(7).random((28, 5))
    reservoir = QuantumReservoir(5, qrc_config_for_seed(2026))
    first = reservoir_image_row_features(reservoir, sequence)
    repeated = reservoir_image_row_features(reservoir, sequence)

    row_reset_reservoir = QuantumReservoir(5, qrc_config_for_seed(2026))
    row_reset = []
    for row in sequence:
        row_reset_reservoir.reset_state()
        row_reset.append(row_reset_reservoir.step(row))

    assert np.array_equal(first, repeated)
    assert not np.allclose(first, np.vstack(row_reset))


def test_qrc_features_are_seed_deterministic_and_seed_sensitive() -> None:
    sequence = np.random.default_rng(11).random((28, 5))
    first = temporal_summary_features(
        reservoir_image_row_features(QuantumReservoir(5, qrc_config_for_seed(2026)), sequence)
    )
    repeated = temporal_summary_features(
        reservoir_image_row_features(QuantumReservoir(5, qrc_config_for_seed(2026)), sequence)
    )
    changed = temporal_summary_features(
        reservoir_image_row_features(QuantumReservoir(5, qrc_config_for_seed(2027)), sequence)
    )

    assert first.shape == (IMAGE_FEATURE_DIMENSION,)
    assert ROW_FEATURE_DIMENSION == 20
    assert IMAGE_FEATURE_DIMENSION == 140
    assert ESN_FEATURE_DIMENSION == 224
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, changed)


def test_feature_cache_is_deterministic_and_reused(tmp_path: Path) -> None:
    base = load_mnist_study_config(_config_path())
    config = replace(base, output_root=tmp_path / "results/qrc_mnist")
    data = _fake_benchmark_data()
    condition = _condition_from_config(config, "analytic", 2026)

    first = generate_or_load_mnist_qrc_features(config, data, condition, mode="smoke")
    repeated = generate_or_load_mnist_qrc_features(config, data, condition, mode="smoke")

    assert first.cache_hit is False
    assert repeated.cache_hit is True
    assert first.metadata["cache_key"]["checksum"] == repeated.metadata["cache_key"]["checksum"]
    assert np.array_equal(first.train, repeated.train)


def test_validation_only_regularisation_and_finite_probabilities() -> None:
    rng = np.random.default_rng(13)
    train_features = rng.normal(size=(40, 8))
    train_labels = np.tile(np.arange(10), 4)
    validation_features = rng.normal(size=(20, 8))
    validation_labels = np.tile(np.arange(10), 2)

    fit = fit_validation_selected_readout(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )
    probabilities = mnist_experiment._probabilities(fit.scaler, fit.estimator, validation_features)

    assert fit.selected_c in READOUT_C_GRID
    assert all(row["selection_data"] == "validation only" for row in fit.selection_rows)
    assert np.isfinite(fit.estimator.coef_).all()
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_digit_metrics_cover_every_class_and_confusion_shape() -> None:
    labels = np.arange(10, dtype=int)
    probabilities = np.eye(10, dtype=float)

    metrics = digit_classification_metrics(labels, probabilities)

    assert metrics["accuracy"] == 1.0
    assert set(metrics["per_class_precision"]) == {str(value) for value in DIGITS}
    assert set(metrics["per_class_recall"]) == {str(value) for value in DIGITS}
    assert set(metrics["per_class_f1"]) == {str(value) for value in DIGITS}
    assert np.asarray(metrics["confusion_matrix"]).shape == (10, 10)


def test_smoke_and_full_sizes_and_frozen_configuration() -> None:
    config = load_mnist_study_config(_config_path())
    qrc = config.raw["qrc"]

    assert FULL_COUNTS == (600, 100, 100)
    assert SMOKE_COUNTS == (20, 5, 5)
    assert config.reservoir_seeds == (2026, 2027, 2028)
    assert qrc["n_qubits"] == 5
    assert qrc["virtual_nodes"] == 2
    assert qrc["image_state_policy"] == "reset_between_images_carry_within_image"


def test_run_resumes_complete_and_refits_partial_classifier(tmp_path: Path) -> None:
    base = load_mnist_study_config(_config_path())
    config = replace(base, output_root=tmp_path / "results/qrc_mnist")
    data = _fake_benchmark_data()
    rng = np.random.default_rng(21)
    bundle = mnist_experiment.MNISTFeatureBundle(
        train=rng.normal(size=(20, 12)),
        validation=rng.normal(size=(10, 12)),
        test=rng.normal(size=(10, 12)),
        metadata={
            "cache_key": {"checksum": "cache"},
            "feature_generation_wall_seconds": 0.01,
        },
        cache_dir=config.output_root / "feature_cache/cache",
        cache_hit=True,
    )
    first_rows, _first_readout, first_resumed = run_mnist_classifier(
        config,
        data,
        model_name="test_model",
        condition_id="analytic",
        reservoir_seed=2026,
        feature_bundle=bundle,
        mode="smoke",
        resume=True,
    )
    second_rows, _second_readout, second_resumed = run_mnist_classifier(
        config,
        data,
        model_name="test_model",
        condition_id="analytic",
        reservoir_seed=2026,
        feature_bundle=bundle,
        mode="smoke",
        resume=True,
    )

    assert first_resumed is False
    assert second_resumed is True
    assert first_rows == second_rows

    run_directory = Path(str(first_rows[0]["run_directory"]))
    incomplete_artifact = run_directory / "test_predictions.csv"
    incomplete_artifact.unlink()
    _third_rows, _third_readout, third_resumed = run_mnist_classifier(
        config,
        data,
        model_name="test_model",
        condition_id="analytic",
        reservoir_seed=2026,
        feature_bundle=bundle,
        mode="smoke",
        resume=True,
    )

    assert third_resumed is False
    assert incomplete_artifact.is_file()


def test_synthetic_digits_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    data = replace(_fake_benchmark_data(), is_synthetic=True)
    monkeypatch.setattr(mnist_experiment, "load_official_mnist", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        mnist_experiment,
        "build_mnist_benchmark_data",
        lambda *args, **kwargs: data,
    )

    with pytest.raises(ValueError, match="refuses synthetic"):
        mnist_experiment.run_qrc_mnist_benchmark(_config_path(), smoke=True)


def test_mnist_outputs_are_isolated_from_all_previous_results() -> None:
    config = load_mnist_study_config(_config_path())
    previous = (
        "final_financial_qrc",
        "qrc_noise_robustness",
        "qrc_state_memory_ablation",
        "qrc_encoding_density",
        "qrc_qubit_scaling",
        "qrc_public_pilot",
        "garch_baseline",
        "public_market",
    )

    assert config.output_root == _root() / "results/qrc_mnist"
    assert all(config.output_root != _root() / "results" / name for name in previous)
    assert json.loads(json.dumps(config.raw))["study"]["id"].startswith("qrc_mnist")
