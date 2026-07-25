"""Common MNIST digit-classification benchmark for the isolated QRC workflow."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from qtyche_qrc.data.mnist import (
    COLUMN_BANDS,
    DIGITS,
    TEMPORAL_WINDOWS,
    MNISTBenchmarkData,
    MNISTSelectedSplit,
    MNISTSourceFile,
    build_mnist_benchmark_data,
    class_counts,
    download_mnist,
    load_official_mnist,
    parse_source_files,
    sha256_path,
)
from qtyche_qrc.experiments.qrc_capacity import effective_feature_rank
from qtyche_qrc.experiments.run import _git_metadata, _write_json
from qtyche_qrc.models.baselines.esn import ESNConfig, ESNReservoir
from qtyche_qrc.models.qrc.encoding import array_checksum
from qtyche_qrc.models.qrc.noise import QRCMeasurementConfig
from qtyche_qrc.models.qrc.reservoir import QRCConfig, QuantumReservoir
from qtyche_qrc.models.qrc.robust_features import RobustQuantumReservoir
from qtyche_qrc.runtime import runtime_metadata

STUDY_ID = "qrc_mnist_common_benchmark_v1"
FEATURE_VERSION = "mnist_image_sequence_summary_v1"
CLASS_COUNT = 10
ROW_FEATURE_DIMENSION = 20
IMAGE_FEATURE_DIMENSION = 140
ESN_FEATURE_DIMENSION = 224
FULL_COUNTS = (600, 100, 100)
SMOKE_COUNTS = (20, 5, 5)
RESERVOIR_SEEDS = (2026, 2027, 2028)
READOUT_C_GRID = (0.01, 0.1, 1.0, 10.0)
ROBUSTNESS_CONDITIONS = (
    "analytic",
    "shots_2048",
    "depolarizing_0_01",
    "measurement_flip_0_02",
)
QRCFeatureReservoir = Union[QuantumReservoir, RobustQuantumReservoir]


def _manifest_path(path: Path, project_root: Path) -> str:
    """Prefer portable project-relative paths while supporting external roots."""

    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


@dataclass(frozen=True)
class MNISTStudyConfig:
    """Checksum-pinned paths and controls for the common benchmark."""

    source: Path
    project_root: Path
    output_root: Path
    dataset_cache: Path
    selection_seed: int
    reservoir_seeds: tuple[int, ...]
    smoke_reservoir_seeds: tuple[int, ...]
    sources: tuple[MNISTSourceFile, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class MNISTFeatureCondition:
    """One exact or controlled-measurement QRC feature condition."""

    condition_id: str
    reservoir_seed: int
    shots: int | None
    measurement_seed: int | None
    depolarizing_probability: float
    measurement_bit_flip_probability: float

    @property
    def measurement_config(self) -> QRCMeasurementConfig:
        return QRCMeasurementConfig(
            shots=self.shots,
            measurement_seed=self.measurement_seed,
            depolarizing_probability=self.depolarizing_probability,
            measurement_bit_flip_probability=self.measurement_bit_flip_probability,
        )

    @property
    def analytic(self) -> bool:
        return self.shots is None


@dataclass(frozen=True)
class MNISTFeatureBundle:
    """Fixed-length image features with cache and simulator provenance."""

    train: NDArray[np.float64]
    validation: NDArray[np.float64]
    test: NDArray[np.float64]
    metadata: dict[str, Any]
    cache_dir: Path
    cache_hit: bool


@dataclass(frozen=True)
class ReadoutFit:
    """Validation-selected ten-class logistic readout and diagnostics."""

    scaler: StandardScaler
    estimator: LogisticRegression
    selected_c: float
    selection_rows: list[dict[str, Any]]
    selection_seconds: float
    training_seconds: float
    converged: bool
    convergence_warnings: tuple[str, ...]


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    return dict(value)


def _int_tuple(value: object, location: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{location} must be a non-empty integer list")
    result = tuple(int(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{location} contains duplicates")
    return result


def _resolved(project_root: Path, value: object, location: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty path")
    return (project_root / value).resolve()


def _json_checksum(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_mnist_study_config(path: Path) -> MNISTStudyConfig:
    """Load and strictly validate the isolated MNIST scientific contract."""

    source = path.resolve()
    root = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "configuration")
    if root.get("schema_version") != 1:
        raise ValueError("MNIST configuration schema_version must be 1")
    study = _mapping(root.get("study"), "study")
    dataset = _mapping(root.get("dataset"), "dataset")
    full = _mapping(root.get("full"), "full")
    smoke = _mapping(root.get("smoke"), "smoke")
    preprocessing = _mapping(root.get("preprocessing"), "preprocessing")
    qrc = _mapping(root.get("qrc"), "qrc")
    readout = _mapping(root.get("readout"), "readout")
    baselines = _mapping(root.get("baselines"), "baselines")
    robustness = _mapping(root.get("robustness"), "robustness")
    project_setting = study.get("project_root")
    if not isinstance(project_setting, str):
        raise ValueError("study.project_root must be a path")
    project_root = (source.parent / project_setting).resolve()
    config = MNISTStudyConfig(
        source=source,
        project_root=project_root,
        output_root=_resolved(project_root, study.get("output_root"), "study.output_root"),
        dataset_cache=_resolved(project_root, study.get("dataset_cache"), "study.dataset_cache"),
        selection_seed=int(study.get("selection_seed", -1)),
        reservoir_seeds=_int_tuple(study.get("reservoir_seeds"), "study.reservoir_seeds"),
        smoke_reservoir_seeds=_int_tuple(smoke.get("reservoir_seeds"), "smoke.reservoir_seeds"),
        sources=parse_source_files(dataset.get("files")),
        raw=root,
    )
    if study.get("id") != STUDY_ID:
        raise ValueError(f"MNIST study ID must remain {STUDY_ID}")
    if config.selection_seed != 2026 or config.reservoir_seeds != RESERVOIR_SEEDS:
        raise ValueError("MNIST selection or reservoir seed grid changed")
    if config.smoke_reservoir_seeds != (2026,):
        raise ValueError("MNIST smoke reservoir seed must remain 2026")
    if (
        int(dataset.get("official_training_partition_size", -1)) != 60_000
        or int(dataset.get("official_test_partition_size", -1)) != 10_000
    ):
        raise ValueError("official MNIST partition sizes changed")
    if tuple(
        int(full[key]) for key in ("train_per_digit", "validation_per_digit", "test_per_digit")
    ) != (FULL_COUNTS):
        raise ValueError("full MNIST balanced subset sizes changed")
    if (
        tuple(
            int(smoke[key]) for key in ("train_per_digit", "validation_per_digit", "test_per_digit")
        )
        != SMOKE_COUNTS
    ):
        raise ValueError("smoke MNIST balanced subset sizes changed")
    configured_bands = tuple(
        tuple(int(value) for value in pair)
        for pair in cast(list[list[int]], preprocessing.get("column_bands"))
    )
    configured_windows = tuple(
        tuple(int(value) for value in pair)
        for pair in cast(list[list[int]], preprocessing.get("temporal_windows"))
    )
    if configured_bands != COLUMN_BANDS or configured_windows != TEMPORAL_WINDOWS:
        raise ValueError("MNIST band boundaries or temporal windows changed")
    expected_qrc = {
        "n_qubits": 5,
        "virtual_nodes": 2,
        "tau": 1.0,
        "graph": "ring",
        "j_strength": 1.0,
        "h_strength": 1.0,
        "input_scaling": 0.5,
        "backend": "numpy_density_matrix_exact",
        "encoding": "Ry",
        "image_state_policy": "reset_between_images_carry_within_image",
        "input_reinjection": "partial_input_qubit_reinjection",
        "row_feature_dimension": ROW_FEATURE_DIMENSION,
        "image_feature_dimension": IMAGE_FEATURE_DIMENSION,
    }
    for name, expected in expected_qrc.items():
        if qrc.get(name) != expected:
            raise ValueError(f"MNIST QRC control changed: {name}")
    if tuple(float(value) for value in cast(list[float], readout.get("regularization_c_grid"))) != (
        READOUT_C_GRID
    ):
        raise ValueError("MNIST validation regularisation grid changed")
    if (
        readout.get("selection_split") != "validation"
        or readout.get("selection_metric") != "macro_f1"
        or readout.get("type") != "multinomial_logistic_regression"
    ):
        raise ValueError("MNIST readout selection contract changed")
    logistic = _mapping(baselines.get("logistic"), "baselines.logistic")
    esn = _mapping(baselines.get("esn"), "baselines.esn")
    if int(logistic.get("feature_dimension", -1)) != 140:
        raise ValueError("flattened band baseline dimension changed")
    if (
        int(esn.get("reservoir_size", -1)) != 32
        or int(esn.get("feature_dimension", -1)) != ESN_FEATURE_DIMENSION
    ):
        raise ValueError("size-controlled MNIST ESN contract changed")
    conditions = cast(list[dict[str, Any]], robustness.get("conditions"))
    if tuple(str(item["id"]) for item in conditions) != ROBUSTNESS_CONDITIONS:
        raise ValueError("MNIST limited robustness conditions changed")
    if (
        int(robustness.get("reservoir_seed", -1)) != 2026
        or int(robustness.get("measurement_seed", -1)) != 2026
    ):
        raise ValueError("MNIST robustness seeds changed")
    return config


def qrc_config_for_seed(seed: int) -> QRCConfig:
    """Return the challenge-prescribed five-qubit image-sequence dynamics."""

    if seed not in RESERVOIR_SEEDS:
        raise ValueError(f"unsupported MNIST reservoir seed: {seed}")
    return QRCConfig(
        n_qubits=5,
        graph="ring",
        virtual_nodes=2,
        j_strength=1.0,
        h_strength=1.0,
        tau=1.0,
        input_scaling=0.5,
        state_policy="reset",
        reservoir_seed=seed,
        backend="numpy_density_matrix_exact",
    )


def temporal_summary_features(
    row_features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Summarize 28 row states by a fixed, test-independent 140-vector."""

    values = np.asarray(row_features, dtype=float)
    if values.shape != (28, ROW_FEATURE_DIMENSION) or not np.isfinite(values).all():
        raise ValueError("MNIST QRC row features must have shape (28,20) and be finite")
    summaries = [
        values[-1],
        values.mean(axis=0),
        values.std(axis=0, ddof=0),
        *(values[start:end].mean(axis=0) for start, end in TEMPORAL_WINDOWS),
    ]
    result = np.concatenate(summaries)
    if result.shape != (IMAGE_FEATURE_DIMENSION,):
        raise ValueError("MNIST temporal summary dimension changed")
    return np.asarray(result, dtype=float)


def esn_temporal_summary_features(
    row_states: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply the same seven temporal summaries to 32 ESN reservoir states."""

    values = np.asarray(row_states, dtype=float)
    if values.shape != (28, 32) or not np.isfinite(values).all():
        raise ValueError("MNIST ESN row states must have shape (28,32)")
    result = np.concatenate(
        [
            values[-1],
            values.mean(axis=0),
            values.std(axis=0, ddof=0),
            *(values[start:end].mean(axis=0) for start, end in TEMPORAL_WINDOWS),
        ]
    )
    if result.shape != (ESN_FEATURE_DIMENSION,):
        raise ValueError("MNIST ESN temporal summary dimension changed")
    return np.asarray(result, dtype=float)


def reservoir_image_row_features(
    reservoir: QRCFeatureReservoir,
    sequence: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reset once per image, then carry state through its 28 row inputs."""

    values = np.asarray(sequence, dtype=float)
    if values.shape != (28, 5) or not np.isfinite(values).all():
        raise ValueError("MNIST image sequence must have shape (28,5)")
    reservoir.reset_state()
    rows = np.vstack([reservoir.step(row) for row in values])
    if rows.shape != (28, ROW_FEATURE_DIMENSION):
        raise ValueError("MNIST reservoir emitted an unexpected row feature shape")
    return np.asarray(rows, dtype=float)


def _condition_from_config(
    config: MNISTStudyConfig, condition_id: str, reservoir_seed: int
) -> MNISTFeatureCondition:
    robustness = _mapping(config.raw["robustness"], "robustness")
    conditions = cast(list[dict[str, Any]], robustness["conditions"])
    raw = next(
        (record for record in conditions if record.get("id") == condition_id),
        None,
    )
    if raw is None:
        raise ValueError(f"unsupported MNIST robustness condition: {condition_id}")
    shots = raw.get("shots")
    measurement_seed = None if shots is None else int(robustness["measurement_seed"])
    condition = MNISTFeatureCondition(
        condition_id=condition_id,
        reservoir_seed=reservoir_seed,
        shots=None if shots is None else int(shots),
        measurement_seed=measurement_seed,
        depolarizing_probability=float(raw["depolarizing_probability"]),
        measurement_bit_flip_probability=float(raw["measurement_bit_flip_probability"]),
    )
    condition.measurement_config.validate()
    return condition


def _feature_key(
    data: MNISTBenchmarkData,
    condition: MNISTFeatureCondition,
) -> dict[str, Any]:
    qrc = qrc_config_for_seed(condition.reservoir_seed)
    payload = {
        "schema_version": 1,
        "feature_version": FEATURE_VERSION,
        "dataset_subset_checksum": data.subset_checksum,
        "preprocessing_checksum": data.preprocessing_manifest["checksum"],
        "qrc_configuration_checksum": qrc.checksum,
        "measurement_configuration_checksum": condition.measurement_config.checksum,
        "image_state_policy": "reset_between_images_carry_within_image",
        "temporal_windows": [list(value) for value in TEMPORAL_WINDOWS],
        "image_feature_dimension": IMAGE_FEATURE_DIMENSION,
    }
    return {**payload, "checksum": _json_checksum(payload)}


def _load_feature_cache(cache_dir: Path, key: dict[str, Any]) -> MNISTFeatureBundle | None:
    arrays_path = cache_dir / "features.npz"
    metadata_path = cache_dir / "metadata.json"
    if not cache_dir.exists():
        return None
    if not arrays_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"incomplete MNIST QRC feature cache: {cache_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("cache_key") != key:
        raise ValueError("MNIST QRC feature cache key mismatch")
    with np.load(arrays_path) as arrays:
        loaded = {
            split: np.asarray(arrays[split], dtype=float)
            for split in ("train", "validation", "test")
        }
    for split, values in loaded.items():
        if array_checksum(values) != metadata["array_checksums"][split]:
            raise ValueError(f"MNIST feature cache array checksum mismatch: {split}")
    return MNISTFeatureBundle(
        loaded["train"],
        loaded["validation"],
        loaded["test"],
        metadata,
        cache_dir,
        True,
    )


def _generate_split_features(
    reservoir: QRCFeatureReservoir,
    split: MNISTSelectedSplit,
) -> NDArray[np.float64]:
    output = np.empty((len(split.labels), IMAGE_FEATURE_DIMENSION), dtype=float)
    for index, sequence in enumerate(split.sequences):
        output[index] = temporal_summary_features(reservoir.transform(sequence, reset=True))
    return output


def generate_or_load_mnist_qrc_features(
    config: MNISTStudyConfig,
    data: MNISTBenchmarkData,
    condition: MNISTFeatureCondition,
    *,
    mode: str,
) -> MNISTFeatureBundle:
    """Generate/reset image-sequence features or checksum-load the isolated cache."""

    key = _feature_key(data, condition)
    cache_dir = config.output_root / "feature_cache" / mode / str(key["checksum"])
    cached = _load_feature_cache(cache_dir, key)
    if cached is not None:
        return cached
    qrc = qrc_config_for_seed(condition.reservoir_seed)
    reservoir: QRCFeatureReservoir
    if condition.analytic:
        reservoir = QuantumReservoir(5, qrc)
    else:
        reservoir = RobustQuantumReservoir(5, qrc, condition.measurement_config)
    started = time.perf_counter()
    train = _generate_split_features(reservoir, data.train)
    validation = _generate_split_features(reservoir, data.validation)
    test = _generate_split_features(reservoir, data.test)
    wall_seconds = time.perf_counter() - started
    if not all(
        np.isfinite(values).all() and values.shape[1] == IMAGE_FEATURE_DIMENSION
        for values in (train, validation, test)
    ):
        raise ValueError("MNIST QRC feature generation produced invalid values")
    cache_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        cache_dir / "features.npz",
        train=train,
        validation=validation,
        test=test,
    )
    resource = reservoir.resource_metadata()
    numerical = (
        reservoir.numerical_diagnostics()
        if isinstance(reservoir, QuantumReservoir)
        else reservoir.diagnostics()
    )
    metadata = {
        "schema_version": 1,
        "cache_key": key,
        "condition": asdict(condition),
        "measurement_configuration": condition.measurement_config.metadata(),
        "label_free_generation": True,
        "labels_consumed": False,
        "image_state_policy": "reset_between_images_carry_within_image",
        "row_steps_per_image": 28,
        "row_feature_dimension": ROW_FEATURE_DIMENSION,
        "image_feature_dimension": IMAGE_FEATURE_DIMENSION,
        "temporal_summaries": (
            "final row; full mean; full population standard deviation; "
            "means over [0,7), [7,14), [14,21), [21,28)"
        ),
        "split_shapes": {
            "train": list(train.shape),
            "validation": list(validation.shape),
            "test": list(test.shape),
        },
        "array_checksums": {
            "train": array_checksum(train),
            "validation": array_checksum(validation),
            "test": array_checksum(test),
        },
        "feature_generation_wall_seconds": wall_seconds,
        "resource_metadata": resource,
        "numerical_diagnostics": numerical,
    }
    _write_json(cache_dir / "metadata.json", metadata)
    return MNISTFeatureBundle(train, validation, test, metadata, cache_dir, False)


def _esn_config(config: MNISTStudyConfig) -> ESNConfig:
    raw = _mapping(
        _mapping(config.raw["baselines"], "baselines")["esn"],
        "baselines.esn",
    )
    return ESNConfig(
        reservoir_size=int(raw["reservoir_size"]),
        spectral_radius=float(raw["spectral_radius"]),
        input_scaling=float(raw["input_scaling"]),
        leaking_rate=float(raw["leaking_rate"]),
        sparsity=float(raw["sparsity"]),
        washout=0,
        ridge_alpha=1e-3,
        seed=int(raw["reservoir_seed"]),
        state_policy="reset",
    )


def generate_or_load_esn_features(
    config: MNISTStudyConfig,
    data: MNISTBenchmarkData,
    *,
    mode: str,
) -> MNISTFeatureBundle:
    """Create deterministic size-controlled ESN image summaries."""

    esn_config = _esn_config(config)
    key_payload = {
        "schema_version": 1,
        "feature_version": "mnist_esn_image_summary_v1",
        "dataset_subset_checksum": data.subset_checksum,
        "configuration": asdict(esn_config),
        "feature_dimension": ESN_FEATURE_DIMENSION,
    }
    key = {**key_payload, "checksum": _json_checksum(key_payload)}
    cache_dir = config.output_root / "baseline_cache" / mode / str(key["checksum"])
    cached = _load_feature_cache(cache_dir, key)
    if cached is not None:
        return cached
    reservoir = ESNReservoir(5, esn_config)
    started = time.perf_counter()

    def transform(split: MNISTSelectedSplit) -> NDArray[np.float64]:
        output = np.empty((len(split.labels), ESN_FEATURE_DIMENSION), dtype=float)
        for index, sequence in enumerate(split.sequences):
            states = reservoir.transform_sequence(sequence, reset=True)
            output[index] = esn_temporal_summary_features(states)
        return output

    train = transform(data.train)
    validation = transform(data.validation)
    test = transform(data.test)
    wall_seconds = time.perf_counter() - started
    cache_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        cache_dir / "features.npz",
        train=train,
        validation=validation,
        test=test,
    )
    metadata = {
        "schema_version": 1,
        "cache_key": key,
        "condition": {"condition_id": "esn", "reservoir_seed": esn_config.seed},
        "label_free_generation": True,
        "labels_consumed": False,
        "image_state_policy": "reset_between_images_carry_within_image",
        "row_steps_per_image": 28,
        "row_feature_dimension": esn_config.reservoir_size,
        "image_feature_dimension": ESN_FEATURE_DIMENSION,
        "split_shapes": {
            "train": list(train.shape),
            "validation": list(validation.shape),
            "test": list(test.shape),
        },
        "array_checksums": {
            "train": array_checksum(train),
            "validation": array_checksum(validation),
            "test": array_checksum(test),
        },
        "feature_generation_wall_seconds": wall_seconds,
        "resource_metadata": {
            "reservoir_size": esn_config.reservoir_size,
            "measured_spectral_radius": reservoir.measured_spectral_radius,
        },
        "numerical_diagnostics": {},
    }
    _write_json(cache_dir / "metadata.json", metadata)
    return MNISTFeatureBundle(train, validation, test, metadata, cache_dir, False)


def generate_or_load_flattened_features(
    config: MNISTStudyConfig,
    data: MNISTBenchmarkData,
    *,
    mode: str,
) -> MNISTFeatureBundle:
    """Cache the label-free flattened 28-by-5 baseline representation."""

    key_payload = {
        "schema_version": 1,
        "feature_version": "mnist_flattened_28_by_5_v1",
        "dataset_subset_checksum": data.subset_checksum,
        "preprocessing_checksum": data.preprocessing_manifest["checksum"],
        "feature_dimension": IMAGE_FEATURE_DIMENSION,
    }
    key = {**key_payload, "checksum": _json_checksum(key_payload)}
    cache_dir = config.output_root / "baseline_cache" / mode / str(key["checksum"])
    cached = _load_feature_cache(cache_dir, key)
    if cached is not None:
        return cached
    started = time.perf_counter()
    train = data.train.sequences.reshape(len(data.train.labels), -1).copy()
    validation = data.validation.sequences.reshape(len(data.validation.labels), -1).copy()
    test = data.test.sequences.reshape(len(data.test.labels), -1).copy()
    wall_seconds = time.perf_counter() - started
    cache_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        cache_dir / "features.npz",
        train=train,
        validation=validation,
        test=test,
    )
    metadata = {
        "schema_version": 1,
        "cache_key": key,
        "condition": {"condition_id": "flattened_28_by_5", "reservoir_seed": None},
        "label_free_generation": True,
        "labels_consumed": False,
        "row_steps_per_image": 28,
        "row_feature_dimension": len(COLUMN_BANDS),
        "image_feature_dimension": IMAGE_FEATURE_DIMENSION,
        "split_shapes": {
            "train": list(train.shape),
            "validation": list(validation.shape),
            "test": list(test.shape),
        },
        "array_checksums": {
            "train": array_checksum(train),
            "validation": array_checksum(validation),
            "test": array_checksum(test),
        },
        "feature_generation_wall_seconds": wall_seconds,
        "resource_metadata": {},
        "numerical_diagnostics": {},
    }
    _write_json(cache_dir / "metadata.json", metadata)
    return MNISTFeatureBundle(train, validation, test, metadata, cache_dir, False)


def digit_classification_metrics(
    labels: NDArray[np.integer[Any]],
    probabilities: NDArray[np.float64],
) -> dict[str, Any]:
    """Compute the complete ten-class MNIST evaluation contract."""

    truth = np.asarray(labels, dtype=int).reshape(-1)
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(truth), CLASS_COUNT):
        raise ValueError("MNIST probabilities must have shape (n,10)")
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-10)
    ):
        raise ValueError("MNIST probabilities must be finite normalized rows")
    predicted = np.argmax(values, axis=1)
    precision, recall, per_class_f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        labels=list(DIGITS),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, labels=list(DIGITS), average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "per_class_precision": {
            str(digit): float(value) for digit, value in zip(DIGITS, precision)
        },
        "per_class_recall": {str(digit): float(value) for digit, value in zip(DIGITS, recall)},
        "per_class_f1": {str(digit): float(value) for digit, value in zip(DIGITS, per_class_f1)},
        "ovr_macro_roc_auc": float(
            roc_auc_score(
                truth,
                values,
                labels=list(DIGITS),
                multi_class="ovr",
                average="macro",
            )
        ),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(DIGITS)).tolist(),
    }


def _probabilities(
    scaler: StandardScaler,
    estimator: LogisticRegression,
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    transformed = scaler.transform(features)
    raw = np.asarray(estimator.predict_proba(transformed), dtype=float)
    probabilities = np.zeros((len(features), CLASS_COUNT), dtype=float)
    probabilities[:, np.asarray(estimator.classes_, dtype=int)] = raw
    if not np.isfinite(probabilities).all():
        raise ValueError("MNIST readout produced non-finite probabilities")
    return probabilities


def fit_validation_selected_readout(
    train_features: NDArray[np.float64],
    train_labels: NDArray[np.integer[Any]],
    validation_features: NDArray[np.float64],
    validation_labels: NDArray[np.integer[Any]],
    *,
    c_grid: tuple[float, ...] = READOUT_C_GRID,
    seed: int = 2026,
    max_iterations: int = 1000,
) -> ReadoutFit:
    """Select L2 regularisation on validation only and retain the fitted train head."""

    if not c_grid or any(value <= 0 for value in c_grid):
        raise ValueError("MNIST regularisation grid must be positive")
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features)
    if not np.isfinite(scaled_train).all():
        raise ValueError("training-only MNIST scaling produced non-finite values")
    selection_started = time.perf_counter()
    trials: list[dict[str, Any]] = []
    fitted: dict[float, tuple[LogisticRegression, tuple[str, ...], float]] = {}
    for trial, value in enumerate(c_grid, start=1):
        estimator = LogisticRegression(
            C=float(value),
            penalty="l2",
            solver="lbfgs",
            max_iter=max_iterations,
            random_state=seed,
        )
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(scaled_train, np.asarray(train_labels, dtype=int))
        training_seconds = time.perf_counter() - started
        convergence_warnings = tuple(str(item.message) for item in caught)
        probabilities = _probabilities(scaler, estimator, validation_features)
        score = digit_classification_metrics(validation_labels, probabilities)["macro_f1"]
        converged = bool(np.all(estimator.n_iter_ < max_iterations))
        trials.append(
            {
                "trial": trial,
                "regularization_c": float(value),
                "selection_metric": "macro_f1",
                "validation_score": float(score),
                "training_seconds": training_seconds,
                "converged": converged,
                "convergence_warnings": list(convergence_warnings),
                "selection_data": "validation only",
            }
        )
        fitted[float(value)] = (
            estimator,
            convergence_warnings,
            training_seconds,
        )
    selected = max(
        trials,
        key=lambda row: (
            float(row["validation_score"]),
            -float(row["regularization_c"]),
        ),
    )
    selected_c = float(selected["regularization_c"])
    estimator, selected_warnings, training_seconds = fitted[selected_c]
    return ReadoutFit(
        scaler=scaler,
        estimator=estimator,
        selected_c=selected_c,
        selection_rows=trials,
        selection_seconds=time.perf_counter() - selection_started,
        training_seconds=training_seconds,
        converged=bool(np.all(estimator.n_iter_ < max_iterations)),
        convergence_warnings=selected_warnings,
    )


def _prediction_table(
    split: MNISTSelectedSplit,
    probabilities: NDArray[np.float64],
) -> pd.DataFrame:
    predicted = np.argmax(probabilities, axis=1)
    table = pd.DataFrame(
        {
            "official_partition": split.source_partition,
            "official_index": split.official_indices,
            "true_digit": split.labels,
            "predicted_digit": predicted,
        }
    )
    for digit in DIGITS:
        table[f"probability_{digit}"] = probabilities[:, digit]
    return table


def _condition_diagnostics(features: NDArray[np.float64]) -> dict[str, Any]:
    diagnostics, singular_values = effective_feature_rank(features)
    return {
        **diagnostics,
        "singular_value_count": len(singular_values),
        "training_feature_checksum": array_checksum(features),
        "computed_from_split": "train",
        "labels_consumed": False,
    }


def _run_identity(
    *,
    model_name: str,
    condition_id: str,
    reservoir_seed: int | None,
    subset_checksum: str,
    configuration_checksum: str,
    feature_cache_checksum: str,
) -> str:
    payload = {
        "model_name": model_name,
        "condition_id": condition_id,
        "reservoir_seed": reservoir_seed,
        "subset_checksum": subset_checksum,
        "configuration_checksum": configuration_checksum,
        "feature_cache_checksum": feature_cache_checksum,
    }
    return f"{model_name}_{condition_id}_{_json_checksum(payload)[:16]}"


def _load_completed_run(
    directory: Path,
    *,
    subset_checksum: str,
    configuration_checksum: str,
    feature_cache_checksum: str,
) -> dict[str, Any] | None:
    required = (
        directory / "manifest.json",
        directory / "result.json",
        directory / "validation_predictions.csv",
        directory / "test_predictions.csv",
        directory / "model.pkl",
    )
    if not all(path.is_file() for path in required):
        return None
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "success"
        or manifest.get("study_id") != STUDY_ID
        or manifest.get("dataset_subset_checksum") != subset_checksum
        or manifest.get("study_configuration_checksum") != configuration_checksum
        or manifest.get("feature_cache_key_checksum") != feature_cache_checksum
        or manifest.get("is_synthetic") is not False
        or manifest.get("model_selection_data") != "validation only"
        or manifest.get("test_evaluated_after_readout_freeze") is not True
    ):
        return None
    return cast(
        dict[str, Any],
        json.loads((directory / "result.json").read_text(encoding="utf-8")),
    )


def run_mnist_classifier(
    config: MNISTStudyConfig,
    data: MNISTBenchmarkData,
    *,
    model_name: str,
    condition_id: str,
    reservoir_seed: int | None,
    feature_bundle: MNISTFeatureBundle,
    mode: str,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    """Fit/resume one readout and persist predictions, metrics, and provenance."""

    configuration_checksum = sha256_path(config.source)
    feature_cache_checksum = str(feature_bundle.metadata["cache_key"]["checksum"])
    identity = _run_identity(
        model_name=model_name,
        condition_id=condition_id,
        reservoir_seed=reservoir_seed,
        subset_checksum=data.subset_checksum,
        configuration_checksum=configuration_checksum,
        feature_cache_checksum=feature_cache_checksum,
    )
    directory = config.output_root / "runs" / mode / identity
    if resume:
        completed = _load_completed_run(
            directory,
            subset_checksum=data.subset_checksum,
            configuration_checksum=configuration_checksum,
            feature_cache_checksum=feature_cache_checksum,
        )
        if completed is not None:
            return (
                cast(list[dict[str, Any]], completed["rows"]),
                cast(dict[str, Any], completed["readout"]),
                True,
            )
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fit = fit_validation_selected_readout(
        feature_bundle.train,
        data.train.labels,
        feature_bundle.validation,
        data.validation.labels,
        seed=config.selection_seed,
        max_iterations=int(config.raw["readout"]["max_iterations"]),
    )
    prediction_outputs: dict[str, tuple[dict[str, Any], pd.DataFrame, float]] = {}
    for split_name, split, features in (
        ("validation", data.validation, feature_bundle.validation),
        ("test", data.test, feature_bundle.test),
    ):
        inference_started = time.perf_counter()
        probabilities = _probabilities(fit.scaler, fit.estimator, features)
        inference_seconds = time.perf_counter() - inference_started
        metrics = digit_classification_metrics(split.labels, probabilities)
        predictions = _prediction_table(split, probabilities)
        prediction_outputs[split_name] = (
            metrics,
            predictions,
            inference_seconds,
        )
        _write_json(directory / f"{split_name}_metrics.json", metrics)
        predictions.to_csv(directory / f"{split_name}_predictions.csv", index=False)
    coefficient_values = np.asarray(fit.estimator.coef_, dtype=float)
    intercept_values = np.asarray(fit.estimator.intercept_, dtype=float)
    finite_coefficients = bool(
        np.isfinite(coefficient_values).all() and np.isfinite(intercept_values).all()
    )
    finite_predictions = all(
        np.isfinite(
            output[1][[f"probability_{digit}" for digit in DIGITS]].to_numpy(dtype=float)
        ).all()
        for output in prediction_outputs.values()
    )
    if not finite_coefficients or not finite_predictions:
        raise ValueError("MNIST readout coefficients or predictions are non-finite")
    diagnostics = _condition_diagnostics(feature_bundle.train)
    feature_generation_seconds = float(feature_bundle.metadata["feature_generation_wall_seconds"])
    total_seconds = time.perf_counter() - started + feature_generation_seconds
    readout = {
        "selected_regularization_c": fit.selected_c,
        "selection_metric": "macro_f1",
        "selection_data": "validation only",
        "selection_trials": fit.selection_rows,
        "selection_seconds": fit.selection_seconds,
        "training_seconds": fit.training_seconds,
        "converged": fit.converged,
        "convergence_warnings": list(fit.convergence_warnings),
        "coefficient_shape": list(coefficient_values.shape),
        "coefficient_l2_norm": float(np.linalg.norm(coefficient_values)),
        "maximum_absolute_coefficient": float(np.max(np.abs(coefficient_values))),
        "finite_coefficients": finite_coefficients,
        "finite_predictions": finite_predictions,
    }
    rows: list[dict[str, Any]] = []
    for split_name in ("validation", "test"):
        metrics, _predictions, inference_seconds = prediction_outputs[split_name]
        rows.append(
            {
                "model": model_name,
                "condition": condition_id,
                "reservoir_seed": reservoir_seed,
                "split": split_name,
                **metrics,
                "selected_regularization_c": fit.selected_c,
                "coefficient_l2_norm": readout["coefficient_l2_norm"],
                "maximum_absolute_coefficient": readout["maximum_absolute_coefficient"],
                "converged": fit.converged,
                "finite_coefficients": finite_coefficients,
                "finite_predictions": finite_predictions,
                "feature_dimension": int(feature_bundle.train.shape[1]),
                "feature_effective_rank": diagnostics["effective_rank"],
                "feature_numerical_rank": diagnostics["numerical_rank"],
                "feature_condition_number": diagnostics["condition_number"],
                "feature_generation_seconds": feature_generation_seconds,
                "training_seconds": fit.training_seconds,
                "selection_seconds": fit.selection_seconds,
                "inference_seconds": inference_seconds,
                "total_runtime_seconds": total_seconds,
                "cache_hit": feature_bundle.cache_hit,
                "cache_key_checksum": feature_cache_checksum,
                "dataset_subset_checksum": data.subset_checksum,
                "model_selection_data": "validation only",
                "test_evaluated_after_readout_freeze": True,
                "physical_qpu_execution": False,
                "quantum_advantage_claim": False,
                "run_directory": _manifest_path(directory, config.project_root),
            }
        )
    with (directory / "model.pkl").open("wb") as handle:
        pickle.dump(
            {"scaler": fit.scaler, "estimator": fit.estimator},
            handle,
        )
    _write_json(directory / "selection_results.json", {"rows": fit.selection_rows})
    result = {"schema_version": 1, "rows": rows, "readout": readout}
    _write_json(directory / "result.json", result)
    manifest = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "success",
        "mode": mode,
        "model_name": model_name,
        "condition": condition_id,
        "reservoir_seed": reservoir_seed,
        "dataset": "genuine MNIST",
        "is_synthetic": False,
        "official_partitions_preserved": True,
        "dataset_subset_checksum": data.subset_checksum,
        "study_configuration_checksum": configuration_checksum,
        "feature_cache_key_checksum": feature_cache_checksum,
        "feature_cache_directory": _manifest_path(
            feature_bundle.cache_dir,
            config.project_root,
        ),
        "feature_dimension": int(feature_bundle.train.shape[1]),
        "selected_regularization_c": fit.selected_c,
        "model_selection_data": "validation only",
        "test_evaluated_after_readout_freeze": True,
        "physical_qpu_execution": False,
        "quantum_advantage_claim": False,
        "git": _git_metadata(config.project_root),
        **runtime_metadata(),
    }
    _write_json(directory / "manifest.json", manifest)
    return rows, readout, False


def aggregate_exact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate exact QRC metrics over independent reservoir seeds."""

    outputs: list[dict[str, Any]] = []
    frame = pd.DataFrame(rows)
    for split, group in frame.groupby("split", sort=True):
        record: dict[str, Any] = {
            "model": "qrc_exact",
            "condition": "analytic",
            "split": split,
            "reservoir_seeds": sorted(int(value) for value in group["reservoir_seed"].unique()),
            "reservoir_seed_count": int(group["reservoir_seed"].nunique()),
            "per_class_f1_mean": {
                str(digit): float(
                    np.mean(
                        [
                            cast(dict[str, float], value)[str(digit)]
                            for value in group["per_class_f1"]
                        ]
                    )
                )
                for digit in DIGITS
            },
            "confusion_matrix_sum": np.sum(
                np.asarray(group["confusion_matrix"].tolist(), dtype=int),
                axis=0,
            ).tolist(),
        }
        for metric in (
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "ovr_macro_roc_auc",
            "feature_effective_rank",
            "feature_condition_number",
            "feature_generation_seconds",
            "training_seconds",
            "inference_seconds",
            "total_runtime_seconds",
        ):
            values = group[metric].to_numpy(dtype=float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_population_standard_deviation"] = float(values.std(ddof=0))
            record[f"{metric}_minimum"] = float(values.min())
            record[f"{metric}_maximum"] = float(values.max())
        outputs.append(record)
    return outputs


def _csv_safe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        values = dict(row)
        for key, value in tuple(values.items()):
            if isinstance(value, (dict, list, tuple)):
                values[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        result.append(values)
    return result


def _write_table_pair(
    base: Path,
    name: str,
    rows: list[dict[str, Any]],
    **metadata: Any,
) -> dict[str, str]:
    json_path = base / f"{name}.json"
    csv_path = base / f"{name}.csv"
    _write_json(json_path, {"schema_version": 1, **metadata, "rows": rows})
    frame = pd.DataFrame(_csv_safe(rows))
    frame = frame.reindex(sorted(frame.columns), axis=1)
    frame.to_csv(csv_path, index=False)
    return {
        f"{name}_json": json_path.as_posix(),
        f"{name}_csv": csv_path.as_posix(),
    }


def _save_figure(figure: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=240)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def plot_example_sequences(data: MNISTBenchmarkData, destination: Path) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(12.0, 5.1))
    for digit, axis in zip(DIGITS, axes.ravel()):
        index = int(np.flatnonzero(data.train.labels == digit)[0])
        image = data.train.sequences[index].T
        axis.imshow(
            image,
            cmap="gray_r",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        axis.set(title=f"digit {digit}", xlabel="row", ylabel="band")
        axis.set_yticks(range(5), [f"B{value}" for value in range(5)])
    _save_figure(figure, destination)


def plot_qrc_confusion(aggregate: list[dict[str, Any]], destination: Path) -> None:
    test = next(row for row in aggregate if row["split"] == "test")
    matrix = np.asarray(test["confusion_matrix_sum"], dtype=int)
    figure, axis = plt.subplots(figsize=(6.2, 5.4))
    handle = axis.imshow(matrix, cmap="Blues")
    for row in range(10):
        for column in range(10):
            color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    axis.set(
        title="Exact QRC test confusion matrix (sum over three seeds)",
        xlabel="predicted digit",
        ylabel="true digit",
        xticks=range(10),
        yticks=range(10),
    )
    figure.colorbar(handle, ax=axis)
    _save_figure(figure, destination)


def plot_metric_comparison(benchmark: list[dict[str, Any]], destination: Path) -> None:
    test = [row for row in benchmark if row["split"] == "test"]
    labels = [str(row["display_name"]) for row in test]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.4, 4.4))
    axis.bar(
        positions - width / 2,
        [float(row["accuracy"]) for row in test],
        width,
        label="accuracy",
        color="#4c78a8",
    )
    axis.bar(
        positions + width / 2,
        [float(row["macro_f1"]) for row in test],
        width,
        label="macro-F1",
        color="#f58518",
    )
    qrc_index = next(index for index, row in enumerate(test) if row["model"] == "qrc_exact_mean")
    qrc = test[qrc_index]
    axis.errorbar(
        [positions[qrc_index] - width / 2, positions[qrc_index] + width / 2],
        [qrc["accuracy"], qrc["macro_f1"]],
        yerr=[
            qrc["accuracy_population_standard_deviation"],
            qrc["macro_f1_population_standard_deviation"],
        ],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylim(0.0, 1.1)
    axis.set_ylabel("test score")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, loc="upper left", ncols=2)
    _save_figure(figure, destination)


def plot_per_digit_f1(
    exact_aggregate: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    qrc = next(row for row in exact_aggregate if row["split"] == "test")
    baselines = [row for row in baseline_rows if row["split"] == "test"]
    series = [
        ("QRC mean", cast(dict[str, float], qrc["per_class_f1_mean"])),
        *[
            (
                "logistic" if row["model"] == "logistic_baseline" else "ESN",
                cast(dict[str, float], row["per_class_f1"]),
            )
            for row in baselines
        ],
    ]
    positions = np.arange(10)
    width = 0.25
    figure, axis = plt.subplots(figsize=(9.0, 4.3))
    for index, (label, values) in enumerate(series):
        axis.bar(
            positions + (index - 1) * width,
            [values[str(digit)] for digit in DIGITS],
            width,
            label=label,
        )
    axis.set(
        xlabel="digit",
        ylabel="test F1",
        xticks=positions,
        ylim=(0.0, 1.08),
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, loc="upper center", ncols=3)
    _save_figure(figure, destination)


def plot_robustness(rows: list[dict[str, Any]], destination: Path) -> None:
    test = [row for row in rows if row["split"] == "test"]
    conditions = list(ROBUSTNESS_CONDITIONS)
    labels = ["analytic", "2048 shots", "depolarising 0.01", "bit flip 0.02"]
    figure, axis = plt.subplots(figsize=(9.0, 4.4))
    positions = np.arange(len(conditions))
    width = 0.36
    selected = {
        condition: next(row for row in test if row["condition"] == condition)
        for condition in conditions
    }
    axis.bar(
        positions - width / 2,
        [selected[value]["accuracy"] for value in conditions],
        width,
        label="accuracy",
        color="#4c78a8",
    )
    axis.bar(
        positions + width / 2,
        [selected[value]["macro_f1"] for value in conditions],
        width,
        label="macro-F1",
        color="#f58518",
    )
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Seed 2026 simulated measurement robustness")
    axis.set_ylabel("test score")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    _save_figure(figure, destination)


def plot_runtime(
    exact_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    selected = [
        next(
            row
            for row in baseline_rows
            if row["model"] == "logistic_baseline" and row["split"] == "test"
        ),
        next(
            row
            for row in baseline_rows
            if row["model"] == "esn_baseline" and row["split"] == "test"
        ),
        *[row for row in exact_rows if row["split"] == "test"],
        *[
            row
            for row in robustness_rows
            if row["split"] == "test" and row["condition"] != "analytic"
        ],
    ]
    labels = []
    for row in selected:
        if row["model"] == "logistic_baseline":
            labels.append("logistic")
        elif row["model"] == "esn_baseline":
            labels.append("ESN")
        elif row["model"] == "qrc_exact":
            labels.append(f"QRC {row['reservoir_seed']}")
        else:
            labels.append(
                {
                    "shots_2048": "2048 shots",
                    "depolarizing_0_01": "depolarising",
                    "measurement_flip_0_02": "bit flip",
                }[str(row["condition"])]
            )
    figure, axis = plt.subplots(figsize=(9.4, 4.4))
    axis.bar(
        range(len(selected)),
        [float(row["total_runtime_seconds"]) for row in selected],
        color="#4c78a8",
    )
    axis.set_yscale("log")
    axis.set_xticks(range(len(selected)), labels, rotation=25, ha="right")
    axis.set_ylabel("feature + fit + inference seconds (log scale)")
    axis.grid(axis="y", alpha=0.2)
    _save_figure(figure, destination)


def plot_conditioning(exact_rows: list[dict[str, Any]], destination: Path) -> None:
    test = pd.DataFrame([row for row in exact_rows if row["split"] == "test"]).sort_values(
        "reservoir_seed"
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    axes[0].bar(
        test["reservoir_seed"].astype(str),
        test["feature_effective_rank"],
        color="#4c78a8",
    )
    axes[0].set(
        xlabel="reservoir seed",
        ylabel="training-feature effective rank",
    )
    axes[1].bar(
        test["reservoir_seed"].astype(str),
        test["feature_condition_number"],
        color="#f58518",
    )
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="reservoir seed",
        ylabel="training-feature condition number",
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    _save_figure(figure, destination)


def _publication_outputs(
    config: MNISTStudyConfig,
    data: MNISTBenchmarkData,
    exact_rows: list[dict[str, Any]],
    exact_aggregate: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
    benchmark: list[dict[str, Any]],
) -> dict[str, str]:
    figures = config.output_root / "figures"
    destinations = {
        "example_compressed_sequences": figures / "example_compressed_mnist_sequences",
        "qrc_confusion_matrix": figures / "qrc_test_confusion_matrix",
        "accuracy_macro_f1_comparison": figures / "accuracy_macro_f1_comparison",
        "per_digit_f1_comparison": figures / "per_digit_f1_comparison",
        "exact_finite_shot_noise_comparison": figures / "exact_finite_shot_noise_comparison",
        "runtime_comparison": figures / "runtime_comparison",
        "feature_rank_conditioning": figures / "feature_rank_conditioning",
    }
    plot_example_sequences(data, destinations["example_compressed_sequences"])
    plot_qrc_confusion(exact_aggregate, destinations["qrc_confusion_matrix"])
    plot_metric_comparison(benchmark, destinations["accuracy_macro_f1_comparison"])
    plot_per_digit_f1(
        exact_aggregate,
        baseline_rows,
        destinations["per_digit_f1_comparison"],
    )
    plot_robustness(
        robustness_rows,
        destinations["exact_finite_shot_noise_comparison"],
    )
    plot_runtime(
        exact_rows,
        baseline_rows,
        robustness_rows,
        destinations["runtime_comparison"],
    )
    plot_conditioning(exact_rows, destinations["feature_rank_conditioning"])
    return {
        f"{name}_{extension}": _manifest_path(
            path.with_suffix(f".{extension}"),
            config.project_root,
        )
        for name, path in destinations.items()
        for extension in ("png", "pdf")
    }


def _benchmark_rows(
    exact_aggregate: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        {
            "model": row["model"],
            "display_name": (
                "flattened logistic"
                if row["model"] == "logistic_baseline"
                else "size-controlled ESN"
            ),
            "split": row["split"],
            "accuracy": row["accuracy"],
            "macro_f1": row["macro_f1"],
            "balanced_accuracy": row["balanced_accuracy"],
            "ovr_macro_roc_auc": row["ovr_macro_roc_auc"],
            "accuracy_population_standard_deviation": 0.0,
            "macro_f1_population_standard_deviation": 0.0,
        }
        for row in baselines
    ]
    rows.extend(
        {
            "model": "qrc_exact_mean",
            "display_name": "exact QRC mean",
            "split": row["split"],
            "accuracy": row["accuracy_mean"],
            "macro_f1": row["macro_f1_mean"],
            "balanced_accuracy": row["balanced_accuracy_mean"],
            "ovr_macro_roc_auc": row["ovr_macro_roc_auc_mean"],
            "accuracy_population_standard_deviation": row["accuracy_population_standard_deviation"],
            "macro_f1_population_standard_deviation": row["macro_f1_population_standard_deviation"],
        }
        for row in exact_aggregate
    )
    return rows


def _resource_estimates(
    data: MNISTBenchmarkData,
    *,
    qrc_feature_cache_count: int,
) -> dict[str, Any]:
    images = len(data.train.labels) + len(data.validation.labels) + len(data.test.labels)
    bytes_per_cache = images * IMAGE_FEATURE_DIMENSION * 8
    return {
        "selected_images": images,
        "row_steps_per_image": 28,
        "virtual_nodes": 2,
        "state_evolutions_per_qrc_condition": images * 28 * 2,
        "hilbert_dimension": 32,
        "density_matrix_elements": 1024,
        "estimated_peak_density_matrix_bytes": 3 * 32 * 32 * 16,
        "image_feature_dimension": IMAGE_FEATURE_DIMENSION,
        "bytes_per_uncompressed_qrc_feature_cache": bytes_per_cache,
        "qrc_feature_cache_count": qrc_feature_cache_count,
        "estimated_total_uncompressed_qrc_feature_cache_bytes": (
            bytes_per_cache * qrc_feature_cache_count
        ),
        "finite_shot_conditions": 3,
        "shots_per_virtual_node_state": 2048,
        "sampled_bitstrings_per_finite_shot_condition": images * 28 * 2 * 2048,
        "prediction_artifacts_and_library_overhead_included": False,
    }


def _write_dataset_artifacts(
    config: MNISTStudyConfig,
    data: MNISTBenchmarkData,
    *,
    mode: str,
) -> dict[str, str]:
    dataset_dir = config.output_root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "dataset_manifest": dataset_dir / "dataset_manifest.json",
        "selected_indices": dataset_dir / "selected_indices.json",
        "preprocessing_manifest": dataset_dir / "preprocessing_manifest.json",
    }
    _write_json(
        paths["dataset_manifest"],
        {
            **data.source_manifest,
            "mode": mode,
            "subset_checksum": data.subset_checksum,
            "class_counts": {
                split.name: class_counts(split.labels)
                for split in (data.train, data.validation, data.test)
            },
            "split_sizes": {
                split.name: len(split.labels) for split in (data.train, data.validation, data.test)
            },
            "synthetic_data": False,
        },
    )
    _write_json(paths["selected_indices"], data.index_manifest)
    _write_json(paths["preprocessing_manifest"], data.preprocessing_manifest)
    return {name: _manifest_path(path, config.project_root) for name, path in paths.items()}


def run_qrc_mnist_benchmark(
    config_path: Path,
    *,
    smoke: bool = False,
    resume: bool = True,
    download: bool = False,
    download_only: bool = False,
) -> Path:
    """Run/resume the genuine-MNIST exact, baseline, and limited-noise study."""

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config = load_mnist_study_config(config_path)
    if download_only:
        manifest = download_mnist(config.dataset_cache, config.sources)
        config.output_root.mkdir(parents=True, exist_ok=True)
        path = config.output_root / "mnist_download_manifest.json"
        _write_json(path, manifest)
        return path
    official = load_official_mnist(config.dataset_cache, config.sources, download=download)
    counts = SMOKE_COUNTS if smoke else FULL_COUNTS
    data = build_mnist_benchmark_data(
        official,
        train_per_digit=counts[0],
        validation_per_digit=counts[1],
        test_per_digit=counts[2],
        seed=config.selection_seed,
    )
    if data.is_synthetic:
        raise ValueError("MNIST benchmark refuses synthetic or substituted digits")
    mode = "smoke" if smoke else "full"
    config.output_root.mkdir(parents=True, exist_ok=True)
    dataset_outputs = _write_dataset_artifacts(config, data, mode=mode)
    seeds = config.smoke_reservoir_seeds if smoke else config.reservoir_seeds
    exact_rows: list[dict[str, Any]] = []
    exact_readouts: list[dict[str, Any]] = []
    execution: list[dict[str, Any]] = []
    for seed in seeds:
        condition = _condition_from_config(config, "analytic", seed)
        bundle = generate_or_load_mnist_qrc_features(config, data, condition, mode=mode)
        rows, readout, resumed = run_mnist_classifier(
            config,
            data,
            model_name="qrc_exact",
            condition_id="analytic",
            reservoir_seed=seed,
            feature_bundle=bundle,
            mode=mode,
            resume=resume,
        )
        exact_rows.extend(rows)
        exact_readouts.append({"reservoir_seed": seed, **readout})
        execution.append(
            {
                "model": "qrc_exact",
                "condition": "analytic",
                "reservoir_seed": seed,
                "feature_cache_hit": bundle.cache_hit,
                "readout_resumed": resumed,
            }
        )
    exact_aggregate = aggregate_exact_rows(exact_rows)
    flattened = generate_or_load_flattened_features(config, data, mode=mode)
    baseline_rows: list[dict[str, Any]] = []
    baseline_readouts: list[dict[str, Any]] = []
    logistic_rows, logistic_readout, logistic_resumed = run_mnist_classifier(
        config,
        data,
        model_name="logistic_baseline",
        condition_id="flattened_28_by_5",
        reservoir_seed=None,
        feature_bundle=flattened,
        mode=mode,
        resume=resume,
    )
    baseline_rows.extend(logistic_rows)
    baseline_readouts.append({"model": "logistic_baseline", **logistic_readout})
    execution.append(
        {
            "model": "logistic_baseline",
            "condition": "flattened_28_by_5",
            "reservoir_seed": None,
            "feature_cache_hit": flattened.cache_hit,
            "readout_resumed": logistic_resumed,
        }
    )
    esn_bundle = generate_or_load_esn_features(config, data, mode=mode)
    esn_rows, esn_readout, esn_resumed = run_mnist_classifier(
        config,
        data,
        model_name="esn_baseline",
        condition_id="size_controlled_32",
        reservoir_seed=int(config.raw["baselines"]["esn"]["reservoir_seed"]),
        feature_bundle=esn_bundle,
        mode=mode,
        resume=resume,
    )
    baseline_rows.extend(esn_rows)
    baseline_readouts.append({"model": "esn_baseline", **esn_readout})
    execution.append(
        {
            "model": "esn_baseline",
            "condition": "size_controlled_32",
            "reservoir_seed": 2026,
            "feature_cache_hit": esn_bundle.cache_hit,
            "readout_resumed": esn_resumed,
        }
    )
    robustness_rows: list[dict[str, Any]] = [
        {
            **row,
            "condition": "analytic",
            "model": "qrc_robustness",
        }
        for row in exact_rows
        if row["reservoir_seed"] == 2026
    ]
    robustness_readouts: list[dict[str, Any]] = [
        {
            "condition": "analytic",
            **next(item for item in exact_readouts if item["reservoir_seed"] == 2026),
        }
    ]
    for condition_id in ROBUSTNESS_CONDITIONS[1:]:
        condition = _condition_from_config(config, condition_id, 2026)
        bundle = generate_or_load_mnist_qrc_features(config, data, condition, mode=mode)
        rows, readout, resumed = run_mnist_classifier(
            config,
            data,
            model_name="qrc_robustness",
            condition_id=condition_id,
            reservoir_seed=2026,
            feature_bundle=bundle,
            mode=mode,
            resume=resume,
        )
        robustness_rows.extend(rows)
        robustness_readouts.append({"condition": condition_id, "reservoir_seed": 2026, **readout})
        execution.append(
            {
                "model": "qrc_robustness",
                "condition": condition_id,
                "reservoir_seed": 2026,
                "feature_cache_hit": bundle.cache_hit,
                "readout_resumed": resumed,
            }
        )
    benchmark = _benchmark_rows(exact_aggregate, baseline_rows)
    tables = config.output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for name, rows, metadata in (
        (
            "mnist_qrc_exact_per_run",
            exact_rows,
            {"readouts": exact_readouts},
        ),
        ("mnist_qrc_exact_aggregate", exact_aggregate, {}),
        (
            "mnist_qrc_finite_shot_noise",
            robustness_rows,
            {"readouts": robustness_readouts},
        ),
        (
            "mnist_classical_baselines",
            baseline_rows,
            {"readouts": baseline_readouts},
        ),
        (
            "mnist_final_benchmark",
            benchmark,
            {
                "formal_significance_tests_run": False,
                "directly_comparable_selected_images": True,
            },
        ),
    ):
        outputs = _write_table_pair(tables, name, rows, **metadata)
        output_paths.update(
            {key: _manifest_path(Path(path), config.project_root) for key, path in outputs.items()}
        )
    figures = _publication_outputs(
        config,
        data,
        exact_rows,
        exact_aggregate,
        baseline_rows,
        robustness_rows,
        benchmark,
    )
    resources = _resource_estimates(data, qrc_feature_cache_count=len(seeds) + 3)
    environment_path = config.output_root / "environment_manifest.json"
    _write_json(
        environment_path,
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "configuration": {
                "path": _manifest_path(config.source, config.project_root),
                "sha256": sha256_path(config.source),
            },
            "dataset_subset_checksum": data.subset_checksum,
            "source_checksums": {
                key: value["sha256"] for key, value in data.source_manifest["files"].items()
            },
            "git": _git_metadata(config.project_root),
            **runtime_metadata(),
            "qbraid_python_3_12_compatible": True,
            "physical_qpu_execution": False,
            "quantum_advantage_claim": False,
        },
    )
    summary_path = config.output_root / "run_summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "status": "success",
            "mode": mode,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.perf_counter() - started,
            "resume_enabled": resume,
            "dataset": {
                "name": "genuine MNIST",
                "official_partitions_preserved": True,
                "synthetic_data": False,
                "subset_checksum": data.subset_checksum,
                "split_sizes": {
                    "train": len(data.train.labels),
                    "validation": len(data.validation.labels),
                    "test": len(data.test.labels),
                },
                "class_counts": {
                    split.name: class_counts(split.labels)
                    for split in (
                        data.train,
                        data.validation,
                        data.test,
                    )
                },
            },
            "qrc_architecture": {
                **asdict(qrc_config_for_seed(2026)),
                "encoding": "Ry",
                "image_state_policy": (
                    "reset entire reservoir between images; carry state across "
                    "28 rows within each image"
                ),
                "row_feature_dimension": ROW_FEATURE_DIMENSION,
                "image_feature_dimension": IMAGE_FEATURE_DIMENSION,
            },
            "execution": execution,
            "resources": resources,
            "outputs": {
                **dataset_outputs,
                **output_paths,
                **figures,
                "environment_manifest": _manifest_path(
                    environment_path,
                    config.project_root,
                ),
            },
            "interpretation": (
                "exact classical density-matrix and controlled finite-shot/noise "
                "simulation; no physical-QPU execution or quantum-advantage claim"
            ),
        },
    )
    return summary_path
