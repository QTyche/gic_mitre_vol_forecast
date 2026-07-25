"""Controlled exact-noiseless QRC temporal-multiplexing density study."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from qtyche_qrc.data.config import load_data_config
from qtyche_qrc.data.download import sha256_file, verify_public_snapshot
from qtyche_qrc.experiments.model_config import ModelExperimentConfig, load_model_config
from qtyche_qrc.experiments.qrc_capacity import effective_feature_rank
from qtyche_qrc.experiments.qrc_run import (
    generate_qrc_features,
    qrc_config_from_model,
    run_qrc_experiment,
)
from qtyche_qrc.experiments.run import SyntheticResultsError, _git_metadata, _write_json
from qtyche_qrc.models.dataset import ModelDataset, load_model_dataset
from qtyche_qrc.models.qrc.encoding import array_checksum
from qtyche_qrc.models.qrc.hamiltonian import ring_edges
from qtyche_qrc.models.qrc.reservoir import QRCConfig
from qtyche_qrc.runtime import runtime_metadata

STUDY_ID = "exact_qrc_encoding_density_v1"
FULL_VIRTUAL_NODE_GRID = (1, 2, 4, 8)
FULL_SEED_GRID = (2026, 2027, 2028)
TASKS = ("regime_classification", "rv_regression")
MODEL_TYPES = {
    "regime_classification": "qrc_classifier",
    "rv_regression": "qrc_regressor",
}
METRICS_BY_TASK = {
    "regime_classification": (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "multiclass_brier_score",
        "transition_accuracy",
        "transition_balanced_accuracy",
        "transition_f1",
        "transition_roc_auc",
        "transition_pr_auc",
        "transition_brier_score",
    ),
    "rv_regression": (
        "rmse",
        "mae",
        "qlike",
        "r_squared",
        "prediction_mean",
        "prediction_median",
        "prediction_minimum",
        "prediction_maximum",
        "non_finite_prediction_count",
        "floored_prediction_count",
    ),
}
RESOURCE_METRICS = (
    "selected_ridge_alpha",
    "state_generation_seconds",
    "feature_generation_seconds",
    "readout_fitting_seconds",
    "total_runtime_seconds",
    "raw_feature_dimension",
    "trainable_readout_parameters",
    "effective_rank",
    "numerical_rank",
    "condition_number",
    "largest_singular_value",
    "smallest_retained_singular_value",
)


@dataclass(frozen=True, order=True)
class EncodingDensityPoint:
    """One temporal readout density and independent reservoir seed."""

    virtual_nodes: int
    reservoir_seed: int

    @property
    def key(self) -> str:
        return f"v{self.virtual_nodes}_seed{self.reservoir_seed}"


@dataclass(frozen=True)
class EncodingDensityStudyConfig:
    """Validated paths and frozen controls for the encoding-density study."""

    source: Path
    project_root: Path
    study_id: str
    output_root: Path
    classifier_reference: Path
    classifier_reference_sha256: str
    regressor_reference: Path
    regressor_reference_sha256: str
    data_snapshot_id: str
    virtual_nodes: tuple[int, ...]
    seeds: tuple[int, ...]
    smoke_virtual_nodes: tuple[int, ...]
    smoke_seeds: tuple[int, ...]
    fixed_qrc: dict[str, Any]
    raw: dict[str, Any]


RunIdentity = tuple[int, int, str]


def temporal_sampling_times(tau: float, virtual_nodes: int) -> tuple[float, ...]:
    """Return the equal cumulative substep endpoints within one fixed interval."""

    if tau <= 0:
        raise ValueError("tau must be positive")
    if virtual_nodes <= 0:
        raise ValueError("virtual_nodes must be positive")
    delta_tau = tau / virtual_nodes
    return tuple(delta_tau * index for index in range(1, virtual_nodes + 1))


def expected_raw_feature_dimension(n_qubits: int, virtual_nodes: int) -> int:
    """Return V times all Z_i and unique ring-edge Z_i Z_j observables."""

    if virtual_nodes <= 0:
        raise ValueError("virtual_nodes must be positive")
    return virtual_nodes * (n_qubits + len(ring_edges(n_qubits)))


def build_encoding_density_grid(
    virtual_nodes: tuple[int, ...] | list[int],
    seeds: tuple[int, ...] | list[int],
) -> tuple[EncodingDensityPoint, ...]:
    """Validate and return a deterministic density-major study grid."""

    node_values = tuple(int(value) for value in virtual_nodes)
    seed_values = tuple(int(value) for value in seeds)
    if not node_values or not seed_values:
        raise ValueError("virtual-node and seed grids must both be non-empty")
    if len(set(node_values)) != len(node_values):
        raise ValueError("virtual-node grid contains duplicates")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("reservoir seed grid contains duplicates")
    unsupported_nodes = sorted(set(node_values) - set(FULL_VIRTUAL_NODE_GRID))
    unsupported_seeds = sorted(set(seed_values) - set(FULL_SEED_GRID))
    if unsupported_nodes:
        raise ValueError(f"unsupported virtual-node counts: {unsupported_nodes}")
    if unsupported_seeds:
        raise ValueError(f"unsupported reservoir seeds: {unsupported_seeds}")
    return tuple(
        EncodingDensityPoint(nodes, seed)
        for nodes in sorted(node_values)
        for seed in sorted(seed_values)
    )


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a YAML mapping")
    return dict(value)


def _text(mapping: dict[str, Any], key: str, location: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location}.{key} must be a non-empty string")
    return value


def _integer_tuple(mapping: dict[str, Any], key: str, location: str) -> tuple[int, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}.{key} must be a non-empty integer list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{location}.{key} must be a non-empty integer list")
    return tuple(int(item) for item in value)


def _assert_reference_contracts(
    config: EncodingDensityStudyConfig,
) -> tuple[ModelExperimentConfig, ModelExperimentConfig]:
    for path, expected in (
        (config.classifier_reference, config.classifier_reference_sha256),
        (config.regressor_reference, config.regressor_reference_sha256),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"frozen encoding-density reference checksum mismatch for {path}: {actual}"
            )
    classifier = load_model_config(config.classifier_reference)
    regressor = load_model_config(config.regressor_reference)
    if classifier.model_type != "qrc_classifier" or classifier.task != TASKS[0]:
        raise ValueError("classifier reference is not the frozen public QRC classifier")
    if regressor.model_type != "qrc_regressor" or regressor.task != TASKS[1]:
        raise ValueError("regressor reference is not the frozen public QRC regressor")
    if classifier.processed_dir != regressor.processed_dir:
        raise ValueError("reference readouts disagree on the frozen processed dataset")
    classifier_qrc = qrc_config_from_model(classifier)
    regressor_qrc = qrc_config_from_model(regressor)
    if classifier_qrc != regressor_qrc:
        raise ValueError("reference readouts disagree on frozen reservoir dynamics")
    if classifier_qrc.virtual_nodes != 2:
        raise ValueError("reference QRC virtual_nodes must remain 2")
    for name in (
        "graph",
        "j_strength",
        "h_strength",
        "tau",
        "input_scaling",
        "state_policy",
        "backend",
    ):
        if getattr(classifier_qrc, name) != config.fixed_qrc.get(name):
            raise ValueError(f"frozen QRC reference disagrees with fixed_qrc.{name}")
    if config.fixed_qrc.get("n_qubits") != 2:
        raise ValueError("encoding-density study must fix n_qubits at 2")
    if config.fixed_qrc.get("exact_noiseless") is not True:
        raise ValueError("encoding-density study requires exact_noiseless: true")
    return classifier, regressor


def load_encoding_density_study_config(path: Path) -> EncodingDensityStudyConfig:
    """Load the study contract and verify frozen reference configuration hashes."""

    source = path.resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "configuration")
    if root.get("schema_version") != 1:
        raise ValueError("QRC encoding-density configuration schema_version must be 1")
    study = _mapping(root.get("study"), "study")
    fixed_qrc = _mapping(root.get("fixed_qrc"), "fixed_qrc")
    project_root = (source.parent / _text(study, "project_root", "study")).resolve()
    config = EncodingDensityStudyConfig(
        source=source,
        project_root=project_root,
        study_id=_text(study, "id", "study"),
        output_root=(project_root / _text(study, "output_root", "study")).resolve(),
        classifier_reference=(
            project_root / _text(study, "classifier_reference", "study")
        ).resolve(),
        classifier_reference_sha256=_text(study, "classifier_reference_sha256", "study"),
        regressor_reference=(project_root / _text(study, "regressor_reference", "study")).resolve(),
        regressor_reference_sha256=_text(study, "regressor_reference_sha256", "study"),
        data_snapshot_id=_text(study, "data_snapshot_id", "study"),
        virtual_nodes=_integer_tuple(study, "virtual_nodes", "study"),
        seeds=_integer_tuple(study, "reservoir_seeds", "study"),
        smoke_virtual_nodes=_integer_tuple(study, "smoke_virtual_nodes", "study"),
        smoke_seeds=_integer_tuple(study, "smoke_seeds", "study"),
        fixed_qrc=fixed_qrc,
        raw=root,
    )
    if config.study_id != STUDY_ID:
        raise ValueError(f"encoding-density study ID must remain {STUDY_ID}")
    build_encoding_density_grid(config.virtual_nodes, config.seeds)
    build_encoding_density_grid(config.smoke_virtual_nodes, config.smoke_seeds)
    if config.virtual_nodes != FULL_VIRTUAL_NODE_GRID or config.seeds != FULL_SEED_GRID:
        raise ValueError("full grid must remain V=1,2,4,8 and seeds 2026,2027,2028")
    _assert_reference_contracts(config)
    return config


def verify_encoding_density_public_data(
    config: EncodingDensityStudyConfig,
    classifier_reference: ModelExperimentConfig,
) -> tuple[ModelDataset, dict[str, Any]]:
    """Verify frozen raw and processed checksums and reject fixture data."""

    data_config = load_data_config(config.project_root / "configs/data_public_market.yaml")
    snapshot = verify_public_snapshot(data_config)
    if snapshot.get("snapshot_id") != config.data_snapshot_id:
        raise ValueError("public snapshot ID disagrees with encoding-density configuration")
    if data_config.snapshot_manifest_path is None:
        raise FileNotFoundError("public snapshot configuration has no manifest")
    dataset = load_model_dataset(classifier_reference.processed_dir)
    if dataset.is_synthetic or dataset.data_source_type != "public_market":
        raise SyntheticResultsError(
            "QRC encoding-density study requires verified non-synthetic public-market data"
        )
    if dataset.manifest.get("source_snapshot_id") != config.data_snapshot_id:
        raise ValueError("processed data manifest disagrees with encoding-density snapshot ID")
    return dataset, {
        "snapshot_id": config.data_snapshot_id,
        "snapshot_manifest_sha256": sha256_file(data_config.snapshot_manifest_path),
        "raw_file_checksums": {
            name: str(record["sha256"]) for name, record in sorted(snapshot["files"].items())
        },
        "processed_manifest_sha256": dataset.processed_checksums["data_manifest.json"],
        "processed_checksums": dataset.processed_checksums,
        "split_row_counts": {
            "train": len(dataset.train.X),
            "validation": len(dataset.validation.X),
            "test": len(dataset.test.X),
        },
    }


def _reference_for_task(config: EncodingDensityStudyConfig, task: str) -> tuple[Path, str]:
    if task == TASKS[0]:
        return config.classifier_reference, config.classifier_reference_sha256
    if task == TASKS[1]:
        return config.regressor_reference, config.regressor_reference_sha256
    raise ValueError(f"unsupported encoding-density task: {task}")


def write_encoding_density_model_config(
    config: EncodingDensityStudyConfig,
    point: EncodingDensityPoint,
    task: str,
) -> Path:
    """Derive one isolated config while changing only n=2, V, seed, and identity."""

    reference_path, reference_sha256 = _reference_for_task(config, task)
    raw = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("frozen QRC reference config must be a mapping")
    model_type = MODEL_TYPES[task]
    config_dir = config.output_root / "generated_configs"
    project_root_setting = os.path.relpath(config.project_root, config_dir)
    output_root_setting = os.path.relpath(config.output_root / "runs", config.project_root)
    feature_cache_setting = os.path.relpath(
        config.output_root / "feature_cache", config.project_root
    )
    raw["experiment"]["name"] = (
        f"qrc_encoding_density_v{point.virtual_nodes}_{model_type}_seed{point.reservoir_seed}"
    )
    raw["experiment"]["project_root"] = Path(project_root_setting).as_posix()
    raw["experiment"]["seed"] = point.reservoir_seed
    raw["experiment"]["output_root"] = Path(output_root_setting).as_posix()
    raw["model"]["parameters"]["n_qubits"] = int(config.fixed_qrc["n_qubits"])
    raw["model"]["parameters"]["virtual_nodes"] = point.virtual_nodes
    raw["model"]["parameters"]["reservoir_seed"] = point.reservoir_seed
    raw["qrc"]["feature_cache"] = Path(feature_cache_setting).as_posix()
    raw["qrc"]["reservoir_seeds"] = list(config.seeds)
    raw["encoding_density_study"] = {
        "id": config.study_id,
        "experimental_variable": "virtual_nodes",
        "virtual_nodes": point.virtual_nodes,
        "reservoir_seed": point.reservoir_seed,
        "n_qubits": int(config.fixed_qrc["n_qubits"]),
        "total_evolution_time_per_input": float(config.fixed_qrc["tau"]),
        "sampling_times": list(
            temporal_sampling_times(float(config.fixed_qrc["tau"]), point.virtual_nodes)
        ),
        "reference_config": reference_path.relative_to(config.project_root).as_posix(),
        "reference_config_sha256": reference_sha256,
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    destination = config_dir / f"{point.key}_{model_type}.yaml"
    destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    derived = load_model_config(destination)
    reference = load_model_config(reference_path)
    derived_qrc = qrc_config_from_model(derived)
    reference_qrc = qrc_config_from_model(reference)
    for name, value in asdict(reference_qrc).items():
        if name in {"n_qubits", "virtual_nodes", "reservoir_seed"}:
            continue
        if asdict(derived_qrc)[name] != value:
            raise ValueError(f"encoding-density config unexpectedly changed QRC field {name}")
    if derived_qrc.n_qubits != 2:
        raise ValueError("derived encoding-density config changed fixed n_qubits=2")
    if derived_qrc.virtual_nodes != point.virtual_nodes:
        raise ValueError("derived encoding-density config has the wrong virtual-node count")
    if derived_qrc.reservoir_seed != point.reservoir_seed:
        raise ValueError("derived encoding-density config has the wrong reservoir seed")
    if derived_qrc.delta_tau * derived_qrc.virtual_nodes != derived_qrc.tau:
        raise ValueError("derived encoding-density config changed total evolution time")
    if derived.output_root != config.output_root / "runs":
        raise ValueError("derived config does not use the isolated output directory")
    return destination


def _completed_identity(
    experiment_dir: Path,
    manifest: dict[str, Any],
    *,
    study_id: str,
    snapshot_id: str,
) -> RunIdentity | None:
    if manifest.get("status") != "success":
        return None
    required = (
        experiment_dir / "config.yaml",
        experiment_dir / "validation_metrics.json",
        experiment_dir / "test_metrics.json",
    )
    if not all(path.is_file() for path in required):
        return None
    raw = yaml.safe_load(required[0].read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    marker = raw.get("encoding_density_study")
    if not isinstance(marker, dict) or marker.get("id") != study_id:
        return None
    if manifest.get("is_synthetic") or manifest.get("data_source_type") != "public_market":
        raise SyntheticResultsError("completed encoding-density run contains synthetic data")
    if manifest.get("data_snapshot_id") != snapshot_id:
        raise ValueError("completed encoding-density run uses a different public snapshot")
    if manifest.get("backend") != "numpy_density_matrix_exact":
        raise ValueError("completed encoding-density run uses a non-reference backend")
    if manifest.get("exact_noiseless") is not True:
        raise ValueError("completed encoding-density run is not exact and noiseless")
    if manifest.get("model_selection_data") != "validation only":
        raise ValueError("completed run did not preserve validation-only model selection")
    if manifest.get("test_evaluated_after_readout_freeze") is not True:
        raise ValueError("completed run evaluated test before readout freeze")
    qrc = manifest.get("qrc_configuration")
    if not isinstance(qrc, dict) or int(qrc.get("n_qubits", -1)) != 2:
        raise ValueError("completed encoding-density run changed fixed n_qubits=2")
    virtual_nodes = int(qrc["virtual_nodes"])
    seed = int(manifest["reservoir_seed"])
    task = str(manifest["task"])
    if (
        virtual_nodes not in FULL_VIRTUAL_NODE_GRID
        or seed not in FULL_SEED_GRID
        or task not in TASKS
    ):
        raise ValueError("completed encoding-density run has an unsupported grid identity")
    return virtual_nodes, seed, task


def discover_completed_encoding_density_runs(
    runs_root: Path,
    *,
    study_id: str,
    snapshot_id: str,
) -> dict[RunIdentity, Path]:
    """Discover latest complete, provenance-valid runs for partial resumption."""

    completed: dict[RunIdentity, Path] = {}
    if not runs_root.is_dir():
        return completed
    for manifest_path in sorted(runs_root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        experiment_dir = manifest_path.parent
        identity = _completed_identity(
            experiment_dir,
            manifest,
            study_id=study_id,
            snapshot_id=snapshot_id,
        )
        if identity is None:
            continue
        previous = completed.get(identity)
        if previous is None or experiment_dir.name > previous.name:
            completed[identity] = experiment_dir
    return completed


def pending_encoding_density_runs(
    grid: tuple[EncodingDensityPoint, ...],
    completed: dict[RunIdentity, Path],
) -> tuple[tuple[EncodingDensityPoint, str], ...]:
    """Return requested task runs absent from a partially completed grid."""

    return tuple(
        (point, task)
        for point in grid
        for task in TASKS
        if (point.virtual_nodes, point.reservoir_seed, task) not in completed
    )


def _validate_run_against_feature_cache(
    experiment_dir: Path,
    *,
    expected_key: str,
    expected_virtual_nodes: int,
) -> dict[str, Any]:
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["qrc_feature_cache_key_checksum"] != expected_key:
        raise ValueError("classifier/regressor did not reuse the prepared feature cache")
    qrc = manifest["qrc_configuration"]
    if int(qrc["n_qubits"]) != 2 or int(qrc["virtual_nodes"]) != expected_virtual_nodes:
        raise ValueError("encoding-density run changed a fixed or requested QRC control")
    if float(qrc["tau"]) != 1.0 or float(qrc["tau"]) / int(qrc["virtual_nodes"]) != float(
        QRCConfig(**qrc).delta_tau
    ):
        raise ValueError("encoding-density run changed the fixed total evolution interval")
    if manifest["qrc_features_generated_without_labels"] is not True:
        raise ValueError("encoding-density run does not prove label-free feature generation")
    if manifest["test_evaluated_after_readout_freeze"] is not True:
        raise ValueError("encoding-density run evaluated test before the readout was frozen")
    return cast(dict[str, Any], manifest)


def _feature_condition_diagnostics(features: np.ndarray[Any, Any]) -> dict[str, Any]:
    diagnostics, singular_values = effective_feature_rank(np.asarray(features, dtype=float))
    return {
        **diagnostics,
        "singular_value_count": len(singular_values),
        "training_feature_checksum": array_checksum(np.asarray(features, dtype=float)),
        "computed_from_split": "train",
        "labels_consumed": False,
    }


def compare_v2_with_qubit_scaling_reference(
    *,
    project_root: Path,
    cache_key_checksum: str,
    encoding_cache_dir: Path,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Compare V=2 features with the existing exact two-qubit scaling cache if present."""

    reference_dir = project_root / "results/qrc_qubit_scaling/feature_cache" / cache_key_checksum
    result: dict[str, Any] = {
        "reference": "existing exact two-qubit V=2 qubit-scaling cache",
        "cache_key_checksum": cache_key_checksum,
        "tolerance": tolerance,
        "reference_cache_available": reference_dir.is_dir(),
        "compared": False,
        "within_tolerance": None,
    }
    if not reference_dir.is_dir():
        return result
    reference_arrays = reference_dir / "qrc_features.npz"
    encoding_arrays = encoding_cache_dir / "qrc_features.npz"
    if not reference_arrays.is_file() or not encoding_arrays.is_file():
        raise FileNotFoundError("V=2 reference comparison found an incomplete feature cache")
    maximum = 0.0
    checksums_match = True
    split_checksums: dict[str, dict[str, str]] = {}
    with np.load(reference_arrays) as reference, np.load(encoding_arrays) as encoding:
        for split in ("train", "validation", "test"):
            reference_values = np.asarray(reference[split], dtype=float)
            encoding_values = np.asarray(encoding[split], dtype=float)
            if reference_values.shape != encoding_values.shape:
                raise ValueError(f"V=2 reference feature shape differs for {split}")
            difference = float(np.max(np.abs(reference_values - encoding_values), initial=0.0))
            maximum = max(maximum, difference)
            reference_checksum = array_checksum(reference_values)
            encoding_checksum = array_checksum(encoding_values)
            checksums_match = checksums_match and reference_checksum == encoding_checksum
            split_checksums[split] = {
                "reference": reference_checksum,
                "encoding_density": encoding_checksum,
            }
    result.update(
        {
            "compared": True,
            "within_tolerance": maximum <= tolerance,
            "maximum_absolute_difference": maximum,
            "array_checksums_match": checksums_match,
            "split_checksums": split_checksums,
            "reference_cache_directory": reference_dir.relative_to(project_root).as_posix(),
        }
    )
    if maximum > tolerance:
        raise ValueError(
            f"V=2 encoding-density features disagree with exact two-qubit reference: {maximum}"
        )
    return result


def collect_encoding_density_rows(
    run_dirs: dict[RunIdentity, Path],
    *,
    point_metadata: dict[tuple[int, int], dict[str, Any]],
    cache_hits: dict[RunIdentity, bool] | None = None,
    repository_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Collect split metrics, timings, conditioning, checksums, and provenance."""

    required_manifest_fields = {
        "configuration_checksum",
        "processed_data_checksums",
        "selected_hyperparameters",
        "qrc_raw_feature_dimension",
        "readout_shape",
        "trainable_readout_parameters",
        "state_generation_time",
        "readout_fitting_time",
        "qrc_feature_cache_hit",
        "qrc_feature_cache_key_checksum",
        "data_snapshot_id",
        "data_manifest_checksum",
        "backend",
        "exact_noiseless",
        "git",
        "python_version",
        "operating_system",
        "execution_platform",
        "package_versions",
        "qrc_configuration",
        "model_selection_data",
        "test_evaluated_after_readout_freeze",
    }
    rows: list[dict[str, Any]] = []
    for identity, experiment_dir in sorted(run_dirs.items()):
        virtual_nodes, seed, task = identity
        manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
        missing = sorted(required_manifest_fields - set(manifest))
        if missing:
            raise ValueError(f"encoding-density run manifest omits required fields: {missing}")
        if manifest["model_selection_data"] != "validation only":
            raise ValueError("encoding-density row violates validation-only model selection")
        if manifest["test_evaluated_after_readout_freeze"] is not True:
            raise ValueError("encoding-density row violates frozen-readout test evaluation")
        metadata = point_metadata[(virtual_nodes, seed)]
        diagnostics = metadata["condition_diagnostics"]
        timing = json.loads((experiment_dir / "timing.json").read_text(encoding="utf-8"))
        state_seconds = float(metadata["state_generation_seconds"])
        readout_seconds = float(manifest["readout_fitting_time"])
        task_runtime = state_seconds + sum(
            float(value)
            for name, value in timing.items()
            if name != "state_generation_seconds" and isinstance(value, (int, float))
        )
        experiment_directory = experiment_dir
        if repository_root is not None:
            experiment_directory = experiment_dir.relative_to(repository_root)
        common: dict[str, Any] = {
            "experiment_id": manifest["experiment_id"],
            "experiment_directory": experiment_directory.as_posix(),
            "virtual_nodes": virtual_nodes,
            "n_qubits": 2,
            "reservoir_seed": seed,
            "task": task,
            "model_type": manifest["model_type"],
            "selected_ridge_alpha": manifest["selected_hyperparameters"]["ridge_alpha"],
            "raw_feature_dimension": manifest["qrc_raw_feature_dimension"],
            "readout_shape": manifest["readout_shape"],
            "trainable_readout_parameters": manifest["trainable_readout_parameters"],
            "state_generation_seconds": state_seconds,
            "feature_generation_seconds": state_seconds,
            "feature_preparation_wall_seconds": metadata["feature_preparation_wall_seconds"],
            "readout_fitting_seconds": readout_seconds,
            "total_runtime_seconds": task_runtime,
            "cache_hit": (
                cache_hits[identity]
                if cache_hits is not None and identity in cache_hits
                else manifest["qrc_feature_cache_hit"]
            ),
            "run_cache_hit": manifest["qrc_feature_cache_hit"],
            "cache_key_checksum": manifest["qrc_feature_cache_key_checksum"],
            "cache_array_checksums": metadata["array_checksums"],
            "study_configuration_checksum": metadata["study_configuration_checksum"],
            "run_configuration_checksum": manifest["configuration_checksum"],
            "data_snapshot_id": manifest["data_snapshot_id"],
            "data_manifest_checksum": manifest["data_manifest_checksum"],
            "processed_data_checksums": manifest["processed_data_checksums"],
            "backend": manifest["backend"],
            "exact_noiseless": manifest["exact_noiseless"],
            "git_commit": manifest["git"]["commit"],
            "git_dirty": manifest["git"]["dirty"],
            "python_version": manifest["python_version"],
            "platform": manifest["operating_system"],
            "operating_system": manifest["operating_system"],
            "execution_platform": manifest["execution_platform"],
            "package_versions": manifest["package_versions"],
            "model_selection_data": manifest["model_selection_data"],
            "test_evaluated_after_readout_freeze": manifest["test_evaluated_after_readout_freeze"],
            "total_evolution_time_per_input": float(manifest["qrc_configuration"]["tau"]),
            "substep_evolution_time": float(manifest["qrc_configuration"]["tau"]) / virtual_nodes,
            "sampling_times": list(
                temporal_sampling_times(float(manifest["qrc_configuration"]["tau"]), virtual_nodes)
            ),
            "condition_number": diagnostics["condition_number"],
            "effective_rank": diagnostics["effective_rank"],
            "numerical_rank": diagnostics["numerical_rank"],
            "largest_singular_value": diagnostics["largest_singular_value"],
            "smallest_retained_singular_value": diagnostics["smallest_retained_singular_value"],
            "rank_tolerance": diagnostics["rank_tolerance"],
            "numerical_diagnostics": metadata["numerical_diagnostics"],
            "v2_reference_agreement": metadata.get("v2_reference_agreement"),
        }
        for split in ("validation", "test"):
            metrics = json.loads(
                (experiment_dir / f"{split}_metrics.json").read_text(encoding="utf-8")
            )
            task_metrics = {
                name: value
                for name, value in metrics.items()
                if name not in {"data_source_type", "is_synthetic", "data_warning"}
            }
            rows.append({**common, "split": split, **task_metrics})
    return rows


def aggregate_encoding_density_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate performance and resource metrics over seeds for each V."""

    aggregated: list[dict[str, Any]] = []
    for virtual_nodes in sorted({int(row["virtual_nodes"]) for row in rows}):
        for task in TASKS:
            for split in ("validation", "test"):
                subset = [
                    row
                    for row in rows
                    if int(row["virtual_nodes"]) == virtual_nodes
                    and row["task"] == task
                    and row["split"] == split
                ]
                if not subset:
                    continue
                for metric in (*METRICS_BY_TASK[task], *RESOURCE_METRICS):
                    values = np.asarray(
                        [float(row[metric]) for row in subset if row.get(metric) is not None],
                        dtype=float,
                    )
                    if not len(values):
                        continue
                    package_versions = subset[0]["package_versions"]
                    if any(row["package_versions"] != package_versions for row in subset):
                        package_versions = {
                            "note": "multiple package environments; inspect per-run table"
                        }
                    aggregated.append(
                        {
                            "virtual_nodes": virtual_nodes,
                            "n_qubits": 2,
                            "task": task,
                            "split": split,
                            "metric": metric,
                            "mean": float(values.mean()),
                            "standard_deviation": float(values.std(ddof=0)),
                            "minimum": float(values.min()),
                            "maximum": float(values.max()),
                            "seed_count": len(values),
                            "reservoir_seeds": sorted(int(row["reservoir_seed"]) for row in subset),
                            "data_snapshot_id": subset[0]["data_snapshot_id"],
                            "backend": subset[0]["backend"],
                            "exact_noiseless": subset[0]["exact_noiseless"],
                            "git_commits": sorted({str(row["git_commit"]) for row in subset}),
                            "package_versions": package_versions,
                        }
                    )
    return aggregated


def build_validation_candidate_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize validation-only quality, cost, and seed variability without selecting."""

    candidates: list[dict[str, Any]] = []
    for virtual_nodes in sorted({int(row["virtual_nodes"]) for row in rows}):
        classification = [
            row
            for row in rows
            if row["virtual_nodes"] == virtual_nodes
            and row["task"] == TASKS[0]
            and row["split"] == "validation"
        ]
        regression = [
            row
            for row in rows
            if row["virtual_nodes"] == virtual_nodes
            and row["task"] == TASKS[1]
            and row["split"] == "validation"
        ]
        if not classification or not regression:
            continue

        def summary(source: list[dict[str, Any]], name: str) -> tuple[float, float]:
            values = np.asarray([float(row[name]) for row in source], dtype=float)
            return float(values.mean()), float(values.std(ddof=0))

        macro_mean, macro_sd = summary(classification, "macro_f1")
        transition_mean, transition_sd = summary(classification, "transition_pr_auc")
        qlike_mean, qlike_sd = summary(regression, "qlike")
        runtime_mean, runtime_sd = summary(classification, "feature_generation_seconds")
        condition_values = np.asarray(
            [
                float(row["condition_number"])
                for row in classification
                if row.get("condition_number") is not None
            ],
            dtype=float,
        )
        candidates.append(
            {
                "virtual_nodes": virtual_nodes,
                "n_qubits": 2,
                "validation_macro_f1_mean": macro_mean,
                "validation_macro_f1_seed_standard_deviation": macro_sd,
                "validation_transition_pr_auc_mean": transition_mean,
                "validation_transition_pr_auc_seed_standard_deviation": transition_sd,
                "validation_qlike_mean": qlike_mean,
                "validation_qlike_seed_standard_deviation": qlike_sd,
                "raw_feature_dimension": int(classification[0]["raw_feature_dimension"]),
                "feature_generation_seconds_mean": runtime_mean,
                "feature_generation_seconds_seed_standard_deviation": runtime_sd,
                "condition_number_mean": (
                    float(condition_values.mean()) if len(condition_values) else None
                ),
                "reservoir_seeds": sorted(int(row["reservoir_seed"]) for row in classification),
                "seed_count": len(classification),
                "selection_basis": "validation only",
                "test_metrics_used": False,
                "architecture_frozen": False,
                "next_required_ablation": "state-memory policy",
            }
        )
    return candidates


def encoding_density_resource_estimates(
    virtual_nodes: tuple[int, ...] | list[int],
    *,
    split_rows: int,
    train_rows: int,
) -> list[dict[str, int]]:
    """Return analytical exact-state and feature-cache memory estimates."""

    hilbert_dimension = 2**2
    rows: list[dict[str, int]] = []
    for nodes in sorted(virtual_nodes):
        raw_dimension = expected_raw_feature_dimension(2, nodes)
        rows.append(
            {
                "virtual_nodes": nodes,
                "n_qubits": 2,
                "temporal_substeps_per_input": nodes,
                "raw_feature_dimension": raw_dimension,
                "hilbert_dimension": hilbert_dimension,
                "density_matrix_elements": hilbert_dimension * hilbert_dimension,
                "estimated_peak_density_matrix_bytes": (
                    3 * hilbert_dimension * hilbert_dimension * 16
                ),
                "estimated_cached_feature_bytes": split_rows * raw_dimension * 8,
                "estimated_train_condition_matrix_bytes": train_rows * raw_dimension * 8,
            }
        )
    return rows


def _csv_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for row in rows:
        values = dict(row)
        for name, value in tuple(values.items()):
            if isinstance(value, (dict, list, tuple)):
                values[name] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        safe.append(values)
    return safe


def _plot_metric(
    table: pd.DataFrame,
    *,
    metric: str,
    task: str,
    split: str,
    ylabel: str,
    destination: Path,
    log_scale: bool = False,
) -> None:
    subset = table.loc[table["task"].eq(task) & table["split"].eq(split)].copy()
    if subset.empty or metric not in subset:
        raise ValueError(f"encoding-density results omit {split} metric {metric}")
    subset = subset.loc[subset[metric].notna()]
    if subset.empty:
        raise ValueError(f"encoding-density results have no finite {metric} values")
    figure, axis = plt.subplots(figsize=(5.8, 3.8))
    for seed, seed_rows in subset.groupby("reservoir_seed"):
        ordered = seed_rows.sort_values("virtual_nodes")
        axis.plot(
            ordered["virtual_nodes"],
            ordered[metric],
            marker="o",
            linewidth=0.8,
            alpha=0.38,
            label=f"seed {seed}",
        )
    grouped = subset.groupby("virtual_nodes")[metric]
    mean = grouped.mean()
    standard_deviation = grouped.std(ddof=0).fillna(0.0)
    axis.errorbar(
        mean.index,
        mean.to_numpy(dtype=float),
        yerr=standard_deviation.to_numpy(dtype=float),
        color="black",
        linewidth=1.8,
        marker="o",
        capsize=3,
        label="mean ± population SD",
        zorder=4,
    )
    axis.set(
        xlabel="virtual nodes per fixed input interval",
        ylabel=ylabel,
        xticks=sorted(int(value) for value in subset["virtual_nodes"].unique()),
    )
    if log_scale and bool((subset[metric].astype(float) > 0).all()):
        axis.set_yscale("log")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=220)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def plot_encoding_density_figures(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    """Write the nine required seed-level plus mean/uncertainty figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(_csv_safe_rows(rows))
    specifications = {
        "validation_macro_f1_vs_virtual_nodes": (
            "macro_f1",
            TASKS[0],
            "validation",
            "validation macro F1",
            False,
        ),
        "validation_transition_pr_auc_vs_virtual_nodes": (
            "transition_pr_auc",
            TASKS[0],
            "validation",
            "validation transition PR-AUC",
            False,
        ),
        "validation_qlike_vs_virtual_nodes": (
            "qlike",
            TASKS[1],
            "validation",
            "validation QLIKE",
            False,
        ),
        "test_macro_f1_vs_virtual_nodes": (
            "macro_f1",
            TASKS[0],
            "test",
            "test macro F1",
            False,
        ),
        "test_transition_pr_auc_vs_virtual_nodes": (
            "transition_pr_auc",
            TASKS[0],
            "test",
            "test transition PR-AUC",
            False,
        ),
        "test_qlike_vs_virtual_nodes": (
            "qlike",
            TASKS[1],
            "test",
            "test QLIKE",
            False,
        ),
        "feature_dimension_vs_virtual_nodes": (
            "raw_feature_dimension",
            TASKS[0],
            "validation",
            "raw QRC feature dimension",
            False,
        ),
        "feature_generation_time_vs_virtual_nodes": (
            "feature_generation_seconds",
            TASKS[0],
            "validation",
            "feature-generation time (seconds)",
            False,
        ),
        "condition_number_vs_virtual_nodes": (
            "condition_number",
            TASKS[0],
            "validation",
            "training-feature condition number",
            True,
        ),
    }
    outputs: dict[str, Path] = {}
    for name, (metric, task, split, ylabel, log_scale) in specifications.items():
        destination = output_dir / name
        _plot_metric(
            table,
            metric=metric,
            task=task,
            split=split,
            ylabel=ylabel,
            destination=destination,
            log_scale=log_scale,
        )
        outputs[f"{name}_png"] = destination.with_suffix(".png")
        outputs[f"{name}_pdf"] = destination.with_suffix(".pdf")
    return outputs


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "points": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("points"), dict):
        raise ValueError("encoding-density state file is invalid")
    return value


def _write_tables(
    output_root: Path,
    rows: list[dict[str, Any]],
    aggregated: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    resources: list[dict[str, int]],
) -> dict[str, Path]:
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_run_json": tables / "qrc_encoding_density_per_run.json",
        "per_run_csv": tables / "qrc_encoding_density_per_run.csv",
        "aggregate_json": tables / "qrc_encoding_density_aggregate.json",
        "aggregate_csv": tables / "qrc_encoding_density_aggregate.csv",
        "validation_candidates_json": tables / "qrc_encoding_density_validation_candidates.json",
        "validation_candidates_csv": tables / "qrc_encoding_density_validation_candidates.csv",
        "resources_json": tables / "qrc_encoding_density_resources.json",
        "resources_csv": tables / "qrc_encoding_density_resources.csv",
    }
    _write_json(outputs["per_run_json"], {"schema_version": 1, "rows": rows})
    pd.DataFrame(_csv_safe_rows(rows)).to_csv(outputs["per_run_csv"], index=False)
    _write_json(outputs["aggregate_json"], {"schema_version": 1, "rows": aggregated})
    pd.DataFrame(_csv_safe_rows(aggregated)).to_csv(outputs["aggregate_csv"], index=False)
    _write_json(
        outputs["validation_candidates_json"],
        {
            "schema_version": 1,
            "selection_basis": "validation only",
            "test_metrics_used": False,
            "architecture_frozen": False,
            "rows": candidates,
        },
    )
    pd.DataFrame(_csv_safe_rows(candidates)).to_csv(
        outputs["validation_candidates_csv"], index=False
    )
    _write_json(outputs["resources_json"], {"schema_version": 1, "rows": resources})
    pd.DataFrame(resources).to_csv(outputs["resources_csv"], index=False)
    return outputs


def run_encoding_density(
    config_path: Path,
    *,
    virtual_nodes: tuple[int, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    smoke: bool = False,
    resume: bool = True,
) -> Path:
    """Run or resume the controlled exact two-qubit encoding-density grid."""

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config = load_encoding_density_study_config(config_path)
    selected_nodes = virtual_nodes or (
        config.smoke_virtual_nodes if smoke else config.virtual_nodes
    )
    selected_seeds = seeds or (config.smoke_seeds if smoke else config.seeds)
    grid = build_encoding_density_grid(selected_nodes, selected_seeds)
    if smoke and (len(selected_nodes) > 3 or len(selected_seeds) > 1):
        raise ValueError("smoke mode permits at most three densities and one seed")
    classifier_reference, _ = _assert_reference_contracts(config)
    dataset, data_provenance = verify_encoding_density_public_data(config, classifier_reference)
    config.output_root.mkdir(parents=True, exist_ok=True)
    runs_root = config.output_root / "runs"
    completed = discover_completed_encoding_density_runs(
        runs_root,
        study_id=config.study_id,
        snapshot_id=config.data_snapshot_id,
    )
    initially_completed = dict(completed)
    state_path = config.output_root / "encoding_density_state.json"
    state = _load_state(state_path)
    point_state = state["points"]
    if not isinstance(point_state, dict):
        raise ValueError("encoding-density state points must be a mapping")
    executed: list[dict[str, Any]] = []
    resumed: list[dict[str, Any]] = []
    cache_hits: dict[RunIdentity, bool] = {}
    point_metadata: dict[tuple[int, int], dict[str, Any]] = {}
    reference_agreements: dict[str, dict[str, Any]] = {}
    study_checksum = sha256_file(config.source)

    for point in grid:
        classifier_path = write_encoding_density_model_config(config, point, TASKS[0])
        regressor_path = write_encoding_density_model_config(config, point, TASKS[1])
        feature_started = time.perf_counter()
        bundle = generate_qrc_features(
            classifier_path,
            allow_synthetic_results=False,
            reservoir_seed=point.reservoir_seed,
        )
        feature_preparation_wall_seconds = time.perf_counter() - feature_started
        cache_key = str(bundle.metadata["cache_key_checksum"])
        cache_directory = bundle.cache_dir.relative_to(config.project_root).as_posix()
        expected_dimension = expected_raw_feature_dimension(2, point.virtual_nodes)
        if bundle.train.shape[1] != expected_dimension:
            raise ValueError(
                f"V={point.virtual_nodes} produced {bundle.train.shape[1]} features; "
                f"expected {expected_dimension}"
            )
        condition_diagnostics = _feature_condition_diagnostics(bundle.train)
        v2_agreement: dict[str, Any] | None = None
        if point.virtual_nodes == 2:
            v2_agreement = compare_v2_with_qubit_scaling_reference(
                project_root=config.project_root,
                cache_key_checksum=cache_key,
                encoding_cache_dir=bundle.cache_dir,
            )
            reference_agreements[point.key] = v2_agreement
        metadata = {
            "virtual_nodes": point.virtual_nodes,
            "n_qubits": 2,
            "reservoir_seed": point.reservoir_seed,
            "feature_preparation_cache_hit": bundle.cache_hit,
            "feature_preparation_wall_seconds": feature_preparation_wall_seconds,
            "state_generation_seconds": float(
                bundle.metadata["resource_metadata"]["state_generation_seconds"]
            ),
            "cache_key_checksum": cache_key,
            "cache_directory": cache_directory,
            "array_checksums": bundle.metadata["array_checksums"],
            "study_configuration_checksum": study_checksum,
            "condition_diagnostics": condition_diagnostics,
            "numerical_diagnostics": bundle.metadata["numerical_diagnostics"],
            "total_evolution_time_per_input": float(config.fixed_qrc["tau"]),
            "substep_evolution_time": float(config.fixed_qrc["tau"]) / point.virtual_nodes,
            "sampling_times": list(
                temporal_sampling_times(float(config.fixed_qrc["tau"]), point.virtual_nodes)
            ),
            "v2_reference_agreement": v2_agreement,
        }
        existing_point_state = point_state.get(point.key)
        if isinstance(existing_point_state, dict):
            if existing_point_state.get("cache_key_checksum") != cache_key:
                raise ValueError("resumed encoding-density point cache key changed")
            if existing_point_state.get("cache_directory") != cache_directory:
                raise ValueError("resumed encoding-density point cache directory changed")
        point_state[point.key] = metadata
        point_metadata[(point.virtual_nodes, point.reservoir_seed)] = metadata
        diagnostics_path = config.output_root / "condition_diagnostics" / f"{point.key}.json"
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(diagnostics_path, metadata)
        _write_json(state_path, state)

        for task, model_config_path in (
            (TASKS[0], classifier_path),
            (TASKS[1], regressor_path),
        ):
            identity = (point.virtual_nodes, point.reservoir_seed, task)
            if resume and identity in completed:
                experiment_dir = completed[identity]
                resumed.append(
                    {
                        "virtual_nodes": point.virtual_nodes,
                        "reservoir_seed": point.reservoir_seed,
                        "task": task,
                        "experiment_directory": experiment_dir.relative_to(
                            config.project_root
                        ).as_posix(),
                    }
                )
            else:
                experiment_dir = run_qrc_experiment(
                    model_config_path,
                    allow_synthetic_results=False,
                    reservoir_seed=point.reservoir_seed,
                )
                completed[identity] = experiment_dir
                executed.append(
                    {
                        "virtual_nodes": point.virtual_nodes,
                        "reservoir_seed": point.reservoir_seed,
                        "task": task,
                        "experiment_directory": experiment_dir.relative_to(
                            config.project_root
                        ).as_posix(),
                    }
                )
            _validate_run_against_feature_cache(
                experiment_dir,
                expected_key=cache_key,
                expected_virtual_nodes=point.virtual_nodes,
            )
            cache_hits[identity] = (
                bool(metadata["feature_preparation_cache_hit"]) if task == TASKS[0] else True
            )

    requested_runs = {
        identity: completed[identity]
        for point in grid
        for task in TASKS
        if (identity := (point.virtual_nodes, point.reservoir_seed, task)) in completed
    }
    expected_run_count = len(grid) * len(TASKS)
    if len(requested_runs) != expected_run_count:
        raise RuntimeError(
            f"encoding-density grid incomplete after execution: {len(requested_runs)} "
            f"of {expected_run_count} task runs"
        )
    rows = collect_encoding_density_rows(
        requested_runs,
        point_metadata=point_metadata,
        cache_hits=cache_hits,
        repository_root=config.project_root,
    )
    aggregated = aggregate_encoding_density_rows(rows)
    candidates = build_validation_candidate_table(rows)
    split_rows = len(dataset.train.X) + len(dataset.validation.X) + len(dataset.test.X)
    resources = encoding_density_resource_estimates(
        tuple(sorted({point.virtual_nodes for point in grid})),
        split_rows=split_rows,
        train_rows=len(dataset.train.X),
    )
    outputs = _write_tables(config.output_root, rows, aggregated, candidates, resources)
    figures = plot_encoding_density_figures(rows, config.output_root / "figures")
    summary_path = config.output_root / "encoding_density_run_summary.json"
    summary = {
        "schema_version": 1,
        "study_id": config.study_id,
        "status": "success",
        "mode": "smoke" if smoke else "full_or_custom",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        **runtime_metadata(),
        "git": _git_metadata(config.project_root),
        "configuration": {
            "path": config.source.relative_to(config.project_root).as_posix(),
            "sha256": study_checksum,
        },
        "reference_configs": {
            "classifier": {
                "path": config.classifier_reference.relative_to(config.project_root).as_posix(),
                "sha256": config.classifier_reference_sha256,
            },
            "regressor": {
                "path": config.regressor_reference.relative_to(config.project_root).as_posix(),
                "sha256": config.regressor_reference_sha256,
            },
        },
        "experimental_variable": "virtual_nodes",
        "fixed_qrc": config.fixed_qrc,
        "temporal_multiplexing_contract": {
            "total_evolution_time_per_market_input": float(config.fixed_qrc["tau"]),
            "substep_duration_definition": "tau / virtual_nodes",
            "sampling_time_definition": "tau * k / virtual_nodes for k=1,...,V",
            "input_state_policy": "carry_inputs",
        },
        "grid": [asdict(point) for point in grid],
        "data_provenance": data_provenance,
        "resume_enabled": resume,
        "completed_before_run": len(initially_completed),
        "executed_runs": executed,
        "resumed_runs": resumed,
        "feature_cache_points": {point.key: point_state[point.key] for point in grid},
        "v2_reference_agreements": reference_agreements,
        "per_run_row_count": len(rows),
        "aggregate_row_count": len(aggregated),
        "validation_candidate_row_count": len(candidates),
        "outputs": {
            **{
                name: path.relative_to(config.project_root).as_posix()
                for name, path in outputs.items()
            },
            **{
                f"figure_{name}": path.relative_to(config.project_root).as_posix()
                for name, path in figures.items()
            },
        },
        "architecture_selection": {
            "selection_basis": "validation only",
            "test_metrics_used": False,
            "architecture_frozen": False,
            "reason": "state-memory policy ablation remains outstanding",
        },
        "interpretation": (
            "controlled exact classical simulation of a quantum reservoir; "
            "not physical-QPU execution or a quantum-advantage claim"
        ),
    }
    _write_json(summary_path, summary)
    return summary_path
