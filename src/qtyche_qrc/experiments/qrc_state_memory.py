"""Controlled exact-noiseless QRC cross-observation state-memory ablation."""

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
from numpy.typing import NDArray

from qtyche_qrc.data.config import load_data_config
from qtyche_qrc.data.download import sha256_file, verify_public_snapshot
from qtyche_qrc.experiments.model_config import ModelExperimentConfig, load_model_config
from qtyche_qrc.experiments.qrc_capacity import (
    effective_feature_rank,
    feature_autocorrelation,
)
from qtyche_qrc.experiments.qrc_run import (
    generate_qrc_features,
    qrc_config_from_model,
    run_qrc_experiment,
)
from qtyche_qrc.experiments.run import SyntheticResultsError, _git_metadata, _write_json
from qtyche_qrc.models.dataset import ModelDataset, load_model_dataset
from qtyche_qrc.models.qrc.backends import trace_distance
from qtyche_qrc.models.qrc.encoding import array_checksum
from qtyche_qrc.models.qrc.reservoir import QRCConfig, QuantumReservoir
from qtyche_qrc.runtime import runtime_metadata

STUDY_ID = "exact_qrc_state_memory_ablation_v1"
STATE_POLICIES = ("carry_inputs", "reset_each_input")
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
    "feature_generation_seconds",
    "readout_fitting_seconds",
    "total_runtime_seconds",
    "raw_feature_dimension",
    "trainable_readout_parameters",
    "effective_rank",
    "numerical_rank",
    "condition_number",
    "lag1_mean_absolute_feature_autocorrelation",
    "lag1_mean_absolute_input_feature_correlation",
    "perturbation_final_trace_distance",
    "perturbation_trace_distance_auc",
)
PAIRED_METRICS = {
    "macro_f1": ("regime_classification", False),
    "balanced_accuracy": ("regime_classification", False),
    "transition_pr_auc": ("regime_classification", False),
    "qlike": ("rv_regression", True),
    "rmse": ("rv_regression", True),
}


@dataclass(frozen=True, order=True)
class StateMemoryPoint:
    """One state policy and independent reservoir seed."""

    state_policy: str
    reservoir_seed: int

    @property
    def key(self) -> str:
        return f"{self.state_policy}_seed{self.reservoir_seed}"


@dataclass(frozen=True)
class StateMemoryStudyConfig:
    """Validated paths, selected architecture, grids, and frozen controls."""

    source: Path
    project_root: Path
    study_id: str
    output_root: Path
    classifier_reference: Path
    classifier_reference_sha256: str
    regressor_reference: Path
    regressor_reference_sha256: str
    encoding_density_candidates: Path
    encoding_density_candidates_sha256: str
    data_snapshot_id: str
    selected_virtual_nodes: int
    state_policies: tuple[str, ...]
    seeds: tuple[int, ...]
    smoke_state_policies: tuple[str, ...]
    smoke_seeds: tuple[int, ...]
    maximum_lag: int
    perturbation_steps: int
    fixed_qrc: dict[str, Any]
    raw: dict[str, Any]


RunIdentity = tuple[str, int, str]


def build_state_memory_grid(
    state_policies: tuple[str, ...] | list[str],
    seeds: tuple[int, ...] | list[int],
) -> tuple[StateMemoryPoint, ...]:
    """Validate and return a deterministic policy-major ablation grid."""

    policies = tuple(str(value) for value in state_policies)
    seed_values = tuple(int(value) for value in seeds)
    if not policies or not seed_values:
        raise ValueError("state-policy and seed grids must both be non-empty")
    if len(set(policies)) != len(policies):
        raise ValueError("state-policy grid contains duplicates")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("reservoir seed grid contains duplicates")
    unsupported_policies = sorted(set(policies) - set(STATE_POLICIES))
    unsupported_seeds = sorted(set(seed_values) - set(FULL_SEED_GRID))
    if unsupported_policies:
        raise ValueError(f"unsupported state policies: {unsupported_policies}")
    if unsupported_seeds:
        raise ValueError(f"unsupported reservoir seeds: {unsupported_seeds}")
    return tuple(
        StateMemoryPoint(policy, seed)
        for policy in STATE_POLICIES
        if policy in policies
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


def _integer(mapping: dict[str, Any], key: str, location: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location}.{key} must be an integer")
    return int(value)


def _integer_tuple(mapping: dict[str, Any], key: str, location: str) -> tuple[int, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}.{key} must be a non-empty integer list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{location}.{key} must be a non-empty integer list")
    return tuple(int(item) for item in value)


def _string_tuple(mapping: dict[str, Any], key: str, location: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}.{key} must be a non-empty string list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{location}.{key} must be a non-empty string list")
    return tuple(str(item) for item in value)


def _assert_reference_contracts(
    config: StateMemoryStudyConfig,
) -> tuple[ModelExperimentConfig, ModelExperimentConfig]:
    for path, expected in (
        (config.classifier_reference, config.classifier_reference_sha256),
        (config.regressor_reference, config.regressor_reference_sha256),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"frozen state-memory reference checksum mismatch for {path}: {actual}"
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
    if classifier_qrc.state_policy != "carry_inputs":
        raise ValueError("frozen public reference must retain carry_inputs")
    for name in (
        "virtual_nodes",
        "graph",
        "j_strength",
        "h_strength",
        "tau",
        "input_scaling",
        "backend",
    ):
        if getattr(classifier_qrc, name) != config.fixed_qrc.get(name):
            raise ValueError(f"frozen QRC reference disagrees with fixed_qrc.{name}")
    if config.fixed_qrc.get("n_qubits") != 2:
        raise ValueError("state-memory ablation must fix n_qubits at 2")
    if config.fixed_qrc.get("exact_noiseless") is not True:
        raise ValueError("state-memory ablation requires exact_noiseless: true")
    return classifier, regressor


def _verify_selected_virtual_nodes(config: StateMemoryStudyConfig) -> dict[str, Any]:
    actual = sha256_file(config.encoding_density_candidates)
    if actual != config.encoding_density_candidates_sha256:
        raise ValueError(
            "encoding-density validation candidate checksum mismatch: "
            f"{actual} != {config.encoding_density_candidates_sha256}"
        )
    evidence = json.loads(config.encoding_density_candidates.read_text(encoding="utf-8"))
    if evidence.get("selection_basis") != "validation only":
        raise ValueError("encoding-density architecture evidence is not validation-only")
    if evidence.get("test_metrics_used") is not False:
        raise ValueError("encoding-density architecture evidence used test metrics")
    rows = evidence.get("rows")
    if not isinstance(rows, list):
        raise ValueError("encoding-density candidate evidence omits rows")
    selected = next(
        (
            row
            for row in rows
            if isinstance(row, dict)
            and int(row.get("virtual_nodes", -1)) == config.selected_virtual_nodes
        ),
        None,
    )
    if selected is None:
        raise ValueError("selected virtual-node count is absent from validation candidates")
    if config.selected_virtual_nodes != 2:
        raise ValueError("state-memory protocol requires validation-selected V=2")
    if int(config.fixed_qrc.get("virtual_nodes", -1)) != config.selected_virtual_nodes:
        raise ValueError("fixed QRC virtual_nodes disagrees with selected architecture")
    return {
        "path": config.encoding_density_candidates.relative_to(config.project_root).as_posix(),
        "sha256": actual,
        "selection_basis": "validation only",
        "test_metrics_used": False,
        "selected_virtual_nodes": config.selected_virtual_nodes,
        "selected_candidate": selected,
        "decision_scope": "protocol-pinned validation/performance-cost candidate",
    }


def load_state_memory_study_config(path: Path) -> StateMemoryStudyConfig:
    """Load and verify the state-memory study contract and selected architecture."""

    source = path.resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "configuration")
    if root.get("schema_version") != 1:
        raise ValueError("QRC state-memory configuration schema_version must be 1")
    study = _mapping(root.get("study"), "study")
    fixed_qrc = _mapping(root.get("fixed_qrc"), "fixed_qrc")
    diagnostics = _mapping(root.get("diagnostics"), "diagnostics")
    project_root = (source.parent / _text(study, "project_root", "study")).resolve()
    config = StateMemoryStudyConfig(
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
        encoding_density_candidates=(
            project_root / _text(study, "encoding_density_candidates", "study")
        ).resolve(),
        encoding_density_candidates_sha256=_text(
            study, "encoding_density_candidates_sha256", "study"
        ),
        data_snapshot_id=_text(study, "data_snapshot_id", "study"),
        selected_virtual_nodes=_integer(study, "selected_virtual_nodes", "study"),
        state_policies=_string_tuple(study, "state_policies", "study"),
        seeds=_integer_tuple(study, "reservoir_seeds", "study"),
        smoke_state_policies=_string_tuple(study, "smoke_state_policies", "study"),
        smoke_seeds=_integer_tuple(study, "smoke_seeds", "study"),
        maximum_lag=_integer(diagnostics, "maximum_lag", "diagnostics"),
        perturbation_steps=_integer(diagnostics, "perturbation_steps", "diagnostics"),
        fixed_qrc=fixed_qrc,
        raw=root,
    )
    if config.study_id != STUDY_ID:
        raise ValueError(f"state-memory study ID must remain {STUDY_ID}")
    if config.state_policies != STATE_POLICIES or config.seeds != FULL_SEED_GRID:
        raise ValueError("full grid must remain both state policies and seeds 2026-2028")
    build_state_memory_grid(config.state_policies, config.seeds)
    build_state_memory_grid(config.smoke_state_policies, config.smoke_seeds)
    if config.maximum_lag <= 0 or config.perturbation_steps <= 0:
        raise ValueError("memory diagnostic lengths must be positive")
    _assert_reference_contracts(config)
    _verify_selected_virtual_nodes(config)
    return config


def verify_state_memory_public_data(
    config: StateMemoryStudyConfig,
    classifier_reference: ModelExperimentConfig,
) -> tuple[ModelDataset, dict[str, Any]]:
    """Verify frozen raw and processed checksums and reject fixture data."""

    data_config = load_data_config(config.project_root / "configs/data_public_market.yaml")
    snapshot = verify_public_snapshot(data_config)
    if snapshot.get("snapshot_id") != config.data_snapshot_id:
        raise ValueError("public snapshot ID disagrees with state-memory configuration")
    if data_config.snapshot_manifest_path is None:
        raise FileNotFoundError("public snapshot configuration has no manifest")
    dataset = load_model_dataset(classifier_reference.processed_dir)
    if dataset.is_synthetic or dataset.data_source_type != "public_market":
        raise SyntheticResultsError(
            "QRC state-memory ablation requires verified non-synthetic public-market data"
        )
    if dataset.manifest.get("source_snapshot_id") != config.data_snapshot_id:
        raise ValueError("processed data manifest disagrees with state-memory snapshot ID")
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


def _reference_for_task(config: StateMemoryStudyConfig, task: str) -> tuple[Path, str]:
    if task == TASKS[0]:
        return config.classifier_reference, config.classifier_reference_sha256
    if task == TASKS[1]:
        return config.regressor_reference, config.regressor_reference_sha256
    raise ValueError(f"unsupported state-memory task: {task}")


def write_state_memory_model_config(
    config: StateMemoryStudyConfig,
    point: StateMemoryPoint,
    task: str,
) -> Path:
    """Derive an isolated model config while changing only fixed n=2, policy, and seed."""

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
        f"qrc_state_memory_{point.state_policy}_{model_type}_seed{point.reservoir_seed}"
    )
    raw["experiment"]["project_root"] = Path(project_root_setting).as_posix()
    raw["experiment"]["seed"] = point.reservoir_seed
    raw["experiment"]["output_root"] = Path(output_root_setting).as_posix()
    raw["model"]["parameters"]["n_qubits"] = int(config.fixed_qrc["n_qubits"])
    raw["model"]["parameters"]["virtual_nodes"] = config.selected_virtual_nodes
    raw["model"]["parameters"]["state_policy"] = point.state_policy
    raw["model"]["parameters"]["reservoir_seed"] = point.reservoir_seed
    raw["qrc"]["feature_cache"] = Path(feature_cache_setting).as_posix()
    raw["qrc"]["reservoir_seeds"] = list(config.seeds)
    raw["state_memory_study"] = {
        "id": config.study_id,
        "experimental_variable": "state_policy",
        "state_policy": point.state_policy,
        "reservoir_seed": point.reservoir_seed,
        "n_qubits": int(config.fixed_qrc["n_qubits"]),
        "virtual_nodes": config.selected_virtual_nodes,
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
        if name in {"n_qubits", "state_policy", "reservoir_seed"}:
            continue
        if asdict(derived_qrc)[name] != value:
            raise ValueError(f"state-memory config unexpectedly changed QRC field {name}")
    if derived_qrc.n_qubits != 2 or derived_qrc.virtual_nodes != config.selected_virtual_nodes:
        raise ValueError("derived state-memory config changed the selected architecture")
    if derived_qrc.state_policy != point.state_policy:
        raise ValueError("derived state-memory config has the wrong state policy")
    if derived_qrc.reservoir_seed != point.reservoir_seed:
        raise ValueError("derived state-memory config has the wrong reservoir seed")
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
    marker = raw.get("state_memory_study")
    if not isinstance(marker, dict) or marker.get("id") != study_id:
        return None
    if manifest.get("is_synthetic") or manifest.get("data_source_type") != "public_market":
        raise SyntheticResultsError("completed state-memory run contains synthetic data")
    if manifest.get("data_snapshot_id") != snapshot_id:
        raise ValueError("completed state-memory run uses a different public snapshot")
    if manifest.get("backend") != "numpy_density_matrix_exact":
        raise ValueError("completed state-memory run uses a non-reference backend")
    if manifest.get("exact_noiseless") is not True:
        raise ValueError("completed state-memory run is not exact and noiseless")
    if manifest.get("model_selection_data") != "validation only":
        raise ValueError("completed run did not preserve validation-only readout selection")
    if manifest.get("test_evaluated_after_readout_freeze") is not True:
        raise ValueError("completed run evaluated test before readout freeze")
    qrc = manifest.get("qrc_configuration")
    if (
        not isinstance(qrc, dict)
        or int(qrc.get("n_qubits", -1)) != 2
        or int(qrc.get("virtual_nodes", -1)) != 2
    ):
        raise ValueError("completed state-memory run changed the selected architecture")
    state_policy = str(qrc["state_policy"])
    seed = int(manifest["reservoir_seed"])
    task = str(manifest["task"])
    if state_policy not in STATE_POLICIES or seed not in FULL_SEED_GRID or task not in TASKS:
        raise ValueError("completed state-memory run has an unsupported grid identity")
    return state_policy, seed, task


def discover_completed_state_memory_runs(
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


def pending_state_memory_runs(
    grid: tuple[StateMemoryPoint, ...],
    completed: dict[RunIdentity, Path],
) -> tuple[tuple[StateMemoryPoint, str], ...]:
    """Return requested task runs absent from a partially completed grid."""

    return tuple(
        (point, task)
        for point in grid
        for task in TASKS
        if (point.state_policy, point.reservoir_seed, task) not in completed
    )


def _validate_run_against_feature_cache(
    experiment_dir: Path,
    *,
    expected_key: str,
    expected_policy: str,
) -> dict[str, Any]:
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["qrc_feature_cache_key_checksum"] != expected_key:
        raise ValueError("classifier/regressor did not reuse the prepared feature cache")
    qrc = manifest["qrc_configuration"]
    if (
        int(qrc["n_qubits"]) != 2
        or int(qrc["virtual_nodes"]) != 2
        or str(qrc["state_policy"]) != expected_policy
    ):
        raise ValueError("state-memory run changed a fixed or requested QRC control")
    if manifest["qrc_features_generated_without_labels"] is not True:
        raise ValueError("state-memory run does not prove label-free feature generation")
    if manifest["test_evaluated_after_readout_freeze"] is not True:
        raise ValueError("state-memory run evaluated test before the readout was frozen")
    return cast(dict[str, Any], manifest)


def _feature_condition_diagnostics(
    features: NDArray[np.float64],
) -> dict[str, Any]:
    diagnostics, singular_values = effective_feature_rank(features)
    return {
        **diagnostics,
        "singular_value_count": len(singular_values),
        "training_feature_checksum": array_checksum(features),
        "computed_from_split": "train",
        "labels_consumed": False,
    }


def lagged_input_feature_correlations(
    features: NDArray[np.float64],
    inputs: NDArray[np.float64],
    *,
    maximum_lag: int,
) -> list[dict[str, float | int]]:
    """Summarize absolute training-feature correlations with lagged training inputs."""

    feature_values = np.asarray(features, dtype=float)
    input_values = np.asarray(inputs, dtype=float)
    if (
        feature_values.ndim != 2
        or input_values.ndim != 2
        or len(feature_values) != len(input_values)
    ):
        raise ValueError("lagged input-feature arrays must be aligned matrices")
    if not np.isfinite(feature_values).all() or not np.isfinite(input_values).all():
        raise ValueError("lagged input-feature arrays must be finite")
    if maximum_lag <= 0 or maximum_lag >= len(feature_values):
        raise ValueError("maximum lag must be positive and shorter than the training sequence")
    rows: list[dict[str, float | int]] = []
    for lag in range(1, maximum_lag + 1):
        current_features = feature_values[lag:]
        lagged_inputs = input_values[:-lag]
        correlations: list[float] = []
        for feature_column in range(current_features.shape[1]):
            feature = current_features[:, feature_column]
            if float(np.std(feature)) <= 1e-15:
                continue
            for input_column in range(lagged_inputs.shape[1]):
                input_feature = lagged_inputs[:, input_column]
                if float(np.std(input_feature)) <= 1e-15:
                    continue
                correlation = float(np.corrcoef(feature, input_feature)[0, 1])
                if np.isfinite(correlation):
                    correlations.append(abs(correlation))
        rows.append(
            {
                "lag": lag,
                "mean_absolute_correlation": (
                    float(np.mean(correlations)) if correlations else 0.0
                ),
                "maximum_absolute_correlation": (
                    float(np.max(correlations)) if correlations else 0.0
                ),
                "correlation_pair_count": len(correlations),
            }
        )
    return rows


def perturbation_decay_diagnostics(
    config: QRCConfig,
    inputs: NDArray[np.float64],
    *,
    steps: int,
) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    """Track decay of one initial full-state perturbation under a configured policy."""

    values = np.asarray(inputs, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("perturbation inputs must be a non-empty matrix")
    steps = min(steps, len(values))
    first = QuantumReservoir(values.shape[1], config)
    second = QuantumReservoir(values.shape[1], config)
    dimension = 2**config.n_qubits
    alternate = np.zeros((dimension, dimension), dtype=complex)
    alternate[-1, -1] = 1.0
    second.set_state(np.asarray(alternate, dtype=complex))
    distances = [trace_distance(first.get_state(), second.get_state())]
    for row in values[:steps]:
        if config.state_policy == "reset_each_input":
            first.reset_state()
            second.reset_state()
        first.step(row)
        second.step(row)
        distances.append(trace_distance(first.get_state(), second.get_state()))
    curve = [
        {"step": step, "trace_distance": float(distance)} for step, distance in enumerate(distances)
    ]
    threshold = 1e-12
    threshold_step = next(
        (index for index, value in enumerate(distances) if index > 0 and value <= threshold),
        None,
    )
    distance_values = np.asarray(distances, dtype=float)
    trapezoids = 0.5 * (distance_values[:-1] + distance_values[1:])
    summary = {
        "initial_trace_distance": float(distances[0]),
        "final_trace_distance": float(distances[-1]),
        "trace_distance_auc": float(np.sum(trapezoids)),
        "steps": steps,
        "first_step_at_or_below_1e-12": threshold_step,
        "cross_observation_memory_removed_by_construction": (
            config.state_policy == "reset_each_input"
        ),
        "labels_consumed": False,
    }
    return curve, summary


def compare_carry_with_encoding_density_reference(
    *,
    project_root: Path,
    cache_key_checksum: str,
    state_memory_cache_dir: Path,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Compare carry-input features with the selected V=2 encoding-density cache."""

    reference_dir = project_root / "results/qrc_encoding_density/feature_cache" / cache_key_checksum
    result: dict[str, Any] = {
        "reference": "selected exact two-qubit V=2 encoding-density cache",
        "cache_key_checksum": cache_key_checksum,
        "tolerance": tolerance,
        "reference_cache_available": reference_dir.is_dir(),
        "compared": False,
        "within_tolerance": None,
    }
    if not reference_dir.is_dir():
        return result
    reference_arrays = reference_dir / "qrc_features.npz"
    memory_arrays = state_memory_cache_dir / "qrc_features.npz"
    if not reference_arrays.is_file() or not memory_arrays.is_file():
        raise FileNotFoundError("carry-input reference comparison found an incomplete cache")
    maximum = 0.0
    checksums_match = True
    split_checksums: dict[str, dict[str, str]] = {}
    with np.load(reference_arrays) as reference, np.load(memory_arrays) as memory:
        for split in ("train", "validation", "test"):
            reference_values = np.asarray(reference[split], dtype=float)
            memory_values = np.asarray(memory[split], dtype=float)
            if reference_values.shape != memory_values.shape:
                raise ValueError(f"carry-input reference feature shape differs for {split}")
            difference = float(np.max(np.abs(reference_values - memory_values), initial=0.0))
            maximum = max(maximum, difference)
            reference_checksum = array_checksum(reference_values)
            memory_checksum = array_checksum(memory_values)
            checksums_match = checksums_match and reference_checksum == memory_checksum
            split_checksums[split] = {
                "reference": reference_checksum,
                "state_memory": memory_checksum,
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
            "carry_inputs features disagree with selected exact reference: "
            f"maximum absolute difference {maximum}"
        )
    return result


def select_state_policy_from_validation(
    run_dirs: dict[RunIdentity, Path],
) -> dict[str, Any]:
    """Select a policy lexicographically from validation metrics without reading test."""

    evidence: dict[str, dict[str, list[float]]] = {
        policy: {
            "macro_f1": [],
            "transition_pr_auc": [],
            "qlike": [],
            "feature_generation_seconds": [],
        }
        for policy in STATE_POLICIES
    }
    for (policy, _seed, task), experiment_dir in sorted(run_dirs.items()):
        validation = json.loads(
            (experiment_dir / "validation_metrics.json").read_text(encoding="utf-8")
        )
        if task == TASKS[0]:
            evidence[policy]["macro_f1"].append(float(validation["macro_f1"]))
            evidence[policy]["transition_pr_auc"].append(float(validation["transition_pr_auc"]))
        else:
            evidence[policy]["qlike"].append(float(validation["qlike"]))
        timing = json.loads((experiment_dir / "timing.json").read_text(encoding="utf-8"))
        evidence[policy]["feature_generation_seconds"].append(
            float(timing["state_generation_seconds"])
        )
    candidate_rows: list[dict[str, Any]] = []
    for policy in STATE_POLICIES:
        values = evidence[policy]
        if not all(values[name] for name in ("macro_f1", "transition_pr_auc", "qlike")):
            raise ValueError(f"validation selection evidence is incomplete for {policy}")
        candidate_rows.append(
            {
                "state_policy": policy,
                "validation_macro_f1_mean": float(np.mean(values["macro_f1"])),
                "validation_macro_f1_seed_standard_deviation": float(
                    np.std(values["macro_f1"], ddof=0)
                ),
                "validation_transition_pr_auc_mean": float(np.mean(values["transition_pr_auc"])),
                "validation_transition_pr_auc_seed_standard_deviation": float(
                    np.std(values["transition_pr_auc"], ddof=0)
                ),
                "validation_qlike_mean": float(np.mean(values["qlike"])),
                "validation_qlike_seed_standard_deviation": float(np.std(values["qlike"], ddof=0)),
                "feature_generation_seconds_mean": float(
                    np.mean(values["feature_generation_seconds"])
                ),
                "seed_count": len(values["macro_f1"]),
            }
        )
    by_policy = {str(row["state_policy"]): row for row in candidate_rows}
    tolerance = 1e-12
    criteria = (
        ("validation_macro_f1_mean", "higher"),
        ("validation_transition_pr_auc_mean", "higher"),
        ("validation_qlike_mean", "lower"),
        ("validation_macro_f1_seed_standard_deviation", "lower"),
        ("validation_transition_pr_auc_seed_standard_deviation", "lower"),
        ("validation_qlike_seed_standard_deviation", "lower"),
        ("feature_generation_seconds_mean", "lower"),
    )
    selected: str | None = None
    decisive_criterion: str | None = None
    decision_trace: list[dict[str, Any]] = []
    carry = by_policy["carry_inputs"]
    reset = by_policy["reset_each_input"]
    for name, direction in criteria:
        carry_value = float(carry[name])
        reset_value = float(reset[name])
        difference = carry_value - reset_value
        favored: str | None = None
        if abs(difference) > tolerance:
            if direction == "higher":
                favored = "carry_inputs" if difference > 0 else "reset_each_input"
            else:
                favored = "carry_inputs" if difference < 0 else "reset_each_input"
        decision_trace.append(
            {
                "criterion": name,
                "direction": direction,
                "carry_inputs": carry_value,
                "reset_each_input": reset_value,
                "difference_carry_minus_reset": difference,
                "favored_policy": favored,
            }
        )
        if selected is None and favored is not None:
            selected = favored
            decisive_criterion = name
    if selected is None:
        selected = "reset_each_input"
        decisive_criterion = "simplicity_exact_tie_break"
    return {
        "selection_basis": "validation only",
        "test_metrics_read": False,
        "criteria_order": [name for name, _direction in criteria],
        "candidate_rows": candidate_rows,
        "decision_trace": decision_trace,
        "selected_state_policy": selected,
        "decisive_criterion": decisive_criterion,
        "selected_before_test_reporting": True,
        "qlike_direction": "lower is better",
    }


def collect_state_memory_rows(
    run_dirs: dict[RunIdentity, Path],
    *,
    point_metadata: dict[tuple[str, int], dict[str, Any]],
    cache_hits: dict[RunIdentity, bool] | None = None,
    repository_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Collect split metrics, timings, memory diagnostics, checksums, and provenance."""

    required_manifest_fields = {
        "configuration_checksum",
        "processed_data_checksums",
        "selected_hyperparameters",
        "qrc_raw_feature_dimension",
        "readout_shape",
        "trainable_readout_parameters",
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
        policy, seed, task = identity
        manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
        missing = sorted(required_manifest_fields - set(manifest))
        if missing:
            raise ValueError(f"state-memory run manifest omits required fields: {missing}")
        if manifest["model_selection_data"] != "validation only":
            raise ValueError("state-memory row violates validation-only readout selection")
        if manifest["test_evaluated_after_readout_freeze"] is not True:
            raise ValueError("state-memory row violates frozen-readout test evaluation")
        metadata = point_metadata[(policy, seed)]
        diagnostics = metadata["condition_diagnostics"]
        perturbation = metadata["perturbation_summary"]
        timing = json.loads((experiment_dir / "timing.json").read_text(encoding="utf-8"))
        feature_seconds = float(metadata["feature_generation_seconds"])
        readout_seconds = float(manifest["readout_fitting_time"])
        total_runtime = feature_seconds + sum(
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
            "state_policy": policy,
            "n_qubits": 2,
            "virtual_nodes": 2,
            "reservoir_seed": seed,
            "task": task,
            "model_type": manifest["model_type"],
            "selected_ridge_alpha": manifest["selected_hyperparameters"]["ridge_alpha"],
            "raw_feature_dimension": manifest["qrc_raw_feature_dimension"],
            "readout_shape": manifest["readout_shape"],
            "trainable_readout_parameters": manifest["trainable_readout_parameters"],
            "feature_generation_seconds": feature_seconds,
            "feature_preparation_wall_seconds": metadata["feature_preparation_wall_seconds"],
            "readout_fitting_seconds": readout_seconds,
            "total_runtime_seconds": total_runtime,
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
            "condition_number": diagnostics["condition_number"],
            "effective_rank": diagnostics["effective_rank"],
            "numerical_rank": diagnostics["numerical_rank"],
            "largest_singular_value": diagnostics["largest_singular_value"],
            "smallest_retained_singular_value": diagnostics["smallest_retained_singular_value"],
            "rank_tolerance": diagnostics["rank_tolerance"],
            "lag1_mean_absolute_feature_autocorrelation": metadata["feature_autocorrelation"][0][
                "mean_absolute_autocorrelation"
            ],
            "lag1_mean_absolute_input_feature_correlation": metadata[
                "lagged_input_feature_correlations"
            ][0]["mean_absolute_correlation"],
            "perturbation_final_trace_distance": perturbation["final_trace_distance"],
            "perturbation_trace_distance_auc": perturbation["trace_distance_auc"],
            "numerical_diagnostics": metadata["numerical_diagnostics"],
            "carry_reference_agreement": metadata.get("carry_reference_agreement"),
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


def aggregate_state_memory_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate performance, resources, and memory diagnostics over seeds."""

    aggregated: list[dict[str, Any]] = []
    for policy in STATE_POLICIES:
        for task in TASKS:
            for split in ("validation", "test"):
                subset = [
                    row
                    for row in rows
                    if row["state_policy"] == policy
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
                    aggregated.append(
                        {
                            "state_policy": policy,
                            "n_qubits": 2,
                            "virtual_nodes": 2,
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
                            "package_versions": subset[0]["package_versions"],
                        }
                    )
    return aggregated


def paired_state_policy_differences(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return per-seed and aggregate carry-minus-reset metric differences."""

    indexed = {
        (
            str(row["state_policy"]),
            int(row["reservoir_seed"]),
            str(row["task"]),
            str(row["split"]),
        ): row
        for row in rows
    }
    paired: list[dict[str, Any]] = []
    seeds = sorted({int(row["reservoir_seed"]) for row in rows})
    for split in ("validation", "test"):
        for metric, (task, lower_is_better) in PAIRED_METRICS.items():
            for seed in seeds:
                carry = indexed[("carry_inputs", seed, task, split)]
                reset = indexed[("reset_each_input", seed, task, split)]
                carry_value = float(carry[metric])
                reset_value = float(reset[metric])
                delta = carry_value - reset_value
                directional = -delta if lower_is_better else delta
                paired.append(
                    {
                        "split": split,
                        "task": task,
                        "metric": metric,
                        "reservoir_seed": seed,
                        "carry_inputs": carry_value,
                        "reset_each_input": reset_value,
                        "delta_carry_minus_reset": delta,
                        "lower_is_better": lower_is_better,
                        "directional_improvement_for_carry": directional,
                        "favored_policy": (
                            "carry_inputs"
                            if directional > 0
                            else "reset_each_input"
                            if directional < 0
                            else "tie"
                        ),
                        "qlike_interpretation": (
                            "negative carry-minus-reset favors carry_inputs"
                            if metric == "qlike"
                            else None
                        ),
                    }
                )
    summary: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for metric, (task, lower_is_better) in PAIRED_METRICS.items():
            subset = [row for row in paired if row["split"] == split and row["metric"] == metric]
            deltas = np.asarray(
                [float(row["delta_carry_minus_reset"]) for row in subset], dtype=float
            )
            directional_values = np.asarray(
                [float(row["directional_improvement_for_carry"]) for row in subset],
                dtype=float,
            )
            summary.append(
                {
                    "split": split,
                    "task": task,
                    "metric": metric,
                    "mean_delta_carry_minus_reset": float(deltas.mean()),
                    "standard_deviation": float(deltas.std(ddof=0)),
                    "minimum": float(deltas.min()),
                    "maximum": float(deltas.max()),
                    "seed_count": len(deltas),
                    "reservoir_seeds": [int(row["reservoir_seed"]) for row in subset],
                    "lower_is_better": lower_is_better,
                    "mean_directional_improvement_for_carry": float(directional_values.mean()),
                    "favored_policy_by_mean": (
                        "carry_inputs"
                        if directional_values.mean() > 0
                        else "reset_each_input"
                        if directional_values.mean() < 0
                        else "tie"
                    ),
                    "qlike_interpretation": (
                        "negative carry-minus-reset favors carry_inputs"
                        if metric == "qlike"
                        else None
                    ),
                }
            )
    return paired, summary


def state_memory_resource_estimates(
    *,
    split_rows: int,
    train_rows: int,
) -> list[dict[str, Any]]:
    """Return analytical exact-state and feature-matrix memory estimates."""

    raw_dimension = 6
    hilbert_dimension = 4
    return [
        {
            "state_policy": policy,
            "n_qubits": 2,
            "virtual_nodes": 2,
            "raw_feature_dimension": raw_dimension,
            "hilbert_dimension": hilbert_dimension,
            "density_matrix_elements": hilbert_dimension * hilbert_dimension,
            "estimated_peak_density_matrix_bytes": (3 * hilbert_dimension * hilbert_dimension * 16),
            "estimated_cached_feature_bytes": split_rows * raw_dimension * 8,
            "estimated_train_condition_matrix_bytes": train_rows * raw_dimension * 8,
        }
        for policy in STATE_POLICIES
    ]


def _csv_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for row in rows:
        values = dict(row)
        for name, value in tuple(values.items()):
            if isinstance(value, (dict, list, tuple)):
                values[name] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        safe.append(values)
    return safe


def _policy_axis(
    axis: Any,
    rows: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
) -> None:
    for position, policy in enumerate(STATE_POLICIES):
        values = np.asarray(
            [float(row[metric]) for row in rows if row["state_policy"] == policy],
            dtype=float,
        )
        jitter = np.linspace(-0.055, 0.055, len(values))
        axis.scatter(
            np.full(len(values), position, dtype=float) + jitter,
            values,
            alpha=0.55,
            s=28,
        )
        axis.errorbar(
            [position],
            [values.mean()],
            yerr=[values.std(ddof=0)],
            color="black",
            marker="o",
            capsize=4,
            linewidth=1.6,
            zorder=4,
        )
    axis.set(
        xticks=range(len(STATE_POLICIES)),
        xticklabels=["carry inputs", "reset each input"],
        ylabel=ylabel,
    )
    axis.grid(axis="y", alpha=0.2)


def _plot_headline_metrics(
    rows: list[dict[str, Any]],
    *,
    split: str,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.7))
    specifications = (
        ("macro_f1", TASKS[0], f"{split} macro F1"),
        ("transition_pr_auc", TASKS[0], f"{split} transition PR-AUC"),
        ("qlike", TASKS[1], f"{split} QLIKE (lower is better)"),
    )
    for axis, (metric, task, ylabel) in zip(axes, specifications):
        subset = [row for row in rows if row["split"] == split and row["task"] == task]
        _policy_axis(axis, subset, metric=metric, ylabel=ylabel)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=220)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def _plot_paired_differences(
    paired: list[dict[str, Any]],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(11.4, 7.0))
    for axis, metric in zip(axes.flat, PAIRED_METRICS):
        subset = [row for row in paired if row["split"] == "test" and row["metric"] == metric]
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.plot(
            [int(row["reservoir_seed"]) for row in subset],
            [float(row["delta_carry_minus_reset"]) for row in subset],
            marker="o",
        )
        direction = "lower favors carry" if PAIRED_METRICS[metric][1] else "higher favors carry"
        axis.set(
            xlabel="reservoir seed",
            ylabel="carry - reset",
            title=f"{metric}\n{direction}",
            xticks=[int(row["reservoir_seed"]) for row in subset],
        )
        axis.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=220)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def plot_state_memory_figures(
    rows: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    """Write the five required policy and paired-comparison figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for split in ("validation", "test"):
        name = f"{split}_headline_metrics_by_state_policy"
        destination = output_dir / name
        _plot_headline_metrics(rows, split=split, destination=destination)
        outputs[f"{name}_png"] = destination.with_suffix(".png")
        outputs[f"{name}_pdf"] = destination.with_suffix(".pdf")
    paired_destination = output_dir / "per_seed_paired_metric_differences"
    _plot_paired_differences(paired, paired_destination)
    outputs["per_seed_paired_metric_differences_png"] = paired_destination.with_suffix(".png")
    outputs["per_seed_paired_metric_differences_pdf"] = paired_destination.with_suffix(".pdf")
    for name, metric, ylabel in (
        (
            "feature_conditioning_by_state_policy",
            "condition_number",
            "training-feature condition number",
        ),
        (
            "feature_generation_runtime_by_state_policy",
            "feature_generation_seconds",
            "feature-generation time (seconds)",
        ),
    ):
        destination = output_dir / name
        figure, axis = plt.subplots(figsize=(5.8, 3.8))
        subset = [row for row in rows if row["task"] == TASKS[0] and row["split"] == "validation"]
        _policy_axis(axis, subset, metric=metric, ylabel=ylabel)
        if metric == "condition_number":
            axis.set_yscale("log")
        figure.tight_layout()
        figure.savefig(destination.with_suffix(".png"), dpi=220)
        figure.savefig(destination.with_suffix(".pdf"))
        plt.close(figure)
        outputs[f"{name}_png"] = destination.with_suffix(".png")
        outputs[f"{name}_pdf"] = destination.with_suffix(".pdf")
    return outputs


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "points": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("points"), dict):
        raise ValueError("state-memory state file is invalid")
    return value


def _write_table_pair(
    json_path: Path,
    csv_path: Path,
    rows: list[dict[str, Any]],
    **metadata: Any,
) -> None:
    _write_json(json_path, {"schema_version": 1, **metadata, "rows": rows})
    pd.DataFrame(_csv_safe_rows(rows)).to_csv(csv_path, index=False)


def _write_tables(
    output_root: Path,
    rows: list[dict[str, Any]],
    aggregated: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    paired_summary: list[dict[str, Any]],
    selection: dict[str, Any],
    resources: list[dict[str, Any]],
    feature_autocorrelations: list[dict[str, Any]],
    lagged_correlations: list[dict[str, Any]],
    perturbation_curves: list[dict[str, Any]],
) -> dict[str, Path]:
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    names = (
        "per_run",
        "aggregate",
        "paired_differences",
        "paired_summary",
        "validation_policy_selection",
        "resources",
        "feature_autocorrelation",
        "lagged_input_feature_correlation",
        "perturbation_decay",
    )
    outputs = {
        f"{name}_{suffix}": tables / f"qrc_state_memory_{name}.{suffix}"
        for name in names
        for suffix in ("json", "csv")
    }
    _write_table_pair(outputs["per_run_json"], outputs["per_run_csv"], rows)
    _write_table_pair(outputs["aggregate_json"], outputs["aggregate_csv"], aggregated)
    _write_table_pair(
        outputs["paired_differences_json"],
        outputs["paired_differences_csv"],
        paired,
        delta_definition="carry_inputs minus reset_each_input",
        qlike_direction="lower is better; a negative delta favors carry_inputs",
    )
    _write_table_pair(
        outputs["paired_summary_json"],
        outputs["paired_summary_csv"],
        paired_summary,
        delta_definition="carry_inputs minus reset_each_input",
        qlike_direction="lower is better; a negative delta favors carry_inputs",
    )
    selection_rows = cast(list[dict[str, Any]], selection["candidate_rows"])
    _write_table_pair(
        outputs["validation_policy_selection_json"],
        outputs["validation_policy_selection_csv"],
        selection_rows,
        selection_basis="validation only",
        test_metrics_read=False,
        selected_state_policy=selection["selected_state_policy"],
        decisive_criterion=selection["decisive_criterion"],
        decision_trace=selection["decision_trace"],
    )
    _write_table_pair(outputs["resources_json"], outputs["resources_csv"], resources)
    _write_table_pair(
        outputs["feature_autocorrelation_json"],
        outputs["feature_autocorrelation_csv"],
        feature_autocorrelations,
        labels_consumed=False,
        split="train",
    )
    _write_table_pair(
        outputs["lagged_input_feature_correlation_json"],
        outputs["lagged_input_feature_correlation_csv"],
        lagged_correlations,
        labels_consumed=False,
        split="train",
    )
    _write_table_pair(
        outputs["perturbation_decay_json"],
        outputs["perturbation_decay_csv"],
        perturbation_curves,
        labels_consumed=False,
        diagnostic_scope="initial-state perturbation under observed training inputs",
    )
    return outputs


def run_state_memory_ablation(
    config_path: Path,
    *,
    state_policies: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    smoke: bool = False,
    resume: bool = True,
) -> Path:
    """Run or resume the exact two-policy, two-qubit state-memory ablation."""

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config = load_state_memory_study_config(config_path)
    selected_policies = state_policies or (
        config.smoke_state_policies if smoke else config.state_policies
    )
    selected_seeds = seeds or (config.smoke_seeds if smoke else config.seeds)
    grid = build_state_memory_grid(selected_policies, selected_seeds)
    if smoke and (len(selected_policies) > 2 or len(selected_seeds) > 1):
        raise ValueError("smoke mode permits both policies and at most one seed")
    classifier_reference, _ = _assert_reference_contracts(config)
    architecture_evidence = _verify_selected_virtual_nodes(config)
    dataset, data_provenance = verify_state_memory_public_data(config, classifier_reference)
    config.output_root.mkdir(parents=True, exist_ok=True)
    runs_root = config.output_root / "runs"
    completed = discover_completed_state_memory_runs(
        runs_root,
        study_id=config.study_id,
        snapshot_id=config.data_snapshot_id,
    )
    initially_completed = dict(completed)
    state_path = config.output_root / "state_memory_state.json"
    state = _load_state(state_path)
    point_state = state["points"]
    if not isinstance(point_state, dict):
        raise ValueError("state-memory state points must be a mapping")
    executed: list[dict[str, Any]] = []
    resumed: list[dict[str, Any]] = []
    cache_hits: dict[RunIdentity, bool] = {}
    point_metadata: dict[tuple[str, int], dict[str, Any]] = {}
    carry_reference_agreements: dict[str, dict[str, Any]] = {}
    all_autocorrelation_rows: list[dict[str, Any]] = []
    all_lagged_correlation_rows: list[dict[str, Any]] = []
    all_perturbation_rows: list[dict[str, Any]] = []
    study_checksum = sha256_file(config.source)

    for point in grid:
        classifier_path = write_state_memory_model_config(config, point, TASKS[0])
        regressor_path = write_state_memory_model_config(config, point, TASKS[1])
        feature_started = time.perf_counter()
        bundle = generate_qrc_features(
            classifier_path,
            allow_synthetic_results=False,
            reservoir_seed=point.reservoir_seed,
        )
        feature_preparation_wall_seconds = time.perf_counter() - feature_started
        if bundle.train.shape[1] != 6:
            raise ValueError(
                f"{point.state_policy} produced {bundle.train.shape[1]} features; expected 6"
            )
        cache_key = str(bundle.metadata["cache_key_checksum"])
        cache_directory = bundle.cache_dir.relative_to(config.project_root).as_posix()
        qrc_config = qrc_config_from_model(load_model_config(classifier_path))
        condition_diagnostics = _feature_condition_diagnostics(bundle.train)
        autocorrelation_rows = cast(
            list[dict[str, Any]],
            feature_autocorrelation(bundle.train, config.maximum_lag).to_dict(orient="records"),
        )
        lagged_rows: list[dict[str, Any]] = [
            dict(row)
            for row in lagged_input_feature_correlations(
                bundle.train,
                dataset.train.X,
                maximum_lag=config.maximum_lag,
            )
        ]
        perturbation_values, perturbation_summary = perturbation_decay_diagnostics(
            qrc_config,
            dataset.train.X,
            steps=config.perturbation_steps,
        )
        perturbation_curve: list[dict[str, Any]] = [dict(row) for row in perturbation_values]
        for row in autocorrelation_rows:
            row.update(
                {
                    "state_policy": point.state_policy,
                    "reservoir_seed": point.reservoir_seed,
                }
            )
        for row in lagged_rows:
            row.update(
                {
                    "state_policy": point.state_policy,
                    "reservoir_seed": point.reservoir_seed,
                }
            )
        for row in perturbation_curve:
            row.update(
                {
                    "state_policy": point.state_policy,
                    "reservoir_seed": point.reservoir_seed,
                }
            )
        all_autocorrelation_rows.extend(autocorrelation_rows)
        all_lagged_correlation_rows.extend(lagged_rows)
        all_perturbation_rows.extend(perturbation_curve)
        carry_agreement: dict[str, Any] | None = None
        if point.state_policy == "carry_inputs":
            carry_agreement = compare_carry_with_encoding_density_reference(
                project_root=config.project_root,
                cache_key_checksum=cache_key,
                state_memory_cache_dir=bundle.cache_dir,
            )
            carry_reference_agreements[point.key] = carry_agreement
        metadata = {
            "state_policy": point.state_policy,
            "n_qubits": 2,
            "virtual_nodes": 2,
            "reservoir_seed": point.reservoir_seed,
            "feature_preparation_cache_hit": bundle.cache_hit,
            "feature_preparation_wall_seconds": feature_preparation_wall_seconds,
            "feature_generation_seconds": float(
                bundle.metadata["resource_metadata"]["state_generation_seconds"]
            ),
            "cache_key_checksum": cache_key,
            "cache_directory": cache_directory,
            "array_checksums": bundle.metadata["array_checksums"],
            "study_configuration_checksum": study_checksum,
            "condition_diagnostics": condition_diagnostics,
            "feature_autocorrelation": autocorrelation_rows,
            "lagged_input_feature_correlations": lagged_rows,
            "perturbation_summary": perturbation_summary,
            "numerical_diagnostics": bundle.metadata["numerical_diagnostics"],
            "carry_reference_agreement": carry_agreement,
        }
        existing_point_state = point_state.get(point.key)
        if isinstance(existing_point_state, dict):
            if existing_point_state.get("cache_key_checksum") != cache_key:
                raise ValueError("resumed state-memory point cache key changed")
            if existing_point_state.get("cache_directory") != cache_directory:
                raise ValueError("resumed state-memory point cache directory changed")
        point_state[point.key] = metadata
        point_metadata[(point.state_policy, point.reservoir_seed)] = metadata
        diagnostics_path = config.output_root / "memory_diagnostics" / f"{point.key}.json"
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(diagnostics_path, metadata)
        _write_json(state_path, state)

        for task, model_config_path in (
            (TASKS[0], classifier_path),
            (TASKS[1], regressor_path),
        ):
            identity = (point.state_policy, point.reservoir_seed, task)
            if resume and identity in completed:
                experiment_dir = completed[identity]
                resumed.append(
                    {
                        "state_policy": point.state_policy,
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
                        "state_policy": point.state_policy,
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
                expected_policy=point.state_policy,
            )
            cache_hits[identity] = (
                bool(metadata["feature_preparation_cache_hit"]) if task == TASKS[0] else True
            )

    requested_runs = {
        identity: completed[identity]
        for point in grid
        for task in TASKS
        if (identity := (point.state_policy, point.reservoir_seed, task)) in completed
    }
    expected_run_count = len(grid) * len(TASKS)
    if len(requested_runs) != expected_run_count:
        raise RuntimeError(
            f"state-memory grid incomplete after execution: {len(requested_runs)} "
            f"of {expected_run_count} task runs"
        )
    policy_selection = select_state_policy_from_validation(requested_runs)
    rows = collect_state_memory_rows(
        requested_runs,
        point_metadata=point_metadata,
        cache_hits=cache_hits,
        repository_root=config.project_root,
    )
    aggregated = aggregate_state_memory_rows(rows)
    paired, paired_summary = paired_state_policy_differences(rows)
    resources = state_memory_resource_estimates(
        split_rows=len(dataset.train.X) + len(dataset.validation.X) + len(dataset.test.X),
        train_rows=len(dataset.train.X),
    )
    outputs = _write_tables(
        config.output_root,
        rows,
        aggregated,
        paired,
        paired_summary,
        policy_selection,
        resources,
        all_autocorrelation_rows,
        all_lagged_correlation_rows,
        all_perturbation_rows,
    )
    figures = plot_state_memory_figures(rows, paired, config.output_root / "figures")
    summary_path = config.output_root / "state_memory_run_summary.json"
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
        "selected_architecture_evidence": architecture_evidence,
        "experimental_variable": "state_policy",
        "state_policy_definitions": {
            "carry_inputs": (
                "carry the full post-input state to the next observation while retaining "
                "the established partial input-qubit reset at every virtual substep"
            ),
            "reset_each_input": (
                "reset the full state before every observation, then retain established "
                "within-input partial reinjection and virtual-substep evolution"
            ),
        },
        "fixed_qrc": config.fixed_qrc,
        "grid": [asdict(point) for point in grid],
        "data_provenance": data_provenance,
        "resume_enabled": resume,
        "completed_before_run": len(initially_completed),
        "executed_runs": executed,
        "resumed_runs": resumed,
        "feature_cache_points": {point.key: point_state[point.key] for point in grid},
        "carry_reference_agreements": carry_reference_agreements,
        "validation_only_state_policy_selection": policy_selection,
        "test_metrics_reported_after_validation_selection": True,
        "per_run_row_count": len(rows),
        "aggregate_row_count": len(aggregated),
        "paired_row_count": len(paired),
        "paired_summary_row_count": len(paired_summary),
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
        "interpretation": (
            "controlled exact classical simulation of quantum-reservoir state memory; "
            "not physical-QPU execution or a quantum-advantage claim"
        ),
    }
    _write_json(summary_path, summary)
    return summary_path
