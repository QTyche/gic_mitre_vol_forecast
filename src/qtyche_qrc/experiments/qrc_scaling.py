"""Controlled exact-noiseless QRC qubit-scaling experiment orchestration."""

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
from qtyche_qrc.experiments.qrc_run import (
    generate_qrc_features,
    qrc_config_from_model,
    run_qrc_experiment,
)
from qtyche_qrc.experiments.run import SyntheticResultsError, _git_metadata, _write_json
from qtyche_qrc.models.dataset import ModelDataset, load_model_dataset
from qtyche_qrc.models.qrc.hamiltonian import ring_edges
from qtyche_qrc.runtime import runtime_metadata

FULL_QUBIT_GRID = (2, 3, 4, 5, 6)
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
    "state_generation_seconds",
    "readout_fitting_seconds",
    "raw_feature_dimension",
    "trainable_readout_parameters",
)


@dataclass(frozen=True, order=True)
class ScalingPoint:
    """One independent reservoir size and seed combination."""

    n_qubits: int
    reservoir_seed: int

    @property
    def key(self) -> str:
        return f"q{self.n_qubits}_seed{self.reservoir_seed}"


@dataclass(frozen=True)
class ScalingStudyConfig:
    """Validated paths and immutable contracts for the scaling study."""

    source: Path
    project_root: Path
    study_id: str
    output_root: Path
    classifier_reference: Path
    classifier_reference_sha256: str
    regressor_reference: Path
    regressor_reference_sha256: str
    data_snapshot_id: str
    qubits: tuple[int, ...]
    seeds: tuple[int, ...]
    smoke_qubits: tuple[int, ...]
    smoke_seeds: tuple[int, ...]
    fixed_qrc: dict[str, Any]
    raw: dict[str, Any]


def build_scaling_grid(
    qubits: tuple[int, ...] | list[int],
    seeds: tuple[int, ...] | list[int],
) -> tuple[ScalingPoint, ...]:
    """Validate and return a deterministic qubit-major scaling grid."""

    qubit_values = tuple(int(value) for value in qubits)
    seed_values = tuple(int(value) for value in seeds)
    if not qubit_values or not seed_values:
        raise ValueError("qubit and seed grids must both be non-empty")
    if len(set(qubit_values)) != len(qubit_values):
        raise ValueError("qubit grid contains duplicates")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("reservoir seed grid contains duplicates")
    unsupported_qubits = sorted(set(qubit_values) - set(FULL_QUBIT_GRID))
    unsupported_seeds = sorted(set(seed_values) - set(FULL_SEED_GRID))
    if unsupported_qubits:
        raise ValueError(f"unsupported qubit counts: {unsupported_qubits}")
    if unsupported_seeds:
        raise ValueError(f"unsupported reservoir seeds: {unsupported_seeds}")
    return tuple(
        ScalingPoint(n_qubits, seed)
        for n_qubits in sorted(qubit_values)
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
    config: ScalingStudyConfig,
) -> tuple[ModelExperimentConfig, ModelExperimentConfig]:
    references = (
        (config.classifier_reference, config.classifier_reference_sha256),
        (config.regressor_reference, config.regressor_reference_sha256),
    )
    for path, expected in references:
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen scaling reference checksum mismatch for {path}: {actual}")
    classifier = load_model_config(config.classifier_reference)
    regressor = load_model_config(config.regressor_reference)
    if classifier.task != "regime_classification" or classifier.model_type != "qrc_classifier":
        raise ValueError("scaling classifier reference is not the frozen public QRC classifier")
    if regressor.task != "rv_regression" or regressor.model_type != "qrc_regressor":
        raise ValueError("scaling regressor reference is not the frozen public QRC regressor")
    if classifier.processed_dir != regressor.processed_dir:
        raise ValueError("scaling references disagree on the processed public dataset")
    classifier_qrc = qrc_config_from_model(classifier)
    regressor_qrc = qrc_config_from_model(regressor)
    if classifier_qrc != regressor_qrc:
        raise ValueError("classifier and regressor references disagree on reservoir dynamics")
    fixed = config.fixed_qrc
    for name in (
        "virtual_nodes",
        "graph",
        "j_strength",
        "h_strength",
        "tau",
        "input_scaling",
        "state_policy",
        "backend",
    ):
        if getattr(classifier_qrc, name) != fixed.get(name):
            raise ValueError(f"frozen QRC reference disagrees with fixed_qrc.{name}")
    if fixed.get("exact_noiseless") is not True:
        raise ValueError("scaling study requires fixed_qrc.exact_noiseless: true")
    if classifier_qrc.n_qubits != 6:
        raise ValueError("scaling reference must remain the frozen six-qubit pilot")
    return classifier, regressor


def load_scaling_study_config(path: Path) -> ScalingStudyConfig:
    """Load the scaling contract and verify the frozen pilot configuration hashes."""

    source = path.resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "configuration")
    if root.get("schema_version") != 1:
        raise ValueError("QRC scaling configuration schema_version must be 1")
    study = _mapping(root.get("study"), "study")
    fixed_qrc = _mapping(root.get("fixed_qrc"), "fixed_qrc")
    project_root = (source.parent / _text(study, "project_root", "study")).resolve()
    config = ScalingStudyConfig(
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
        qubits=_integer_tuple(study, "qubits", "study"),
        seeds=_integer_tuple(study, "reservoir_seeds", "study"),
        smoke_qubits=_integer_tuple(study, "smoke_qubits", "study"),
        smoke_seeds=_integer_tuple(study, "smoke_seeds", "study"),
        fixed_qrc=fixed_qrc,
        raw=root,
    )
    build_scaling_grid(config.qubits, config.seeds)
    build_scaling_grid(config.smoke_qubits, config.smoke_seeds)
    if config.qubits != FULL_QUBIT_GRID or config.seeds != FULL_SEED_GRID:
        raise ValueError("full scaling grid must remain qubits 2-6 and seeds 2026-2028")
    _assert_reference_contracts(config)
    return config


def verify_scaling_public_data(
    config: ScalingStudyConfig,
    classifier_reference: ModelExperimentConfig,
) -> tuple[ModelDataset, dict[str, Any]]:
    """Verify the raw snapshot and every processed checksum without network access."""

    data_config = load_data_config(config.project_root / "configs/data_public_market.yaml")
    snapshot = verify_public_snapshot(data_config)
    if snapshot.get("snapshot_id") != config.data_snapshot_id:
        raise ValueError("public snapshot ID disagrees with scaling configuration")
    if data_config.snapshot_manifest_path is None:
        raise FileNotFoundError("public snapshot configuration has no manifest")
    dataset = load_model_dataset(classifier_reference.processed_dir)
    if dataset.is_synthetic or dataset.data_source_type != "public_market":
        raise SyntheticResultsError(
            "QRC qubit scaling requires verified non-synthetic public-market data"
        )
    if dataset.manifest.get("source_snapshot_id") != config.data_snapshot_id:
        raise ValueError("processed data manifest disagrees with scaling snapshot ID")
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


def scaling_resource_estimates(
    qubits: tuple[int, ...] | list[int],
    *,
    virtual_nodes: int,
    split_rows: int,
) -> list[dict[str, int]]:
    """Return analytical exact-state and cached-feature memory estimates."""

    rows: list[dict[str, int]] = []
    for n_qubits in sorted(qubits):
        hilbert_dimension = 2**n_qubits
        raw_dimension = virtual_nodes * (n_qubits + len(ring_edges(n_qubits)))
        rows.append(
            {
                "n_qubits": n_qubits,
                "virtual_nodes": virtual_nodes,
                "hilbert_dimension": hilbert_dimension,
                "density_matrix_elements": hilbert_dimension * hilbert_dimension,
                "estimated_peak_density_matrix_bytes": (
                    3 * hilbert_dimension * hilbert_dimension * 16
                ),
                "raw_feature_dimension": raw_dimension,
                "estimated_cached_feature_bytes": split_rows * raw_dimension * 8,
            }
        )
    return rows


def _reference_for_task(config: ScalingStudyConfig, task: str) -> tuple[Path, str]:
    if task == "regime_classification":
        return config.classifier_reference, config.classifier_reference_sha256
    if task == "rv_regression":
        return config.regressor_reference, config.regressor_reference_sha256
    raise ValueError(f"unsupported scaling task: {task}")


def write_scaling_model_config(
    config: ScalingStudyConfig,
    point: ScalingPoint,
    task: str,
) -> Path:
    """Derive a model config by changing only qubit count, seed, and output identity."""

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
        f"qrc_qubit_scaling_q{point.n_qubits}_{model_type}_seed{point.reservoir_seed}"
    )
    raw["experiment"]["project_root"] = Path(project_root_setting).as_posix()
    raw["experiment"]["seed"] = point.reservoir_seed
    raw["experiment"]["output_root"] = Path(output_root_setting).as_posix()
    raw["model"]["parameters"]["n_qubits"] = point.n_qubits
    raw["model"]["parameters"]["reservoir_seed"] = point.reservoir_seed
    raw["qrc"]["feature_cache"] = Path(feature_cache_setting).as_posix()
    raw["qrc"]["reservoir_seeds"] = list(config.seeds)
    raw["scaling_study"] = {
        "id": config.study_id,
        "experimental_variable": "n_qubits",
        "n_qubits": point.n_qubits,
        "reservoir_seed": point.reservoir_seed,
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
        if name in {"n_qubits", "reservoir_seed"}:
            continue
        if asdict(derived_qrc)[name] != value:
            raise ValueError(f"scaling config unexpectedly changed QRC field {name}")
    if derived_qrc.n_qubits != point.n_qubits:
        raise ValueError("derived scaling config has the wrong qubit count")
    if derived_qrc.reservoir_seed != point.reservoir_seed:
        raise ValueError("derived scaling config has the wrong reservoir seed")
    if derived.output_root != config.output_root / "runs":
        raise ValueError("derived scaling config does not use the isolated output directory")
    return destination


RunIdentity = tuple[int, int, str]


def _completed_identity(
    experiment_dir: Path,
    manifest: dict[str, Any],
    *,
    study_id: str,
    snapshot_id: str,
) -> RunIdentity | None:
    if manifest.get("status") != "success":
        return None
    config_path = experiment_dir / "config.yaml"
    required_paths = (
        config_path,
        experiment_dir / "validation_metrics.json",
        experiment_dir / "test_metrics.json",
    )
    if not all(path.is_file() for path in required_paths):
        return None
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    scaling = raw.get("scaling_study")
    if not isinstance(scaling, dict) or scaling.get("id") != study_id:
        return None
    if manifest.get("is_synthetic") or manifest.get("data_source_type") != "public_market":
        raise SyntheticResultsError("completed scaling run contains synthetic data")
    if manifest.get("data_snapshot_id") != snapshot_id:
        raise ValueError("completed scaling run uses a different public snapshot")
    if manifest.get("backend") != "numpy_density_matrix_exact":
        raise ValueError("completed scaling run uses a non-reference backend")
    if manifest.get("exact_noiseless") is not True:
        raise ValueError("completed scaling run is not marked exact and noiseless")
    if manifest.get("model_selection_data") != "validation only":
        raise ValueError("completed scaling run did not preserve validation-only selection")
    if manifest.get("test_evaluated_after_readout_freeze") is not True:
        raise ValueError("completed scaling run evaluated test before readout freeze")
    qrc = manifest.get("qrc_configuration")
    if not isinstance(qrc, dict):
        raise ValueError("completed scaling run omits QRC configuration")
    n_qubits = int(qrc["n_qubits"])
    seed = int(manifest["reservoir_seed"])
    task = str(manifest["task"])
    if n_qubits not in FULL_QUBIT_GRID or seed not in FULL_SEED_GRID or task not in TASKS:
        raise ValueError("completed scaling run has an unsupported grid identity")
    return n_qubits, seed, task


def discover_completed_scaling_runs(
    runs_root: Path,
    *,
    study_id: str,
    snapshot_id: str,
) -> dict[RunIdentity, Path]:
    """Discover latest complete, provenance-valid runs for graceful resumption."""

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


def pending_scaling_runs(
    grid: tuple[ScalingPoint, ...],
    completed: dict[RunIdentity, Path],
) -> tuple[tuple[ScalingPoint, str], ...]:
    """Return only task runs absent from a partially completed grid."""

    return tuple(
        (point, task)
        for point in grid
        for task in TASKS
        if (point.n_qubits, point.reservoir_seed, task) not in completed
    )


def _validate_run_against_feature_cache(
    experiment_dir: Path,
    *,
    expected_key: str,
    expected_virtual_nodes: int,
) -> dict[str, Any]:
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["qrc_feature_cache_key_checksum"] != expected_key:
        raise ValueError("classifier/regressor run did not reuse the prepared feature cache")
    if int(manifest["qrc_configuration"]["virtual_nodes"]) != expected_virtual_nodes:
        raise ValueError("scaling run changed the fixed virtual-node count")
    if manifest["qrc_features_generated_without_labels"] is not True:
        raise ValueError("scaling run does not prove label-free feature generation")
    if manifest["test_evaluated_after_readout_freeze"] is not True:
        raise ValueError("scaling run evaluated test before the readout was frozen")
    return cast(dict[str, Any], manifest)


def collect_scaling_rows(
    run_dirs: dict[RunIdentity, Path],
    *,
    cache_hits: dict[RunIdentity, bool] | None = None,
    repository_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Collect split-level metrics and complete provenance from scaling run artifacts."""

    rows: list[dict[str, Any]] = []
    required_manifest_fields = {
        "selected_hyperparameters",
        "qrc_raw_feature_dimension",
        "readout_shape",
        "trainable_readout_parameters",
        "state_generation_time",
        "readout_fitting_time",
        "qrc_feature_cache_hit",
        "qrc_feature_cache_key_checksum",
        "data_snapshot_id",
        "backend",
        "exact_noiseless",
        "git",
        "package_versions",
        "qrc_configuration",
        "model_selection_data",
        "test_evaluated_after_readout_freeze",
    }
    for identity, experiment_dir in sorted(run_dirs.items()):
        manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
        missing = sorted(required_manifest_fields - set(manifest))
        if missing:
            raise ValueError(f"scaling run manifest omits required fields: {missing}")
        n_qubits, seed, task = identity
        if manifest["model_selection_data"] != "validation only":
            raise ValueError("scaling row violates validation-only model selection")
        if manifest["test_evaluated_after_readout_freeze"] is not True:
            raise ValueError("scaling row violates frozen-readout test evaluation")
        experiment_directory = experiment_dir
        if repository_root is not None:
            experiment_directory = experiment_dir.relative_to(repository_root)
        common: dict[str, Any] = {
            "experiment_id": manifest["experiment_id"],
            "experiment_directory": experiment_directory.as_posix(),
            "n_qubits": n_qubits,
            "virtual_nodes": int(manifest["qrc_configuration"]["virtual_nodes"]),
            "reservoir_seed": seed,
            "task": task,
            "model_type": manifest["model_type"],
            "selected_ridge_alpha": manifest["selected_hyperparameters"]["ridge_alpha"],
            "raw_feature_dimension": manifest["qrc_raw_feature_dimension"],
            "readout_shape": manifest["readout_shape"],
            "trainable_readout_parameters": manifest["trainable_readout_parameters"],
            "state_generation_seconds": manifest["state_generation_time"],
            "readout_fitting_seconds": manifest["readout_fitting_time"],
            "cache_hit": (
                cache_hits[identity]
                if cache_hits is not None and identity in cache_hits
                else manifest["qrc_feature_cache_hit"]
            ),
            "run_cache_hit": manifest["qrc_feature_cache_hit"],
            "cache_key_checksum": manifest["qrc_feature_cache_key_checksum"],
            "data_snapshot_id": manifest["data_snapshot_id"],
            "data_manifest_checksum": manifest["data_manifest_checksum"],
            "backend": manifest["backend"],
            "exact_noiseless": manifest["exact_noiseless"],
            "git_commit": manifest["git"]["commit"],
            "git_dirty": manifest["git"]["dirty"],
            "package_versions": manifest["package_versions"],
            "model_selection_data": manifest["model_selection_data"],
            "test_evaluated_after_readout_freeze": manifest["test_evaluated_after_readout_freeze"],
        }
        for split in ("validation", "test"):
            metrics = json.loads(
                (experiment_dir / f"{split}_metrics.json").read_text(encoding="utf-8")
            )
            numeric_metrics = {
                name: value
                for name, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            rows.append({**common, "split": split, **numeric_metrics})
    return rows


def aggregate_scaling_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate metrics over independent seeds for each qubit count."""

    aggregated: list[dict[str, Any]] = []
    for n_qubits in sorted({int(row["n_qubits"]) for row in rows}):
        for task in TASKS:
            for split in ("validation", "test"):
                subset = [
                    row
                    for row in rows
                    if row["n_qubits"] == n_qubits and row["task"] == task and row["split"] == split
                ]
                if not subset:
                    continue
                metric_names = (*METRICS_BY_TASK[task], *RESOURCE_METRICS)
                for metric in metric_names:
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
                            "n_qubits": n_qubits,
                            "virtual_nodes": subset[0]["virtual_nodes"],
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
    ylabel: str,
    destination: Path,
) -> None:
    subset = table.loc[table["task"].eq(task) & table["split"].eq("test")].copy()
    if subset.empty or metric not in subset:
        raise ValueError(f"scaling results omit test metric {metric}")
    figure, axis = plt.subplots(figsize=(5.8, 3.8))
    for seed, seed_rows in subset.groupby("reservoir_seed"):
        ordered = seed_rows.sort_values("n_qubits")
        axis.plot(
            ordered["n_qubits"],
            ordered[metric],
            marker="o",
            linewidth=0.8,
            alpha=0.38,
            label=f"seed {seed}",
        )
    grouped = subset.groupby("n_qubits")[metric]
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
        xlabel="number of reservoir qubits",
        ylabel=ylabel,
        xticks=sorted(int(value) for value in subset["n_qubits"].unique()),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=220)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def plot_scaling_figures(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    """Write seed-level plus mean/uncertainty publication figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(_csv_safe_rows(rows))
    specifications = {
        "test_macro_f1_vs_qubits": (
            "macro_f1",
            "regime_classification",
            "test macro F1",
        ),
        "test_transition_pr_auc_vs_qubits": (
            "transition_pr_auc",
            "regime_classification",
            "test transition PR-AUC",
        ),
        "test_qlike_vs_qubits": ("qlike", "rv_regression", "test QLIKE"),
        "state_generation_time_vs_qubits": (
            "state_generation_seconds",
            "regime_classification",
            "state-generation time (seconds)",
        ),
        "qrc_feature_dimension_vs_qubits": (
            "raw_feature_dimension",
            "regime_classification",
            "raw QRC feature dimension",
        ),
    }
    outputs: dict[str, Path] = {}
    for name, (metric, task, ylabel) in specifications.items():
        destination = output_dir / name
        _plot_metric(
            table,
            metric=metric,
            task=task,
            ylabel=ylabel,
            destination=destination,
        )
        outputs[f"{name}_png"] = destination.with_suffix(".png")
        outputs[f"{name}_pdf"] = destination.with_suffix(".pdf")
    return outputs


def _load_scaling_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "points": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("points"), dict):
        raise ValueError("scaling state file is invalid")
    return value


def _write_tables(
    output_root: Path,
    rows: list[dict[str, Any]],
    aggregated: list[dict[str, Any]],
    resources: list[dict[str, int]],
) -> dict[str, Path]:
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_run_json": tables / "qrc_qubit_scaling_per_run.json",
        "per_run_csv": tables / "qrc_qubit_scaling_per_run.csv",
        "aggregate_json": tables / "qrc_qubit_scaling_aggregate.json",
        "aggregate_csv": tables / "qrc_qubit_scaling_aggregate.csv",
        "resources_json": tables / "qrc_qubit_scaling_resources.json",
        "resources_csv": tables / "qrc_qubit_scaling_resources.csv",
    }
    _write_json(outputs["per_run_json"], {"schema_version": 1, "rows": rows})
    pd.DataFrame(_csv_safe_rows(rows)).to_csv(outputs["per_run_csv"], index=False)
    _write_json(outputs["aggregate_json"], {"schema_version": 1, "rows": aggregated})
    pd.DataFrame(_csv_safe_rows(aggregated)).to_csv(outputs["aggregate_csv"], index=False)
    _write_json(outputs["resources_json"], {"schema_version": 1, "rows": resources})
    pd.DataFrame(resources).to_csv(outputs["resources_csv"], index=False)
    return outputs


def run_qubit_scaling(
    config_path: Path,
    *,
    qubits: tuple[int, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    smoke: bool = False,
    resume: bool = True,
) -> Path:
    """Run or resume the exact QRC scaling grid and write all aggregate artifacts."""

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config = load_scaling_study_config(config_path)
    selected_qubits = qubits or (config.smoke_qubits if smoke else config.qubits)
    selected_seeds = seeds or (config.smoke_seeds if smoke else config.seeds)
    grid = build_scaling_grid(selected_qubits, selected_seeds)
    if smoke and (len(selected_qubits) > 3 or len(selected_seeds) > 1):
        raise ValueError("smoke mode permits at most three qubit counts and one seed")
    classifier_reference, _ = _assert_reference_contracts(config)
    dataset, data_provenance = verify_scaling_public_data(config, classifier_reference)
    config.output_root.mkdir(parents=True, exist_ok=True)
    runs_root = config.output_root / "runs"
    completed = discover_completed_scaling_runs(
        runs_root,
        study_id=config.study_id,
        snapshot_id=config.data_snapshot_id,
    )
    initially_completed = dict(completed)
    state_path = config.output_root / "scaling_state.json"
    state = _load_scaling_state(state_path)
    point_state = state["points"]
    if not isinstance(point_state, dict):
        raise ValueError("scaling state points must be a mapping")
    executed: list[dict[str, Any]] = []
    resumed: list[dict[str, Any]] = []
    cache_hits: dict[RunIdentity, bool] = {}

    for point in grid:
        classifier_path = write_scaling_model_config(config, point, "regime_classification")
        regressor_path = write_scaling_model_config(config, point, "rv_regression")
        bundle = generate_qrc_features(
            classifier_path,
            allow_synthetic_results=False,
            reservoir_seed=point.reservoir_seed,
        )
        cache_key = str(bundle.metadata["cache_key_checksum"])
        cache_directory = bundle.cache_dir.relative_to(config.project_root).as_posix()
        existing_point_state = point_state.get(point.key)
        if (
            not isinstance(existing_point_state, dict)
            or existing_point_state.get("cache_directory") != cache_directory
        ):
            existing_point_state = {
                "n_qubits": point.n_qubits,
                "reservoir_seed": point.reservoir_seed,
                "feature_preparation_cache_hit": bundle.cache_hit,
                "cache_key_checksum": cache_key,
                "cache_directory": cache_directory,
            }
            point_state[point.key] = existing_point_state
            _write_json(state_path, state)
        elif existing_point_state.get("cache_key_checksum") != cache_key:
            raise ValueError("resumed scaling point cache key changed")

        for task, model_config_path in (
            ("regime_classification", classifier_path),
            ("rv_regression", regressor_path),
        ):
            identity = (point.n_qubits, point.reservoir_seed, task)
            if resume and identity in completed:
                experiment_dir = completed[identity]
                resumed.append(
                    {
                        "n_qubits": point.n_qubits,
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
                        "n_qubits": point.n_qubits,
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
                expected_virtual_nodes=int(config.fixed_qrc["virtual_nodes"]),
            )
            if task == "regime_classification":
                cache_hits[identity] = bool(existing_point_state["feature_preparation_cache_hit"])
            else:
                cache_hits[identity] = True

    requested_runs = {
        identity: completed[identity]
        for point in grid
        for task in TASKS
        if (identity := (point.n_qubits, point.reservoir_seed, task)) in completed
    }
    expected_run_count = len(grid) * len(TASKS)
    if len(requested_runs) != expected_run_count:
        raise RuntimeError(
            f"scaling grid incomplete after execution: {len(requested_runs)} "
            f"of {expected_run_count} task runs"
        )
    rows = collect_scaling_rows(
        requested_runs,
        cache_hits=cache_hits,
        repository_root=config.project_root,
    )
    aggregated = aggregate_scaling_rows(rows)
    split_rows = len(dataset.train.X) + len(dataset.validation.X) + len(dataset.test.X)
    resources = scaling_resource_estimates(
        tuple(point.n_qubits for point in grid),
        virtual_nodes=int(config.fixed_qrc["virtual_nodes"]),
        split_rows=split_rows,
    )
    unique_resources = {row["n_qubits"]: row for row in resources}
    outputs = _write_tables(
        config.output_root,
        rows,
        aggregated,
        [unique_resources[key] for key in sorted(unique_resources)],
    )
    figures = plot_scaling_figures(rows, config.output_root / "figures")
    summary_path = config.output_root / "scaling_run_summary.json"
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
            "sha256": sha256_file(config.source),
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
        "experimental_variable": "n_qubits",
        "fixed_qrc": config.fixed_qrc,
        "grid": [asdict(point) for point in grid],
        "data_provenance": data_provenance,
        "resume_enabled": resume,
        "completed_before_run": len(initially_completed),
        "executed_runs": executed,
        "resumed_runs": resumed,
        "feature_cache_points": {point.key: point_state[point.key] for point in grid},
        "per_run_row_count": len(rows),
        "aggregate_row_count": len(aggregated),
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
            "controlled exact classical simulation scaling evidence; "
            "not a physical-QPU result or quantum-advantage claim"
        ),
    }
    _write_json(summary_path, summary)
    return summary_path
