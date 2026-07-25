"""Freeze, execute, and report the validation-selected financial QRC."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
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
from qtyche_qrc.experiments.qrc_robustness import run_qrc_noise_robustness
from qtyche_qrc.experiments.qrc_run import (
    generate_qrc_features,
    qrc_config_from_model,
    run_qrc_experiment,
)
from qtyche_qrc.experiments.qrc_state_memory import _feature_condition_diagnostics
from qtyche_qrc.experiments.run import SyntheticResultsError, _git_metadata, _write_json
from qtyche_qrc.models.dataset import ModelDataset, load_model_dataset
from qtyche_qrc.runtime import runtime_metadata

STUDY_ID = "final_financial_qrc_v1"
TASKS = ("regime_classification", "rv_regression")
SEEDS = (2026, 2027, 2028)
RIDGE_GRID = (1.0e-5, 1.0e-3, 1.0e-1, 1.0)
EXACT_METRICS = {
    "regime_classification": (
        "macro_f1",
        "balanced_accuracy",
        "transition_pr_auc",
        "confusion_matrix",
    ),
    "rv_regression": (
        "qlike",
        "rmse",
        "mae",
        "non_finite_prediction_count",
        "floored_prediction_count",
    ),
}


@dataclass(frozen=True)
class FinalQRCConfig:
    """Paths and immutable scientific controls for Stage 1D."""

    source: Path
    project_root: Path
    output_root: Path
    snapshot_id: str
    data_config: Path
    classifier_config: Path
    regressor_config: Path
    robustness_config: Path
    seeds: tuple[int, ...]
    smoke_seeds: tuple[int, ...]
    raw: dict[str, Any]


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    return dict(value)


def _resolve(root: Path, value: object, location: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty path")
    return (root / value).resolve()


def _verify_checksum(path: Path, expected: object, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} checksum must be a SHA-256 string")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} checksum mismatch: {actual} != {expected}")


def _validate_model_config(
    config: ModelExperimentConfig,
    *,
    task: str,
) -> None:
    expected_type = "qrc_classifier" if task == "regime_classification" else "qrc_regressor"
    if config.model_type != expected_type or config.task != task:
        raise ValueError(f"final {task} configuration has the wrong model contract")
    qrc = qrc_config_from_model(config)
    expected = {
        "n_qubits": 2,
        "virtual_nodes": 2,
        "state_policy": "reset_each_input",
        "tau": 1.0,
        "graph": "ring",
        "j_strength": 1.0,
        "h_strength": 1.0,
        "input_scaling": 0.5,
        "backend": "numpy_density_matrix_exact",
    }
    for key, value in expected.items():
        if getattr(qrc, key) != value:
            raise ValueError(f"final architecture changed {key}")
    if tuple(float(value) for value in config.search_space["ridge_alpha"]) != RIDGE_GRID:
        raise ValueError("final readout ridge grid changed")
    if not config.search_enabled or config.maximum_trials != len(RIDGE_GRID):
        raise ValueError("final readout must select across the complete frozen ridge grid")
    marker = _mapping(config.raw.get("final_architecture"), "final_architecture")
    if marker.get("architecture_frozen") is not True:
        raise ValueError("final model config is not marked architecture_frozen")
    if marker.get("no_further_test_tuning") is not True:
        raise ValueError("final model config permits further test tuning")
    if marker.get("selection_data") != "validation only":
        raise ValueError("final model selection is not validation-only")


def load_final_qrc_config(path: Path) -> FinalQRCConfig:
    """Load and verify the complete frozen-architecture configuration."""

    source = path.resolve()
    root = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "configuration")
    if root.get("schema_version") != 1:
        raise ValueError("final QRC configuration schema_version must be 1")
    study = _mapping(root.get("study"), "study")
    freeze = _mapping(root.get("freeze"), "freeze")
    architecture = _mapping(root.get("architecture"), "architecture")
    readout = _mapping(root.get("readout"), "readout")
    if study.get("id") != STUDY_ID:
        raise ValueError(f"final QRC study ID must remain {STUDY_ID}")
    project_setting = study.get("project_root")
    if not isinstance(project_setting, str):
        raise ValueError("study.project_root must be a path")
    project_root = (source.parent / project_setting).resolve()
    config = FinalQRCConfig(
        source=source,
        project_root=project_root,
        output_root=_resolve(project_root, study.get("output_root"), "study.output_root"),
        snapshot_id=str(study.get("data_snapshot_id")),
        data_config=_resolve(project_root, study.get("data_config"), "study.data_config"),
        classifier_config=_resolve(
            project_root, study.get("classifier_config"), "study.classifier_config"
        ),
        regressor_config=_resolve(
            project_root, study.get("regressor_config"), "study.regressor_config"
        ),
        robustness_config=_resolve(
            project_root, study.get("robustness_config"), "study.robustness_config"
        ),
        seeds=tuple(int(value) for value in cast(list[int], study.get("reservoir_seeds"))),
        smoke_seeds=tuple(int(value) for value in cast(list[int], study.get("smoke_seeds"))),
        raw=root,
    )
    for key, path_value in (
        ("data_config", config.data_config),
        ("classifier_config", config.classifier_config),
        ("regressor_config", config.regressor_config),
        ("robustness_config", config.robustness_config),
    ):
        _verify_checksum(path_value, study.get(f"{key}_sha256"), key)
    evidence = _mapping(root.get("selection_evidence"), "selection_evidence")
    for name, record_value in evidence.items():
        record = _mapping(record_value, f"selection_evidence.{name}")
        evidence_path = _resolve(
            project_root, record.get("path"), f"selection_evidence.{name}.path"
        )
        _verify_checksum(evidence_path, record.get("sha256"), f"selection_evidence.{name}")
    expected_architecture = {
        "n_qubits": 2,
        "virtual_nodes": 2,
        "state_policy": "reset_each_input",
        "tau": 1.0,
        "graph": "ring",
        "j_strength": 1.0,
        "h_strength": 1.0,
        "input_scaling": 0.5,
        "backend": "numpy_density_matrix_exact",
        "exact_noiseless": True,
        "raw_feature_dimension": 6,
    }
    for key, value in expected_architecture.items():
        if architecture.get(key) != value:
            raise ValueError(f"frozen architecture changed {key}")
    observables = _mapping(architecture.get("observables"), "architecture.observables")
    if observables != {
        "single_qubit": "analytic_Z_i",
        "two_qubit": "analytic_unique_Z_i_Z_j",
    }:
        raise ValueError("final analytic observable family changed")
    if architecture.get("input_reinjection") != "partial_input_qubit_reinjection":
        raise ValueError("final partial input-qubit reinjection changed")
    if tuple(float(value) for value in cast(list[float], readout["ridge_alpha_grid"])) != (
        RIDGE_GRID
    ):
        raise ValueError("final ridge grid changed")
    if (
        readout.get("preprocessing_fit_split") != "train"
        or readout.get("selection_split") != "validation"
        or readout.get("test_evaluated_after_freeze") is not True
    ):
        raise ValueError("final temporal evaluation contract changed")
    if config.seeds != SEEDS or set(config.smoke_seeds) - set(SEEDS):
        raise ValueError("final seed grid changed")
    if freeze != {
        "architecture_frozen": True,
        "no_further_test_tuning": True,
        "selection_data": "validation only",
        "test_metrics_inspected_after_validation_decisions": True,
        "test_metrics_used_for_selection": False,
    }:
        raise ValueError("final freeze declaration changed")
    _validate_model_config(
        load_model_config(config.classifier_config), task="regime_classification"
    )
    _validate_model_config(load_model_config(config.regressor_config), task="rv_regression")
    return config


def verify_final_public_data(
    config: FinalQRCConfig,
) -> tuple[ModelDataset, dict[str, Any]]:
    """Verify source and processed checksums and reject fixture data."""

    data_config = load_data_config(config.data_config)
    snapshot = verify_public_snapshot(data_config)
    if snapshot.get("snapshot_id") != config.snapshot_id:
        raise ValueError("final snapshot ID changed")
    classifier = load_model_config(config.classifier_config)
    dataset = load_model_dataset(classifier.processed_dir)
    if dataset.is_synthetic or dataset.data_source_type != "public_market":
        raise SyntheticResultsError(
            "final financial QRC requires verified non-synthetic public-market data"
        )
    if dataset.manifest.get("source_snapshot_id") != config.snapshot_id:
        raise ValueError("processed data do not derive from the frozen snapshot")
    if data_config.snapshot_manifest_path is None:
        raise FileNotFoundError("public data config omits the snapshot manifest")
    return dataset, {
        "data_snapshot_id": config.snapshot_id,
        "data_config_sha256": sha256_file(config.data_config),
        "source_snapshot_manifest_sha256": sha256_file(data_config.snapshot_manifest_path),
        "raw_file_checksums": {
            name: str(record["sha256"])
            for name, record in sorted(cast(dict[str, Any], snapshot["files"]).items())
        },
        "processed_manifest_sha256": dataset.processed_checksums["data_manifest.json"],
        "processed_checksums": dataset.processed_checksums,
        "split_row_counts": {
            "train": len(dataset.train.X),
            "validation": len(dataset.validation.X),
            "test": len(dataset.test.X),
        },
    }


def _completed_run_identity(directory: Path, snapshot_id: str) -> tuple[int, str] | None:
    manifest_path = directory / "manifest.json"
    config_path = directory / "config.yaml"
    required = (
        manifest_path,
        config_path,
        directory / "validation_metrics.json",
        directory / "test_metrics.json",
        directory / "validation_predictions.csv",
        directory / "test_predictions.csv",
        directory / "model/readout.npz",
    )
    if not all(path.is_file() for path in required):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        return None
    raw = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "run config")
    marker = raw.get("final_architecture")
    if not isinstance(marker, dict) or marker.get("id") != STUDY_ID:
        return None
    if manifest.get("is_synthetic") or manifest.get("data_source_type") != "public_market":
        raise SyntheticResultsError("completed final QRC run contains synthetic data")
    qrc = _mapping(manifest.get("qrc_configuration"), "manifest.qrc_configuration")
    if (
        manifest.get("data_snapshot_id") != snapshot_id
        or manifest.get("backend") != "numpy_density_matrix_exact"
        or manifest.get("exact_noiseless") is not True
        or manifest.get("model_selection_data") != "validation only"
        or manifest.get("test_evaluated_after_readout_freeze") is not True
        or qrc.get("n_qubits") != 2
        or qrc.get("virtual_nodes") != 2
        or qrc.get("state_policy") != "reset_each_input"
    ):
        raise ValueError("completed final QRC run violates the frozen contract")
    seed = int(manifest["reservoir_seed"])
    task = str(manifest["task"])
    return (seed, task) if seed in SEEDS and task in TASKS else None


def discover_completed_final_runs(
    runs_root: Path, *, snapshot_id: str
) -> dict[tuple[int, str], Path]:
    """Find the latest complete matching run for deterministic partial resumption."""

    completed: dict[tuple[int, str], Path] = {}
    if not runs_root.is_dir():
        return completed
    for directory in sorted(path.parent for path in runs_root.rglob("manifest.json")):
        identity = _completed_run_identity(directory, snapshot_id)
        if identity is not None:
            completed[identity] = directory
    return completed


def _prediction_correlation(path: Path) -> float:
    table = pd.read_csv(path)
    actual = table["true_rv_5d"].to_numpy(dtype=float)
    predicted = table["predicted_rv_5d"].to_numpy(dtype=float)
    return float(np.corrcoef(actual, predicted)[0, 1])


def _run_row(
    directory: Path,
    *,
    split: str,
    diagnostics: dict[str, Any],
    reference_rows: dict[tuple[int, str, str], dict[str, Any]],
    repository_root: Path,
) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / f"{split}_metrics.json").read_text(encoding="utf-8"))
    timing = json.loads((directory / "timing.json").read_text(encoding="utf-8"))
    with np.load(directory / "model/readout.npz") as payload:
        coefficients = np.asarray(payload["readout"], dtype=float)
    predictions = pd.read_csv(directory / f"{split}_predictions.csv")
    numeric_predictions = predictions.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    seed = int(manifest["reservoir_seed"])
    task = str(manifest["task"])
    row: dict[str, Any] = {
        "reservoir_seed": seed,
        "task": task,
        "split": split,
        "n_qubits": 2,
        "virtual_nodes": 2,
        "state_policy": "reset_each_input",
        "selected_ridge_alpha": float(manifest["selected_hyperparameters"]["ridge_alpha"]),
        "raw_feature_dimension": int(manifest["qrc_raw_feature_dimension"]),
        "readout_shape": manifest["readout_shape"],
        "trainable_readout_parameters": int(manifest["trainable_readout_parameters"]),
        "feature_generation_seconds": float(timing["state_generation_seconds"]),
        "readout_fitting_seconds": float(timing["readout_fitting_seconds"]),
        "cache_key_checksum": str(manifest["qrc_feature_cache_key_checksum"]),
        "cache_hit": bool(manifest["qrc_feature_cache_hit"]),
        "effective_rank": float(diagnostics["effective_rank"]),
        "numerical_rank": int(diagnostics["numerical_rank"]),
        "condition_number": float(diagnostics["condition_number"]),
        "largest_singular_value": float(diagnostics["largest_singular_value"]),
        "smallest_retained_singular_value": float(diagnostics["smallest_retained_singular_value"]),
        "readout_coefficient_l2_norm": float(np.linalg.norm(coefficients)),
        "maximum_absolute_readout_coefficient": float(np.max(np.abs(coefficients))),
        "finite_coefficients": bool(np.isfinite(coefficients).all()),
        "finite_predictions": bool(np.isfinite(numeric_predictions).all()),
        "data_snapshot_id": manifest["data_snapshot_id"],
        "backend": manifest["backend"],
        "exact_noiseless": manifest["exact_noiseless"],
        "model_selection_data": manifest["model_selection_data"],
        "test_evaluated_after_readout_freeze": manifest["test_evaluated_after_readout_freeze"],
        "experiment_directory": directory.relative_to(repository_root).as_posix(),
    }
    for metric in EXACT_METRICS[task]:
        row[metric] = metrics[metric]
    if task == "rv_regression":
        row["correlation"] = _prediction_correlation(directory / f"{split}_predictions.csv")
    reference = reference_rows.get((seed, task, split))
    checked = (
        ("macro_f1", "balanced_accuracy", "transition_pr_auc")
        if task == "regime_classification"
        else ("qlike", "rmse", "mae")
    )
    deltas = (
        {metric: abs(float(row[metric]) - float(reference[metric])) for metric in checked}
        if reference is not None
        else {}
    )
    row["reset_ablation_reference_available"] = reference is not None
    row["reset_ablation_maximum_metric_delta"] = max(deltas.values(), default=None)
    row["reset_ablation_reproduced_within_1e_12"] = (
        max(deltas.values(), default=float("inf")) <= 1e-12
    )
    return row


def _reference_rows(config: FinalQRCConfig) -> dict[tuple[int, str, str], dict[str, Any]]:
    record = _mapping(
        _mapping(config.raw["selection_evidence"], "selection_evidence")["reset_reference"],
        "selection_evidence.reset_reference",
    )
    path = _resolve(config.project_root, record["path"], "reset reference")
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "reset reference")
    rows = cast(list[dict[str, Any]], payload["rows"])
    return {
        (int(row["reservoir_seed"]), str(row["task"]), str(row["split"])): row
        for row in rows
        if row.get("state_policy") == "reset_each_input"
    }


def aggregate_exact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate scalar exact metrics across reservoir seeds by task and split."""

    outputs: list[dict[str, Any]] = []
    for (task, split), group in pd.DataFrame(rows).groupby(["task", "split"], sort=True):
        subset = group
        record: dict[str, Any] = {
            "task": task,
            "split": split,
            "reservoir_seed_count": int(subset["reservoir_seed"].nunique()),
            "reservoir_seeds": sorted(int(value) for value in subset["reservoir_seed"].unique()),
            "confusion_matrices_by_seed": (
                [
                    {
                        "reservoir_seed": int(row["reservoir_seed"]),
                        "confusion_matrix": row["confusion_matrix"],
                    }
                    for row in rows
                    if row["task"] == task and row["split"] == split
                ]
                if task == "regime_classification"
                else None
            ),
        }
        metrics = (
            (
                "macro_f1",
                "balanced_accuracy",
                "transition_pr_auc",
                "condition_number",
                "effective_rank",
                "readout_coefficient_l2_norm",
            )
            if task == "regime_classification"
            else (
                "qlike",
                "rmse",
                "mae",
                "correlation",
                "condition_number",
                "effective_rank",
                "readout_coefficient_l2_norm",
            )
        )
        for metric in metrics:
            values = subset[metric].to_numpy(dtype=float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_standard_deviation"] = float(values.std(ddof=0))
            record[f"{metric}_minimum"] = float(values.min())
            record[f"{metric}_maximum"] = float(values.max())
        outputs.append(record)
    return outputs


def _csv_safe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for row in rows:
        safe = dict(row)
        for key, value in safe.items():
            if isinstance(value, (dict, list, tuple)):
                safe[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        outputs.append(safe)
    return outputs


def _validation_only_payload(value: Any) -> Any:
    """Remove held-out-selection guard fields from published validation evidence."""

    if isinstance(value, dict):
        return {
            key: _validation_only_payload(item)
            for key, item in value.items()
            if not key.startswith("test_")
        }
    if isinstance(value, list):
        return [_validation_only_payload(item) for item in value]
    return value


def _validation_selection(config: FinalQRCConfig) -> dict[str, Any]:
    state_path = (
        config.project_root / "results/qrc_state_memory_ablation/tables/"
        "qrc_state_memory_validation_policy_selection.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    encoding_path = (
        config.project_root / "results/qrc_encoding_density/tables/"
        "qrc_encoding_density_validation_candidates.json"
    )
    encoding = json.loads(encoding_path.read_text(encoding="utf-8"))
    scaling_path = (
        config.project_root / "results/qrc_qubit_scaling/tables/qrc_qubit_scaling_aggregate.json"
    )
    scaling = json.loads(scaling_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "selection_basis": "validation only",
        "architecture_frozen": True,
        "selected": {
            "n_qubits": 2,
            "virtual_nodes": 2,
            "state_policy": "reset_each_input",
        },
        "qubit_scaling_validation_evidence": {
            "schema_version": scaling["schema_version"],
            "rows": [row for row in scaling["rows"] if row.get("split") == "validation"],
        },
        "encoding_density_validation_evidence": _validation_only_payload(encoding),
        "state_memory_validation_evidence": _validation_only_payload(state),
        "held_out_metrics_excluded": True,
        "selection_complete": True,
    }


def _find_classical_runs(config: FinalQRCConfig) -> list[dict[str, Any]]:
    public_root = config.project_root / "results/public_market"
    rows: list[dict[str, Any]] = []
    names = {
        "majority_classifier": "majority classifier",
        "regime_persistence": "regime persistence",
        "logistic_regression": "logistic regression",
        "esn_classifier": "ESN classifier",
        "rv_persistence": "RV persistence",
        "esn_regressor": "ESN regressor",
    }
    for directory in sorted(path for path in public_root.iterdir() if path.is_dir()):
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_type = str(manifest["model_type"])
        if model_type not in names:
            continue
        for split in ("validation", "test"):
            metrics = json.loads((directory / f"{split}_metrics.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": names[model_type],
                    "model_type": model_type,
                    "task": manifest["task"],
                    "split": split,
                    "reservoir_seed": None,
                    **{
                        key: metrics.get(key)
                        for key in (
                            "macro_f1",
                            "balanced_accuracy",
                            "transition_pr_auc",
                            "qlike",
                            "rmse",
                            "mae",
                        )
                    },
                }
            )
    garch_runs = sorted(
        path
        for path in (config.project_root / "results/garch_baseline/runs").glob(
            "*_gaussian_garch_1_1_full"
        )
        if (path / "manifest.json").is_file()
    )
    if not garch_runs:
        raise FileNotFoundError("full GARCH baseline is required for final comparison")
    garch = garch_runs[-1]
    for split in ("validation", "test"):
        metrics = json.loads((garch / f"{split}_metrics.json").read_text(encoding="utf-8"))
        common = {
            "model": "GARCH(1,1)",
            "model_type": "gaussian_garch_1_1",
            "split": split,
            "reservoir_seed": None,
        }
        rows.extend(
            (
                {
                    **common,
                    "task": "regime_classification",
                    "macro_f1": metrics.get("macro_f1"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "transition_pr_auc": None,
                    "transition_pr_auc_applicable": False,
                    "qlike": None,
                    "rmse": None,
                    "mae": None,
                },
                {
                    **common,
                    "task": "rv_regression",
                    "macro_f1": None,
                    "balanced_accuracy": None,
                    "transition_pr_auc": None,
                    "transition_pr_auc_applicable": False,
                    "qlike": metrics.get("qlike"),
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                },
            )
        )
    return rows


def build_final_benchmark(
    config: FinalQRCConfig, exact_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build a directly comparable table without formal significance tests."""

    rows = _find_classical_runs(config)
    for row in exact_rows:
        rows.append(
            {
                "model": "frozen final QRC",
                "model_type": (
                    "qrc_classifier" if row["task"] == "regime_classification" else "qrc_regressor"
                ),
                "task": row["task"],
                "split": row["split"],
                "reservoir_seed": row["reservoir_seed"],
                **{
                    key: row.get(key)
                    for key in (
                        "macro_f1",
                        "balanced_accuracy",
                        "transition_pr_auc",
                        "qlike",
                        "rmse",
                        "mae",
                    )
                },
            }
        )
    return rows


def _save_figure(figure: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=240)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def _benchmark_figure(rows: list[dict[str, Any]], *, task: str, destination: Path) -> None:
    table = pd.DataFrame(rows)
    subset = table.loc[table["task"].eq(task) & table["split"].eq("test")].copy()
    metrics = (
        ("macro_f1", "balanced_accuracy", "transition_pr_auc")
        if task == "regime_classification"
        else ("qlike", "rmse", "mae")
    )
    models = list(dict.fromkeys(str(value) for value in subset["model"]))
    figure, axes = plt.subplots(1, 3, figsize=(12.3, 3.8))
    for axis, metric in zip(axes, metrics):
        means: list[float] = []
        deviations: list[float] = []
        labels: list[str] = []
        for model in models:
            values = pd.to_numeric(
                subset.loc[subset["model"].eq(model), metric], errors="coerce"
            ).dropna()
            if values.empty:
                continue
            labels.append(model)
            means.append(float(values.mean()))
            deviations.append(float(values.std(ddof=0)))
        positions = np.arange(len(labels))
        axis.bar(positions, means, yerr=deviations, color="#4c78a8", capsize=3)
        axis.set_xticks(positions, labels, rotation=35, ha="right", fontsize=8)
        axis.set_ylabel(metric.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
    _save_figure(figure, destination)


def _conditioning_figure(rows: list[dict[str, Any]], destination: Path) -> None:
    table = pd.DataFrame(rows).drop_duplicates("reservoir_seed")
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    axes[0].bar(
        table["reservoir_seed"].astype(str),
        table["condition_number"],
        color="#4c78a8",
    )
    axes[0].set_yscale("log")
    axes[0].set(xlabel="reservoir seed", ylabel="training-feature condition number")
    axes[1].bar(
        table["reservoir_seed"].astype(str),
        table["readout_coefficient_l2_norm"],
        color="#f58518",
    )
    axes[1].set(xlabel="reservoir seed", ylabel="classifier coefficient L2 norm")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    _save_figure(figure, destination)


def _robustness_figure(
    table: pd.DataFrame,
    *,
    study_type: str,
    x_column: str,
    xlabel: str,
    destination: Path,
    log_x: bool = False,
) -> None:
    specs = (
        ("regime_classification", "macro_f1", "macro-F1"),
        ("regime_classification", "transition_pr_auc", "transition PR-AUC"),
        ("rv_regression", "qlike", "QLIKE"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))
    for axis, (task, metric, ylabel) in zip(axes, specs):
        subset = table.loc[
            table["study_type"].eq(study_type) & table["task"].eq(task) & table["split"].eq("test")
        ]
        axis.scatter(subset[x_column], subset[metric], s=14, alpha=0.25)
        grouped = subset.groupby(x_column)[metric]
        means = grouped.mean()
        deviations = grouped.std(ddof=0).fillna(0.0)
        axis.errorbar(
            means.index,
            means,
            yerr=deviations,
            color="black",
            marker="o",
            capsize=3,
        )
        reference = table.loc[
            table["study_type"].eq("analytic_reference")
            & table["task"].eq(task)
            & table["split"].eq("test"),
            metric,
        ]
        axis.axhline(float(reference.mean()), color="#d62728", linestyle="--")
        if log_x:
            axis.set_xscale("log", base=2)
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=0.2)
    _save_figure(figure, destination)


def write_publication_figures(
    config: FinalQRCConfig,
    *,
    exact_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Write the six requested PNG/PDF publication figures."""

    root = config.output_root / "figures"
    destinations = {
        "classification_benchmark": root / "final_classification_benchmark",
        "regression_benchmark": root / "final_regression_benchmark",
        "numerical_conditioning": root / "final_numerical_conditioning",
    }
    _benchmark_figure(
        benchmark_rows,
        task="regime_classification",
        destination=destinations["classification_benchmark"],
    )
    _benchmark_figure(
        benchmark_rows,
        task="rv_regression",
        destination=destinations["regression_benchmark"],
    )
    _conditioning_figure(
        [row for row in exact_rows if row["task"] == "regime_classification"],
        destinations["numerical_conditioning"],
    )
    robustness_path = config.output_root / "robustness/tables/qrc_noise_robustness_per_run.json"
    if robustness_path.is_file():
        robustness_rows = json.loads(robustness_path.read_text(encoding="utf-8"))["rows"]
        table = pd.DataFrame(robustness_rows)
        robustness_specs = (
            (
                "shot_convergence",
                "finite_shot",
                "shot_count",
                "shots per virtual-node state",
                True,
            ),
            (
                "depolarizing_robustness",
                "depolarizing_noise",
                "depolarizing_probability",
                "local depolarizing probability",
                False,
            ),
            (
                "measurement_robustness",
                "measurement_noise",
                "measurement_bit_flip_probability",
                "measurement bit-flip probability",
                False,
            ),
        )
        for name, study_type, column, xlabel, log_x in robustness_specs:
            destination = root / f"final_qrc_{name}"
            _robustness_figure(
                table,
                study_type=study_type,
                x_column=column,
                xlabel=xlabel,
                destination=destination,
                log_x=log_x,
            )
            destinations[name] = destination
    return {
        f"{name}_{extension}": path.with_suffix(f".{extension}")
        .relative_to(config.project_root)
        .as_posix()
        for name, path in destinations.items()
        for extension in ("png", "pdf")
        if path.with_suffix(f".{extension}").is_file()
    }


def run_final_financial_qrc(
    config_path: Path,
    *,
    smoke: bool = False,
    resume: bool = True,
    refresh_robustness: bool = True,
) -> Path:
    """Run/resume the final exact benchmark and isolated robustness refresh."""

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config = load_final_qrc_config(config_path)
    _dataset, provenance = verify_final_public_data(config)
    seeds = config.smoke_seeds if smoke else config.seeds
    config.output_root.mkdir(parents=True, exist_ok=True)
    completed = discover_completed_final_runs(
        config.output_root / "exact/runs", snapshot_id=config.snapshot_id
    )
    completed_before = len({(seed, task) for seed in seeds for task in TASKS} & set(completed))
    diagnostics_by_seed: dict[int, dict[str, Any]] = {}
    cache_keys_by_seed: dict[int, str] = {}
    executed: list[dict[str, Any]] = []
    resumed: list[dict[str, Any]] = []
    for seed in seeds:
        bundle = generate_qrc_features(
            config.classifier_config,
            allow_synthetic_results=False,
            reservoir_seed=seed,
        )
        if bundle.train.shape[1] != 6:
            raise ValueError("final QRC did not produce six raw features")
        diagnostics_by_seed[seed] = _feature_condition_diagnostics(bundle.train)
        cache_key = str(bundle.metadata["cache_key_checksum"])
        cache_keys_by_seed[seed] = cache_key
        for task, model_config in (
            (TASKS[0], config.classifier_config),
            (TASKS[1], config.regressor_config),
        ):
            identity = (seed, task)
            if resume and identity in completed:
                directory = completed[identity]
                resumed.append(
                    {
                        "reservoir_seed": seed,
                        "task": task,
                        "experiment_directory": directory.relative_to(
                            config.project_root
                        ).as_posix(),
                    }
                )
            else:
                directory = run_qrc_experiment(
                    model_config,
                    allow_synthetic_results=False,
                    reservoir_seed=seed,
                )
                completed[identity] = directory
                executed.append(
                    {
                        "reservoir_seed": seed,
                        "task": task,
                        "experiment_directory": directory.relative_to(
                            config.project_root
                        ).as_posix(),
                    }
                )
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest["qrc_feature_cache_key_checksum"] != cache_key:
                raise ValueError("final classifier and regressor did not share features")
    requested = {
        identity: completed[identity]
        for identity in ((seed, task) for seed in seeds for task in TASKS)
    }
    references = _reference_rows(config)
    rows = [
        _run_row(
            directory,
            split=split,
            diagnostics=diagnostics_by_seed[seed],
            reference_rows=references,
            repository_root=config.project_root,
        )
        for (seed, _task), directory in requested.items()
        for split in ("validation", "test")
    ]
    if not all(
        row["finite_coefficients"]
        and row["finite_predictions"]
        and row["reset_ablation_reproduced_within_1e_12"]
        for row in rows
    ):
        raise ValueError("final exact run failed finiteness or reset-reference reproduction")
    for seed in seeds:
        seed_rows = [row for row in rows if row["reservoir_seed"] == seed]
        if len({row["cache_key_checksum"] for row in seed_rows}) != 1:
            raise ValueError("final classifier/regressor feature caches differ")
    aggregate = aggregate_exact_rows(rows)
    tables = config.output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    _write_json(tables / "final_qrc_exact_per_run.json", {"schema_version": 1, "rows": rows})
    pd.DataFrame(_csv_safe(rows)).to_csv(tables / "final_qrc_exact_per_run.csv", index=False)
    _write_json(
        tables / "final_qrc_exact_aggregate.json",
        {"schema_version": 1, "rows": aggregate},
    )
    pd.DataFrame(_csv_safe(aggregate)).to_csv(tables / "final_qrc_exact_aggregate.csv", index=False)
    validation_selection = _validation_selection(config)
    _write_json(
        config.output_root / "final_validation_selection.json",
        validation_selection,
    )
    existing_robustness_summary = config.output_root / "robustness/robustness_run_summary.json"
    robustness_summary: str | None = (
        existing_robustness_summary.relative_to(config.project_root).as_posix()
        if existing_robustness_summary.is_file()
        else None
    )
    if refresh_robustness:
        robustness_summary = (
            run_qrc_noise_robustness(config.robustness_config, smoke=smoke, resume=resume)
            .relative_to(config.project_root)
            .as_posix()
        )
    benchmark = build_final_benchmark(config, rows)
    _write_json(
        tables / "final_financial_benchmark.json",
        {
            "schema_version": 1,
            "formal_significance_tests_run": False,
            "rows": benchmark,
        },
    )
    pd.DataFrame(_csv_safe(benchmark)).to_csv(tables / "final_financial_benchmark.csv", index=False)
    figures = write_publication_figures(config, exact_rows=rows, benchmark_rows=benchmark)
    selected_ridges = [
        {
            "reservoir_seed": seed,
            "task": task,
            "ridge_alpha": float(
                json.loads((directory / "manifest.json").read_text(encoding="utf-8"))[
                    "selected_hyperparameters"
                ]["ridge_alpha"]
            ),
        }
        for (seed, task), directory in requested.items()
    ]
    manifest_path = config.output_root / "final_architecture_manifest.json"
    manifest = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "success",
        "architecture_frozen": True,
        "no_further_test_tuning": True,
        "architecture": config.raw["architecture"],
        "readout": {
            **cast(dict[str, Any], config.raw["readout"]),
            "selected_ridge_values": selected_ridges,
        },
        "configuration": {
            "path": config.source.relative_to(config.project_root).as_posix(),
            "sha256": sha256_file(config.source),
        },
        "data_provenance": provenance,
        "selection_lineage": {
            "n_qubits": "selected from qubit-scaling validation/performance-cost evidence",
            "virtual_nodes": "selected from validation-only encoding-density evidence",
            "state_policy": "selected from validation-only state-memory evidence",
            "selection_evidence": config.raw["selection_evidence"],
            "test_metrics_used_for_selection": False,
            "test_metrics_inspected_only_after_validation_decisions": True,
            "further_architecture_tuning_permitted": False,
        },
        "exact_run_cache_keys_by_seed": cache_keys_by_seed,
        "near_singular_seed_recorded": any(
            float(value["condition_number"]) > 1e12 for value in diagnostics_by_seed.values()
        ),
        "resource_estimates": {
            "hilbert_dimension": 4,
            "raw_feature_dimension": 6,
            "split_rows": sum(provenance["split_row_counts"].values()),
            "estimated_peak_density_matrix_bytes": 768,
            "estimated_uncompressed_feature_cache_bytes_per_seed": (
                sum(provenance["split_row_counts"].values()) * 6 * 8
            ),
            "estimated_uncompressed_feature_cache_bytes_selected_seeds": (
                sum(provenance["split_row_counts"].values()) * 6 * 8 * len(seeds)
            ),
            "prediction_artifacts_and_library_overhead_included": False,
        },
        "numerical_limitation": (
            "The reset-policy seed 2027 training-feature matrix is near singular. "
            "The frozen architecture and ridge grid were not changed; selected "
            "regularisation kept coefficients and predictions finite."
        ),
        "exact_classical_density_matrix_simulation": True,
        "physical_qpu_execution": False,
        "quantum_advantage_claim": False,
        "git": _git_metadata(config.project_root),
        **runtime_metadata(),
        "outputs": {
            "validation_selection": ("results/final_financial_qrc/final_validation_selection.json"),
            "exact_per_run": ("results/final_financial_qrc/tables/final_qrc_exact_per_run.json"),
            "exact_aggregate": (
                "results/final_financial_qrc/tables/final_qrc_exact_aggregate.json"
            ),
            "benchmark": ("results/final_financial_qrc/tables/final_financial_benchmark.json"),
            "robustness_summary": robustness_summary,
            "figures": figures,
        },
    }
    _write_json(manifest_path, manifest)
    summary_path = config.output_root / "final_run_summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "status": "success",
            "mode": "smoke" if smoke else "full",
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.perf_counter() - started,
            "resume_enabled": resume,
            "completed_before_run": completed_before,
            "executed_runs": executed,
            "resumed_runs": resumed,
            "architecture_manifest": manifest_path.relative_to(config.project_root).as_posix(),
            "architecture_manifest_sha256": sha256_file(manifest_path),
            "robustness_summary": robustness_summary,
            "interpretation": (
                "exact classical simulation of a quantum reservoir; not physical-QPU "
                "execution and not evidence of quantum advantage"
            ),
        },
    )
    return summary_path
