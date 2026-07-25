"""Leakage-safe public-market Gaussian GARCH(1,1) baseline orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from qtyche_qrc.data.config import load_data_config
from qtyche_qrc.data.download import sha256_file, verify_public_snapshot
from qtyche_qrc.evaluation.metrics import regression_metrics
from qtyche_qrc.experiments.run import SyntheticResultsError, _git_metadata, _write_json
from qtyche_qrc.models.baselines.garch import GARCHFitResult, GaussianGARCH11
from qtyche_qrc.models.dataset import ModelDataset, ModelSplit, load_model_dataset
from qtyche_qrc.models.qrc.encoding import array_checksum
from qtyche_qrc.runtime import runtime_metadata

STUDY_ID = "gaussian_garch11_public_v1"
MODEL_TYPE = "gaussian_garch_1_1"
TASK = "rv_regression"
COMPARISON_MODELS = {
    "rv_persistence": "Realized-variance persistence",
    "esn_regressor": "ESN regressor",
    "gaussian_garch_1_1": "Gaussian GARCH(1,1)",
    "qrc_regressor": "Final QRC",
}
COMPARISON_METRICS = ("qlike", "rmse", "mae", "prediction_correlation")


@dataclass(frozen=True)
class GARCHStudyConfig:
    """Validated study paths, model controls, smoke controls, and references."""

    source: Path
    project_root: Path
    study_id: str
    output_root: Path
    data_config: Path
    data_config_sha256: str
    data_snapshot_id: str
    processed_manifest_sha256: str
    qrc_selection_summary: Path
    qrc_selection_summary_sha256: str
    public_baseline_results: Path
    horizon: int
    annualization: float
    stationarity_margin: float
    parameter_variance_floor: float
    evaluation_variance_floor: float
    maximum_iterations: int
    optimiser_tolerance: float
    maximum_starts: int
    smoke_training_return_count: int
    smoke_maximum_starts: int
    raw: dict[str, Any]


@dataclass(frozen=True)
class GARCHReturnInputs:
    """Raw return stream and exact frozen evaluation-row alignment."""

    training_dates: NDArray[np.datetime64]
    training_returns: NDArray[np.float64]
    post_training_dates: NDArray[np.datetime64]
    post_training_returns: NDArray[np.float64]
    validation_indices: NDArray[np.int_]
    test_indices: NDArray[np.int_]
    raw_return_checksum: str
    training_return_checksum: str
    post_training_return_checksum: str
    evaluation_date_checksum: str
    processed_return_maximum_absolute_difference: float
    reconstructed_target_maximum_absolute_difference: float
    raw_spy_path: Path


@dataclass(frozen=True)
class CompletedGARCHRun:
    """Resume identity and complete run directory."""

    mode: str
    experiment_dir: Path


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


def _number(mapping: dict[str, Any], key: str, location: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}.{key} must be numeric")
    return float(value)


def load_garch_study_config(path: Path) -> GARCHStudyConfig:
    """Load and validate the frozen GARCH study contract."""

    source = path.resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "configuration")
    if root.get("schema_version") != 1:
        raise ValueError("GARCH baseline configuration schema_version must be 1")
    study = _mapping(root.get("study"), "study")
    model = _mapping(root.get("model"), "model")
    smoke = _mapping(root.get("smoke"), "smoke")
    project_root = (source.parent / _text(study, "project_root", "study")).resolve()
    config = GARCHStudyConfig(
        source=source,
        project_root=project_root,
        study_id=_text(study, "id", "study"),
        output_root=(project_root / _text(study, "output_root", "study")).resolve(),
        data_config=(project_root / _text(study, "data_config", "study")).resolve(),
        data_config_sha256=_text(study, "data_config_sha256", "study"),
        data_snapshot_id=_text(study, "data_snapshot_id", "study"),
        processed_manifest_sha256=_text(study, "processed_manifest_sha256", "study"),
        qrc_selection_summary=(
            project_root / _text(study, "qrc_selection_summary", "study")
        ).resolve(),
        qrc_selection_summary_sha256=_text(study, "qrc_selection_summary_sha256", "study"),
        public_baseline_results=(
            project_root / _text(study, "public_baseline_results", "study")
        ).resolve(),
        horizon=_integer(model, "horizon", "model"),
        annualization=_number(model, "annualization", "model"),
        stationarity_margin=_number(model, "stationarity_margin", "model"),
        parameter_variance_floor=_number(model, "parameter_variance_floor", "model"),
        evaluation_variance_floor=_number(model, "evaluation_variance_floor", "model"),
        maximum_iterations=_integer(model, "maximum_iterations", "model"),
        optimiser_tolerance=_number(model, "optimiser_tolerance", "model"),
        maximum_starts=_integer(model, "maximum_starts", "model"),
        smoke_training_return_count=_integer(smoke, "training_return_count", "smoke"),
        smoke_maximum_starts=_integer(smoke, "maximum_starts", "smoke"),
        raw=root,
    )
    if config.study_id != STUDY_ID:
        raise ValueError(f"GARCH study ID must remain {STUDY_ID}")
    if config.horizon != 5 or config.annualization != 252.0:
        raise ValueError("GARCH target contract requires horizon=5 and annualization=252")
    if (
        config.maximum_starts <= 0
        or config.smoke_maximum_starts <= 0
        or config.smoke_maximum_starts > config.maximum_starts
        or config.smoke_training_return_count < 50
    ):
        raise ValueError("GARCH full/smoke optimiser grids are invalid")
    if sha256_file(config.data_config) != config.data_config_sha256:
        raise ValueError("frozen public-data configuration checksum mismatch")
    if sha256_file(config.qrc_selection_summary) != config.qrc_selection_summary_sha256:
        raise ValueError("frozen final-QRC selection summary checksum mismatch")
    qrc_summary = json.loads(config.qrc_selection_summary.read_text(encoding="utf-8"))
    selection = qrc_summary.get("validation_only_state_policy_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("selection_basis") != "validation only"
        or selection.get("test_metrics_read") is not False
        or selection.get("selected_state_policy") != "reset_each_input"
    ):
        raise ValueError("final QRC reference is not the validation-selected reset policy")
    return config


def verify_garch_public_data(
    config: GARCHStudyConfig,
) -> tuple[ModelDataset, dict[str, Any]]:
    """Verify raw snapshot and processed checksums, then reject fixture data."""

    data_config = load_data_config(config.data_config)
    snapshot = verify_public_snapshot(data_config)
    if snapshot.get("snapshot_id") != config.data_snapshot_id:
        raise ValueError("public snapshot ID disagrees with GARCH configuration")
    if data_config.snapshot_manifest_path is None:
        raise FileNotFoundError("public snapshot configuration has no manifest")
    dataset = load_model_dataset(data_config.processed_path)
    if dataset.is_synthetic or dataset.data_source_type != "public_market":
        raise SyntheticResultsError(
            "GARCH baseline requires verified non-synthetic public-market data"
        )
    if dataset.manifest.get("source_snapshot_id") != config.data_snapshot_id:
        raise ValueError("processed data manifest disagrees with GARCH snapshot ID")
    actual_manifest = dataset.processed_checksums["data_manifest.json"]
    if actual_manifest != config.processed_manifest_sha256:
        raise ValueError("frozen processed-data manifest checksum mismatch")
    return dataset, {
        "snapshot_id": config.data_snapshot_id,
        "snapshot_manifest_sha256": sha256_file(data_config.snapshot_manifest_path),
        "raw_file_checksums": {
            name: str(record["sha256"]) for name, record in sorted(snapshot["files"].items())
        },
        "processed_manifest_sha256": actual_manifest,
        "processed_checksums": dataset.processed_checksums,
        "split_row_counts": {
            "train": len(dataset.train.X),
            "validation": len(dataset.validation.X),
            "test": len(dataset.test.X),
        },
    }


def _date_checksum(dates: NDArray[np.datetime64]) -> str:
    payload = "\n".join(pd.to_datetime(dates).strftime("%Y-%m-%d").tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _indices_for_dates(
    source_dates: NDArray[np.datetime64],
    requested_dates: NDArray[np.datetime64],
) -> NDArray[np.int_]:
    lookup = {pd.Timestamp(value).normalize(): index for index, value in enumerate(source_dates)}
    indices: list[int] = []
    for value in requested_dates:
        key = pd.Timestamp(value).normalize()
        if key not in lookup:
            raise ValueError(f"GARCH return stream omits evaluation date {key.date()}")
        indices.append(lookup[key])
    result = np.asarray(indices, dtype=int)
    if len(result) > 1 and np.any(np.diff(result) <= 0):
        raise ValueError("GARCH evaluation dates are not strictly chronological")
    return result


def _reconstructed_targets(
    raw_dates: NDArray[np.datetime64],
    raw_returns: NDArray[np.float64],
    evaluation_dates: NDArray[np.datetime64],
    *,
    horizon: int,
    annualization: float,
) -> NDArray[np.float64]:
    lookup = {pd.Timestamp(value).normalize(): index for index, value in enumerate(raw_dates)}
    values: list[float] = []
    for date in evaluation_dates:
        key = pd.Timestamp(date).normalize()
        index = lookup.get(key)
        if index is None or index + horizon >= len(raw_returns):
            raise ValueError(f"cannot reconstruct forward target for {key.date()}")
        future = raw_returns[index + 1 : index + horizon + 1]
        if not np.isfinite(future).all():
            raise ValueError(f"future target return window is incomplete for {key.date()}")
        values.append(float((annualization / horizon) * np.sum(future**2)))
    return np.asarray(values, dtype=float)


def load_garch_return_inputs(
    config: GARCHStudyConfig,
    dataset: ModelDataset,
) -> GARCHReturnInputs:
    """Load the pipeline's raw SPY return stream and prove row/unit alignment."""

    source_files = dataset.manifest.get("source_files")
    if not isinstance(source_files, dict) or not isinstance(source_files.get("spy"), dict):
        raise ValueError("processed data manifest omits the SPY source record")
    spy_setting = str(source_files["spy"]["path"])
    spy_path = (config.project_root / spy_setting).resolve()
    if sha256_file(spy_path) != str(source_files["spy"]["sha256"]):
        raise ValueError("SPY source checksum disagrees with processed manifest")
    raw = pd.read_csv(spy_path, parse_dates=["date"])
    if (
        raw["date"].duplicated().any()
        or not raw["date"].is_monotonic_increasing
        or (raw["close"] <= 0).any()
    ):
        raise ValueError("raw SPY dates/prices violate the frozen return contract")
    raw["return"] = np.log(raw["close"] / raw["close"].shift(1))
    raw_dates = raw["date"].to_numpy(dtype="datetime64[ns]")
    raw_returns = raw["return"].to_numpy(dtype=float)
    split_boundaries = dataset.manifest.get("split_boundaries")
    if not isinstance(split_boundaries, dict):
        raise ValueError("processed data manifest omits split boundaries")
    train_boundary = split_boundaries.get("train")
    if not isinstance(train_boundary, dict):
        raise ValueError("processed data manifest omits training boundaries")
    train_start = pd.Timestamp(str(train_boundary["start"]))
    train_end = pd.Timestamp(str(train_boundary["end"]))
    training_mask = raw["date"].between(train_start, train_end, inclusive="both")
    training = raw.loc[training_mask & raw["return"].notna(), ["date", "return"]]
    if training.empty or training["date"].max() > train_end:
        raise ValueError("GARCH training return extraction violated the training boundary")
    maximum_origin = pd.Timestamp(dataset.test.dates.max())
    post = raw.loc[
        raw["date"].gt(train_end) & raw["date"].le(maximum_origin) & raw["return"].notna(),
        ["date", "return"],
    ]
    if post.empty:
        raise ValueError("GARCH post-training return stream is empty")
    post_dates = post["date"].to_numpy(dtype="datetime64[ns]")
    post_returns = post["return"].to_numpy(dtype=float)
    validation_indices = _indices_for_dates(post_dates, dataset.validation.dates)
    test_indices = _indices_for_dates(post_dates, dataset.test.dates)

    unscaled_path = config.project_root / "data/processed/public_market/features_unscaled.csv"
    unscaled = pd.read_csv(unscaled_path, parse_dates=["date"]).set_index("date")
    evaluation_dates = np.concatenate((dataset.validation.dates, dataset.test.dates))
    processed_returns = unscaled.loc[
        pd.to_datetime(evaluation_dates), "spy_log_return_1d"
    ].to_numpy(dtype=float)
    raw_lookup = pd.Series(raw_returns, index=pd.to_datetime(raw_dates))
    aligned_raw_returns = raw_lookup.loc[pd.to_datetime(evaluation_dates)].to_numpy(dtype=float)
    return_difference = float(np.max(np.abs(processed_returns - aligned_raw_returns), initial=0.0))
    if return_difference > 1e-10:
        raise ValueError("raw SPY returns disagree with pipeline spy_log_return_1d")

    reconstructed = _reconstructed_targets(
        raw_dates,
        raw_returns,
        evaluation_dates,
        horizon=config.horizon,
        annualization=config.annualization,
    )
    frozen_targets = np.concatenate((dataset.validation.y_rv, dataset.test.y_rv))
    target_difference = float(np.max(np.abs(reconstructed - frozen_targets), initial=0.0))
    if target_difference > 1e-10:
        raise ValueError("reconstructed GARCH target units disagree with frozen target_rv_5d")
    return GARCHReturnInputs(
        training_dates=training["date"].to_numpy(dtype="datetime64[ns]"),
        training_returns=training["return"].to_numpy(dtype=float),
        post_training_dates=post_dates,
        post_training_returns=post_returns,
        validation_indices=validation_indices,
        test_indices=test_indices,
        raw_return_checksum=array_checksum(np.asarray(raw_returns[1:], dtype=float)),
        training_return_checksum=array_checksum(training["return"].to_numpy(dtype=float)),
        post_training_return_checksum=array_checksum(post_returns),
        evaluation_date_checksum=_date_checksum(evaluation_dates),
        processed_return_maximum_absolute_difference=return_difference,
        reconstructed_target_maximum_absolute_difference=target_difference,
        raw_spy_path=spy_path,
    )


def deterministic_regime_metrics(
    true_regime: NDArray[np.int_],
    predicted_variance: NDArray[np.float64],
    *,
    low_medium: float,
    medium_high: float,
) -> tuple[dict[str, Any], NDArray[np.int_]]:
    """Map deterministic variance forecasts through frozen training thresholds."""

    truth = np.asarray(true_regime, dtype=int).reshape(-1)
    predictions = np.asarray(predicted_variance, dtype=float).reshape(-1)
    if len(truth) != len(predictions) or not np.isfinite(predictions).all():
        raise ValueError("GARCH regime evaluation arrays must align and be finite")
    if not 0 < low_medium < medium_high:
        raise ValueError("GARCH regime thresholds must be positive and ordered")
    predicted = np.select(
        [predictions <= low_medium, predictions <= medium_high],
        [0, 1],
        default=2,
    ).astype(int)
    metrics = {
        "accuracy": float(np.mean(predicted == truth)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
        "regime_thresholds": {
            "low_medium": low_medium,
            "medium_high": medium_high,
            "fit_split": "train",
        },
        "transition_pr_auc": None,
        "transition_pr_auc_applicable": False,
        "transition_score_definition": None,
        "transition_metric_reason": (
            "deterministic thresholded GARCH forecasts do not define calibrated "
            "class or transition probabilities"
        ),
    }
    return metrics, np.asarray(predicted, dtype=int)


def _prediction_correlation(
    truth: NDArray[np.float64],
    predictions: NDArray[np.float64],
) -> float | None:
    if len(truth) < 2 or float(np.std(truth)) <= 1e-15 or float(np.std(predictions)) <= 1e-15:
        return None
    value = float(np.corrcoef(truth, predictions)[0, 1])
    return value if np.isfinite(value) else None


def _evaluate_split(
    split_name: str,
    split: ModelSplit,
    indices: NDArray[np.int_],
    inputs: GARCHReturnInputs,
    forecast: Any,
    *,
    variance_floor: float,
    low_medium: float,
    medium_high: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    raw_predictions = np.asarray(forecast.target_unit_forecast[indices], dtype=float)
    regression = regression_metrics(split.y_rv, raw_predictions, variance_floor)
    metrics = dict(regression.metrics)
    metrics["prediction_correlation"] = _prediction_correlation(split.y_rv, regression.predictions)
    regime_metrics, predicted_regime = deterministic_regime_metrics(
        split.y_regime,
        regression.predictions,
        low_medium=low_medium,
        medium_high=medium_high,
    )
    metrics["regime_classification"] = regime_metrics
    metrics["macro_f1"] = regime_metrics["macro_f1"]
    metrics["balanced_accuracy"] = regime_metrics["balanced_accuracy"]
    metrics["confusion_matrix"] = regime_metrics["confusion_matrix"]
    metrics["transition_pr_auc"] = None
    metrics["transition_pr_auc_applicable"] = False
    metrics["target_units"] = "annualized five-day average realized variance"
    metrics["forecast_conversion"] = "(252 / 5) * five_day_cumulative_variance"
    metrics["split"] = split_name
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(split.dates).strftime("%Y-%m-%d"),
            "observed_return_at_origin": inputs.post_training_returns[indices],
            "filtered_variance_at_origin": forecast.filtered_variance_at_origin[indices],
            "one_day_conditional_variance": forecast.one_day_variance[indices],
            "annualized_one_day_variance": 252.0 * forecast.one_day_variance[indices],
            "five_day_cumulative_variance": forecast.five_day_cumulative_variance[indices],
            "raw_predicted_rv_5d": raw_predictions,
            "predicted_rv_5d": regression.predictions,
            "prediction_was_floored": regression.floored.astype(int),
            "true_rv_5d": split.y_rv,
            "current_regime": split.current_regime,
            "true_regime": split.y_regime,
            "predicted_regime": predicted_regime,
        }
    )
    if not np.array_equal(
        pd.to_datetime(predictions["date"]).to_numpy(dtype="datetime64[ns]"),
        split.dates,
    ):
        raise ValueError(f"GARCH {split_name} prediction dates are misaligned")
    return metrics, predictions


def _experiment_directory(config: GARCHStudyConfig, mode: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = config.output_root / "runs" / f"{timestamp}_{MODEL_TYPE}_{mode}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "figures").mkdir()
    (directory / "logs").mkdir()
    return directory


def _required_run_paths(experiment_dir: Path) -> tuple[Path, ...]:
    return (
        experiment_dir / "fitted_parameters.json",
        experiment_dir / "validation_predictions.csv",
        experiment_dir / "test_predictions.csv",
        experiment_dir / "validation_metrics.json",
        experiment_dir / "test_metrics.json",
        experiment_dir / "conditional_variance_path.csv",
        experiment_dir / "manifest.json",
    )


def discover_completed_garch_runs(
    runs_root: Path,
    *,
    study_id: str,
    snapshot_id: str,
) -> dict[str, Path]:
    """Discover latest complete smoke/full runs for graceful resumption."""

    completed: dict[str, Path] = {}
    if not runs_root.is_dir():
        return completed
    for manifest_path in sorted(runs_root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        experiment_dir = manifest_path.parent
        if manifest.get("study_id") != study_id or manifest.get("status") != "success":
            continue
        if not all(path.is_file() for path in _required_run_paths(experiment_dir)):
            continue
        if manifest.get("data_source_type") != "public_market" or manifest.get("is_synthetic"):
            raise SyntheticResultsError("completed GARCH run contains synthetic data")
        if manifest.get("data_snapshot_id") != snapshot_id:
            raise ValueError("completed GARCH run uses a different public snapshot")
        if manifest.get("training_only_parameter_fit") is not True:
            raise ValueError("completed GARCH run lacks training-only fit evidence")
        if manifest.get("future_return_lookahead") is not False:
            raise ValueError("completed GARCH run does not reject future-return lookahead")
        mode = str(manifest.get("mode"))
        if mode not in {"smoke", "full"}:
            raise ValueError("completed GARCH run has an unsupported mode")
        previous = completed.get(mode)
        if previous is None or experiment_dir.name > previous.name:
            completed[mode] = experiment_dir
    return completed


def pending_garch_modes(
    requested_mode: str,
    completed: dict[str, Path],
) -> tuple[str, ...]:
    """Return an empty tuple only when the requested mode is complete."""

    if requested_mode not in {"smoke", "full"}:
        raise ValueError("GARCH mode must be smoke or full")
    return () if requested_mode in completed else (requested_mode,)


def _write_fit_artifact(
    path: Path,
    fit: GARCHFitResult,
    *,
    mode: str,
    fit_dates: NDArray[np.datetime64],
    fit_return_checksum: str,
) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "model": MODEL_TYPE,
            "mode": mode,
            **fit.as_dict(),
            "fit_split": "train",
            "fit_date_start": pd.Timestamp(fit_dates.min()).date().isoformat(),
            "fit_date_end": pd.Timestamp(fit_dates.max()).date().isoformat(),
            "fit_return_checksum": fit_return_checksum,
            "parameters_frozen_after_fit": True,
            "gaussian_qml": True,
            "parameter_constraints": (
                "omega > 0; alpha >= 0; beta >= 0; alpha + beta < 1 "
                "by transformed unconstrained parameters"
            ),
        },
    )


def _execute_garch_run(
    config: GARCHStudyConfig,
    dataset: ModelDataset,
    inputs: GARCHReturnInputs,
    *,
    smoke: bool,
    data_provenance: dict[str, Any],
) -> Path:
    mode = "smoke" if smoke else "full"
    experiment_dir = _experiment_directory(config, mode)
    fit_returns = inputs.training_returns
    fit_dates = inputs.training_dates
    maximum_starts = config.maximum_starts
    if smoke:
        fit_returns = fit_returns[-config.smoke_training_return_count :]
        fit_dates = fit_dates[-config.smoke_training_return_count :]
        maximum_starts = config.smoke_maximum_starts
    model = GaussianGARCH11(
        horizon=config.horizon,
        annualization=config.annualization,
        stationarity_margin=config.stationarity_margin,
        variance_floor=config.parameter_variance_floor,
        maximum_iterations=config.maximum_iterations,
        tolerance=config.optimiser_tolerance,
    )
    started_fit = time.perf_counter()
    fit = model.fit(fit_returns, maximum_starts=maximum_starts)
    fitting_seconds = time.perf_counter() - started_fit
    started_forecast = time.perf_counter()
    forecast = model.forecast_sequence(
        inputs.post_training_returns,
        initial_variance=fit.next_conditional_variance,
    )
    forecasting_seconds = time.perf_counter() - started_forecast
    threshold_path = config.project_root / "data/processed/public_market/regime_thresholds.json"
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    validation_metrics, validation_predictions = _evaluate_split(
        "validation",
        dataset.validation,
        inputs.validation_indices,
        inputs,
        forecast,
        variance_floor=config.evaluation_variance_floor,
        low_medium=float(thresholds["low_medium"]),
        medium_high=float(thresholds["medium_high"]),
    )
    test_metrics, test_predictions = _evaluate_split(
        "test",
        dataset.test,
        inputs.test_indices,
        inputs,
        forecast,
        variance_floor=config.evaluation_variance_floor,
        low_medium=float(thresholds["low_medium"]),
        medium_high=float(thresholds["medium_high"]),
    )
    validation_predictions.to_csv(experiment_dir / "validation_predictions.csv", index=False)
    test_predictions.to_csv(experiment_dir / "test_predictions.csv", index=False)
    _write_json(experiment_dir / "validation_metrics.json", validation_metrics)
    _write_json(experiment_dir / "test_metrics.json", test_metrics)
    _write_fit_artifact(
        experiment_dir / "fitted_parameters.json",
        fit,
        mode=mode,
        fit_dates=fit_dates,
        fit_return_checksum=array_checksum(fit_returns),
    )
    evaluation_split = np.full(len(inputs.post_training_dates), "none", dtype=object)
    evaluation_split[inputs.validation_indices] = "validation"
    evaluation_split[inputs.test_indices] = "test"
    conditional_path = pd.DataFrame(
        {
            "date": pd.to_datetime(inputs.post_training_dates).strftime("%Y-%m-%d"),
            "evaluation_split": evaluation_split,
            "observed_return_at_origin": inputs.post_training_returns,
            "filtered_variance_at_origin": forecast.filtered_variance_at_origin,
            "one_day_conditional_variance": forecast.one_day_variance,
            "annualized_one_day_variance": 252.0 * forecast.one_day_variance,
            "five_day_cumulative_variance": forecast.five_day_cumulative_variance,
            "predicted_rv_5d": forecast.target_unit_forecast,
        }
    )
    conditional_path.to_csv(experiment_dir / "conditional_variance_path.csv", index=False)
    timing = {
        "parameter_fitting_seconds": fitting_seconds,
        "recursive_forecasting_seconds": forecasting_seconds,
        "total_model_seconds": fitting_seconds + forecasting_seconds,
    }
    _write_json(experiment_dir / "timing.json", timing)
    manifest = {
        "schema_version": 1,
        "study_id": config.study_id,
        "experiment_id": experiment_dir.name,
        "status": "success",
        "mode": mode,
        "model_type": MODEL_TYPE,
        "task": TASK,
        "seed": 2026,
        "git": _git_metadata(config.project_root),
        **runtime_metadata(),
        "configuration": {
            "path": config.source.relative_to(config.project_root).as_posix(),
            "sha256": sha256_file(config.source),
        },
        "data_source_type": dataset.data_source_type,
        "is_synthetic": dataset.is_synthetic,
        "data_snapshot_id": config.data_snapshot_id,
        "data_manifest_checksum": dataset.processed_checksums["data_manifest.json"],
        "processed_data_checksums": dataset.processed_checksums,
        "data_provenance": data_provenance,
        "raw_spy_path": inputs.raw_spy_path.relative_to(config.project_root).as_posix(),
        "raw_return_checksum": inputs.raw_return_checksum,
        "training_return_checksum": inputs.training_return_checksum,
        "post_training_return_checksum": inputs.post_training_return_checksum,
        "evaluation_date_checksum": inputs.evaluation_date_checksum,
        "training_return_count": len(fit_returns),
        "training_return_date_range": {
            "start": pd.Timestamp(fit_dates.min()).date().isoformat(),
            "end": pd.Timestamp(fit_dates.max()).date().isoformat(),
        },
        "validation_prediction_count": len(validation_predictions),
        "test_prediction_count": len(test_predictions),
        "processed_return_maximum_absolute_difference": (
            inputs.processed_return_maximum_absolute_difference
        ),
        "reconstructed_target_maximum_absolute_difference": (
            inputs.reconstructed_target_maximum_absolute_difference
        ),
        "training_only_parameter_fit": True,
        "parameters_frozen_after_training": True,
        "validation_or_test_parameter_refit": False,
        "future_return_lookahead": False,
        "forecast_origin_information": (
            "fixed parameters, current filtered variance, and returns observed "
            "at or before each origin"
        ),
        "post_training_filter_consumes_targets": False,
        "test_data_roles": ["forecast evaluation only"],
        "target_definition": {
            "name": "target_rv_5d",
            "horizon": config.horizon,
            "source_returns": "SPY close-to-close log returns",
            "formula": "(252 / 5) * sum(r_(t+1)^2,...,r_(t+5)^2)",
            "forecast_formula": "(252 / 5) * sum(h_(t+1),...,h_(t+5))",
            "units": "annualized five-day average realized variance",
        },
        "evaluation_variance_floor": config.evaluation_variance_floor,
        "validation_floored_prediction_count": validation_metrics["floored_prediction_count"],
        "test_floored_prediction_count": test_metrics["floored_prediction_count"],
        "fit": fit.as_dict(),
        "timing": timing,
    }
    _write_json(experiment_dir / "manifest.json", manifest)
    return experiment_dir


def _latest_manifest_by_model(
    root: Path,
    *,
    allowed_models: set[str],
) -> dict[tuple[str, int], tuple[Path, dict[str, Any]]]:
    latest: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_type = str(manifest.get("model_type"))
        if (
            model_type not in allowed_models
            or manifest.get("task") != TASK
            or manifest.get("status") != "success"
            or manifest.get("data_source_type") != "public_market"
            or manifest.get("is_synthetic")
        ):
            continue
        seed = int(manifest.get("seed", manifest.get("reservoir_seed", 0)))
        identity = (model_type, seed)
        previous = latest.get(identity)
        if previous is None or str(manifest["experiment_id"]) > str(previous[1]["experiment_id"]):
            latest[identity] = (manifest_path.parent, manifest)
    return latest


def _latest_final_qrc_runs(
    config: GARCHStudyConfig,
) -> dict[int, tuple[Path, dict[str, Any]]]:
    summary = json.loads(config.qrc_selection_summary.read_text(encoding="utf-8"))
    selection = summary["validation_only_state_policy_selection"]
    policy = str(selection["selected_state_policy"])
    if policy != "reset_each_input":
        raise ValueError("GARCH comparison expected final QRC reset_each_input policy")
    latest: dict[int, tuple[Path, dict[str, Any]]] = {}
    runs_root = config.qrc_selection_summary.parent / "runs"
    for manifest_path in sorted(runs_root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        qrc = manifest.get("qrc_configuration")
        if (
            manifest.get("status") != "success"
            or manifest.get("task") != TASK
            or manifest.get("model_type") != "qrc_regressor"
            or manifest.get("data_source_type") != "public_market"
            or manifest.get("is_synthetic")
            or manifest.get("data_snapshot_id") != config.data_snapshot_id
            or manifest.get("backend") != "numpy_density_matrix_exact"
            or manifest.get("exact_noiseless") is not True
            or manifest.get("model_selection_data") != "validation only"
            or manifest.get("test_evaluated_after_readout_freeze") is not True
            or not isinstance(qrc, dict)
            or qrc.get("n_qubits") != 2
            or qrc.get("virtual_nodes") != 2
            or qrc.get("tau") != 1.0
            or qrc.get("state_policy") != policy
        ):
            continue
        seed = int(manifest["reservoir_seed"])
        previous = latest.get(seed)
        if previous is None or str(manifest["experiment_id"]) > str(previous[1]["experiment_id"]):
            latest[seed] = (manifest_path.parent, manifest)
    if sorted(latest) != [2026, 2027, 2028]:
        raise ValueError("final QRC comparison requires seeds 2026, 2027, and 2028")
    return latest


def _prediction_metric_row(
    experiment_dir: Path,
    manifest: dict[str, Any],
    *,
    model_label: str,
    split: str,
    expected_dates: NDArray[np.datetime64],
    expected_truth: NDArray[np.float64],
    variance_floor: float,
) -> dict[str, Any]:
    predictions = pd.read_csv(experiment_dir / f"{split}_predictions.csv", parse_dates=["date"])
    dates = predictions["date"].to_numpy(dtype="datetime64[ns]")
    if not np.array_equal(dates, expected_dates):
        raise ValueError(f"{model_label} {split} prediction dates disagree with GARCH")
    truth = predictions["true_rv_5d"].to_numpy(dtype=float)
    if not np.allclose(truth, expected_truth, rtol=0.0, atol=1e-12):
        raise ValueError(f"{model_label} {split} targets disagree with frozen rows")
    raw = predictions["predicted_rv_5d"].to_numpy(dtype=float)
    evaluated = regression_metrics(expected_truth, raw, variance_floor)
    return {
        "model": model_label,
        "model_type": manifest["model_type"],
        "split": split,
        "seed": int(manifest.get("seed", manifest.get("reservoir_seed", 2026))),
        "qlike": evaluated.metrics["qlike"],
        "rmse": evaluated.metrics["rmse"],
        "mae": evaluated.metrics["mae"],
        "prediction_correlation": _prediction_correlation(expected_truth, evaluated.predictions),
        "non_finite_prediction_count": evaluated.metrics["non_finite_prediction_count"],
        "floored_prediction_count": evaluated.metrics["floored_prediction_count"],
        "prediction_count": len(predictions),
        "evaluation_date_checksum": _date_checksum(expected_dates),
        "experiment_id": manifest["experiment_id"],
        "git_commit": manifest.get("git", {}).get("commit"),
        "data_snapshot_id": manifest.get("data_snapshot_id"),
    }


def build_garch_comparison(
    config: GARCHStudyConfig,
    experiment_dir: Path,
    dataset: ModelDataset,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute aligned persistence, ESN, GARCH, and final-QRC comparisons."""

    garch_manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    classical = _latest_manifest_by_model(
        config.public_baseline_results,
        allowed_models={
            "rv_persistence",
            "esn_regressor",
            "historical_variance",
            "linear_variance",
        },
    )
    for required in (("rv_persistence", 2026), ("esn_regressor", 2026)):
        if required not in classical:
            raise ValueError(f"GARCH comparison is missing public baseline {required[0]}")
    sources: list[tuple[Path, dict[str, Any], str]] = [
        (
            classical[("rv_persistence", 2026)][0],
            classical[("rv_persistence", 2026)][1],
            COMPARISON_MODELS["rv_persistence"],
        ),
        (
            classical[("esn_regressor", 2026)][0],
            classical[("esn_regressor", 2026)][1],
            COMPARISON_MODELS["esn_regressor"],
        ),
        (experiment_dir, garch_manifest, COMPARISON_MODELS[MODEL_TYPE]),
    ]
    for (model_type, _seed), (directory, manifest) in classical.items():
        if model_type in {"historical_variance", "linear_variance"}:
            sources.append((directory, manifest, model_type.replace("_", " ").title()))
    for _seed, (directory, manifest) in sorted(_latest_final_qrc_runs(config).items()):
        sources.append((directory, manifest, COMPARISON_MODELS["qrc_regressor"]))
    per_run: list[dict[str, Any]] = []
    for split_name, split in (
        ("validation", dataset.validation),
        ("test", dataset.test),
    ):
        for directory, manifest, label in sources:
            per_run.append(
                _prediction_metric_row(
                    directory,
                    manifest,
                    model_label=label,
                    split=split_name,
                    expected_dates=split.dates,
                    expected_truth=split.y_rv,
                    variance_floor=config.evaluation_variance_floor,
                )
            )
    aggregate: list[dict[str, Any]] = []
    for aggregate_split in ("validation", "test"):
        for model_label in dict.fromkeys(row["model"] for row in per_run):
            subset = [
                row
                for row in per_run
                if row["split"] == aggregate_split and row["model"] == model_label
            ]
            if not subset:
                continue
            row: dict[str, Any] = {
                "model": model_label,
                "split": aggregate_split,
                "seed_count": len(subset),
                "seeds": sorted(int(item["seed"]) for item in subset),
                "prediction_count": int(subset[0]["prediction_count"]),
                "evaluation_date_checksum": subset[0]["evaluation_date_checksum"],
                "data_snapshot_id": config.data_snapshot_id,
                "formal_significance_claim": False,
            }
            for metric in COMPARISON_METRICS:
                values = np.asarray(
                    [float(item[metric]) for item in subset if item[metric] is not None],
                    dtype=float,
                )
                row[metric] = float(values.mean()) if len(values) else None
                row[f"{metric}_standard_deviation"] = (
                    float(values.std(ddof=0)) if len(values) else None
                )
                row[f"{metric}_minimum"] = float(values.min()) if len(values) else None
                row[f"{metric}_maximum"] = float(values.max()) if len(values) else None
            row["non_finite_prediction_count"] = int(
                sum(int(item["non_finite_prediction_count"]) for item in subset)
            )
            row["floored_prediction_count"] = int(
                sum(int(item["floored_prediction_count"]) for item in subset)
            )
            aggregate.append(row)
    return per_run, aggregate


def _plot_time_series(
    experiment_dir: Path,
    destination: Path,
) -> None:
    predictions = pd.read_csv(experiment_dir / "test_predictions.csv", parse_dates=["date"])
    figure, axis = plt.subplots(figsize=(9.0, 4.0))
    axis.plot(
        predictions["date"],
        predictions["true_rv_5d"],
        label="realized variance",
        linewidth=1.0,
    )
    axis.plot(
        predictions["date"],
        predictions["predicted_rv_5d"],
        label="GARCH forecast",
        linewidth=1.1,
    )
    axis.set(xlabel="forecast origin", ylabel="annualized five-day realized variance")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=220)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def _plot_comparison_metric(
    comparison: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    destination: Path,
) -> None:
    rows = [row for row in comparison if row["split"] == "test"]
    labels = [str(row["model"]) for row in rows]
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    errors = np.asarray(
        [float(row[f"{metric}_standard_deviation"] or 0.0) for row in rows],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    positions = np.arange(len(rows))
    axis.bar(positions, values, yerr=errors, capsize=3, color="#4978a8")
    axis.set(xticks=positions, xticklabels=labels, ylabel=ylabel)
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=220)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def _plot_forecast_by_regime(experiment_dir: Path, destination: Path) -> None:
    predictions = pd.read_csv(experiment_dir / "test_predictions.csv")
    groups = [
        predictions.loc[predictions["true_regime"].eq(regime), "predicted_rv_5d"].to_numpy(
            dtype=float
        )
        for regime in (0, 1, 2)
    ]
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.boxplot(groups, tick_labels=["low", "medium", "high"], showfliers=False)
    axis.set(
        xlabel="frozen realized regime",
        ylabel="GARCH annualized five-day variance forecast",
    )
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=220)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def _plot_conditional_path(experiment_dir: Path, destination: Path) -> None:
    path = pd.read_csv(experiment_dir / "conditional_variance_path.csv", parse_dates=["date"])
    test = path.loc[path["evaluation_split"].eq("test")]
    figure, axis = plt.subplots(figsize=(9.0, 4.0))
    axis.plot(
        test["date"],
        test["annualized_one_day_variance"],
        label="annualized h(t+1)",
        linewidth=1.0,
    )
    axis.plot(
        test["date"],
        test["predicted_rv_5d"],
        label="five-day target-unit forecast",
        linewidth=1.0,
    )
    axis.set(xlabel="forecast origin", ylabel="annualized variance")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=220)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def write_garch_outputs(
    config: GARCHStudyConfig,
    experiment_dir: Path,
    dataset: ModelDataset,
) -> dict[str, Path]:
    """Write aligned aggregate comparisons and all five publication figures."""

    per_run, aggregate = build_garch_comparison(config, experiment_dir, dataset)
    comparison_csv = experiment_dir / "aggregate_comparison.csv"
    comparison_json = experiment_dir / "aggregate_comparison.json"
    comparison_runs_csv = experiment_dir / "comparison_per_run.csv"
    comparison_runs_json = experiment_dir / "comparison_per_run.json"
    pd.DataFrame(aggregate).to_csv(comparison_csv, index=False)
    _write_json(
        comparison_json,
        {
            "schema_version": 1,
            "formal_significance_claim": False,
            "rows": aggregate,
        },
    )
    pd.DataFrame(per_run).to_csv(comparison_runs_csv, index=False)
    _write_json(
        comparison_runs_json,
        {"schema_version": 1, "rows": per_run},
    )
    figures = experiment_dir / "figures"
    specifications: dict[str, Callable[[Path], None]] = {
        "test_realized_variance_and_garch_forecast": lambda path: _plot_time_series(
            experiment_dir, path
        ),
        "test_qlike_comparison": lambda path: _plot_comparison_metric(
            aggregate,
            metric="qlike",
            ylabel="test QLIKE (lower is better)",
            destination=path,
        ),
        "test_rmse_comparison": lambda path: _plot_comparison_metric(
            aggregate,
            metric="rmse",
            ylabel="test RMSE (lower is better)",
            destination=path,
        ),
        "garch_forecast_by_realized_regime": lambda path: _plot_forecast_by_regime(
            experiment_dir, path
        ),
        "conditional_variance_and_five_day_forecast": lambda path: _plot_conditional_path(
            experiment_dir, path
        ),
    }
    outputs = {
        "aggregate_comparison_csv": comparison_csv,
        "aggregate_comparison_json": comparison_json,
        "comparison_per_run_csv": comparison_runs_csv,
        "comparison_per_run_json": comparison_runs_json,
    }
    for name, writer in specifications.items():
        destination = figures / name
        writer(destination)
        outputs[f"figure_{name}_png"] = destination.with_suffix(".png")
        outputs[f"figure_{name}_pdf"] = destination.with_suffix(".pdf")
    return outputs


def run_garch_baseline(
    config_path: Path,
    *,
    smoke: bool = False,
    resume: bool = True,
) -> Path:
    """Run or resume the leakage-safe GARCH baseline and aligned comparison."""

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config = load_garch_study_config(config_path)
    dataset, data_provenance = verify_garch_public_data(config)
    inputs = load_garch_return_inputs(config, dataset)
    config.output_root.mkdir(parents=True, exist_ok=True)
    mode = "smoke" if smoke else "full"
    completed = discover_completed_garch_runs(
        config.output_root / "runs",
        study_id=config.study_id,
        snapshot_id=config.data_snapshot_id,
    )
    resumed = resume and mode in completed
    if resumed:
        experiment_dir = completed[mode]
    else:
        experiment_dir = _execute_garch_run(
            config,
            dataset,
            inputs,
            smoke=smoke,
            data_provenance=data_provenance,
        )
    outputs = write_garch_outputs(config, experiment_dir, dataset)
    summary_path = config.output_root / "garch_run_summary.json"
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = {
        "schema_version": 1,
        "study_id": config.study_id,
        "status": "success",
        "mode": mode,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        **runtime_metadata(),
        "git": _git_metadata(config.project_root),
        "configuration": {
            "path": config.source.relative_to(config.project_root).as_posix(),
            "sha256": sha256_file(config.source),
        },
        "data_provenance": data_provenance,
        "resume_enabled": resume,
        "resumed": resumed,
        "experiment_directory": experiment_dir.relative_to(config.project_root).as_posix(),
        "fit": manifest["fit"],
        "timing": manifest["timing"],
        "leakage_controls": {
            "training_only_parameter_fit": True,
            "parameters_frozen_after_training": True,
            "validation_or_test_parameter_refit": False,
            "future_return_lookahead": False,
            "post_training_filter_consumes_targets": False,
            "test_data_roles": ["forecast evaluation only"],
        },
        "alignment": {
            "evaluation_date_checksum": inputs.evaluation_date_checksum,
            "processed_return_maximum_absolute_difference": (
                inputs.processed_return_maximum_absolute_difference
            ),
            "reconstructed_target_maximum_absolute_difference": (
                inputs.reconstructed_target_maximum_absolute_difference
            ),
        },
        "outputs": {
            **{
                name: path.relative_to(config.project_root).as_posix()
                for name, path in outputs.items()
            },
            "fitted_parameters": (experiment_dir / "fitted_parameters.json")
            .relative_to(config.project_root)
            .as_posix(),
            "validation_predictions": (experiment_dir / "validation_predictions.csv")
            .relative_to(config.project_root)
            .as_posix(),
            "test_predictions": (experiment_dir / "test_predictions.csv")
            .relative_to(config.project_root)
            .as_posix(),
            "validation_metrics": (experiment_dir / "validation_metrics.json")
            .relative_to(config.project_root)
            .as_posix(),
            "test_metrics": (experiment_dir / "test_metrics.json")
            .relative_to(config.project_root)
            .as_posix(),
            "manifest": (experiment_dir / "manifest.json")
            .relative_to(config.project_root)
            .as_posix(),
        },
        "interpretation": (
            "classical Gaussian QML GARCH(1,1) benchmark; no formal significance, "
            "physical-QPU, or quantum-advantage claim"
        ),
    }
    _write_json(summary_path, summary)
    return summary_path
