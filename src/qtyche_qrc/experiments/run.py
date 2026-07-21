"""Reproducible classical-baseline experiment execution and artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import socket
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from qtyche_qrc.config import ProjectConfig
from qtyche_qrc.evaluation.metrics import (
    classification_metrics,
    regression_metrics,
    transition_metrics,
)
from qtyche_qrc.evaluation.plots import (
    SYNTHETIC_WARNING,
    plot_confusion_matrix,
    plot_rv_series,
    plot_transition_calibration,
    plot_transition_series,
)
from qtyche_qrc.experiments.model_config import ModelExperimentConfig, load_model_config
from qtyche_qrc.experiments.sweep import (
    CandidateResult,
    candidate_rows,
    search_esn_classifier,
    search_esn_regressor,
    search_logistic,
)
from qtyche_qrc.models.base import ForecastModel
from qtyche_qrc.models.baselines.esn import ESNClassifier, ESNConfig, ESNRegressor
from qtyche_qrc.models.baselines.logistic import MultinomialLogisticClassifier
from qtyche_qrc.models.baselines.persistence import (
    CurrentRegimePersistenceClassifier,
    MajorityClassClassifier,
    RealizedVariancePersistenceRegressor,
)
from qtyche_qrc.models.dataset import ModelDataset, ModelSplit, load_model_dataset


class ExperimentRunner(ABC):
    """Legacy generic runner contract retained for future backend implementations."""

    @abstractmethod
    def run(self, config: ProjectConfig) -> list[Path]:
        """Execute the experiment and return all persisted output paths."""


class SyntheticResultsError(ValueError):
    """Raised when a headline-capable command is pointed at fixture data."""


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_metadata(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "scikit-learn", "matplotlib", "qtyche-qrc"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _experiment_directory(config: ModelExperimentConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    identifier = f"{timestamp}_{config.model_type}_{config.task}_seed{config.seed}"
    path = config.output_root / identifier
    path.mkdir(parents=True, exist_ok=False)
    for directory in ("model", "figures", "logs"):
        (path / directory).mkdir()
    return path


def _esn_configuration(parameters: dict[str, Any], seed: int) -> ESNConfig:
    values = dict(parameters)
    values["seed"] = seed
    return ESNConfig(**values)


def _classification_predictions(
    split: ModelSplit,
    probabilities: np.ndarray[Any, np.dtype[np.float64]],
    transition_threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    class_metrics = classification_metrics(split.y_regime, probabilities)
    transition_values, transition_probability, predicted_transition = transition_metrics(
        split.y_transition,
        probabilities,
        split.current_regime,
        transition_threshold,
    )
    metrics = {**class_metrics, **transition_values}
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(split.dates).strftime("%Y-%m-%d"),
            "current_regime": split.current_regime,
            "true_regime": split.y_regime,
            "predicted_regime": np.argmax(probabilities, axis=1),
            "probability_low": probabilities[:, 0],
            "probability_medium": probabilities[:, 1],
            "probability_high": probabilities[:, 2],
            "true_transition": split.y_transition,
            "predicted_transition_probability": transition_probability,
            "predicted_transition": predicted_transition,
        }
    )
    return metrics, predictions


def _classification_model_outputs(
    config: ModelExperimentConfig,
    data: ModelDataset,
    selected: dict[str, Any],
) -> tuple[
    ForecastModel,
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    dict[str, float],
]:
    started = time.perf_counter()
    model: ForecastModel
    if config.model_type == "majority_classifier":
        majority_model = MajorityClassClassifier(data.feature_names, config.seed)
        majority_model.fit(data.train.X, data.train.y_regime)
        model = majority_model
        training_time = time.perf_counter() - started
        prediction_started = time.perf_counter()
        validation = majority_model.predict_proba(data.validation.X)
        validation_time = time.perf_counter() - prediction_started
        prediction_started = time.perf_counter()
        test = majority_model.predict_proba(data.test.X)
        test_time = time.perf_counter() - prediction_started
    elif config.model_type == "regime_persistence":
        persistence_model = CurrentRegimePersistenceClassifier(config.seed)
        persistence_model.fit(data.train.current_regime[:, None].astype(float), data.train.y_regime)
        model = persistence_model
        training_time = time.perf_counter() - started
        prediction_started = time.perf_counter()
        validation = persistence_model.predict_proba(
            data.validation.current_regime[:, None].astype(float)
        )
        validation_time = time.perf_counter() - prediction_started
        prediction_started = time.perf_counter()
        test = persistence_model.predict_proba(data.test.current_regime[:, None].astype(float))
        test_time = time.perf_counter() - prediction_started
    elif config.model_type == "logistic_regression":
        logistic_model = MultinomialLogisticClassifier(
            data.feature_names,
            float(selected.get("regularization_c", 1.0)),
            selected.get("class_weight"),
            int(selected.get("max_iterations", 500)),
            config.seed,
        )
        logistic_model.fit(data.train.X, data.train.y_regime)
        model = logistic_model
        training_time = time.perf_counter() - started
        prediction_started = time.perf_counter()
        validation = logistic_model.predict_proba(data.validation.X)
        validation_time = time.perf_counter() - prediction_started
        prediction_started = time.perf_counter()
        test = logistic_model.predict_proba(data.test.X)
        test_time = time.perf_counter() - prediction_started
    elif config.model_type == "esn_classifier":
        esn_model = ESNClassifier(data.feature_names, _esn_configuration(selected, config.seed))
        train_states = esn_model.transform_sequence(data.train.X, reset=True)
        esn_model.fit_readout(train_states, data.train.y_regime)
        model = esn_model
        training_time = time.perf_counter() - started
        prediction_started = time.perf_counter()
        validation_states = esn_model.transform_sequence(
            data.validation.X, reset=esn_model.config.state_policy == "reset"
        )
        validation = esn_model.predict_proba_from_states(validation_states)
        validation_time = time.perf_counter() - prediction_started
        prediction_started = time.perf_counter()
        test_states = esn_model.transform_sequence(
            data.test.X, reset=esn_model.config.state_policy == "reset"
        )
        test = esn_model.predict_proba_from_states(test_states)
        test_time = time.perf_counter() - prediction_started
    else:
        raise ValueError(f"{config.model_type} is not a classification model")
    return (
        model,
        validation,
        test,
        {
            "training_seconds": training_time,
            "validation_prediction_seconds": validation_time,
            "test_prediction_seconds": test_time,
        },
    )


def _regression_model_outputs(
    config: ModelExperimentConfig,
    data: ModelDataset,
    selected: dict[str, Any],
) -> tuple[
    ForecastModel,
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    dict[str, float],
]:
    started = time.perf_counter()
    model: ForecastModel
    if config.model_type == "rv_persistence":
        persistence_model = RealizedVariancePersistenceRegressor(config.seed)
        persistence_model.fit(data.train.current_rv_unscaled[:, None], data.train.y_rv)
        model = persistence_model
        training_time = time.perf_counter() - started
        prediction_started = time.perf_counter()
        validation = persistence_model.predict(data.validation.current_rv_unscaled[:, None])
        validation_time = time.perf_counter() - prediction_started
        prediction_started = time.perf_counter()
        test = persistence_model.predict(data.test.current_rv_unscaled[:, None])
        test_time = time.perf_counter() - prediction_started
    elif config.model_type == "esn_regressor":
        esn_model = ESNRegressor(data.feature_names, _esn_configuration(selected, config.seed))
        train_states = esn_model.transform_sequence(data.train.X, reset=True)
        esn_model.fit_readout(train_states, data.train.y_rv)
        model = esn_model
        training_time = time.perf_counter() - started
        prediction_started = time.perf_counter()
        validation_states = esn_model.transform_sequence(
            data.validation.X, reset=esn_model.config.state_policy == "reset"
        )
        validation = esn_model.predict_from_states(validation_states)
        validation_time = time.perf_counter() - prediction_started
        prediction_started = time.perf_counter()
        test_states = esn_model.transform_sequence(
            data.test.X, reset=esn_model.config.state_policy == "reset"
        )
        test = esn_model.predict_from_states(test_states)
        test_time = time.perf_counter() - prediction_started
    else:
        raise ValueError(f"{config.model_type} is not a regression model")
    return (
        model,
        validation,
        test,
        {
            "training_seconds": training_time,
            "validation_prediction_seconds": validation_time,
            "test_prediction_seconds": test_time,
        },
    )


def _regression_predictions(
    split: ModelSplit, raw_predictions: np.ndarray[Any, np.dtype[np.float64]], epsilon: float
) -> tuple[dict[str, Any], pd.DataFrame]:
    evaluated = regression_metrics(split.y_rv, raw_predictions, epsilon)
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(split.dates).strftime("%Y-%m-%d"),
            "true_rv_5d": split.y_rv,
            "predicted_rv_5d": evaluated.predictions,
            "prediction_was_floored": evaluated.floored.astype(int),
        }
    )
    return evaluated.metrics, predictions


def _selection(
    config: ModelExperimentConfig, data: ModelDataset
) -> tuple[dict[str, Any], list[CandidateResult]]:
    if not config.search_enabled:
        return dict(config.parameters), []
    selection_data = data.for_selection()
    if config.model_type == "logistic_regression":
        return search_logistic(
            selection_data,
            config.parameters,
            config.search_space,
            config.maximum_trials,
            config.seed,
        )
    if config.model_type == "esn_classifier":
        return search_esn_classifier(
            selection_data,
            config.parameters,
            config.search_space,
            config.maximum_trials,
            config.seed,
        )
    if config.model_type == "esn_regressor":
        return search_esn_regressor(
            selection_data,
            config.parameters,
            config.search_space,
            config.maximum_trials,
            config.seed,
            config.variance_floor,
        )
    raise ValueError(f"search is not supported for {config.model_type}")


def run_baseline_experiment(
    config_path: Path,
    *,
    allow_synthetic_results: bool = False,
) -> Path:
    """Select on validation, freeze, evaluate test once, and persist all artifacts."""

    config = load_model_config(config_path)
    data = load_model_dataset(config.processed_dir)
    if data.is_synthetic and not allow_synthetic_results:
        raise SyntheticResultsError(
            "fixture data cannot produce headline results; pass --allow-synthetic-results "
            "only for smoke or integration experiments"
        )
    experiment_dir = _experiment_directory(config)
    shutil.copyfile(config.source, experiment_dir / "config.yaml")
    warning = SYNTHETIC_WARNING if data.is_synthetic else None
    status = "success"
    warnings_list = [warning] if warning else []
    try:
        selected, trials = _selection(config, data)
        if config.task == "regime_classification":
            model, validation_raw, test_raw, timing = _classification_model_outputs(
                config, data, selected
            )
            validation_metrics, validation_predictions = _classification_predictions(
                data.validation, validation_raw, config.transition_threshold
            )
            test_metrics, test_predictions = _classification_predictions(
                data.test, test_raw, config.transition_threshold
            )
            plot_confusion_matrix(
                test_metrics["confusion_matrix"],
                experiment_dir / "figures" / "confusion_matrix.png",
                data.is_synthetic,
            )
            plot_transition_series(
                test_predictions,
                experiment_dir / "figures" / "transition_probability.png",
                data.is_synthetic,
            )
            plot_transition_calibration(
                data.test.y_transition,
                test_predictions["predicted_transition_probability"].to_numpy(dtype=float),
                experiment_dir / "figures" / "transition_calibration.png",
                data.is_synthetic,
            )
        else:
            model, validation_raw, test_raw, timing = _regression_model_outputs(
                config, data, selected
            )
            validation_metrics, validation_predictions = _regression_predictions(
                data.validation, validation_raw, config.variance_floor
            )
            test_metrics, test_predictions = _regression_predictions(
                data.test, test_raw, config.variance_floor
            )
            plot_rv_series(
                test_predictions,
                experiment_dir / "figures" / "realized_variance.png",
                data.is_synthetic,
            )

        if not trials:
            validation_score = float(validation_metrics[config.selection_metric])
            trials = [
                CandidateResult(
                    1,
                    selected,
                    config.selection_metric,
                    validation_score,
                    "success",
                )
            ]
        trial_table = pd.DataFrame(candidate_rows(trials))
        trial_table.insert(0, "data_source_type", data.data_source_type)
        trial_table.insert(1, "is_synthetic", data.is_synthetic)
        trial_table.insert(2, "data_warning", warning)
        trial_table.to_csv(experiment_dir / "selection_results.csv", index=False)
        validation_predictions.to_csv(experiment_dir / "validation_predictions.csv", index=False)
        test_predictions.to_csv(experiment_dir / "test_predictions.csv", index=False)
        _write_json(
            experiment_dir / "validation_metrics.json",
            {
                "data_source_type": data.data_source_type,
                "is_synthetic": data.is_synthetic,
                "data_warning": warning,
                **validation_metrics,
            },
        )
        _write_json(
            experiment_dir / "test_metrics.json",
            {
                "data_source_type": data.data_source_type,
                "is_synthetic": data.is_synthetic,
                "data_warning": warning,
                **test_metrics,
            },
        )
        _write_json(experiment_dir / "timing.json", timing)
        model.save(experiment_dir / "model")
        model_metadata = model.get_model_metadata()
        _write_json(experiment_dir / "model_metadata.json", model_metadata)
    except Exception as exc:
        status = "failure"
        warnings_list.append(str(exc))
        (experiment_dir / "logs" / "failure.txt").write_text(str(exc) + "\n", encoding="utf-8")
        raise
    finally:
        manifest = {
            "schema_version": 1,
            "experiment_id": experiment_dir.name,
            "git": _git_metadata(config.project_root),
            "model_type": config.model_type,
            "task": config.task,
            "configuration_checksum": _sha256(experiment_dir / "config.yaml"),
            "processed_data_checksums": data.processed_checksums,
            "data_source_type": data.data_source_type,
            "is_synthetic": data.is_synthetic,
            "data_warning": warning,
            "split_row_counts": {
                "train": len(data.train.X),
                "validation": len(data.validation.X),
                "test": len(data.test.X),
            },
            "selected_features": list(data.feature_names),
            "target": "target_regime_5d"
            if config.task == "regime_classification"
            else "target_rv_5d",
            "seed": config.seed,
            "selected_hyperparameters": locals().get("selected"),
            "training_time": locals().get("timing", {}).get("training_seconds"),
            "validation_prediction_time": locals()
            .get("timing", {})
            .get("validation_prediction_seconds"),
            "test_prediction_time": locals().get("timing", {}).get("test_prediction_seconds"),
            "package_versions": _package_versions(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "model_selection_metric": config.selection_metric,
            "model_selection_trial_count": len(locals().get("trials", [])),
            "status": status,
            "warnings": warnings_list,
            "synthetic_override_used": allow_synthetic_results and data.is_synthetic,
        }
        _write_json(experiment_dir / "manifest.json", manifest)
    return experiment_dir


def evaluate_experiment(experiment_dir: Path) -> dict[str, Any]:
    """Recompute persisted validation and test metrics without model selection."""

    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((experiment_dir / "config.yaml").read_text(encoding="utf-8"))
    threshold = float(config.get("evaluation", {}).get("transition_threshold", 0.5))
    epsilon = float(config.get("evaluation", {}).get("variance_floor", 1e-12))
    summaries: dict[str, Any] = {}
    for split_name in ("validation", "test"):
        frame = pd.read_csv(experiment_dir / f"{split_name}_predictions.csv")
        if manifest["task"] == "regime_classification":
            probabilities = frame[
                ["probability_low", "probability_medium", "probability_high"]
            ].to_numpy(dtype=float)
            class_values = classification_metrics(
                frame["true_regime"].to_numpy(dtype=int), probabilities
            )
            transition_values, _, _ = transition_metrics(
                frame["true_transition"].to_numpy(dtype=int),
                probabilities,
                frame["current_regime"].to_numpy(dtype=int),
                threshold,
            )
            metrics = {**class_values, **transition_values}
        else:
            evaluated = regression_metrics(
                frame["true_rv_5d"].to_numpy(dtype=float),
                frame["predicted_rv_5d"].to_numpy(dtype=float),
                epsilon,
            )
            metrics = evaluated.metrics
            saved_metrics = json.loads(
                (experiment_dir / f"{split_name}_metrics.json").read_text(encoding="utf-8")
            )
            for key in (
                "non_finite_prediction_count",
                "floored_prediction_count",
                "prediction_floor",
                "floor_policy",
            ):
                metrics[key] = saved_metrics[key]
        summaries[split_name] = metrics
    return summaries


def inspect_experiment(experiment_dir: Path) -> dict[str, Any]:
    """Return concise provenance and final metrics for console inspection."""

    return {
        "manifest": json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8")),
        "validation_metrics": json.loads(
            (experiment_dir / "validation_metrics.json").read_text(encoding="utf-8")
        ),
        "test_metrics": json.loads(
            (experiment_dir / "test_metrics.json").read_text(encoding="utf-8")
        ),
    }
