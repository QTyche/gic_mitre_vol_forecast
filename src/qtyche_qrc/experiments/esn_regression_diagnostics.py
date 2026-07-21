"""Validation-only diagnosis and selection of ESN realized-variance heads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtyche_qrc.evaluation.metrics import regression_metrics
from qtyche_qrc.evaluation.plots import plot_regression_diagnostics, plot_rv_series
from qtyche_qrc.experiments.model_config import load_model_config
from qtyche_qrc.models.baselines.esn import ESNConfig, ESNRegressor, split_reservoir_states
from qtyche_qrc.models.dataset import SelectionDataset, load_model_dataset


def _esn_config(parameters: dict[str, Any], seed: int) -> ESNConfig:
    values = dict(parameters)
    values["seed"] = seed
    return ESNConfig(**values)


def _distribution(values: np.ndarray[Any, np.dtype[np.float64]]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q99": float(np.quantile(values, 0.99)),
    }


def select_esn_regression_head(
    data: SelectionDataset,
    base_parameters: dict[str, Any],
    ridge_alphas: list[float],
    transformations: list[str],
    seed: int,
    variance_floor: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Select target transform and ridge using validation data with no test access."""

    state_parameters = dict(base_parameters)
    state_parameters["target_transform"] = "direct_variance"
    reservoir_model = ESNRegressor(data.feature_names, _esn_config(state_parameters, seed))
    train_states, validation_states, _ = split_reservoir_states(
        reservoir_model.reservoir,
        data.train.X,
        data.validation.X,
        None,
        reservoir_model.config.state_policy,
    )
    washout = reservoir_model.config.washout
    design = np.column_stack((np.ones(len(train_states) - washout), train_states[washout:]))
    condition_number = float(np.linalg.cond(design))
    candidates: list[dict[str, Any]] = []
    for transform in transformations:
        for ridge_alpha in ridge_alphas:
            parameters = {
                **base_parameters,
                "target_transform": transform,
                "ridge_alpha": ridge_alpha,
            }
            model = ESNRegressor(data.feature_names, _esn_config(parameters, seed))
            model.fit_readout(train_states, data.train.y_rv)
            readout_predictions = model.predict_readout_from_states(validation_states)
            physical_predictions = model.predict_from_states(validation_states)
            evaluated = regression_metrics(
                data.validation.y_rv,
                physical_predictions,
                variance_floor,
            )
            if model.readout is None:
                raise RuntimeError("diagnostic ESN readout was not fitted")
            candidates.append(
                {
                    "target_transform": transform,
                    "ridge_alpha": ridge_alpha,
                    "validation_rmse": evaluated.metrics["rmse"],
                    "validation_mae": evaluated.metrics["mae"],
                    "validation_qlike": evaluated.metrics["qlike"],
                    "negative_prediction_count": int(np.sum(physical_predictions < 0)),
                    "floored_prediction_count": evaluated.metrics["floored_prediction_count"],
                    "raw_readout_distribution": _distribution(readout_predictions),
                    "physical_prediction_distribution": _distribution(physical_predictions),
                    "readout_coefficient_l2": float(np.linalg.norm(model.readout)),
                    "readout_coefficient_max_abs": float(np.max(np.abs(model.readout))),
                    "status": "success",
                }
            )
    selected_row = min(candidates, key=lambda row: float(row["validation_qlike"]))
    selected = {
        **base_parameters,
        "target_transform": selected_row["target_transform"],
        "ridge_alpha": selected_row["ridge_alpha"],
    }
    diagnostics = {
        "selection_dataset": "training labels and validation metrics only",
        "training_target_distribution": _distribution(data.train.y_rv),
        "validation_target_distribution": _distribution(data.validation.y_rv),
        "ridge_design_condition_number": condition_number,
        "reservoir_state_distribution": {
            "training": _distribution(train_states.reshape(-1)),
            "validation": _distribution(validation_states.reshape(-1)),
        },
        "reservoir_state_dimension": reservoir_model.config.reservoir_size,
        "measured_spectral_radius": reservoir_model.reservoir.measured_spectral_radius,
    }
    return selected, candidates, diagnostics


def run_esn_regression_diagnostics(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Persist head diagnostics, freeze the validation choice, then evaluate test once."""

    config = load_model_config(config_path)
    if config.model_type != "esn_regressor" or config.task != "rv_regression":
        raise ValueError("ESN regression diagnostics require an esn_regressor configuration")
    dataset = load_model_dataset(config.processed_dir)
    transforms = [
        str(value)
        for value in config.search_space.get(
            "target_transform", ["direct_variance", "log_variance"]
        )
    ]
    for required in ("direct_variance", "log_variance"):
        if required not in transforms:
            transforms.append(required)
    ridge_alphas = [
        float(value) for value in config.search_space.get("ridge_alpha", [1e-5, 1e-3, 1e-1])
    ]
    selected, candidates, diagnostics = select_esn_regression_head(
        dataset.for_selection(),
        config.parameters,
        ridge_alphas,
        transforms,
        config.seed,
        config.variance_floor,
    )

    # Test inputs and labels are first accessed after the validation choice is frozen.
    model = ESNRegressor(dataset.feature_names, _esn_config(selected, config.seed))
    train_states, validation_states, test_states = split_reservoir_states(
        model.reservoir,
        dataset.train.X,
        dataset.validation.X,
        dataset.test.X,
        model.config.state_policy,
    )
    if test_states is None:
        raise RuntimeError("diagnostic test states were not constructed")
    model.fit_readout(train_states, dataset.train.y_rv)
    validation_raw = model.predict_from_states(validation_states)
    test_raw = model.predict_from_states(test_states)
    validation_evaluation = regression_metrics(
        dataset.validation.y_rv, validation_raw, config.variance_floor
    )
    test_evaluation = regression_metrics(dataset.test.y_rv, test_raw, config.variance_floor)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates).to_csv(output_dir / "head_candidates.csv", index=False)
    validation_predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(dataset.validation.dates).strftime("%Y-%m-%d"),
            "true_rv_5d": dataset.validation.y_rv,
            "raw_predicted_rv_5d": validation_raw,
            "predicted_rv_5d": validation_evaluation.predictions,
            "prediction_was_floored": validation_evaluation.floored.astype(int),
        }
    )
    test_predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(dataset.test.dates).strftime("%Y-%m-%d"),
            "true_rv_5d": dataset.test.y_rv,
            "raw_predicted_rv_5d": test_raw,
            "predicted_rv_5d": test_evaluation.predictions,
            "prediction_was_floored": test_evaluation.floored.astype(int),
        }
    )
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    plot_rv_series(test_predictions, output_dir / "true_vs_predicted.png", dataset.is_synthetic)
    plot_regression_diagnostics(test_predictions, output_dir, dataset.is_synthetic)
    model.save(output_dir / "selected_model")

    direct = [row for row in candidates if row["target_transform"] == "direct_variance"]
    log = [row for row in candidates if row["target_transform"] == "log_variance"]
    summary = {
        "schema_version": 1,
        "data_source_type": dataset.data_source_type,
        "is_synthetic": dataset.is_synthetic,
        "data_snapshot_id": dataset.manifest.get("source_snapshot_id"),
        "selection_policy": "minimum validation QLIKE; test unavailable until configuration freeze",
        "selected_configuration": selected,
        "diagnostics": diagnostics,
        "direct_variance_best_validation": min(
            direct, key=lambda row: float(row["validation_qlike"])
        ),
        "log_variance_best_validation": min(log, key=lambda row: float(row["validation_qlike"])),
        "final_validation_metrics": validation_evaluation.metrics,
        "final_test_metrics": test_evaluation.metrics,
        "final_validation_negative_predictions": int(np.sum(validation_raw < 0)),
        "final_test_negative_predictions": int(np.sum(test_raw < 0)),
    }
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
