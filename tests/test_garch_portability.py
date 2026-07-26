from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from qtyche_qrc.models.baselines.garch import GARCHParameters
from qtyche_qrc.models.dataset import ModelSplit
from qtyche_qrc.reproducibility.garch_portability import (
    compare_garch_fit,
    compare_garch_forecast_split,
)
from qtyche_qrc.reproducibility.verification import compare_numeric


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _reference() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            (_root() / "configs/reproduction/garch_portability_reference.json").read_text(
                encoding="utf-8"
            )
        ),
    )


def _candidate_fit() -> dict[str, Any]:
    reference = _reference()
    expected = reference["expected_fit"]
    parameters = dict(expected["parameters"])
    return {
        "parameters": parameters,
        "training_log_likelihood": expected["training_log_likelihood"],
        "selected_start_index": expected["selected_start_index"],
        "optimiser_status": expected["optimiser_status"],
        "optimiser_message": expected["optimiser_message"],
        "number_of_iterations": expected["number_of_iterations"],
        "function_evaluations": expected["function_evaluations"],
        "training_return_count": expected["training_return_count"],
        "training_return_mean": expected["selected_initial_parameters"]["mu"],
        "training_return_variance": (
            expected["selected_initial_parameters"]["omega"]
            / (
                1.0
                - expected["selected_initial_parameters"]["alpha"]
                - expected["selected_initial_parameters"]["beta"]
            )
        ),
        "convergence_warnings": [],
        "attempts": [
            {
                "start_index": expected["selected_start_index"],
                "initial_parameters": expected["selected_initial_parameters"],
                "fitted_parameters": {
                    name: parameters[name] for name in ("omega", "alpha", "beta", "mu")
                },
                "negative_log_likelihood": -expected["training_log_likelihood"],
                "number_of_iterations": expected["number_of_iterations"],
                "success": True,
                "status": 0,
                "message": expected["optimiser_message"],
            }
        ],
    }


def _split_and_frame(
    predictions: np.ndarray[Any, np.dtype[np.float64]] | None = None,
) -> tuple[ModelSplit, pd.DataFrame, np.ndarray[Any, np.dtype[np.float64]]]:
    reference = np.asarray([0.0055, 0.0100, 0.0190, 0.0300], dtype=float)
    values = reference.copy() if predictions is None else predictions
    dates = pd.date_range("2024-01-02", periods=4, freq="D").to_numpy(dtype="datetime64[ns]")
    true_regime = np.asarray([0, 1, 1, 2], dtype=int)
    current_regime = np.asarray([0, 0, 1, 2], dtype=int)
    split = ModelSplit(
        X=np.zeros((4, 1), dtype=float),
        y_regime=true_regime,
        y_transition=(true_regime != current_regime).astype(int),
        y_rv=np.asarray([0.006, 0.011, 0.018, 0.031], dtype=float),
        dates=dates,
        current_regime=current_regime,
        current_rv_unscaled=np.asarray([0.005, 0.006, 0.017, 0.029], dtype=float),
    )
    regimes = np.select(
        [values <= 0.006, values <= 0.02],
        [0, 1],
        default=2,
    )
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "raw_predicted_rv_5d": values,
            "predicted_rv_5d": values,
            "prediction_was_floored": np.zeros(4, dtype=int),
            "true_rv_5d": split.y_rv,
            "current_regime": current_regime,
            "true_regime": true_regime,
            "predicted_regime": regimes,
        }
    )
    return split, frame, reference


def test_linux_x86_equivalent_optimizer_variation_is_accepted() -> None:
    reference = _reference()
    candidate = _candidate_fit()
    expected = reference["expected_fit"]["parameters"]
    varied = GARCHParameters(
        omega=float(expected["omega"]) + 1e-11,
        alpha=float(expected["alpha"]) + 1e-6,
        beta=float(expected["beta"]) - 5e-7,
        mu=float(expected["mu"]) + 1e-9,
    )
    candidate["parameters"] = {
        "omega": varied.omega,
        "alpha": varied.alpha,
        "beta": varied.beta,
        "mu": varied.mu,
        "persistence": varied.persistence,
        "unconditional_variance": varied.unconditional_variance,
    }
    candidate["training_log_likelihood"] += 5e-8
    candidate["selected_start_index"] = 4
    candidate["number_of_iterations"] = 47
    training_mean = float(candidate["training_return_mean"])
    training_variance = float(candidate["training_return_variance"])
    candidate["attempts"] = [
        {
            "start_index": 4,
            "initial_parameters": {
                "alpha": 0.2,
                "beta": 0.6,
                "mu": training_mean,
                "omega": training_variance * 0.2,
            },
            "fitted_parameters": {
                "omega": varied.omega,
                "alpha": varied.alpha,
                "beta": varied.beta,
                "mu": varied.mu,
            },
            "negative_log_likelihood": -candidate["training_log_likelihood"],
            "number_of_iterations": candidate["number_of_iterations"],
            "success": True,
            "status": 0,
            "message": candidate["optimiser_message"],
        }
    ]

    report = compare_garch_fit(candidate, reference)

    assert report["passed"] is True
    assert report["selected_start"]["same_selected_start"] is False
    assert report["training_log_likelihood"]["passed"] is True
    assert report["parameter_equivalence_passed"] is True


def test_parameter_drift_is_rejected() -> None:
    reference = _reference()
    candidate = _candidate_fit()
    parameters = dict(candidate["parameters"])
    varied = GARCHParameters(
        omega=float(parameters["omega"]),
        alpha=float(parameters["alpha"]) + 1e-3,
        beta=float(parameters["beta"]),
        mu=float(parameters["mu"]),
    )
    candidate["parameters"] = {
        "omega": varied.omega,
        "alpha": varied.alpha,
        "beta": varied.beta,
        "mu": varied.mu,
        "persistence": varied.persistence,
        "unconditional_variance": varied.unconditional_variance,
    }

    report = compare_garch_fit(candidate, reference)

    assert report["passed"] is False
    assert report["parameter_comparisons"]["alpha"]["passed"] is False


def test_small_forecast_path_variation_without_crossings_is_accepted() -> None:
    reference = _reference()
    split, frame, frozen = _split_and_frame()
    candidate = frozen + np.asarray([1e-8, -2e-8, 3e-8, -4e-8])
    frame["raw_predicted_rv_5d"] = candidate
    frame["predicted_rv_5d"] = candidate

    report = compare_garch_forecast_split(
        frame,
        split_name="test",
        split=split,
        reference_forecast=frozen,
        recomputed_candidate_forecast=candidate,
        low_medium=0.006,
        medium_high=0.02,
        variance_floor=1e-12,
        limits=reference["comparison_contract"]["forecast_limits"],
    )

    assert report["passed"] is True
    assert report["changed_regime_classification_count"] == 0
    assert report["changed_threshold_crossing_count"] == 0


def test_materially_different_forecast_path_is_rejected() -> None:
    reference = _reference()
    split, frame, frozen = _split_and_frame()
    candidate = frozen.copy()
    candidate[-1] += 1e-3
    frame["raw_predicted_rv_5d"] = candidate
    frame["predicted_rv_5d"] = candidate

    report = compare_garch_forecast_split(
        frame,
        split_name="test",
        split=split,
        reference_forecast=frozen,
        recomputed_candidate_forecast=candidate,
        low_medium=0.006,
        medium_high=0.02,
        variance_floor=1e-12,
        limits=reference["comparison_contract"]["forecast_limits"],
    )

    assert report["passed"] is False
    assert report["forecast_limit_checks"]["maximum_absolute_difference"] is False


def test_changed_classification_or_threshold_crossing_is_rejected() -> None:
    reference = _reference()
    frozen = np.asarray([0.0059999, 0.0100, 0.0190, 0.0300], dtype=float)
    candidate = frozen.copy()
    candidate[0] = 0.0060001
    split, frame, _unused = _split_and_frame(candidate)

    report = compare_garch_forecast_split(
        frame,
        split_name="test",
        split=split,
        reference_forecast=frozen,
        recomputed_candidate_forecast=candidate,
        low_medium=0.006,
        medium_high=0.02,
        variance_floor=1e-12,
        limits=reference["comparison_contract"]["forecast_limits"],
    )

    assert report["passed"] is False
    assert report["changed_regime_classification_count"] == 1
    assert report["changed_threshold_crossing_count"] == 1


def test_candidate_forecast_must_match_its_fitted_parameters() -> None:
    reference = _reference()
    split, frame, frozen = _split_and_frame()
    recomputed = frozen.copy()
    recomputed[2] += 1e-6

    report = compare_garch_forecast_split(
        frame,
        split_name="test",
        split=split,
        reference_forecast=frozen,
        recomputed_candidate_forecast=recomputed,
        low_medium=0.006,
        medium_high=0.02,
        variance_floor=1e-12,
        limits=reference["comparison_contract"]["forecast_limits"],
    )

    assert report["passed"] is False
    assert report["forecast_limit_checks"]["candidate_path_matches_its_parameters"] is False


def test_reported_qbraid_metrics_fit_only_the_garch_contract() -> None:
    reference = _reference()
    expected = reference["expected_test_metrics"]
    actual = {
        "qlike": -2.895255342277749,
        "rmse": 0.074042971867777,
        "mae": 0.023328686154586862,
    }
    contract = reference["comparison_contract"]["regression_metric_tolerance"]

    for metric, value in actual.items():
        global_check = compare_numeric(
            value,
            float(expected[metric]),
            absolute_tolerance=1e-10,
            relative_tolerance=1e-9,
        )
        garch_check = compare_numeric(
            value,
            float(expected[metric]),
            absolute_tolerance=float(contract["absolute"]),
            relative_tolerance=float(contract["relative"]),
        )
        assert global_check["passed"] is False
        assert garch_check["passed"] is True


def test_global_metric_tolerance_is_not_modified() -> None:
    reference = _reference()
    original = copy.deepcopy(reference)

    _ = compare_garch_fit(_candidate_fit(), reference)

    assert reference == original
    phase3 = (_root() / "configs/phase3_reproduction.yaml").read_text(encoding="utf-8")
    assert "absolute: 1.0e-10" in phase3
    assert "relative: 1.0e-9" in phase3
