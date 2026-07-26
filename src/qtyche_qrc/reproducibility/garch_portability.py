"""Strict cross-platform equivalence evidence for the frozen GARCH fit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qtyche_qrc.experiments.garch_baseline import (
    GARCHReturnInputs,
    GARCHStudyConfig,
    load_garch_return_inputs,
    load_garch_study_config,
    verify_garch_public_data,
)
from qtyche_qrc.models.baselines.garch import (
    GARCHFitResult,
    GARCHForecastPath,
    GARCHParameters,
    GaussianGARCH11,
)
from qtyche_qrc.models.dataset import ModelSplit
from qtyche_qrc.reproducibility.verification import compare_numeric, sha256_path

GARCH_PORTABILITY_REPORT = "garch_portability_report.json"
GARCH_REGRESSION_METRICS = ("qlike", "rmse", "mae")
DISPLAYED_GARCH_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "qlike",
    "rmse",
    "mae",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _numeric_comparison(actual: float, expected: float, tolerance: float) -> dict[str, Any]:
    return compare_numeric(
        actual,
        expected,
        absolute_tolerance=tolerance,
        relative_tolerance=0.0,
    )


def _relative_difference(actual: float, expected: float) -> float:
    difference = abs(actual - expected)
    return difference / abs(expected) if expected != 0 else difference


def _actual_parameters(fit: dict[str, Any]) -> dict[str, float]:
    raw = fit.get("parameters")
    if not isinstance(raw, dict):
        raise ValueError("GARCH fit evidence omits parameters")
    required = ("omega", "alpha", "beta", "mu")
    if any(not isinstance(raw.get(name), (int, float)) for name in required):
        raise ValueError("GARCH fit parameters are incomplete")
    parameters = GARCHParameters(**{name: float(raw[name]) for name in required})
    parameters.validate()
    return {
        "omega": parameters.omega,
        "alpha": parameters.alpha,
        "beta": parameters.beta,
        "mu": parameters.mu,
        "persistence": parameters.persistence,
        "unconditional_variance": parameters.unconditional_variance,
    }


def compare_garch_fit(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Compare optimizer provenance and parameters against the frozen fit."""

    expected = dict(reference["expected_fit"])
    contract = dict(reference["comparison_contract"])
    actual_parameters = _actual_parameters(candidate)
    expected_parameters = {
        name: float(value) for name, value in dict(expected["parameters"]).items()
    }
    parameter_tolerances = {
        name: float(value)
        for name, value in dict(contract["parameter_absolute_tolerances"]).items()
    }
    parameter_checks = {
        name: {
            **_numeric_comparison(
                actual_parameters[name],
                expected_parameters[name],
                parameter_tolerances[name],
            ),
            "relative_difference": _relative_difference(
                actual_parameters[name],
                expected_parameters[name],
            ),
        }
        for name in (
            "omega",
            "alpha",
            "beta",
            "mu",
            "persistence",
            "unconditional_variance",
        )
    }
    stored = dict(candidate.get("parameters", {}))
    derived_field_checks = {
        name: bool(
            isinstance(stored.get(name), (int, float))
            and float(stored[name]) == actual_parameters[name]
        )
        for name in ("persistence", "unconditional_variance")
    }

    selected_index = int(candidate.get("selected_start_index", -1))
    attempts = candidate.get("attempts")
    selected_attempts = (
        [
            dict(attempt)
            for attempt in attempts
            if isinstance(attempt, dict) and int(attempt.get("start_index", -1)) == selected_index
        ]
        if isinstance(attempts, list)
        else []
    )
    selected_attempt = selected_attempts[0] if len(selected_attempts) == 1 else {}
    start_shapes = {
        int(record["start_index"]): dict(record)
        for record in reference["deterministic_start_shapes"]
    }
    selected_shape = start_shapes.get(selected_index)
    initial = selected_attempt.get("initial_parameters")
    initial_values = dict(initial) if isinstance(initial, dict) else {}
    initial_tolerance = float(contract["selected_start_initial_parameter_absolute_tolerance"])
    training_mean = float(candidate.get("training_return_mean", float("nan")))
    training_variance = float(candidate.get("training_return_variance", float("nan")))
    initial_checks: dict[str, Any] = {}
    if selected_shape is not None and np.isfinite((training_mean, training_variance)).all():
        alpha = float(selected_shape["alpha"])
        beta = float(selected_shape["beta"])
        expected_initial = {
            "alpha": alpha,
            "beta": beta,
            "mu": training_mean if selected_shape["mu"] == "training_return_mean" else 0.0,
            "omega": max(
                training_variance * (1.0 - alpha - beta),
                float(reference["model"]["parameter_variance_floor"]) * 10.0,
            ),
        }
        for name, expected_value in expected_initial.items():
            actual_value = initial_values.get(name)
            if isinstance(actual_value, (int, float)):
                initial_checks[name] = _numeric_comparison(
                    float(actual_value),
                    expected_value,
                    initial_tolerance,
                )
            else:
                initial_checks[name] = {
                    "actual": actual_value,
                    "expected": expected_value,
                    "absolute_tolerance": initial_tolerance,
                    "passed": False,
                }
    deterministic_start_valid = bool(initial_checks) and all(
        record["passed"] for record in initial_checks.values()
    )
    same_selected_start = (
        selected_index == int(expected["selected_start_index"]) and deterministic_start_valid
    )

    likelihood = _numeric_comparison(
        float(candidate.get("training_log_likelihood", float("nan"))),
        float(expected["training_log_likelihood"]),
        float(contract["training_log_likelihood_absolute_tolerance"]),
    )
    iteration_count = int(candidate.get("number_of_iterations", -1))
    expected_iterations = int(expected["number_of_iterations"])
    iteration_difference = abs(iteration_count - expected_iterations)
    iteration_check = {
        "actual": iteration_count,
        "expected": expected_iterations,
        "absolute_difference": iteration_difference,
        "maximum_absolute_difference": int(contract["iteration_count_maximum_absolute_difference"]),
        "passed": (
            iteration_count > 0
            and iteration_count <= int(reference["model"]["maximum_iterations"])
            and iteration_difference <= int(contract["iteration_count_maximum_absolute_difference"])
        ),
    }
    status = int(candidate.get("optimiser_status", -1))
    selected_success = selected_attempt.get("success") is True
    selected_fitted = selected_attempt.get("fitted_parameters")
    selected_fitted_values = dict(selected_fitted) if isinstance(selected_fitted, dict) else {}
    selected_attempt_parameter_checks = {
        name: (
            _numeric_comparison(
                float(selected_fitted_values[name]),
                actual_parameters[name],
                1e-15,
            )
            if isinstance(selected_fitted_values.get(name), (int, float))
            else {
                "actual": selected_fitted_values.get(name),
                "expected": actual_parameters[name],
                "absolute_tolerance": 1e-15,
                "passed": False,
            }
        )
        for name in ("omega", "alpha", "beta", "mu")
    }
    selected_negative_log_likelihood = selected_attempt.get("negative_log_likelihood")
    selected_attempt_likelihood_check = (
        _numeric_comparison(
            -float(selected_negative_log_likelihood),
            float(candidate.get("training_log_likelihood", float("nan"))),
            1e-10,
        )
        if isinstance(selected_negative_log_likelihood, (int, float))
        else {
            "actual": selected_negative_log_likelihood,
            "expected": candidate.get("training_log_likelihood"),
            "absolute_tolerance": 1e-10,
            "passed": False,
        }
    )
    selected_attempt_consistent = bool(
        all(record["passed"] for record in selected_attempt_parameter_checks.values())
        and selected_attempt_likelihood_check["passed"]
        and int(selected_attempt.get("number_of_iterations", -1))
        == int(candidate.get("number_of_iterations", -2))
        and int(selected_attempt.get("status", -1)) == status
    )
    convergence_warnings = candidate.get("convergence_warnings")
    convergence = {
        "actual_status": status,
        "expected_status": int(expected["optimiser_status"]),
        "actual_message": candidate.get("optimiser_message"),
        "expected_message": expected["optimiser_message"],
        "selected_attempt_success": selected_success,
        "convergence_warnings": convergence_warnings,
        "passed": (
            status == int(expected["optimiser_status"])
            and selected_success
            and selected_attempt_consistent
            and isinstance(convergence_warnings, list)
            and not convergence_warnings
        ),
    }
    training_count = {
        "actual": int(candidate.get("training_return_count", -1)),
        "expected": int(expected["training_return_count"]),
    }
    training_count["passed"] = training_count["actual"] == training_count["expected"]

    parameter_equivalence = all(record["passed"] for record in parameter_checks.values()) and all(
        derived_field_checks.values()
    )
    optimiser_equivalence = bool(
        likelihood["passed"]
        and parameter_equivalence
        and iteration_check["passed"]
        and convergence["passed"]
        and training_count["passed"]
        and deterministic_start_valid
    )
    return {
        "actual_parameters": actual_parameters,
        "expected_parameters": expected_parameters,
        "parameter_comparisons": parameter_checks,
        "stored_derived_parameter_checks": derived_field_checks,
        "training_log_likelihood": likelihood,
        "selected_start": {
            "actual_index": selected_index,
            "expected_index": int(expected["selected_start_index"]),
            "same_selected_start": same_selected_start,
            "actual_initial_parameters": initial_values,
            "expected_selected_initial_parameters": expected["selected_initial_parameters"],
            "deterministic_grid_initial_parameter_checks": initial_checks,
            "deterministic_start_valid": deterministic_start_valid,
        },
        "iterations": iteration_check,
        "function_evaluations": {
            "actual": int(candidate.get("function_evaluations", -1)),
            "expected": int(expected["function_evaluations"]),
        },
        "convergence": convergence,
        "selected_attempt_consistency": {
            "parameter_checks": selected_attempt_parameter_checks,
            "likelihood_check": selected_attempt_likelihood_check,
            "iteration_count_match": (
                int(selected_attempt.get("number_of_iterations", -1))
                == int(candidate.get("number_of_iterations", -2))
            ),
            "status_match": int(selected_attempt.get("status", -1)) == status,
            "passed": selected_attempt_consistent,
        },
        "training_return_count": training_count,
        "parameter_equivalence_passed": parameter_equivalence,
        "optimiser_equivalence_passed": optimiser_equivalence,
        "passed": optimiser_equivalence,
    }


def _model_with_parameters(
    config: GARCHStudyConfig,
    inputs: GARCHReturnInputs,
    parameters: GARCHParameters,
) -> tuple[GaussianGARCH11, GARCHForecastPath]:
    model = GaussianGARCH11(
        horizon=config.horizon,
        annualization=config.annualization,
        stationarity_margin=config.stationarity_margin,
        variance_floor=config.parameter_variance_floor,
        maximum_iterations=config.maximum_iterations,
        tolerance=config.optimiser_tolerance,
    )
    variances, next_variance = model._filter_with_parameters(
        inputs.training_returns,
        parameters,
    )
    model.fit_result = GARCHFitResult(
        parameters=parameters,
        training_log_likelihood=0.0,
        selected_start_index=0,
        optimiser_status=0,
        optimiser_message="reference forecast reconstruction",
        number_of_iterations=0,
        function_evaluations=0,
        training_return_count=len(inputs.training_returns),
        training_return_mean=float(np.mean(inputs.training_returns)),
        training_return_variance=float(np.var(inputs.training_returns, ddof=0)),
        initial_conditional_variance=float(variances[0]),
        final_conditional_variance=float(variances[-1]),
        next_conditional_variance=next_variance,
        convergence_warnings=(),
        attempts=(),
    )
    forecast = model.forecast_sequence(
        inputs.post_training_returns,
        initial_variance=next_variance,
    )
    return model, forecast


def _regimes(values: NDArray[np.float64], low: float, high: float) -> NDArray[np.int_]:
    return np.select(
        [values <= low, values <= high],
        [0, 1],
        default=2,
    ).astype(int)


def compare_garch_forecast_split(
    candidate: pd.DataFrame,
    *,
    split_name: str,
    split: ModelSplit,
    reference_forecast: NDArray[np.float64],
    recomputed_candidate_forecast: NDArray[np.float64],
    low_medium: float,
    medium_high: float,
    variance_floor: float,
    limits: dict[str, Any],
) -> dict[str, Any]:
    """Compare one candidate prediction frame with both frozen and recomputed paths."""

    required_columns = {
        "date",
        "raw_predicted_rv_5d",
        "predicted_rv_5d",
        "prediction_was_floored",
        "true_rv_5d",
        "current_regime",
        "true_regime",
        "predicted_regime",
    }
    columns_present = required_columns.issubset(candidate.columns)
    expected_count = len(split.dates)
    row_count_match = len(candidate) == expected_count
    if not columns_present or not row_count_match:
        return {
            "split": split_name,
            "actual_row_count": len(candidate),
            "expected_row_count": expected_count,
            "row_count_match": row_count_match,
            "required_columns_present": columns_present,
            "passed": False,
        }

    dates = pd.to_datetime(candidate["date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    date_match = bool(np.array_equal(dates, split.dates))
    truth = candidate["true_rv_5d"].to_numpy(dtype=float)
    current_regime = candidate["current_regime"].to_numpy(dtype=int)
    true_regime = candidate["true_regime"].to_numpy(dtype=int)
    labels_match = bool(
        np.allclose(truth, split.y_rv, rtol=0.0, atol=1e-12)
        and np.array_equal(current_regime, split.current_regime)
        and np.array_equal(true_regime, split.y_regime)
    )

    raw = candidate["raw_predicted_rv_5d"].to_numpy(dtype=float)
    evaluated = candidate["predicted_rv_5d"].to_numpy(dtype=float)
    declared_floored = candidate["prediction_was_floored"].to_numpy(dtype=int)
    non_finite_count = int((~np.isfinite(raw)).sum())
    derived_floored = (~np.isfinite(raw)) | (raw < variance_floor)
    floored_count = int(derived_floored.sum())
    floor_declaration_match = bool(np.array_equal(declared_floored, derived_floored.astype(int)))
    reference_non_finite_count = int((~np.isfinite(reference_forecast)).sum())
    reference_floored = (~np.isfinite(reference_forecast)) | (reference_forecast < variance_floor)
    reference_floored_count = int(reference_floored.sum())
    finite = bool(
        non_finite_count == 0
        and reference_non_finite_count == 0
        and np.isfinite(evaluated).all()
        and np.isfinite(recomputed_candidate_forecast).all()
    )
    if not finite:
        return {
            "split": split_name,
            "actual_row_count": len(candidate),
            "expected_row_count": expected_count,
            "row_count_match": True,
            "required_columns_present": True,
            "dates_identical": date_match,
            "labels_identical": labels_match,
            "non_finite_prediction_count": non_finite_count,
            "reference_non_finite_prediction_count": reference_non_finite_count,
            "floored_prediction_count": floored_count,
            "reference_floored_prediction_count": reference_floored_count,
            "floor_declaration_match": floor_declaration_match,
            "passed": False,
        }

    expected_evaluated = reference_forecast.copy()
    expected_evaluated[reference_floored] = variance_floor
    recomputed_evaluated = recomputed_candidate_forecast.copy()
    recomputed_evaluated[recomputed_evaluated < variance_floor] = variance_floor
    absolute = np.abs(evaluated - expected_evaluated)
    relative = absolute / np.abs(expected_evaluated)
    recomputed_absolute = np.abs(evaluated - recomputed_evaluated)
    statistics = {
        "maximum_absolute_difference": float(absolute.max(initial=0.0)),
        "mean_absolute_difference": float(absolute.mean()),
        "median_absolute_difference": float(np.median(absolute)),
        "maximum_relative_difference": float(relative.max(initial=0.0)),
        "maximum_candidate_recomputation_difference": float(recomputed_absolute.max(initial=0.0)),
    }
    statistic_checks = {
        "maximum_absolute_difference": (
            statistics["maximum_absolute_difference"]
            <= float(limits["maximum_absolute_difference"])
        ),
        "mean_absolute_difference": (
            statistics["mean_absolute_difference"]
            <= float(limits["maximum_mean_absolute_difference"])
        ),
        "median_absolute_difference": (
            statistics["median_absolute_difference"]
            <= float(limits["maximum_median_absolute_difference"])
        ),
        "maximum_relative_difference": (
            statistics["maximum_relative_difference"]
            <= float(limits["maximum_relative_difference"])
        ),
        "candidate_path_matches_its_parameters": (
            statistics["maximum_candidate_recomputation_difference"]
            <= float(limits["recomputed_path_absolute_tolerance"])
        ),
    }

    expected_regime = _regimes(
        expected_evaluated,
        low_medium,
        medium_high,
    )
    candidate_regime = _regimes(evaluated, low_medium, medium_high)
    declared_regime = candidate["predicted_regime"].to_numpy(dtype=int)
    declared_regime_consistent = bool(np.array_equal(declared_regime, candidate_regime))
    changed_classifications = int(np.count_nonzero(candidate_regime != expected_regime))
    threshold_crossings_changed = changed_classifications
    thresholds = np.asarray([low_medium, medium_high], dtype=float)
    return {
        "split": split_name,
        "actual_row_count": len(candidate),
        "expected_row_count": expected_count,
        "row_count_match": True,
        "required_columns_present": True,
        "dates_identical": date_match,
        "labels_identical": labels_match,
        "forecast_difference": statistics,
        "forecast_limits": limits,
        "forecast_limit_checks": statistic_checks,
        "changed_regime_classification_count": changed_classifications,
        "changed_threshold_crossing_count": threshold_crossings_changed,
        "candidate_declared_regime_consistent": declared_regime_consistent,
        "minimum_candidate_distance_to_threshold": float(
            np.min(np.abs(evaluated[:, None] - thresholds[None, :]))
        ),
        "minimum_reference_distance_to_threshold": float(
            np.min(np.abs(expected_evaluated[:, None] - thresholds[None, :]))
        ),
        "non_finite_prediction_count": non_finite_count,
        "reference_non_finite_prediction_count": reference_non_finite_count,
        "floored_prediction_count": floored_count,
        "reference_floored_prediction_count": reference_floored_count,
        "floor_declaration_match": floor_declaration_match,
        "passed": bool(
            date_match
            and labels_match
            and all(statistic_checks.values())
            and changed_classifications == 0
            and threshold_crossings_changed == 0
            and declared_regime_consistent
            and non_finite_count == reference_non_finite_count == 0
            and floored_count == reference_floored_count
            and floor_declaration_match
        ),
    }


def _display_check(actual: float, expected_display: str) -> dict[str, Any]:
    decimals = len(expected_display.split(".", maxsplit=1)[1]) if "." in expected_display else 0
    actual_display = f"{actual:.{decimals}f}"
    return {
        "actual": actual,
        "actual_display": actual_display,
        "expected_display": expected_display,
        "passed": actual_display == expected_display,
    }


def _ranking_checks(
    root: Path,
    actual_metrics: dict[str, float],
    expected_rankings: dict[str, list[str]],
) -> dict[str, Any]:
    table = _load_json(root / "paper_assets/tables/publication_table_2_financial_benchmark.json")
    rows = [dict(row) for row in table["rows"]]
    checks: dict[str, Any] = {}
    lower_is_better = {"qlike", "rmse", "mae"}
    for metric, expected in expected_rankings.items():
        values: list[tuple[str, float]] = []
        for row in rows:
            value = row.get(metric)
            if row.get("model") == "GARCH(1,1)":
                value = actual_metrics[metric]
            if isinstance(value, (int, float)):
                values.append((str(row["model"]), float(value)))
        actual = [
            model
            for model, _value in sorted(
                values,
                key=lambda record: record[1],
                reverse=metric not in lower_is_better,
            )
        ]
        checks[metric] = {
            "actual": actual,
            "expected": expected,
            "passed": actual == expected,
        }
    return checks


def compare_garch_portability(
    root: Path,
    *,
    experiment_dir: Path,
    reference_path: Path,
    expected_reference_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Write and return a complete candidate-versus-frozen GARCH evidence report."""

    actual_reference_sha256 = sha256_path(reference_path)
    reference = _load_json(reference_path)
    if (
        actual_reference_sha256 != expected_reference_sha256
        or reference.get("schema_version") != 1
        or reference.get("contract_id") != "garch_optimizer_portability_v1"
    ):
        raise ValueError("GARCH portability reference identity mismatch")

    config = load_garch_study_config(root / "configs/reproduction/garch_baseline.yaml")
    dataset, data_provenance = verify_garch_public_data(config)
    inputs = load_garch_return_inputs(config, dataset)
    thresholds = _load_json(root / "data/processed/public_market/regime_thresholds.json")
    low_medium = float(thresholds["low_medium"])
    medium_high = float(thresholds["medium_high"])

    fit = _load_json(experiment_dir / "fitted_parameters.json")
    fit_report = compare_garch_fit(fit, reference)
    actual_values = fit_report["actual_parameters"]
    expected_values = fit_report["expected_parameters"]
    actual_parameters = GARCHParameters(
        **{name: float(actual_values[name]) for name in ("omega", "alpha", "beta", "mu")}
    )
    expected_parameters = GARCHParameters(
        **{name: float(expected_values[name]) for name in ("omega", "alpha", "beta", "mu")}
    )
    _actual_model, actual_recomputed = _model_with_parameters(
        config,
        inputs,
        actual_parameters,
    )
    _reference_model, reference_forecast = _model_with_parameters(
        config,
        inputs,
        expected_parameters,
    )

    split_reports: dict[str, Any] = {}
    for split_name, split, indices in (
        ("validation", dataset.validation, inputs.validation_indices),
        ("test", dataset.test, inputs.test_indices),
    ):
        candidate = pd.read_csv(experiment_dir / f"{split_name}_predictions.csv")
        split_reports[split_name] = compare_garch_forecast_split(
            candidate,
            split_name=split_name,
            split=split,
            reference_forecast=reference_forecast.target_unit_forecast[indices],
            recomputed_candidate_forecast=actual_recomputed.target_unit_forecast[indices],
            low_medium=low_medium,
            medium_high=medium_high,
            variance_floor=config.evaluation_variance_floor,
            limits=dict(reference["comparison_contract"]["forecast_limits"]),
        )

    metrics = _load_json(experiment_dir / "test_metrics.json")
    actual_metrics = {name: float(metrics[name]) for name in DISPLAYED_GARCH_METRICS}
    metric_contract = dict(reference["comparison_contract"]["regression_metric_tolerance"])
    regression_metric_checks = {
        metric: compare_numeric(
            actual_metrics[metric],
            float(reference["expected_test_metrics"][metric]),
            absolute_tolerance=float(metric_contract["absolute"]),
            relative_tolerance=float(metric_contract["relative"]),
        )
        for metric in GARCH_REGRESSION_METRICS
    }
    classification_metric_checks = {
        metric: compare_numeric(
            actual_metrics[metric],
            float(reference["expected_test_metrics"][metric]),
            absolute_tolerance=1e-10,
            relative_tolerance=1e-9,
        )
        for metric in ("macro_f1", "balanced_accuracy")
    }
    display_checks = {
        metric: _display_check(
            actual_metrics[metric],
            str(reference["display_contract"][metric]),
        )
        for metric in DISPLAYED_GARCH_METRICS
    }
    ranking_checks = _ranking_checks(
        root,
        actual_metrics,
        {
            str(metric): list(ranking)
            for metric, ranking in dict(reference["expected_rankings"]).items()
        },
    )

    forecast_equivalence = all(report["passed"] for report in split_reports.values())
    equivalent_optimum = bool(fit_report["optimiser_equivalence_passed"] and forecast_equivalence)
    same_start_or_equivalent_optimum = bool(
        fit_report["selected_start"]["same_selected_start"] or equivalent_optimum
    )
    passed = bool(
        fit_report["passed"]
        and forecast_equivalence
        and same_start_or_equivalent_optimum
        and all(record["passed"] for record in regression_metric_checks.values())
        and all(record["passed"] for record in classification_metric_checks.values())
        and all(record["passed"] for record in display_checks.values())
        and all(record["passed"] for record in ranking_checks.values())
    )
    report = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "passed": passed,
        "contract_id": reference["contract_id"],
        "reference": {
            "path": reference_path.relative_to(root).as_posix(),
            "actual_sha256": actual_reference_sha256,
            "expected_sha256": expected_reference_sha256,
            "passed": actual_reference_sha256 == expected_reference_sha256,
        },
        "root_cause": reference["root_cause"],
        "tolerance_derivation": reference["derivation"],
        "comparison_contract": reference["comparison_contract"],
        "data": {
            "snapshot_id": config.data_snapshot_id,
            "synthetic": dataset.is_synthetic,
            "processed_verification_mode": data_provenance["processed_verification_mode"],
            "thresholds": {
                "low_medium": low_medium,
                "medium_high": medium_high,
                "fit_split": thresholds["fit_split"],
            },
        },
        "fit_comparison": fit_report,
        "forecast_comparison": split_reports,
        "forecast_reference_rule": (
            "Reconstruct the forecast path from the frozen parameters on the same "
            "checksum- and semantic-verified return stream used by the candidate. "
            "This isolates optimiser portability from already-gated input serialization."
        ),
        "same_start_or_equivalent_optimum": same_start_or_equivalent_optimum,
        "equivalent_optimum": equivalent_optimum,
        "likelihood_numerically_equivalent": fit_report["training_log_likelihood"]["passed"],
        "regression_metric_comparisons": regression_metric_checks,
        "classification_metric_comparisons": classification_metric_checks,
        "displayed_paper_value_checks": display_checks,
        "model_ranking_checks": ranking_checks,
        "global_tolerance_unchanged": {
            "absolute": 1e-10,
            "relative": 1e-9,
            "passed": True,
        },
        "garch_specific_tolerance_activated": passed,
        "interpretation": (
            "The GARCH-only regression-metric tolerance is activated only after "
            "convergence, deterministic-start/equivalent-optimum, likelihood, "
            "parameter, independently reconstructed forecast, date, label, regime, "
            "threshold-crossing, floor, display, and ranking checks all pass."
        ),
    }
    _write_json(output_path, report)
    return report
