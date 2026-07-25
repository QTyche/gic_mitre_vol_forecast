"""Frozen regime, tail, transition, and classification diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qtyche_qrc.diagnostics.calibration import (
    variance_bootstrap_intervals,
    variance_point_metrics,
)

REGIME_NAMES = {0: "low", 1: "medium", 2: "high"}


def fixed_regime_labels(
    realised_variance: NDArray[np.floating[Any]],
    *,
    low_medium: float,
    medium_high: float,
) -> NDArray[np.int64]:
    """Apply the frozen training-derived regime rule without fitting thresholds."""

    values = np.asarray(realised_variance, dtype=float).reshape(-1)
    if (
        not len(values)
        or not np.isfinite(values).all()
        or low_medium <= 0.0
        or medium_high <= low_medium
    ):
        raise ValueError("frozen regime thresholds or realised variances are invalid")
    return np.where(values <= low_medium, 0, np.where(values <= medium_high, 1, 2)).astype(np.int64)


def variance_regime_diagnostics(
    truth: NDArray[np.floating[Any]],
    forecasts: NDArray[np.floating[Any]],
    regime_labels: NDArray[np.integer[Any]],
    *,
    bootstrap_counts: NDArray[np.integer[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Calculate variance diagnostics separately for each immutable target regime."""

    observed = np.asarray(truth, dtype=float).reshape(-1)
    predicted = np.asarray(forecasts, dtype=float).reshape(-1)
    regimes = np.asarray(regime_labels, dtype=np.int64).reshape(-1)
    if (
        len(observed) != len(predicted)
        or len(observed) != len(regimes)
        or not set(np.unique(regimes)).issubset(REGIME_NAMES)
    ):
        raise ValueError("regime-conditioned variance inputs do not align")
    rows: list[dict[str, Any]] = []
    for label, name in REGIME_NAMES.items():
        mask = regimes == label
        count = int(mask.sum())
        if not count:
            rows.append(
                {
                    "regime_label": label,
                    "regime": name,
                    "sample_count": 0,
                    "mean_forecast": None,
                    "mean_realised_variance": None,
                    "bias": None,
                    "qlike": None,
                    "rmse": None,
                    "mae": None,
                    "correlation": None,
                }
            )
            continue
        metrics = variance_point_metrics(observed[mask], predicted[mask])
        row = {
            "regime_label": label,
            "regime": name,
            "sample_count": count,
            "mean_forecast": metrics["mean_forecast"],
            "mean_realised_variance": metrics["mean_realised_variance"],
            "bias": metrics["mean_error"],
            "qlike": metrics["qlike"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "correlation": metrics["correlation"],
        }
        if bootstrap_counts is not None:
            intervals = variance_bootstrap_intervals(
                observed,
                predicted,
                bootstrap_counts,
                mask=mask,
            )
            row.update(
                {
                    "bias_ci_lower": intervals["mean_error_ci_lower"],
                    "bias_ci_upper": intervals["mean_error_ci_upper"],
                    "qlike_ci_lower": intervals["qlike_ci_lower"],
                    "qlike_ci_upper": intervals["qlike_ci_upper"],
                    "rmse_ci_lower": intervals["rmse_ci_lower"],
                    "rmse_ci_upper": intervals["rmse_ci_upper"],
                    "mae_ci_lower": intervals["mae_ci_lower"],
                    "mae_ci_upper": intervals["mae_ci_upper"],
                    "bootstrap_valid_count": intervals["bootstrap_valid_count"],
                    "bootstrap_invalid_count": intervals["bootstrap_invalid_count"],
                }
            )
        rows.append(row)
    return rows


def variance_tail_diagnostics(
    truth: NDArray[np.floating[Any]],
    forecasts: NDArray[np.floating[Any]],
    *,
    thresholds: dict[str, float],
    bootstrap_counts: NDArray[np.integer[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Describe fixed training-quantile high-volatility tails."""

    observed = np.asarray(truth, dtype=float).reshape(-1)
    predicted = np.asarray(forecasts, dtype=float).reshape(-1)
    if len(observed) != len(predicted) or not thresholds:
        raise ValueError("tail diagnostic inputs do not align")
    rows: list[dict[str, Any]] = []
    for threshold_id, threshold in thresholds.items():
        if threshold <= 0.0:
            raise ValueError("tail thresholds must be positive")
        mask = observed >= threshold
        count = int(mask.sum())
        if not count:
            raise ValueError(f"tail threshold {threshold_id} selects no observations")
        metrics = variance_point_metrics(observed[mask], predicted[mask])
        relative_underprediction = (observed[mask] - predicted[mask]) / observed[mask]
        row = {
            "tail_threshold_id": threshold_id,
            "training_derived_threshold": float(threshold),
            "sample_count": count,
            "mean_error": metrics["mean_error"],
            "mean_relative_underprediction": float(relative_underprediction.mean()),
            "median_relative_underprediction": float(np.median(relative_underprediction)),
            "qlike": metrics["qlike"],
            "rmse": metrics["rmse"],
            "detection_rate": float(np.mean(predicted[mask] >= threshold)),
            "threshold_fit_split": "train",
            "threshold_optimised": False,
        }
        if bootstrap_counts is not None:
            intervals = variance_bootstrap_intervals(
                observed,
                predicted,
                bootstrap_counts,
                mask=mask,
            )
            row.update(
                {
                    "mean_error_ci_lower": intervals["mean_error_ci_lower"],
                    "mean_error_ci_upper": intervals["mean_error_ci_upper"],
                    "qlike_ci_lower": intervals["qlike_ci_lower"],
                    "qlike_ci_upper": intervals["qlike_ci_upper"],
                    "rmse_ci_lower": intervals["rmse_ci_lower"],
                    "rmse_ci_upper": intervals["rmse_ci_upper"],
                    "bootstrap_valid_count": intervals["bootstrap_valid_count"],
                    "bootstrap_invalid_count": intervals["bootstrap_invalid_count"],
                }
            )
        rows.append(row)
    return rows


def confusion_counts(
    truth: NDArray[np.integer[Any]],
    predictions: NDArray[np.integer[Any]],
    *,
    labels: tuple[int, ...],
    weights: NDArray[np.floating[Any]] | None = None,
) -> NDArray[np.float64]:
    """Return a fixed-label confusion matrix with optional observation weights."""

    observed = np.asarray(truth, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.int64).reshape(-1)
    sample_weights = (
        np.ones(len(observed), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1)
    )
    if (
        len(observed) != len(predicted)
        or len(observed) != len(sample_weights)
        or not np.isfinite(sample_weights).all()
        or np.any(sample_weights < 0.0)
        or not set(np.unique(observed)).issubset(labels)
        or not set(np.unique(predicted)).issubset(labels)
    ):
        raise ValueError("confusion inputs do not align with the fixed labels")
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    label_index = {label: index for index, label in enumerate(labels)}
    for true_value, predicted_value, sample_weight in zip(
        observed,
        predicted,
        sample_weights,
    ):
        matrix[label_index[int(true_value)], label_index[int(predicted_value)]] += sample_weight
    return matrix


def per_class_diagnostics(
    truth: NDArray[np.integer[Any]],
    predictions: NDArray[np.integer[Any]],
    *,
    labels: tuple[int, ...],
) -> list[dict[str, Any]]:
    matrix = confusion_counts(truth, predictions, labels=labels)
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        true_positive = float(matrix[index, index])
        support = float(matrix[index].sum())
        predicted_count = float(matrix[:, index].sum())
        precision = true_positive / predicted_count if predicted_count > 0.0 else None
        recall = true_positive / support if support > 0.0 else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0.0
            else None
        )
        rows.append(
            {
                "class_label": label,
                "class_name": REGIME_NAMES.get(label, str(label)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
                "predicted_count": int(predicted_count),
            }
        )
    return rows


def classification_diagnostics(
    truth: NDArray[np.integer[Any]],
    predictions: NDArray[np.integer[Any]],
    *,
    labels: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any]:
    """Return immutable per-class and distribution diagnostics."""

    observed = np.asarray(truth, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.int64).reshape(-1)
    matrix = confusion_counts(observed, predicted, labels=labels)
    return {
        "observation_count": len(observed),
        "per_class": per_class_diagnostics(observed, predicted, labels=labels),
        "confusion_matrix": matrix.astype(int).tolist(),
        "true_regime_distribution": {
            str(label): int(np.sum(observed == label)) for label in labels
        },
        "predicted_regime_distribution": {
            str(label): int(np.sum(predicted == label)) for label in labels
        },
    }


def transition_type_diagnostics(
    origins: NDArray[np.integer[Any]],
    destinations: NDArray[np.integer[Any]],
    predictions: NDArray[np.integer[Any]],
    supplied_transition_scores: NDArray[np.floating[Any]],
) -> list[dict[str, Any]]:
    """Describe each observed origin-to-destination change using supplied scores."""

    origin = np.asarray(origins, dtype=np.int64).reshape(-1)
    destination = np.asarray(destinations, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.int64).reshape(-1)
    scores = np.asarray(supplied_transition_scores, dtype=float).reshape(-1)
    if (
        len(origin) != len(destination)
        or len(origin) != len(predicted)
        or len(origin) != len(scores)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("transition diagnostic inputs do not align")
    observed_types = sorted(
        {(int(left), int(right)) for left, right in zip(origin, destination) if left != right}
    )
    rows: list[dict[str, Any]] = []
    for left, right in observed_types:
        mask = (origin == left) & (destination == right)
        count = int(mask.sum())
        selected_predictions = predicted[mask]
        selected_scores = scores[mask]
        correctly_predicted = int(np.sum(selected_predictions == right))
        retained = int(np.sum(selected_predictions == left))
        rows.append(
            {
                "origin_regime": left,
                "origin_regime_name": REGIME_NAMES[left],
                "destination_regime": right,
                "destination_regime_name": REGIME_NAMES[right],
                "transition_type": f"{REGIME_NAMES[left]}_to_{REGIME_NAMES[right]}",
                "event_count": count,
                "correctly_predicted_destination_count": correctly_predicted,
                "correctly_predicted_destination_rate": correctly_predicted / count,
                "origin_regime_retention_error_count": retained,
                "origin_regime_retention_error_rate": retained / count,
                "other_destination_error_count": count - correctly_predicted - retained,
                "other_destination_error_rate": (count - correctly_predicted - retained) / count,
                "mean_supplied_transition_score": float(selected_scores.mean()),
                "median_supplied_transition_score": float(np.median(selected_scores)),
                "transition_score_source": "frozen supplied prediction column",
                "derived_transition_score": False,
            }
        )
    return rows


def _macro_metrics(
    truth: NDArray[np.int64],
    predictions: NDArray[np.int64],
    *,
    labels: tuple[int, ...],
) -> tuple[float | None, float | None]:
    rows = per_class_diagnostics(truth, predictions, labels=labels)
    supported = [row for row in rows if int(row["support"]) > 0]
    f1_values = [float(row["f1"]) for row in supported if row["f1"] is not None]
    recalls = [float(row["recall"]) for row in supported if row["recall"] is not None]
    macro_f1 = float(np.mean(f1_values)) if f1_values else None
    balanced_accuracy = float(np.mean(recalls)) if recalls else None
    return macro_f1, balanced_accuracy


def transition_subset_diagnostics(
    truth: NDArray[np.integer[Any]],
    predictions: NDArray[np.integer[Any]],
    transition_truth: NDArray[np.integer[Any]],
    supplied_transition_scores: NDArray[np.floating[Any]],
    *,
    probabilities: NDArray[np.floating[Any]] | None,
) -> list[dict[str, Any]]:
    """Compare frozen classification behaviour on transition and non-transition dates."""

    observed = np.asarray(truth, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.int64).reshape(-1)
    transitions = np.asarray(transition_truth, dtype=np.int64).reshape(-1)
    scores = np.asarray(supplied_transition_scores, dtype=float).reshape(-1)
    probability_values = None if probabilities is None else np.asarray(probabilities, dtype=float)
    if (
        len(observed) != len(predicted)
        or len(observed) != len(transitions)
        or len(observed) != len(scores)
        or not set(np.unique(transitions)).issubset({0, 1})
        or (probability_values is not None and probability_values.shape != (len(observed), 3))
    ):
        raise ValueError("transition-subset diagnostic inputs do not align")
    rows: list[dict[str, Any]] = []
    for transition_value, subset_name in ((0, "non_transition"), (1, "transition")):
        mask = transitions == transition_value
        subset_truth = observed[mask]
        subset_predictions = predicted[mask]
        macro_f1, balanced_accuracy = _macro_metrics(
            subset_truth,
            subset_predictions,
            labels=(0, 1, 2),
        )
        confidence = (
            float(np.max(probability_values[mask], axis=1).mean())
            if probability_values is not None
            else None
        )
        rows.append(
            {
                "subset": subset_name,
                "transition_value": transition_value,
                "sample_count": int(mask.sum()),
                "macro_f1": macro_f1,
                "balanced_accuracy": balanced_accuracy,
                "destination_regime_accuracy": float(np.mean(subset_predictions == subset_truth)),
                "mean_confidence": confidence,
                "mean_transition_score": float(scores[mask].mean()),
                "classification_error_rate": float(np.mean(subset_predictions != subset_truth)),
            }
        )
    return rows


def assess_lead_time_identifiability(
    *,
    prediction_columns: set[str],
    target_horizon: int,
    target_definition: str,
) -> dict[str, Any]:
    """Refuse lead-time analysis unless origin, target, and transition dates are explicit."""

    required = {"forecast_origin_date", "target_date", "transition_date"}
    missing = sorted(required - prediction_columns)
    identifiable = not missing
    reason = (
        "Explicit forecast-origin, target, and transition dates are available."
        if identifiable
        else (
            "Lead time is not identifiable: frozen prediction rows contain only the row date "
            f"and a {target_horizon}-trading-day aggregate {target_definition}; they do not "
            f"contain {', '.join(missing)}. An aggregate future regime has no unique within-"
            "window transition date, so no lead is inferred."
        )
    )
    return {
        "identifiable": identifiable,
        "performed": False,
        "required_columns": sorted(required),
        "missing_columns": missing,
        "target_horizon_trading_days": target_horizon,
        "target_definition": target_definition,
        "reason": reason,
        "lead_times_inferred": False,
    }
