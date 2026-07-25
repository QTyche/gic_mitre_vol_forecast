"""Pairwise losses, classification inference, multiplicity correction, and MZ tests."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest, chi2, norm  # type: ignore[import-untyped]

LossName = Literal["qlike", "squared_error", "absolute_error"]
ClassificationMetric = Literal[
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "transition_pr_auc",
    "macro_roc_auc",
]


def loss_values(
    truth: NDArray[np.floating[Any]],
    predictions: NDArray[np.floating[Any]],
    metric: LossName,
) -> NDArray[np.float64]:
    """Return per-observation losses without altering frozen predictions."""

    observed = np.asarray(truth, dtype=float).reshape(-1)
    forecast = np.asarray(predictions, dtype=float).reshape(-1)
    if (
        len(observed) != len(forecast)
        or not len(observed)
        or not np.isfinite(observed).all()
        or not np.isfinite(forecast).all()
    ):
        raise ValueError("loss inputs must be finite, non-empty, and aligned")
    if metric == "qlike":
        if np.any(observed <= 0.0) or np.any(forecast <= 0.0):
            raise ValueError("QLIKE requires strictly positive targets and forecasts")
        return np.asarray(np.log(forecast) + observed / forecast, dtype=float)
    errors = forecast - observed
    if metric == "squared_error":
        return np.asarray(errors**2, dtype=float)
    if metric == "absolute_error":
        return np.asarray(np.abs(errors), dtype=float)
    raise ValueError(f"unsupported loss metric: {metric}")


def holm_adjust(p_values: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Return Holm step-down family-wise-error adjusted p-values."""

    values = np.asarray(p_values, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm adjustment requires finite p-values in [0,1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    adjusted_sorted = np.empty_like(sorted_values)
    running = 0.0
    total = len(values)
    for rank, value in enumerate(sorted_values):
        running = max(running, (total - rank) * float(value))
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return np.asarray(adjusted, dtype=float)


def _validate_classification_inputs(
    truth: NDArray[np.integer[Any]],
    predictions: NDArray[np.integer[Any]],
    counts: NDArray[np.integer[Any]],
    class_labels: tuple[int, ...],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    observed = np.asarray(truth, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.int64).reshape(-1)
    weights = np.asarray(counts, dtype=float)
    if (
        len(observed) != len(predicted)
        or weights.ndim != 2
        or weights.shape[1] != len(observed)
        or not len(observed)
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or np.any(weights.sum(axis=1) <= 0.0)
    ):
        raise ValueError("classification observations and bootstrap counts do not align")
    allowed = set(class_labels)
    if not set(np.unique(observed)).issubset(allowed) or not set(np.unique(predicted)).issubset(
        allowed
    ):
        raise ValueError("classification labels fall outside class_labels")
    return observed, predicted, weights


def _average_precision_distribution(
    truth: NDArray[np.int64],
    scores: NDArray[np.float64],
    counts: NDArray[np.float64],
) -> NDArray[np.float64]:
    if (
        scores.shape != truth.shape
        or not np.isfinite(scores).all()
        or not set(np.unique(truth)).issubset({0, 1})
    ):
        raise ValueError("transition PR-AUC requires aligned finite binary scores")
    positives = np.einsum("ij,j->i", counts, truth.astype(float))
    totals = counts.sum(axis=1)
    valid = (positives > 0.0) & (positives < totals)
    result = np.full(len(counts), np.nan, dtype=float)
    if not np.any(valid):
        return result
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_truth = truth[order].astype(float)
    ordered_counts = counts[valid][:, order]
    cumulative_positive = np.cumsum(ordered_counts * ordered_truth[None, :], axis=1)
    cumulative_total = np.cumsum(ordered_counts, axis=1)
    group_ends = np.flatnonzero(np.r_[ordered_scores[:-1] != ordered_scores[1:], True])
    group_positive = cumulative_positive[:, group_ends]
    group_total = cumulative_total[:, group_ends]
    increments = np.diff(
        np.concatenate(
            (np.zeros((len(group_positive), 1), dtype=float), group_positive),
            axis=1,
        ),
        axis=1,
    )
    precision = np.divide(
        group_positive,
        group_total,
        out=np.zeros_like(group_positive),
        where=group_total > 0.0,
    )
    result[valid] = np.sum(precision * increments, axis=1) / positives[valid]
    return result


def _binary_auc_distribution(
    truth: NDArray[np.int64],
    scores: NDArray[np.float64],
    counts: NDArray[np.float64],
) -> NDArray[np.float64]:
    positives = np.einsum("ij,j->i", counts, truth.astype(float))
    negatives = counts.sum(axis=1) - positives
    valid = (positives > 0.0) & (negatives > 0.0)
    result = np.full(len(counts), np.nan, dtype=float)
    if not np.any(valid):
        return result
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    ordered_truth = truth[order].astype(float)
    ordered_counts = counts[valid][:, order]
    positive_weights = ordered_counts * ordered_truth[None, :]
    negative_weights = ordered_counts * (1.0 - ordered_truth[None, :])
    cumulative_positive = np.cumsum(positive_weights, axis=1)
    cumulative_negative = np.cumsum(negative_weights, axis=1)
    group_ends = np.flatnonzero(np.r_[ordered_scores[:-1] != ordered_scores[1:], True])
    positive_by_group = np.diff(
        np.concatenate(
            (
                np.zeros((len(ordered_counts), 1), dtype=float),
                cumulative_positive[:, group_ends],
            ),
            axis=1,
        ),
        axis=1,
    )
    negative_by_group = np.diff(
        np.concatenate(
            (
                np.zeros((len(ordered_counts), 1), dtype=float),
                cumulative_negative[:, group_ends],
            ),
            axis=1,
        ),
        axis=1,
    )
    negative_before = np.cumsum(negative_by_group, axis=1) - negative_by_group
    concordant = np.sum(
        positive_by_group * (negative_before + 0.5 * negative_by_group),
        axis=1,
    )
    result[valid] = concordant / (positives[valid] * negatives[valid])
    return result


def classification_metric_distribution(
    truth: NDArray[np.integer[Any]],
    predictions: NDArray[np.integer[Any]],
    counts: NDArray[np.integer[Any]],
    metric: ClassificationMetric,
    *,
    class_labels: tuple[int, ...],
    scores: NDArray[np.floating[Any]] | None = None,
) -> NDArray[np.float64]:
    """Compute a paired metric for every bootstrap multiplicity vector."""

    raw_counts = np.asarray(counts)
    chunk_size = 512
    if raw_counts.ndim == 2 and raw_counts.shape[0] > chunk_size:
        return np.concatenate(
            [
                classification_metric_distribution(
                    truth,
                    predictions,
                    raw_counts[start : start + chunk_size],
                    metric,
                    class_labels=class_labels,
                    scores=scores,
                )
                for start in range(0, len(raw_counts), chunk_size)
            ]
        )
    observed, predicted, weights = _validate_classification_inputs(
        truth,
        predictions,
        counts,
        class_labels,
    )
    if metric == "accuracy":
        correct = (observed == predicted).astype(float)
        return np.asarray(
            np.einsum("ij,j->i", weights, correct) / weights.sum(axis=1),
            dtype=float,
        )
    if metric in {"macro_f1", "balanced_accuracy"}:
        recalls: list[NDArray[np.float64]] = []
        f1_values: list[NDArray[np.float64]] = []
        for label in class_labels:
            true_indicator = (observed == label).astype(float)
            predicted_indicator = (predicted == label).astype(float)
            true_count = np.einsum("ij,j->i", weights, true_indicator)
            predicted_count = np.einsum("ij,j->i", weights, predicted_indicator)
            true_positive = np.einsum(
                "ij,j->i",
                weights,
                true_indicator * predicted_indicator,
            )
            recalls.append(
                np.divide(
                    true_positive,
                    true_count,
                    out=np.zeros_like(true_positive),
                    where=true_count > 0.0,
                )
            )
            denominator = true_count + predicted_count
            f1_values.append(
                np.divide(
                    2.0 * true_positive,
                    denominator,
                    out=np.zeros_like(true_positive),
                    where=denominator > 0.0,
                )
            )
        selected = recalls if metric == "balanced_accuracy" else f1_values
        return np.asarray(np.mean(np.vstack(selected), axis=0), dtype=float)
    if scores is None:
        raise ValueError(f"{metric} requires supplied frozen scores")
    score_values = np.asarray(scores, dtype=float)
    if metric == "transition_pr_auc":
        return _average_precision_distribution(observed, score_values.reshape(-1), weights)
    if metric == "macro_roc_auc":
        if score_values.shape != (len(observed), len(class_labels)):
            raise ValueError("macro ROC-AUC scores must have shape (n, class_count)")
        per_class = [
            _binary_auc_distribution(
                (observed == label).astype(np.int64),
                score_values[:, column],
                weights,
            )
            for column, label in enumerate(class_labels)
        ]
        return np.asarray(np.mean(np.vstack(per_class), axis=0), dtype=float)
    raise ValueError(f"unsupported classification metric: {metric}")


def classification_metric(
    truth: NDArray[np.integer[Any]],
    predictions: NDArray[np.integer[Any]],
    metric: ClassificationMetric,
    *,
    class_labels: tuple[int, ...],
    scores: NDArray[np.floating[Any]] | None = None,
) -> float:
    """Compute one classification metric using the same weighted implementation."""

    counts = np.ones((1, len(np.asarray(truth).reshape(-1))), dtype=np.int32)
    return float(
        classification_metric_distribution(
            truth,
            predictions,
            counts,
            metric,
            class_labels=class_labels,
            scores=scores,
        )[0]
    )


def mcnemar_exact(
    truth: NDArray[np.integer[Any]],
    first_predictions: NDArray[np.integer[Any]],
    second_predictions: NDArray[np.integer[Any]],
) -> dict[str, Any]:
    """Run the exact paired McNemar/binomial test on correctness indicators."""

    observed = np.asarray(truth).reshape(-1)
    first = np.asarray(first_predictions).reshape(-1)
    second = np.asarray(second_predictions).reshape(-1)
    if len(observed) != len(first) or len(observed) != len(second) or not len(observed):
        raise ValueError("McNemar inputs must be non-empty and aligned")
    first_correct = first == observed
    second_correct = second == observed
    first_only = int(np.sum(first_correct & ~second_correct))
    second_only = int(np.sum(~first_correct & second_correct))
    discordant = first_only + second_only
    p_value = (
        float(binomtest(first_only, discordant, 0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )
    asymptotic = float((first_only - second_only) ** 2 / discordant) if discordant else 0.0
    return {
        "first_correct_second_wrong": first_only,
        "first_wrong_second_correct": second_only,
        "discordant_count": discordant,
        "test_statistic": min(first_only, second_only),
        "asymptotic_chi_square_statistic": asymptotic,
        "raw_p_value": p_value,
        "method": "exact two-sided binomial McNemar",
    }


def mincer_zarnowitz(
    truth: NDArray[np.floating[Any]],
    forecasts: NDArray[np.floating[Any]],
    *,
    hac_lag: int = 4,
) -> dict[str, Any]:
    """Fit the frozen-test diagnostic RV = alpha + beta forecast with HAC covariance."""

    observed = np.asarray(truth, dtype=float).reshape(-1)
    predicted = np.asarray(forecasts, dtype=float).reshape(-1)
    if (
        len(observed) != len(predicted)
        or len(observed) < 3
        or not np.isfinite(observed).all()
        or not np.isfinite(predicted).all()
        or hac_lag < 0
        or hac_lag >= len(observed)
    ):
        raise ValueError("Mincer-Zarnowitz inputs or HAC lag are invalid")
    design = np.column_stack((np.ones(len(predicted), dtype=float), predicted))
    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ observed
    residuals = observed - design @ coefficients
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, hac_lag + 1):
        weight = 1.0 - lag / (hac_lag + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat = meat + weight * (cross + cross.T)
    covariance = bread @ meat @ bread
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    alpha = float(coefficients[0])
    beta = float(coefficients[1])
    alpha_se = float(standard_errors[0])
    beta_se = float(standard_errors[1])
    if alpha_se == 0.0 or beta_se == 0.0:
        raise ValueError("Mincer-Zarnowitz HAC standard error is zero")
    alpha_t = alpha / alpha_se
    beta_t = (beta - 1.0) / beta_se
    restrictions = np.asarray([alpha, beta - 1.0], dtype=float)
    joint_statistic = float(restrictions @ np.linalg.inv(covariance) @ restrictions)
    total_sum_squares = float(np.sum((observed - observed.mean()) ** 2))
    residual_sum_squares = float(np.sum(residuals**2))
    return {
        "sample_count": len(observed),
        "hac_lag": hac_lag,
        "alpha": alpha,
        "beta": beta,
        "alpha_hac_standard_error": alpha_se,
        "beta_hac_standard_error": beta_se,
        "alpha_equals_zero_t_statistic": alpha_t,
        "alpha_equals_zero_p_value": float(2.0 * norm.sf(abs(alpha_t))),
        "beta_equals_one_t_statistic": beta_t,
        "beta_equals_one_p_value": float(2.0 * norm.sf(abs(beta_t))),
        "joint_wald_statistic": joint_statistic,
        "joint_wald_degrees_of_freedom": 2,
        "joint_wald_p_value": float(chi2.sf(joint_statistic, df=2)),
        "r_squared": 1.0 - residual_sum_squares / total_sum_squares,
        "diagnostic_only": True,
        "forecast_modified": False,
    }
