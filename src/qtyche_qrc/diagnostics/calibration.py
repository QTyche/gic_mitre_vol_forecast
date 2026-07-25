"""Calibration and descriptive uncertainty calculations for frozen predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _finite_vector(values: NDArray[np.floating[Any]], name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if not len(vector) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return vector


def _probability_matrix(
    probabilities: NDArray[np.floating[Any]],
    *,
    class_count: int,
) -> NDArray[np.float64]:
    values = np.asarray(probabilities, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != class_count
        or not len(values)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-10)
    ):
        raise ValueError("probabilities must be finite aligned rows summing to one")
    return values


def calibration_bin_assignments(
    probabilities: NDArray[np.floating[Any]],
    bin_count: int,
) -> NDArray[np.int64]:
    """Assign values in [0,1] to fixed equal-width bins, including one in the last."""

    values = _finite_vector(probabilities, "calibration probabilities")
    if (
        isinstance(bin_count, bool)
        or not isinstance(bin_count, int)
        or bin_count < 2
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("calibration bins require probabilities in [0,1] and bins >= 2")
    return np.asarray(
        np.minimum(np.floor(values * bin_count).astype(np.int64), bin_count - 1),
        dtype=np.int64,
    )


def reliability_table(
    events: NDArray[np.integer[Any]] | NDArray[np.bool_],
    probabilities: NDArray[np.floating[Any]],
    bin_count: int,
) -> list[dict[str, Any]]:
    """Return every fixed reliability bin, retaining empty bins explicitly."""

    observed = np.asarray(events, dtype=np.int64).reshape(-1)
    predicted = _finite_vector(probabilities, "reliability probabilities")
    if (
        len(observed) != len(predicted)
        or not set(np.unique(observed)).issubset({0, 1})
        or np.any(predicted < 0.0)
        or np.any(predicted > 1.0)
    ):
        raise ValueError("reliability inputs must be aligned binary events and probabilities")
    assignments = calibration_bin_assignments(predicted, bin_count)
    rows: list[dict[str, Any]] = []
    for index in range(bin_count):
        mask = assignments == index
        count = int(mask.sum())
        mean_probability = float(predicted[mask].mean()) if count else None
        event_rate = float(observed[mask].mean()) if count else None
        rows.append(
            {
                "bin_index": index,
                "bin_lower": index / bin_count,
                "bin_upper": (index + 1) / bin_count,
                "upper_inclusive": index == bin_count - 1,
                "observation_count": count,
                "empty_bin": count == 0,
                "mean_probability": mean_probability,
                "event_rate": event_rate,
                "calibration_gap": (
                    float(mean_probability - event_rate)
                    if mean_probability is not None and event_rate is not None
                    else None
                ),
            }
        )
    return rows


def calibration_errors(rows: list[dict[str, Any]]) -> tuple[float, float]:
    """Calculate observation-weighted ECE and maximum absolute calibration error."""

    total = sum(int(row["observation_count"]) for row in rows)
    if total <= 0:
        raise ValueError("calibration table contains no observations")
    populated = [row for row in rows if int(row["observation_count"]) > 0]
    absolute_gaps = [abs(float(row["calibration_gap"])) for row in populated]
    ece = (
        sum(int(row["observation_count"]) * abs(float(row["calibration_gap"])) for row in populated)
        / total
    )
    return float(ece), float(max(absolute_gaps))


def probability_entropy(
    probabilities: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Return Shannon entropy in natural-log units for every probability row."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2:
        raise ValueError("probability entropy requires a two-dimensional array")
    safe = np.clip(values, np.finfo(float).tiny, 1.0)
    return np.asarray(-np.sum(values * np.log(safe), axis=1), dtype=float)


def multiclass_calibration_summary(
    truth: NDArray[np.integer[Any]],
    probabilities: NDArray[np.floating[Any]],
    *,
    bin_count: int,
) -> dict[str, Any]:
    """Describe multiclass calibration without fitting a calibration transform."""

    observed = np.asarray(truth, dtype=np.int64).reshape(-1)
    class_count = np.asarray(probabilities).shape[1] if np.asarray(probabilities).ndim == 2 else 0
    values = _probability_matrix(probabilities, class_count=class_count)
    if (
        len(observed) != len(values)
        or class_count < 2
        or np.any(observed < 0)
        or np.any(observed >= class_count)
    ):
        raise ValueError("multiclass truth and probabilities do not align")
    predicted = np.argmax(values, axis=1)
    confidence = np.max(values, axis=1)
    correct = (predicted == observed).astype(np.int64)
    top_rows = reliability_table(correct, confidence, bin_count)
    top_ece, top_mce = calibration_errors(top_rows)
    one_hot = np.eye(class_count, dtype=float)[observed]
    clipped = np.clip(values[np.arange(len(values)), observed], np.finfo(float).tiny, 1.0)
    return {
        "observation_count": len(observed),
        "bin_count": bin_count,
        "multiclass_brier_score": float(np.mean(np.sum((values - one_hot) ** 2, axis=1))),
        "multiclass_log_loss": float(-np.mean(np.log(clipped))),
        "top_label_expected_calibration_error": top_ece,
        "top_label_maximum_calibration_error": top_mce,
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "confidence_minus_accuracy": float(confidence.mean() - correct.mean()),
        "mean_probability_entropy": float(probability_entropy(values).mean()),
        "top_label_reliability": top_rows,
    }


def classwise_calibration_summary(
    truth: NDArray[np.integer[Any]],
    probabilities: NDArray[np.floating[Any]],
    *,
    bin_count: int,
) -> list[dict[str, Any]]:
    """Return one-vs-rest Brier, ECE, MCE, and reliability rows for every class."""

    observed = np.asarray(truth, dtype=np.int64).reshape(-1)
    class_count = np.asarray(probabilities).shape[1] if np.asarray(probabilities).ndim == 2 else 0
    values = _probability_matrix(probabilities, class_count=class_count)
    if len(observed) != len(values):
        raise ValueError("classwise calibration inputs do not align")
    rows: list[dict[str, Any]] = []
    for label in range(class_count):
        events = (observed == label).astype(np.int64)
        reliability = reliability_table(events, values[:, label], bin_count)
        ece, mce = calibration_errors(reliability)
        rows.append(
            {
                "class_label": label,
                "bin_count": bin_count,
                "one_vs_rest_brier_score": float(np.mean((values[:, label] - events) ** 2)),
                "expected_calibration_error": ece,
                "maximum_calibration_error": mce,
                "mean_probability": float(values[:, label].mean()),
                "event_rate": float(events.mean()),
                "reliability_bins": reliability,
            }
        )
    return rows


def qlike_values(
    truth: NDArray[np.floating[Any]],
    forecasts: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    observed = _finite_vector(truth, "realised variance")
    predicted = _finite_vector(forecasts, "variance forecast")
    if len(observed) != len(predicted) or np.any(observed <= 0.0) or np.any(predicted <= 0.0):
        raise ValueError("QLIKE requires aligned strictly positive values")
    return np.asarray(np.log(predicted) + observed / predicted, dtype=float)


def variance_point_metrics(
    truth: NDArray[np.floating[Any]],
    forecasts: NDArray[np.floating[Any]],
) -> dict[str, Any]:
    """Calculate the requested scalar calibration diagnostics."""

    observed = _finite_vector(truth, "realised variance")
    predicted = _finite_vector(forecasts, "variance forecast")
    if len(observed) != len(predicted) or np.any(observed <= 0.0) or np.any(predicted <= 0.0):
        raise ValueError("variance calibration requires aligned positive observations")
    error = predicted - observed
    correlation = float(np.corrcoef(observed, predicted)[0, 1])
    return {
        "observation_count": len(observed),
        "mean_forecast": float(predicted.mean()),
        "mean_realised_variance": float(observed.mean()),
        "mean_error": float(error.mean()),
        "median_error": float(np.median(error)),
        "forecast_to_realised_mean_ratio": float(predicted.mean() / observed.mean()),
        "percentage_overpredictions": float(100.0 * np.mean(error > 0.0)),
        "percentage_underpredictions": float(100.0 * np.mean(error < 0.0)),
        "percentage_exact_predictions": float(100.0 * np.mean(error == 0.0)),
        "qlike": float(qlike_values(observed, predicted).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "correlation": correlation if np.isfinite(correlation) else None,
        "forecast_standard_deviation": float(np.std(predicted, ddof=0)),
        "realised_variance_standard_deviation": float(np.std(observed, ddof=0)),
    }


def _weighted_mean(
    counts: NDArray[np.float64],
    values: NDArray[np.float64],
    totals: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.divide(
        np.einsum("ij,j->i", counts, values),
        totals,
        out=np.full(len(counts), np.nan, dtype=float),
        where=totals > 0.0,
    )


def variance_bootstrap_intervals(
    truth: NDArray[np.floating[Any]],
    forecasts: NDArray[np.floating[Any]],
    counts: NDArray[np.integer[Any]],
    *,
    mask: NDArray[np.bool_] | None = None,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Return moving-block bootstrap intervals using fixed paired multiplicities."""

    observed = _finite_vector(truth, "bootstrap realised variance")
    predicted = _finite_vector(forecasts, "bootstrap variance forecast")
    multiplicities = np.asarray(counts, dtype=float)
    selected = np.ones(len(observed), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if (
        len(observed) != len(predicted)
        or multiplicities.ndim != 2
        or multiplicities.shape[1] != len(observed)
        or selected.shape != observed.shape
        or not selected.any()
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("variance bootstrap inputs do not align")
    weights = multiplicities[:, selected]
    y = observed[selected]
    f = predicted[selected]
    totals = weights.sum(axis=1)
    error = f - y
    mean_y = _weighted_mean(weights, y, totals)
    mean_f = _weighted_mean(weights, f, totals)
    mean_error = _weighted_mean(weights, error, totals)
    mean_qlike = _weighted_mean(weights, qlike_values(y, f), totals)
    mse = _weighted_mean(weights, error**2, totals)
    mae = _weighted_mean(weights, np.abs(error), totals)
    over = 100.0 * _weighted_mean(weights, (error > 0.0).astype(float), totals)
    under = 100.0 * _weighted_mean(weights, (error < 0.0).astype(float), totals)
    mean_y2 = _weighted_mean(weights, y**2, totals)
    mean_f2 = _weighted_mean(weights, f**2, totals)
    mean_yf = _weighted_mean(weights, y * f, totals)
    variance_y = np.maximum(mean_y2 - mean_y**2, 0.0)
    variance_f = np.maximum(mean_f2 - mean_f**2, 0.0)
    covariance = mean_yf - mean_y * mean_f
    correlation = np.divide(
        covariance,
        np.sqrt(variance_y * variance_f),
        out=np.full(len(weights), np.nan, dtype=float),
        where=(variance_y > 0.0) & (variance_f > 0.0),
    )
    draws = {
        "mean_forecast": mean_f,
        "mean_realised_variance": mean_y,
        "mean_error": mean_error,
        "forecast_to_realised_mean_ratio": mean_f / mean_y,
        "percentage_overpredictions": over,
        "percentage_underpredictions": under,
        "qlike": mean_qlike,
        "rmse": np.sqrt(mse),
        "mae": mae,
        "correlation": correlation,
        "forecast_standard_deviation": np.sqrt(variance_f),
        "realised_variance_standard_deviation": np.sqrt(variance_y),
    }
    alpha = 1.0 - confidence_level
    result: dict[str, Any] = {}
    for name, values in draws.items():
        finite = values[np.isfinite(values)]
        if not len(finite):
            result[f"{name}_ci_lower"] = None
            result[f"{name}_ci_upper"] = None
            continue
        lower, upper = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
        result[f"{name}_ci_lower"] = float(lower)
        result[f"{name}_ci_upper"] = float(upper)
    result["bootstrap_valid_count"] = int(np.sum(totals > 0.0))
    result["bootstrap_invalid_count"] = int(np.sum(totals <= 0.0))
    return result


def variance_calibration_deciles(
    truth: NDArray[np.floating[Any]],
    forecasts: NDArray[np.floating[Any]],
    *,
    decile_count: int = 10,
) -> list[dict[str, Any]]:
    """Build post-hoc split-specific forecast-quantile calibration bins."""

    observed = _finite_vector(truth, "decile realised variance")
    predicted = _finite_vector(forecasts, "decile variance forecast")
    if len(observed) != len(predicted) or decile_count < 2:
        raise ValueError("variance decile inputs do not align")
    edges = np.quantile(predicted, np.linspace(0.0, 1.0, decile_count + 1))
    assignments = np.searchsorted(edges[1:-1], predicted, side="right")
    rows: list[dict[str, Any]] = []
    for index in range(decile_count):
        mask = assignments == index
        count = int(mask.sum())
        metrics = variance_point_metrics(observed[mask], predicted[mask]) if count else None
        rows.append(
            {
                "decile": index + 1,
                "forecast_quantile_lower": index / decile_count,
                "forecast_quantile_upper": (index + 1) / decile_count,
                "forecast_edge_lower": float(edges[index]),
                "forecast_edge_upper": float(edges[index + 1]),
                "observation_count": count,
                "empty_bin": count == 0,
                "mean_forecast": metrics["mean_forecast"] if metrics else None,
                "mean_realised_variance": (metrics["mean_realised_variance"] if metrics else None),
                "median_forecast": float(np.median(predicted[mask])) if count else None,
                "median_realised_variance": float(np.median(observed[mask])) if count else None,
                "bias": metrics["mean_error"] if metrics else None,
                "qlike": metrics["qlike"] if metrics else None,
                "rmse": metrics["rmse"] if metrics else None,
                "mae": metrics["mae"] if metrics else None,
                "post_hoc_diagnostic_bin": True,
                "used_for_selection": False,
            }
        )
    return rows
