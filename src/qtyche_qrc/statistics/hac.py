"""Newey-West/HAC inference for frozen loss differentials."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t  # type: ignore[import-untyped]


def _finite_vector(values: NDArray[np.floating[Any]], name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if len(vector) < 2 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain at least two finite observations")
    return vector


def newey_west_long_run_variance(
    values: NDArray[np.floating[Any]],
    lag: int,
) -> float:
    """Estimate the Bartlett-kernel long-run variance of a scalar series."""

    vector = _finite_vector(values, "HAC series")
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0 or lag >= len(vector):
        raise ValueError("HAC lag must be an integer in [0, n-1]")
    centered = vector - vector.mean()
    count = len(centered)
    estimate = float(np.dot(centered, centered) / count)
    for offset in range(1, lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / count)
        weight = 1.0 - offset / (lag + 1.0)
        estimate += 2.0 * weight * covariance
    if estimate < 0.0 and estimate > -1e-14:
        estimate = 0.0
    if not np.isfinite(estimate) or estimate < 0.0:
        raise ValueError("Newey-West long-run variance is negative or non-finite")
    return estimate


def harvey_leybourne_newbold_factor(count: int, horizon: int) -> float:
    """Return the finite-sample correction for an h-step DM statistic."""

    if count < 2 or horizon < 1:
        raise ValueError("HLN correction requires count >= 2 and horizon >= 1")
    inside = (count + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / count) / count
    if inside <= 0.0:
        raise ValueError("HLN correction is undefined for this sample and horizon")
    return float(np.sqrt(inside))


def hac_mean_test(
    values: NDArray[np.floating[Any]],
    *,
    lag: int,
    confidence_level: float = 0.95,
    forecast_horizon: int = 1,
    apply_hln: bool = False,
) -> dict[str, Any]:
    """Test whether a serially dependent scalar series has zero mean."""

    vector = _finite_vector(values, "mean-test series")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    count = len(vector)
    long_run_variance = newey_west_long_run_variance(vector, lag)
    hac_standard_error = float(np.sqrt(long_run_variance / count))
    correction = harvey_leybourne_newbold_factor(count, forecast_horizon) if apply_hln else 1.0
    reported_standard_error = hac_standard_error / correction
    mean = float(vector.mean())
    if reported_standard_error == 0.0:
        if mean != 0.0:
            raise ValueError("zero HAC standard error with non-zero sample mean")
        statistic = 0.0
        raw_p_value = 1.0
        critical = 0.0
    else:
        statistic = mean / reported_standard_error
        raw_p_value = float(2.0 * student_t.sf(abs(statistic), df=count - 1))
        critical = float(student_t.ppf((1.0 + confidence_level) / 2.0, df=count - 1))
    return {
        "sample_count": count,
        "mean": mean,
        "hac_lag": lag,
        "long_run_variance": long_run_variance,
        "hac_standard_error": hac_standard_error,
        "reported_standard_error": reported_standard_error,
        "test_statistic": statistic,
        "raw_p_value": raw_p_value,
        "confidence_level": confidence_level,
        "confidence_interval_lower": mean - critical * reported_standard_error,
        "confidence_interval_upper": mean + critical * reported_standard_error,
        "harvey_leybourne_newbold_applied": apply_hln,
        "harvey_leybourne_newbold_factor": correction if apply_hln else None,
        "forecast_horizon": forecast_horizon,
        "reference_distribution": f"Student-t(df={count - 1})",
    }
