"""Rolling and cumulative descriptive diagnostics for frozen financial predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qtyche_qrc.diagnostics.calibration import qlike_values


def rolling_variance_diagnostics(
    dates: NDArray[np.str_],
    truth: NDArray[np.floating[Any]],
    forecasts: NDArray[np.floating[Any]],
    *,
    window: int = 60,
) -> list[dict[str, Any]]:
    """Calculate trailing diagnostics aligned to each window's final observation."""

    date_values = np.asarray(dates, dtype=str).reshape(-1)
    observed = np.asarray(truth, dtype=float).reshape(-1)
    predicted = np.asarray(forecasts, dtype=float).reshape(-1)
    if (
        len(date_values) != len(observed)
        or len(date_values) != len(predicted)
        or isinstance(window, bool)
        or not isinstance(window, int)
        or window < 2
        or window > len(observed)
    ):
        raise ValueError("rolling diagnostic inputs or window are invalid")
    losses = qlike_values(observed, predicted)
    errors = predicted - observed
    squared_errors = errors**2
    rows: list[dict[str, Any]] = []
    for end in range(window - 1, len(observed)):
        start = end - window + 1
        rows.append(
            {
                "date": str(date_values[end]),
                "window_start_date": str(date_values[start]),
                "window_end_date": str(date_values[end]),
                "window_observation_count": window,
                "rolling_qlike": float(losses[start : end + 1].mean()),
                "rolling_rmse": float(np.sqrt(squared_errors[start : end + 1].mean())),
                "rolling_bias": float(errors[start : end + 1].mean()),
                "trailing_window": True,
                "used_for_selection": False,
            }
        )
    return rows


def cumulative_loss_difference(
    model_losses: NDArray[np.floating[Any]],
    reference_losses: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Return cumulative model-minus-reference loss; negative favours the model."""

    model = np.asarray(model_losses, dtype=float).reshape(-1)
    reference = np.asarray(reference_losses, dtype=float).reshape(-1)
    if (
        len(model) != len(reference)
        or not len(model)
        or not np.isfinite(model).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("cumulative loss inputs must be finite and aligned")
    return np.asarray(np.cumsum(model - reference), dtype=float)


def cumulative_classification_error_difference(
    truth: NDArray[np.integer[Any]],
    predictions: NDArray[np.integer[Any]],
    reference_predictions: NDArray[np.integer[Any]],
) -> NDArray[np.float64]:
    """Return cumulative classification-error difference relative to a reference."""

    observed = np.asarray(truth).reshape(-1)
    predicted = np.asarray(predictions).reshape(-1)
    reference = np.asarray(reference_predictions).reshape(-1)
    if len(observed) != len(predicted) or len(observed) != len(reference) or not len(observed):
        raise ValueError("classification-error inputs must be non-empty and aligned")
    model_error = (predicted != observed).astype(float)
    reference_error = (reference != observed).astype(float)
    return cumulative_loss_difference(model_error, reference_error)
