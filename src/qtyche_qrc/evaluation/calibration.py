"""Simple deterministic calibration summaries."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def calibration_bins(
    truth: NDArray[np.int_],
    probabilities: NDArray[np.float64],
    bins: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]:
    """Return mean probability, observed frequency, and count for non-empty bins."""

    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    mean_probability: list[float] = []
    observed_frequency: list[float] = []
    counts: list[int] = []
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            mean_probability.append(float(np.mean(probabilities[selected])))
            observed_frequency.append(float(np.mean(truth[selected])))
            counts.append(int(selected.sum()))
    return (
        np.asarray(mean_probability),
        np.asarray(observed_frequency),
        np.asarray(counts),
    )
