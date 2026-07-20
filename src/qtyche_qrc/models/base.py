"""Common forecasting interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ForecastModel(ABC):
    """Fit on training observations and forecast without mutating the data split."""

    @abstractmethod
    def fit(self, features: NDArray[np.float64], targets: NDArray[np.int_]) -> None:
        """Fit the model using training data only."""

    @abstractmethod
    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return class probabilities for each observation."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return serializable model and resource metadata."""
