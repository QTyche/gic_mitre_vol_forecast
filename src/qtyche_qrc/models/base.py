"""Common, inspectable forecasting interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

ClassifierT = TypeVar("ClassifierT", bound="ForecastClassifier")
RegressorT = TypeVar("RegressorT", bound="ForecastRegressor")


class ForecastModel(ABC):
    """Shared metadata and persistence contract for all forecast models."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist all fitted parameters required for exact reload."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return configured hyperparameters."""

    @abstractmethod
    def get_model_metadata(self) -> dict[str, Any]:
        """Return versioned fit status, provenance, and package metadata."""

    def metadata(self) -> dict[str, Any]:
        """Backward-compatible alias for model metadata."""

        return self.get_model_metadata()


class ForecastClassifier(ForecastModel):
    """Three-class probabilistic forecasting interface."""

    @abstractmethod
    def fit(self, features: NDArray[np.float64], targets: NDArray[np.int_]) -> None:
        """Fit using training observations and labels only."""

    @abstractmethod
    def predict(self, features: NDArray[np.float64]) -> NDArray[np.int_]:
        """Return class predictions."""

    @abstractmethod
    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return probabilities ordered as low, medium, high."""


class ForecastRegressor(ForecastModel):
    """Realized-variance forecasting interface."""

    @abstractmethod
    def fit(self, features: NDArray[np.float64], targets: NDArray[np.float64]) -> None:
        """Fit using training observations and targets only."""

    @abstractmethod
    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return raw realized-variance predictions in physical units."""
