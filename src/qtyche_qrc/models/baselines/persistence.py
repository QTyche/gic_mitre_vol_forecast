"""Minimum classification and realized-variance persistence controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qtyche_qrc.models.base import ForecastClassifier, ForecastRegressor
from qtyche_qrc.models.metadata import package_versions, utc_now


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.json").write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class MajorityClassClassifier(ForecastClassifier):
    """Predict the training majority class with empirical training frequencies."""

    def __init__(self, feature_names: tuple[str, ...], seed: int = 0) -> None:
        self.feature_names = feature_names
        self.seed = seed
        self.class_probabilities: NDArray[np.float64] | None = None
        self.majority_class: int | None = None
        self.training_timestamp: str | None = None

    def fit(self, features: NDArray[np.float64], targets: NDArray[np.int_]) -> None:
        if len(features) != len(targets) or not len(targets):
            raise ValueError("majority classifier requires non-empty aligned training data")
        if not set(np.unique(targets)).issubset({0, 1, 2}):
            raise ValueError("majority classifier targets must be in {0, 1, 2}")
        counts = np.bincount(targets.astype(int), minlength=3).astype(float)
        self.class_probabilities = counts / counts.sum()
        self.majority_class = int(np.argmax(counts))
        self.training_timestamp = utc_now()

    def _require_fitted(self) -> tuple[int, NDArray[np.float64]]:
        if self.majority_class is None or self.class_probabilities is None:
            raise RuntimeError("majority classifier is not fitted")
        return self.majority_class, self.class_probabilities

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.int_]:
        majority, _ = self._require_fitted()
        return np.full(len(features), majority, dtype=int)

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        _, probabilities = self._require_fitted()
        return np.tile(probabilities, (len(features), 1))

    def get_params(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "model_name": "majority_class_classifier",
            "model_version": "1.0",
            "task": "regime_classification",
            "feature_names": list(self.feature_names),
            "fitted": self.majority_class is not None,
            "hyperparameters": self.get_params(),
            "random_seed": self.seed,
            "training_timestamp": self.training_timestamp,
            "majority_class": self.majority_class,
            "class_probabilities": (
                self.class_probabilities.tolist() if self.class_probabilities is not None else None
            ),
            "package_versions": package_versions("numpy"),
        }

    def save(self, path: Path) -> None:
        self._require_fitted()
        _write_json(path, self.get_model_metadata())

    @classmethod
    def load(cls, path: Path) -> MajorityClassClassifier:
        value = json.loads((path / "model.json").read_text(encoding="utf-8"))
        model = cls(tuple(value["feature_names"]), int(value["random_seed"]))
        model.majority_class = int(value["majority_class"])
        model.class_probabilities = np.asarray(value["class_probabilities"], dtype=float)
        model.training_timestamp = value["training_timestamp"]
        return model


class CurrentRegimePersistenceClassifier(ForecastClassifier):
    """Predict the explicitly supplied current regime without using target labels."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.fitted = False
        self.training_timestamp: str | None = None

    def fit(self, features: NDArray[np.float64], targets: NDArray[np.int_]) -> None:
        del targets
        self._current_regimes(features)
        self.fitted = True
        self.training_timestamp = utc_now()

    @staticmethod
    def _current_regimes(values: NDArray[np.float64]) -> NDArray[np.int_]:
        regimes = np.asarray(values).reshape(-1)
        if not set(np.unique(regimes)).issubset({0, 1, 2}):
            raise ValueError("current_regime must contain only 0, 1, or 2")
        return regimes.astype(int)

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("current-regime persistence classifier is not fitted")

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.int_]:
        self._require_fitted()
        return self._current_regimes(features)

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        regimes = self.predict(features)
        probabilities = np.zeros((len(regimes), 3), dtype=float)
        probabilities[np.arange(len(regimes)), regimes] = 1.0
        return probabilities

    def get_params(self) -> dict[str, Any]:
        return {"seed": self.seed, "input": "current_regime_explicit"}

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "model_name": "current_regime_persistence",
            "model_version": "1.0",
            "task": "regime_classification",
            "feature_names": ["current_regime"],
            "fitted": self.fitted,
            "hyperparameters": self.get_params(),
            "random_seed": self.seed,
            "training_timestamp": self.training_timestamp,
            "package_versions": package_versions("numpy"),
        }

    def save(self, path: Path) -> None:
        self._require_fitted()
        _write_json(path, self.get_model_metadata())

    @classmethod
    def load(cls, path: Path) -> CurrentRegimePersistenceClassifier:
        value = json.loads((path / "model.json").read_text(encoding="utf-8"))
        model = cls(int(value["random_seed"]))
        model.fitted = bool(value["fitted"])
        model.training_timestamp = value["training_timestamp"]
        return model


class RealizedVariancePersistenceRegressor(ForecastRegressor):
    """Return current unscaled annualized five-day RV as the forward RV forecast."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.fitted = False
        self.training_timestamp: str | None = None

    def fit(self, features: NDArray[np.float64], targets: NDArray[np.float64]) -> None:
        current_rv = np.asarray(features, dtype=float).reshape(-1)
        if len(current_rv) != len(targets) or not len(targets):
            raise ValueError("RV persistence requires non-empty aligned training values")
        if not np.isfinite(current_rv).all() or np.any(current_rv < 0):
            raise ValueError("current unscaled realized variance must be finite and non-negative")
        if not np.isfinite(targets).all() or np.any(targets <= 0):
            raise ValueError("target realized variance must be finite and positive")
        self.fitted = True
        self.training_timestamp = utc_now()

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.fitted:
            raise RuntimeError("realized-variance persistence regressor is not fitted")
        current_rv = np.asarray(features, dtype=float).reshape(-1)
        if not np.isfinite(current_rv).all() or np.any(current_rv < 0):
            raise ValueError("current unscaled realized variance must be finite and non-negative")
        return current_rv.copy()

    def get_params(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "input": "unscaled_spy_rv_5d",
            "annualization": "252/5 trailing squared log returns",
        }

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "model_name": "realized_variance_persistence",
            "model_version": "1.0",
            "task": "rv_regression",
            "feature_names": ["unscaled_spy_rv_5d"],
            "fitted": self.fitted,
            "hyperparameters": self.get_params(),
            "random_seed": self.seed,
            "training_timestamp": self.training_timestamp,
            "package_versions": package_versions("numpy"),
        }

    def save(self, path: Path) -> None:
        if not self.fitted:
            raise RuntimeError("realized-variance persistence regressor is not fitted")
        _write_json(path, self.get_model_metadata())

    @classmethod
    def load(cls, path: Path) -> RealizedVariancePersistenceRegressor:
        value = json.loads((path / "model.json").read_text(encoding="utf-8"))
        model = cls(int(value["random_seed"]))
        model.fitted = bool(value["fitted"])
        model.training_timestamp = value["training_timestamp"]
        return model
