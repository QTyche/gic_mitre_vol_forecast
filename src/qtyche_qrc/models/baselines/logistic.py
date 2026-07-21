"""Deterministic regularized multinomial logistic-regression baseline."""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from qtyche_qrc.models.base import ForecastClassifier
from qtyche_qrc.models.metadata import package_versions, utc_now


class MultinomialLogisticClassifier(ForecastClassifier):
    """L2-regularized three-class logistic regression on frozen scaled features."""

    def __init__(
        self,
        feature_names: tuple[str, ...],
        regularization_c: float = 1.0,
        class_weight: str | None = None,
        max_iterations: int = 500,
        seed: int = 0,
    ) -> None:
        if regularization_c <= 0:
            raise ValueError("regularization_c must be positive")
        self.feature_names = feature_names
        self.regularization_c = regularization_c
        self.class_weight = class_weight
        self.max_iterations = max_iterations
        self.seed = seed
        self.estimator: LogisticRegression | None = None
        self.converged: bool | None = None
        self.convergence_warnings: list[str] = []
        self.training_timestamp: str | None = None

    def fit(self, features: NDArray[np.float64], targets: NDArray[np.int_]) -> None:
        if features.shape[1] != len(self.feature_names):
            raise ValueError("logistic feature count differs from configured feature names")
        estimator = LogisticRegression(
            C=self.regularization_c,
            penalty="l2",
            solver="lbfgs",
            class_weight=self.class_weight,
            max_iter=self.max_iterations,
            random_state=self.seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            estimator.fit(features, targets)
        self.convergence_warnings = [str(item.message) for item in caught]
        self.converged = bool(np.all(estimator.n_iter_ < self.max_iterations))
        self.estimator = estimator
        self.training_timestamp = utc_now()

    def _require_fitted(self) -> LogisticRegression:
        if self.estimator is None:
            raise RuntimeError("multinomial logistic regression is not fitted")
        return self.estimator

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.int_]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            predicted = self._require_fitted().predict(features)
        return np.asarray(predicted, dtype=int)

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        estimator = self._require_fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw = np.asarray(estimator.predict_proba(features), dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("logistic regression produced non-finite probabilities")
        probabilities = np.zeros((len(features), 3), dtype=float)
        probabilities[:, np.asarray(estimator.classes_, dtype=int)] = raw
        return probabilities

    def get_params(self) -> dict[str, Any]:
        return {
            "regularization_c": self.regularization_c,
            "class_weight": self.class_weight,
            "max_iterations": self.max_iterations,
            "seed": self.seed,
        }

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "model_name": "multinomial_logistic_regression",
            "model_version": "1.0",
            "task": "regime_classification",
            "feature_names": list(self.feature_names),
            "fitted": self.estimator is not None,
            "hyperparameters": self.get_params(),
            "random_seed": self.seed,
            "training_timestamp": self.training_timestamp,
            "converged": self.converged,
            "convergence_warnings": self.convergence_warnings,
            "package_versions": package_versions("numpy", "scikit-learn"),
        }

    def save(self, path: Path) -> None:
        estimator = self._require_fitted()
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.pkl").write_bytes(pickle.dumps(estimator))
        (path / "metadata.json").write_text(
            json.dumps(self.get_model_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> MultinomialLogisticClassifier:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        params = metadata["hyperparameters"]
        model = cls(
            tuple(metadata["feature_names"]),
            float(params["regularization_c"]),
            params["class_weight"],
            int(params["max_iterations"]),
            int(params["seed"]),
        )
        model.estimator = pickle.loads((path / "model.pkl").read_bytes())
        model.converged = metadata["converged"]
        model.convergence_warnings = metadata["convergence_warnings"]
        model.training_timestamp = metadata["training_timestamp"]
        return model
