"""Ridge-only classical heads for fixed QRC feature matrices."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qtyche_qrc.models.base import ForecastClassifier, ForecastRegressor
from qtyche_qrc.models.metadata import package_versions, utc_now


def design_matrix(features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Add the unpenalized classical intercept column."""

    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("QRC readout features must be a finite matrix")
    return np.column_stack((np.ones(len(values), dtype=float), values))


def ridge_solution(
    features: NDArray[np.float64], targets: NDArray[np.float64], alpha: float
) -> NDArray[np.float64]:
    """Fit an intercept-unpenalized ridge readout with an augmented least square."""

    if alpha <= 0:
        raise ValueError("ridge_alpha must be positive")
    design = design_matrix(features)
    values = np.asarray(targets, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) != len(design) or not np.isfinite(values).all():
        raise ValueError("QRC ridge targets must be finite and aligned")
    penalty = np.eye(design.shape[1], dtype=float) * np.sqrt(alpha)
    penalty[0, 0] = 0.0
    augmented_features = np.vstack((design, penalty))
    augmented_targets = np.vstack(
        (values, np.zeros((design.shape[1], values.shape[1]), dtype=float))
    )
    solution, _, _, _ = np.linalg.lstsq(augmented_features, augmented_targets, rcond=None)
    return np.asarray(solution, dtype=float)


def stable_softmax(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert finite three-class scores into normalized probabilities."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("QRC classifier scores must be a finite (n, 3) matrix")
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise ValueError("QRC softmax produced non-finite probabilities")
    return np.asarray(probabilities, dtype=float)


@dataclass(frozen=True)
class QRCReadoutConfig:
    """Shared readout settings for a fixed QRC feature representation."""

    ridge_alpha: float = 0.1
    transform_epsilon: float = 1e-12

    def validate(self) -> None:
        if self.ridge_alpha <= 0 or self.transform_epsilon <= 0:
            raise ValueError("QRC ridge alpha and transform epsilon must be positive")


class QRCClassifier(ForecastClassifier):
    """Three-output ridge and softmax head over precomputed QRC features."""

    def __init__(self, feature_names: tuple[str, ...], config: QRCReadoutConfig) -> None:
        config.validate()
        self.feature_names = feature_names
        self.config = config
        self.readout: NDArray[np.float64] | None = None
        self.training_timestamp: str | None = None

    def fit(self, features: NDArray[np.float64], targets: NDArray[np.int_]) -> None:
        labels = np.asarray(targets, dtype=int).reshape(-1)
        if len(labels) != len(features) or not set(np.unique(labels)).issubset({0, 1, 2}):
            raise ValueError("QRC classification labels must align and lie in {0,1,2}")
        one_hot = np.eye(3, dtype=float)[labels]
        self.readout = ridge_solution(features, one_hot, self.config.ridge_alpha)
        self.training_timestamp = utc_now()

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.readout is None:
            raise RuntimeError("QRC classifier is not fitted")
        scores = np.einsum("ij,jk->ik", design_matrix(features), self.readout)
        return stable_softmax(np.asarray(scores, dtype=float))

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.int_]:
        return np.asarray(np.argmax(self.predict_proba(features), axis=1), dtype=int)

    def get_params(self) -> dict[str, Any]:
        return asdict(self.config)

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "model_name": "exact_qrc_ridge_classifier",
            "model_version": "1.0",
            "task": "regime_classification",
            "feature_names": list(self.feature_names),
            "fitted": self.readout is not None,
            "hyperparameters": self.get_params(),
            "training_timestamp": self.training_timestamp,
            "readout_shape": list(self.readout.shape) if self.readout is not None else None,
            "trainable_readout_parameters": int(self.readout.size)
            if self.readout is not None
            else None,
            "package_versions": package_versions("numpy"),
        }

    def save(self, path: Path) -> None:
        if self.readout is None:
            raise RuntimeError("QRC classifier is not fitted")
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "readout.npz", readout=self.readout)
        (path / "readout_metadata.json").write_text(
            json.dumps(self.get_model_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> QRCClassifier:
        metadata = json.loads((path / "readout_metadata.json").read_text(encoding="utf-8"))
        model = cls(
            tuple(metadata["feature_names"]), QRCReadoutConfig(**metadata["hyperparameters"])
        )
        with np.load(path / "readout.npz") as values:
            model.readout = np.asarray(values["readout"], dtype=float)
        model.training_timestamp = metadata["training_timestamp"]
        return model


class QRCRegressor(ForecastRegressor):
    """Ridge head trained on log(target variance + epsilon)."""

    def __init__(self, feature_names: tuple[str, ...], config: QRCReadoutConfig) -> None:
        config.validate()
        self.feature_names = feature_names
        self.config = config
        self.readout: NDArray[np.float64] | None = None
        self.training_timestamp: str | None = None

    def fit(self, features: NDArray[np.float64], targets: NDArray[np.float64]) -> None:
        values = np.asarray(targets, dtype=float).reshape(-1)
        if len(values) != len(features) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("QRC variance targets must be finite, positive, and aligned")
        transformed = np.log(values + self.config.transform_epsilon)
        self.readout = ridge_solution(
            features, transformed[:, None], self.config.ridge_alpha
        ).reshape(-1)
        self.training_timestamp = utc_now()

    def predict_transformed(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.readout is None:
            raise RuntimeError("QRC regressor is not fitted")
        return np.asarray(np.einsum("ij,j->i", design_matrix(features), self.readout), dtype=float)

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        transformed = self.predict_transformed(features)
        with np.errstate(over="ignore", invalid="ignore"):
            predictions = np.exp(transformed) - self.config.transform_epsilon
        return np.asarray(predictions, dtype=float)

    def get_params(self) -> dict[str, Any]:
        return asdict(self.config)

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "model_name": "exact_qrc_log_variance_regressor",
            "model_version": "1.0",
            "task": "rv_regression",
            "feature_names": list(self.feature_names),
            "fitted": self.readout is not None,
            "hyperparameters": self.get_params(),
            "training_timestamp": self.training_timestamp,
            "readout_shape": list(self.readout.shape) if self.readout is not None else None,
            "trainable_readout_parameters": int(self.readout.size)
            if self.readout is not None
            else None,
            "target_transformation": {
                "name": "log_variance",
                "forward": "log(target_rv_5d + epsilon)",
                "inverse": "exp(prediction) - epsilon",
                "epsilon": self.config.transform_epsilon,
            },
            "package_versions": package_versions("numpy"),
        }

    def save(self, path: Path) -> None:
        if self.readout is None:
            raise RuntimeError("QRC regressor is not fitted")
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "readout.npz", readout=self.readout)
        (path / "readout_metadata.json").write_text(
            json.dumps(self.get_model_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> QRCRegressor:
        metadata = json.loads((path / "readout_metadata.json").read_text(encoding="utf-8"))
        model = cls(
            tuple(metadata["feature_names"]), QRCReadoutConfig(**metadata["hyperparameters"])
        )
        with np.load(path / "readout.npz") as values:
            model.readout = np.asarray(values["readout"], dtype=float)
        model.training_timestamp = metadata["training_timestamp"]
        return model
