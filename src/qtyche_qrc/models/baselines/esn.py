"""Inspectable Echo State Network reservoir and ridge readouts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qtyche_qrc.models.base import ForecastClassifier, ForecastRegressor
from qtyche_qrc.models.metadata import package_versions, utc_now


@dataclass(frozen=True)
class ESNConfig:
    """Versioned Echo State Network hyperparameters."""

    reservoir_size: int = 50
    spectral_radius: float = 0.9
    input_scaling: float = 0.5
    leaking_rate: float = 0.4
    sparsity: float = 0.1
    washout: int = 0
    ridge_alpha: float = 1e-3
    seed: int = 0
    state_policy: str = "carry_inputs"

    def validate(self) -> None:
        if self.reservoir_size <= 0:
            raise ValueError("reservoir_size must be positive")
        if self.spectral_radius <= 0 or self.input_scaling <= 0:
            raise ValueError("spectral_radius and input_scaling must be positive")
        if not 0 < self.leaking_rate <= 1:
            raise ValueError("leaking_rate must lie in (0, 1]")
        if not 0 < self.sparsity <= 1:
            raise ValueError("sparsity is the non-zero connection fraction in (0, 1]")
        if self.washout < 0 or self.ridge_alpha <= 0:
            raise ValueError("washout must be non-negative and ridge_alpha positive")
        if self.state_policy not in {"reset", "carry_inputs"}:
            raise ValueError("state_policy must be reset or carry_inputs")


class ESNReservoir:
    """Fixed random recurrent reservoir with explicit mutable temporal state."""

    def __init__(self, input_size: int, config: ESNConfig) -> None:
        config.validate()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        self.input_size = input_size
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.W_in = rng.uniform(
            -config.input_scaling,
            config.input_scaling,
            size=(config.reservoir_size, input_size + 1),
        )
        self.W_res = self._initialize_recurrent(rng)
        self._state = np.zeros(config.reservoir_size, dtype=float)

    def _initialize_recurrent(self, rng: np.random.Generator) -> NDArray[np.float64]:
        size = self.config.reservoir_size
        for _ in range(100):
            mask = rng.random((size, size)) < self.config.sparsity
            raw = rng.uniform(-1.0, 1.0, size=(size, size)) * mask
            radius = float(np.max(np.abs(np.linalg.eigvals(raw))))
            if np.isfinite(radius) and radius > 1e-12:
                return np.asarray(raw * (self.config.spectral_radius / radius), dtype=float)
        raise ValueError("could not initialize a recurrent matrix with non-zero spectral radius")

    @property
    def measured_spectral_radius(self) -> float:
        """Measure the current recurrent matrix spectral radius."""

        return float(np.max(np.abs(np.linalg.eigvals(self.W_res))))

    def reset_state(self) -> None:
        self._state = np.zeros(self.config.reservoir_size, dtype=float)

    def get_state(self) -> NDArray[np.float64]:
        return self._state.copy()

    def set_state(self, state: NDArray[np.float64]) -> None:
        value = np.asarray(state, dtype=float).reshape(-1)
        if value.shape != (self.config.reservoir_size,):
            raise ValueError("reservoir state has the wrong dimension")
        if not np.isfinite(value).all():
            raise ValueError("reservoir state must be finite")
        self._state = value.copy()

    def transform_sequence(
        self,
        inputs: NDArray[np.float64],
        *,
        reset: bool = False,
    ) -> NDArray[np.float64]:
        """Process chronological inputs; state updates never consume labels."""

        values = np.asarray(inputs, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.input_size:
            raise ValueError("ESN input matrix has the wrong shape")
        if not np.isfinite(values).all():
            raise ValueError("ESN inputs must be finite")
        if reset:
            self.reset_state()
        states = np.empty((len(values), self.config.reservoir_size), dtype=float)
        leaking_rate = self.config.leaking_rate
        for index, input_row in enumerate(values):
            augmented = np.concatenate(([1.0], input_row))
            candidate = np.tanh(self.W_in @ augmented + self.W_res @ self._state)
            self._state = (1.0 - leaking_rate) * self._state + leaking_rate * candidate
            states[index] = self._state
        return states


def split_reservoir_states(
    reservoir: ESNReservoir,
    X_train: NDArray[np.float64],
    X_validation: NDArray[np.float64],
    X_test: NDArray[np.float64] | None,
    state_policy: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Create split states under reset or label-free continuous-input policy."""

    if state_policy == "carry_inputs":
        train_states = reservoir.transform_sequence(X_train, reset=True)
        validation_states = reservoir.transform_sequence(X_validation, reset=False)
        test_states = (
            reservoir.transform_sequence(X_test, reset=False) if X_test is not None else None
        )
    elif state_policy == "reset":
        train_states = reservoir.transform_sequence(X_train, reset=True)
        validation_states = reservoir.transform_sequence(X_validation, reset=True)
        test_states = (
            reservoir.transform_sequence(X_test, reset=True) if X_test is not None else None
        )
    else:
        raise ValueError("state_policy must be reset or carry_inputs")
    return train_states, validation_states, test_states


def _design(states: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack((np.ones(len(states), dtype=float), states))


def _ridge_readout(
    states: NDArray[np.float64], targets: NDArray[np.float64], alpha: float
) -> NDArray[np.float64]:
    design = _design(states)
    ridge_rows = np.eye(design.shape[1], dtype=float) * np.sqrt(alpha)
    ridge_rows[0, 0] = 0.0
    augmented_design = np.vstack((design, ridge_rows))
    target_matrix = np.asarray(targets, dtype=float)
    if target_matrix.ndim == 1:
        target_matrix = target_matrix[:, None]
    augmented_targets = np.vstack(
        (target_matrix, np.zeros((design.shape[1], target_matrix.shape[1]), dtype=float))
    )
    solution, _, _, _ = np.linalg.lstsq(augmented_design, augmented_targets, rcond=None)
    return np.asarray(solution, dtype=float)


def _softmax(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return np.asarray(exponentials / exponentials.sum(axis=1, keepdims=True), dtype=float)


class ESNClassifier(ForecastClassifier):
    """ESN reservoir with a three-output ridge classification readout."""

    def __init__(self, feature_names: tuple[str, ...], config: ESNConfig) -> None:
        self.feature_names = feature_names
        self.config = config
        self.reservoir = ESNReservoir(len(feature_names), config)
        self.readout: NDArray[np.float64] | None = None
        self.training_timestamp: str | None = None

    def fit_readout(self, states: NDArray[np.float64], targets: NDArray[np.int_]) -> None:
        if len(states) != len(targets):
            raise ValueError("ESN states and labels must align")
        start = self.config.washout
        if start >= len(states):
            raise ValueError("washout removes all ESN training states")
        one_hot = np.eye(3, dtype=float)[targets.astype(int)]
        self.readout = _ridge_readout(states[start:], one_hot[start:], self.config.ridge_alpha)
        self.training_timestamp = utc_now()

    def fit(self, features: NDArray[np.float64], targets: NDArray[np.int_]) -> None:
        states = self.reservoir.transform_sequence(features, reset=True)
        self.fit_readout(states, targets)

    def transform_sequence(
        self, features: NDArray[np.float64], *, reset: bool = False
    ) -> NDArray[np.float64]:
        return self.reservoir.transform_sequence(features, reset=reset)

    def predict_proba_from_states(self, states: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.readout is None:
            raise RuntimeError("ESN classifier is not fitted")
        scores = np.einsum("ij,jk->ik", _design(states), self.readout)
        return _softmax(np.asarray(scores, dtype=float))

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        states = self.reservoir.transform_sequence(features, reset=True)
        return self.predict_proba_from_states(states)

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.int_]:
        return np.asarray(np.argmax(self.predict_proba(features), axis=1), dtype=int)

    def reset_state(self) -> None:
        self.reservoir.reset_state()

    def get_state(self) -> NDArray[np.float64]:
        return self.reservoir.get_state()

    def set_state(self, state: NDArray[np.float64]) -> None:
        self.reservoir.set_state(state)

    def get_params(self) -> dict[str, Any]:
        return asdict(self.config)

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "model_name": "echo_state_network_classifier",
            "model_version": "1.0",
            "task": "regime_classification",
            "feature_names": list(self.feature_names),
            "fitted": self.readout is not None,
            "hyperparameters": self.get_params(),
            "random_seed": self.config.seed,
            "training_timestamp": self.training_timestamp,
            "reservoir_dimensions": {
                "W_in": list(self.reservoir.W_in.shape),
                "W_res": list(self.reservoir.W_res.shape),
                "readout": list(self.readout.shape) if self.readout is not None else None,
            },
            "measured_spectral_radius": self.reservoir.measured_spectral_radius,
            "state_policy": self.config.state_policy,
            "package_versions": package_versions("numpy"),
        }

    def save(self, path: Path) -> None:
        if self.readout is None:
            raise RuntimeError("ESN classifier is not fitted")
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "weights.npz",
            W_in=self.reservoir.W_in,
            W_res=self.reservoir.W_res,
            readout=self.readout,
            state=self.reservoir.get_state(),
        )
        (path / "metadata.json").write_text(
            json.dumps(self.get_model_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> ESNClassifier:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        model = cls(tuple(metadata["feature_names"]), ESNConfig(**metadata["hyperparameters"]))
        weights = np.load(path / "weights.npz")
        model.reservoir.W_in = np.asarray(weights["W_in"], dtype=float)
        model.reservoir.W_res = np.asarray(weights["W_res"], dtype=float)
        model.readout = np.asarray(weights["readout"], dtype=float)
        model.reservoir.set_state(np.asarray(weights["state"], dtype=float))
        model.training_timestamp = metadata["training_timestamp"]
        return model


class ESNRegressor(ForecastRegressor):
    """ESN reservoir with a physical-unit ridge realized-variance readout."""

    def __init__(self, feature_names: tuple[str, ...], config: ESNConfig) -> None:
        self.feature_names = feature_names
        self.config = config
        self.reservoir = ESNReservoir(len(feature_names), config)
        self.readout: NDArray[np.float64] | None = None
        self.training_timestamp: str | None = None

    def fit_readout(self, states: NDArray[np.float64], targets: NDArray[np.float64]) -> None:
        start = self.config.washout
        if len(states) != len(targets) or start >= len(states):
            raise ValueError("ESN regression states/targets are invalid after washout")
        self.readout = _ridge_readout(
            states[start:], targets[start:, None], self.config.ridge_alpha
        ).reshape(-1)
        self.training_timestamp = utc_now()

    def fit(self, features: NDArray[np.float64], targets: NDArray[np.float64]) -> None:
        states = self.reservoir.transform_sequence(features, reset=True)
        self.fit_readout(states, targets)

    def transform_sequence(
        self, features: NDArray[np.float64], *, reset: bool = False
    ) -> NDArray[np.float64]:
        return self.reservoir.transform_sequence(features, reset=reset)

    def predict_from_states(self, states: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.readout is None:
            raise RuntimeError("ESN regressor is not fitted")
        return np.asarray(np.einsum("ij,j->i", _design(states), self.readout), dtype=float)

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        states = self.reservoir.transform_sequence(features, reset=True)
        return self.predict_from_states(states)

    def reset_state(self) -> None:
        self.reservoir.reset_state()

    def get_state(self) -> NDArray[np.float64]:
        return self.reservoir.get_state()

    def set_state(self, state: NDArray[np.float64]) -> None:
        self.reservoir.set_state(state)

    def get_params(self) -> dict[str, Any]:
        return asdict(self.config)

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "model_name": "echo_state_network_regressor",
            "model_version": "1.0",
            "task": "rv_regression",
            "feature_names": list(self.feature_names),
            "fitted": self.readout is not None,
            "hyperparameters": self.get_params(),
            "random_seed": self.config.seed,
            "training_timestamp": self.training_timestamp,
            "reservoir_dimensions": {
                "W_in": list(self.reservoir.W_in.shape),
                "W_res": list(self.reservoir.W_res.shape),
                "readout": list(self.readout.shape) if self.readout is not None else None,
            },
            "measured_spectral_radius": self.reservoir.measured_spectral_radius,
            "state_policy": self.config.state_policy,
            "package_versions": package_versions("numpy"),
        }

    def save(self, path: Path) -> None:
        if self.readout is None:
            raise RuntimeError("ESN regressor is not fitted")
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "weights.npz",
            W_in=self.reservoir.W_in,
            W_res=self.reservoir.W_res,
            readout=self.readout,
            state=self.reservoir.get_state(),
        )
        (path / "metadata.json").write_text(
            json.dumps(self.get_model_metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> ESNRegressor:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        model = cls(tuple(metadata["feature_names"]), ESNConfig(**metadata["hyperparameters"]))
        weights = np.load(path / "weights.npz")
        model.reservoir.W_in = np.asarray(weights["W_in"], dtype=float)
        model.reservoir.W_res = np.asarray(weights["W_res"], dtype=float)
        model.readout = np.asarray(weights["readout"], dtype=float)
        model.reservoir.set_state(np.asarray(weights["state"], dtype=float))
        model.training_timestamp = metadata["training_timestamp"]
        return model
