"""Sequential reset-and-reinject exact quantum reservoir."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qtyche_qrc.models.qrc.backends import (
    BACKEND_NAME,
    ExactDensityMatrixBackend,
    NumericalTolerances,
)
from qtyche_qrc.models.qrc.encoding import (
    array_checksum,
    initial_density_matrix,
    input_projection,
    reset_and_encode_input,
)
from qtyche_qrc.models.qrc.hamiltonian import Edge, HamiltonianDefinition, generate_hamiltonian
from qtyche_qrc.models.qrc.observables import ObservableSet


@dataclass(frozen=True)
class QRCConfig:
    """Frozen dynamical parameters; classical ridge settings live in the readout."""

    n_qubits: int = 3
    graph: str = "ring"
    virtual_nodes: int = 1
    j_strength: float = 1.0
    h_strength: float = 1.0
    h_min_factor: float = 0.5
    h_max_factor: float = 1.5
    tau: float = 1.0
    input_scaling: float = 0.5
    state_policy: str = "carry_inputs"
    reservoir_seed: int = 2026
    backend: str = BACKEND_NAME
    chords: tuple[Edge, ...] = ()

    def validate(self) -> None:
        if self.n_qubits < 2 or self.n_qubits > 6:
            raise ValueError("exact ring QRC requires 2 <= n_qubits <= 6")
        if self.graph not in {"ring", "ring_plus_chords"}:
            raise ValueError("graph must be ring or ring_plus_chords")
        if self.virtual_nodes <= 0:
            raise ValueError("virtual_nodes must be positive")
        if self.j_strength < 0 or self.h_strength <= 0 or self.tau <= 0:
            raise ValueError("QRC strengths and tau are invalid")
        if self.input_scaling <= 0:
            raise ValueError("input_scaling must be positive")
        if self.state_policy not in {"reset", "carry_inputs", "reset_each_input"}:
            raise ValueError("state_policy must be reset, carry_inputs, or reset_each_input")
        if self.backend != BACKEND_NAME:
            raise ValueError(f"unsupported QRC backend: {self.backend}")

    @property
    def delta_tau(self) -> float:
        return self.tau / self.virtual_nodes

    @property
    def checksum(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class QuantumReservoir:
    """Transform a causal input sequence into label-free exact observables."""

    def __init__(
        self,
        input_size: int,
        config: QRCConfig,
        tolerances: NumericalTolerances | None = None,
    ) -> None:
        config.validate()
        if input_size <= 0:
            raise ValueError("QRC input_size must be positive")
        self.input_size = input_size
        self.config = config
        self.hamiltonian: HamiltonianDefinition = generate_hamiltonian(
            config.n_qubits,
            seed=config.reservoir_seed,
            graph=config.graph,
            chords=config.chords,
            j_strength=config.j_strength,
            h_strength=config.h_strength,
            h_min_factor=config.h_min_factor,
            h_max_factor=config.h_max_factor,
        )
        self.backend = ExactDensityMatrixBackend(
            self.hamiltonian.matrix,
            config.n_qubits,
            config.delta_tau,
            tolerances,
        )
        self.input_projection = input_projection(
            config.virtual_nodes, input_size, config.reservoir_seed
        )
        self.observables = ObservableSet.build(
            config.n_qubits, self.hamiltonian.edges, config.virtual_nodes
        )
        self._state = initial_density_matrix(config.n_qubits)
        self._angles: list[float] = []
        self._connected_absolute_sum = 0.0
        self._connected_count = 0
        self._state_generation_seconds = 0.0

    def reset_state(self) -> None:
        self._state = initial_density_matrix(self.config.n_qubits)

    def get_state(self) -> NDArray[np.complex128]:
        return self._state.copy()

    def set_state(self, state: NDArray[np.complex128]) -> None:
        self._state = self.backend.validate_state(state, context="assigned reservoir state")

    def step(self, input_row: NDArray[np.float64]) -> NDArray[np.float64]:
        """Process one chronological input and return all slice observables."""

        row = np.asarray(input_row, dtype=float).reshape(-1)
        if row.shape != (self.input_size,) or not np.isfinite(row).all():
            raise ValueError("QRC input row has the wrong shape or non-finite values")
        features: list[float] = []
        for projection_row in self.input_projection:
            theta = float(self.config.input_scaling * np.dot(projection_row, row))
            self._angles.append(theta)
            reset_state = reset_and_encode_input(self._state, theta, self.config.n_qubits, q_in=0)
            self._state = self.backend.validate_state(reset_state, context="reset state")
            self._state = self.backend.evolve(self._state)
            values, connected = self.observables.expectations(self._state)
            features.extend(float(value) for value in values)
            self._connected_absolute_sum += float(np.sum(np.abs(connected)))
            self._connected_count += len(connected)
        return np.asarray(features, dtype=float)

    def transform(
        self,
        inputs: NDArray[np.float64],
        *,
        reset: bool = False,
        reset_each_input: bool = False,
    ) -> NDArray[np.float64]:
        """Return chronological features; reservoir dynamics never consume labels."""

        values = np.asarray(inputs, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.input_size:
            raise ValueError("QRC input matrix has the wrong shape")
        if not np.isfinite(values).all():
            raise ValueError("QRC inputs must be finite")
        if reset:
            self.reset_state()
        started = time.perf_counter()
        output = np.empty((len(values), self.observables.raw_feature_dimension), dtype=float)
        for index, row in enumerate(values):
            if reset_each_input:
                self.reset_state()
            output[index] = self.step(row)
        self._state_generation_seconds += time.perf_counter() - started
        return output

    def angle_diagnostics(self) -> dict[str, float | int]:
        values = np.asarray(self._angles, dtype=float)
        if not len(values):
            return {"count": 0}
        absolute = np.abs(values)
        return {
            "count": len(values),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=0)),
            "fraction_absolute_greater_than_pi": float(np.mean(absolute > np.pi)),
            "fraction_absolute_greater_than_2pi": float(np.mean(absolute > 2.0 * np.pi)),
        }

    def numerical_diagnostics(self) -> dict[str, Any]:
        return {
            **self.backend.diagnostics.as_dict(),
            "angles": self.angle_diagnostics(),
            "mean_absolute_connected_correlation": (
                self._connected_absolute_sum / self._connected_count
                if self._connected_count
                else None
            ),
        }

    def resource_metadata(self) -> dict[str, Any]:
        """Return exact backend and feature-resource metadata."""

        return {
            **self.backend.metadata(),
            "reservoir_seed": self.config.reservoir_seed,
            "state_policy": self.config.state_policy,
            "configuration_checksum": self.config.checksum,
            "hamiltonian_checksum": self.hamiltonian.checksum,
            "input_projection_shape": list(self.input_projection.shape),
            "input_projection_checksum": array_checksum(self.input_projection),
            "raw_feature_dimension": self.observables.raw_feature_dimension,
            "observable_checksum": self.observables.checksum,
            "state_generation_seconds": self._state_generation_seconds,
            "labels_consumed": False,
            "explicit_feedback": False,
        }


def split_qrc_features(
    reservoir: QuantumReservoir,
    X_train: NDArray[np.float64],
    X_validation: NDArray[np.float64],
    X_test: NDArray[np.float64] | None,
    state_policy: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Apply reset or continuous-input temporal policy without label access."""

    if state_policy == "carry_inputs":
        train = reservoir.transform(X_train, reset=True)
        validation = reservoir.transform(X_validation, reset=False)
        test = reservoir.transform(X_test, reset=False) if X_test is not None else None
    elif state_policy == "reset":
        train = reservoir.transform(X_train, reset=True)
        validation = reservoir.transform(X_validation, reset=True)
        test = reservoir.transform(X_test, reset=True) if X_test is not None else None
    elif state_policy == "reset_each_input":
        train = reservoir.transform(X_train, reset_each_input=True)
        validation = reservoir.transform(X_validation, reset_each_input=True)
        test = reservoir.transform(X_test, reset_each_input=True) if X_test is not None else None
    else:
        raise ValueError("state_policy must be reset, carry_inputs, or reset_each_input")
    return train, validation, test
