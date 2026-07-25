"""Finite-shot measurement and explicit simulated noise channels for QRC."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qtyche_qrc.models.qrc.hamiltonian import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    Edge,
    operator_on_qubit,
)


@dataclass(frozen=True)
class QRCMeasurementConfig:
    """Measurement budget and controlled simulated channel probabilities."""

    shots: int | None = None
    measurement_seed: int | None = None
    depolarizing_probability: float = 0.0
    measurement_bit_flip_probability: float = 0.0

    def validate(self) -> None:
        if self.shots is None:
            if self.measurement_seed is not None:
                raise ValueError("analytic expectations must not specify a measurement seed")
            if self.measurement_bit_flip_probability != 0.0:
                raise ValueError("measurement bit flips require a finite shot count")
        elif isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots <= 0:
            raise ValueError("shots must be a positive integer or null for analytic expectations")
        elif (
            self.measurement_seed is None
            or isinstance(self.measurement_seed, bool)
            or not isinstance(self.measurement_seed, int)
        ):
            raise ValueError("finite-shot measurement requires an integer measurement seed")
        if self.measurement_seed is not None and self.measurement_seed < 0:
            raise ValueError("measurement_seed must be non-negative")
        for name, probability in (
            ("depolarizing_probability", self.depolarizing_probability),
            ("measurement_bit_flip_probability", self.measurement_bit_flip_probability),
        ):
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

    @property
    def analytic_expectations(self) -> bool:
        return self.shots is None

    @property
    def checksum(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "analytic_expectations": self.analytic_expectations,
            "exact_state_evolution": True,
            "physical_qpu_execution": False,
            "hardware_calibrated_noise": False,
            "depolarizing_channel_location": (
                "after each virtual-node unitary and before measurement"
            ),
            "depolarizing_probability_scope": "per reservoir qubit per virtual node",
            "measurement_bit_flip_scope": "per sampled output bit",
            "joint_commuting_observable_sampling": True,
        }


def measurement_rng(seed: int, *, stream: int = 0) -> np.random.Generator:
    """Return a deterministic measurement RNG with domain and stream separation."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(stream, bool)
        or not isinstance(stream, int)
        or stream < 0
    ):
        raise ValueError("measurement seed and stream must be non-negative integers")
    return np.random.default_rng(np.random.SeedSequence([seed, 0x5152434D, stream]))


@lru_cache(maxsize=6)
def _embedded_paulis(
    n_qubits: int,
) -> tuple[tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]], ...]:
    if n_qubits <= 0 or n_qubits > 6:
        raise ValueError("n_qubits must lie in [1, 6]")
    return tuple(
        (
            operator_on_qubit(PAULI_X, qubit, n_qubits),
            operator_on_qubit(PAULI_Y, qubit, n_qubits),
            operator_on_qubit(PAULI_Z, qubit, n_qubits),
        )
        for qubit in range(n_qubits)
    )


def apply_local_depolarizing_channel(
    state: NDArray[np.complex128],
    *,
    n_qubits: int,
    probability: float,
) -> NDArray[np.complex128]:
    """Apply independent single-qubit replacement-form depolarisation.

    For each qubit, the channel is
    E_p(rho) = (1-p) rho + p I/2 tensor Tr_qubit(rho).
    It is implemented through the equivalent Pauli mixture with probabilities
    (1-3p/4, p/4, p/4, p/4), then composed over all reservoir qubits.
    """

    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("depolarizing probability must lie in [0, 1]")
    if not 1 <= n_qubits <= 6:
        raise ValueError("n_qubits must lie in [1, 6]")
    dimension = 2**n_qubits
    value = np.asarray(state, dtype=complex)
    if value.shape != (dimension, dimension):
        raise ValueError("depolarizing channel state shape disagrees with n_qubits")
    if probability == 0.0:
        return np.asarray(value.copy(), dtype=complex)
    identity_weight = 1.0 - 3.0 * probability / 4.0
    pauli_weight = probability / 4.0
    output = value
    for operators in _embedded_paulis(n_qubits):
        mixed = identity_weight * output
        for operator in operators:
            mixed = mixed + pauli_weight * (operator @ output @ operator)
        output = np.asarray(mixed, dtype=complex)
    return np.asarray(output, dtype=complex)


def computational_basis_probabilities(
    state: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Extract validated computational-basis probabilities from a density matrix."""

    value = np.asarray(state, dtype=complex)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("measurement state must be a square density matrix")
    diagonal = np.diag(value)
    if np.max(np.abs(diagonal.imag), initial=0.0) > 1e-10:
        raise ValueError("density-matrix diagonal has non-negligible imaginary entries")
    probabilities = np.asarray(diagonal.real, dtype=float)
    if not np.isfinite(probabilities).all() or probabilities.min(initial=0.0) < -1e-10:
        raise ValueError("density-matrix diagonal is not a valid probability vector")
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0 or abs(total - 1.0) > 1e-9:
        raise ValueError("computational-basis probabilities do not sum to one")
    return np.asarray(probabilities / total, dtype=float)


def basis_indices_to_bits(indices: NDArray[np.integer[Any]], n_qubits: int) -> NDArray[np.int8]:
    """Decode q0-most-significant computational-basis indices into bits."""

    if not 1 <= n_qubits <= 6:
        raise ValueError("n_qubits must lie in [1, 6]")
    values = np.asarray(indices)
    if values.ndim != 1 or np.any(values < 0) or np.any(values >= 2**n_qubits):
        raise ValueError("sampled basis indices are invalid")
    shifts = np.arange(n_qubits - 1, -1, -1, dtype=np.int64)
    return np.asarray((values[:, None] >> shifts[None, :]) & 1, dtype=np.int8)


def observable_estimates_from_bits(
    bits: NDArray[np.integer[Any]],
    *,
    edges: tuple[Edge, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate all Z and ZZ values jointly from one shared bitstring batch."""

    samples = np.asarray(bits, dtype=np.int8)
    if samples.ndim != 2 or not len(samples) or np.any((samples != 0) & (samples != 1)):
        raise ValueError("measurement bits must be a non-empty two-dimensional binary array")
    z_shots = np.asarray(1 - 2 * samples, dtype=float)
    z_values = np.asarray(z_shots.mean(axis=0), dtype=float)
    zz_values = np.asarray(
        [(z_shots[:, first] * z_shots[:, second]).mean() for first, second in edges],
        dtype=float,
    )
    values = np.concatenate((z_values, zz_values))
    connected = np.asarray(
        [
            value - z_values[first] * z_values[second]
            for value, (first, second) in zip(zz_values, edges)
        ],
        dtype=float,
    )
    return values, connected


def sample_commuting_observables(
    state: NDArray[np.complex128],
    *,
    n_qubits: int,
    edges: tuple[Edge, ...],
    shots: int,
    rng: np.random.Generator,
    measurement_bit_flip_probability: float = 0.0,
    bit_flip_rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample one bitstring batch and reuse it for every diagonal observable."""

    if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if (
        not np.isfinite(measurement_bit_flip_probability)
        or not 0.0 <= measurement_bit_flip_probability <= 1.0
    ):
        raise ValueError("measurement bit-flip probability must lie in [0, 1]")
    probabilities = computational_basis_probabilities(state)
    if len(probabilities) != 2**n_qubits:
        raise ValueError("measurement state dimension disagrees with n_qubits")
    indices = np.asarray(
        rng.choice(len(probabilities), size=shots, replace=True, p=probabilities),
        dtype=np.int64,
    )
    bits = basis_indices_to_bits(indices, n_qubits)
    if measurement_bit_flip_probability > 0.0:
        flip_generator = bit_flip_rng if bit_flip_rng is not None else rng
        flips = flip_generator.random(bits.shape) < measurement_bit_flip_probability
        bits = np.asarray(np.bitwise_xor(bits, flips), dtype=np.int8)
    return observable_estimates_from_bits(bits, edges=edges)
