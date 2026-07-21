"""Inspectable exact density-matrix backend for the first QRC stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm  # type: ignore[import-untyped]

from qtyche_qrc.models.qrc.encoding import initial_density_matrix

BACKEND_NAME = "numpy_density_matrix_exact"
BACKEND_VERSION = "1.0"


@dataclass(frozen=True)
class NumericalTolerances:
    """Explicit thresholds for rejecting or correcting exact-backend states."""

    trace_atol: float = 1e-10
    hermiticity_atol: float = 1e-10
    positivity_atol: float = 1e-10
    unitary_atol: float = 1e-10

    def validate(self) -> None:
        if min(asdict(self).values()) <= 0:
            raise ValueError("all numerical tolerances must be positive")


@dataclass
class NumericalDiagnostics:
    """Accumulated state-validity evidence; failures are never suppressed."""

    states_validated: int = 0
    trace_corrections: int = 0
    non_finite_states: int = 0
    maximum_trace_error_before_correction: float = 0.0
    maximum_hermiticity_error: float = 0.0
    minimum_eigenvalue: float = 1.0
    unitary_error: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "states_validated": self.states_validated,
            "trace_corrections": self.trace_corrections,
            "non_finite_states": self.non_finite_states,
            "maximum_trace_error_before_correction": self.maximum_trace_error_before_correction,
            "maximum_hermiticity_error": self.maximum_hermiticity_error,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "unitary_error": self.unitary_error,
        }


def trace_distance(first: NDArray[np.complex128], second: NDArray[np.complex128]) -> float:
    """Compute one-half the trace norm of the difference of density matrices."""

    left = np.asarray(first, dtype=complex)
    right = np.asarray(second, dtype=complex)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError("trace-distance states must be same-size square matrices")
    difference = (left - right + (left - right).conj().T) / 2.0
    eigenvalues = np.linalg.eigvalsh(difference)
    distance = float(0.5 * np.sum(np.abs(eigenvalues)))
    if distance < -1e-12 or distance > 1.0 + 1e-10:
        raise ValueError(f"invalid density-matrix trace distance: {distance}")
    return float(np.clip(distance, 0.0, 1.0))


class ExactDensityMatrixBackend:
    """Exact noiseless evolution with cached U=exp(-i H delta_tau)."""

    def __init__(
        self,
        hamiltonian: NDArray[np.complex128],
        n_qubits: int,
        delta_tau: float,
        tolerances: NumericalTolerances | None = None,
    ) -> None:
        if n_qubits <= 0 or n_qubits > 6:
            raise ValueError("exact density-matrix QRC safety limit is n_qubits <= 6")
        if not np.isfinite(delta_tau) or delta_tau <= 0:
            raise ValueError("delta_tau must be finite and positive")
        dimension = 2**n_qubits
        self.hamiltonian = np.asarray(hamiltonian, dtype=complex)
        if self.hamiltonian.shape != (dimension, dimension):
            raise ValueError("Hamiltonian dimension disagrees with n_qubits")
        if not np.allclose(self.hamiltonian, self.hamiltonian.conj().T, atol=1e-12):
            raise ValueError("Hamiltonian must be Hermitian")
        self.n_qubits = n_qubits
        self.delta_tau = delta_tau
        self.tolerances = tolerances or NumericalTolerances()
        self.tolerances.validate()
        # Accelerate-backed NumPy can emit spurious floating-point warnings from
        # SciPy's internal Pade matmuls. The finite/unitarity checks immediately
        # below remain authoritative and reject any genuinely invalid result.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            self.unitary = np.asarray(expm(-1.0j * self.hamiltonian * delta_tau), dtype=complex)
        identity = np.eye(dimension, dtype=complex)
        product = np.einsum("ji,jk->ik", self.unitary.conj(), self.unitary)
        unitary_error = float(np.max(np.abs(product - identity)))
        if not np.isfinite(unitary_error) or unitary_error > self.tolerances.unitary_atol:
            raise ValueError(f"cached evolution operator is not unitary: {unitary_error}")
        self.diagnostics = NumericalDiagnostics(unitary_error=unitary_error)
        self.validate_state(initial_density_matrix(n_qubits), context="initial state")

    def validate_state(
        self, state: NDArray[np.complex128], *, context: str
    ) -> NDArray[np.complex128]:
        """Reject invalid states and correct only trace drift within trace_atol."""

        value = np.asarray(state, dtype=complex)
        dimension = 2**self.n_qubits
        if value.shape != (dimension, dimension):
            raise ValueError(f"{context} has the wrong density-matrix shape")
        if not np.isfinite(value).all():
            self.diagnostics.non_finite_states += 1
            raise ValueError(f"{context} contains non-finite density-matrix entries")
        trace = np.trace(value)
        trace_error = float(abs(trace - 1.0))
        hermiticity_error = float(np.max(np.abs(value - value.conj().T)))
        hermitian = (value + value.conj().T) / 2.0
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hermitian)))
        self.diagnostics.states_validated += 1
        self.diagnostics.maximum_trace_error_before_correction = max(
            self.diagnostics.maximum_trace_error_before_correction, trace_error
        )
        self.diagnostics.maximum_hermiticity_error = max(
            self.diagnostics.maximum_hermiticity_error, hermiticity_error
        )
        self.diagnostics.minimum_eigenvalue = min(
            self.diagnostics.minimum_eigenvalue, minimum_eigenvalue
        )
        if (
            abs(float(trace.imag)) > self.tolerances.trace_atol
            or trace_error > self.tolerances.trace_atol
        ):
            raise ValueError(f"{context} trace differs from one by {trace_error}")
        if hermiticity_error > self.tolerances.hermiticity_atol:
            raise ValueError(f"{context} is not Hermitian: {hermiticity_error}")
        if minimum_eigenvalue < -self.tolerances.positivity_atol:
            raise ValueError(f"{context} is not positive semidefinite: {minimum_eigenvalue}")
        if trace_error > 0.0:
            self.diagnostics.trace_corrections += 1
            value = value / trace
        return np.asarray(value, dtype=complex)

    def evolve(self, state: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Apply the cached exact unitary and validate the evolved state."""

        left = np.einsum("ij,jk->ik", self.unitary, np.asarray(state, dtype=complex))
        evolved = np.einsum("ij,jk->ik", left, self.unitary.conj().T)
        return self.validate_state(np.asarray(evolved, dtype=complex), context="evolved state")

    def metadata(self) -> dict[str, Any]:
        """Return backend, scaling, exactness, and tolerance metadata."""

        dimension = 2**self.n_qubits
        return {
            "backend": BACKEND_NAME,
            "backend_version": BACKEND_VERSION,
            "exact": True,
            "noiseless": True,
            "finite_shots": False,
            "physical_noise": False,
            "hardware_execution": False,
            "n_qubits": self.n_qubits,
            "density_matrix_shape": [dimension, dimension],
            "estimated_peak_density_matrix_bytes": 3 * dimension * dimension * 16,
            "delta_tau": self.delta_tau,
            "tolerances": asdict(self.tolerances),
        }
