"""Label-free sequential angle encoding and reset-and-reinject channel."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray


def initial_density_matrix(n_qubits: int) -> NDArray[np.complex128]:
    """Return |0...0><0...0| in q0-most-significant ordering."""

    if n_qubits <= 0 or n_qubits > 6:
        raise ValueError("n_qubits must lie in [1, 6]")
    dimension = 2**n_qubits
    state = np.zeros((dimension, dimension), dtype=complex)
    state[0, 0] = 1.0
    return np.asarray(state, dtype=complex)


def ry_state(theta: float) -> NDArray[np.complex128]:
    """Return Ry(theta)|0> without clipping or wrapping theta."""

    if not np.isfinite(theta):
        raise ValueError("encoding angle must be finite")
    return np.asarray([np.cos(theta / 2.0), np.sin(theta / 2.0)], dtype=complex)


def partial_trace_qubit(
    state: NDArray[np.complex128], qubit: int, n_qubits: int
) -> NDArray[np.complex128]:
    """Trace out one qubit using explicit row/column tensor axes."""

    dimension = 2**n_qubits
    value = np.asarray(state, dtype=complex)
    if value.shape != (dimension, dimension) or not 0 <= qubit < n_qubits:
        raise ValueError("partial-trace state or qubit is invalid")
    tensor = value.reshape((2,) * (2 * n_qubits))
    reduced = np.trace(tensor, axis1=qubit, axis2=n_qubits + qubit)
    return np.asarray(reduced.reshape((dimension // 2, dimension // 2)), dtype=complex)


def reset_and_encode_input(
    previous: NDArray[np.complex128], theta: float, n_qubits: int, q_in: int = 0
) -> NDArray[np.complex128]:
    """Apply |psi><psi|_q0 tensor Tr_q0(previous) exactly."""

    if q_in != 0:
        raise ValueError("this exact QRC stage designates q_in = 0")
    reduced = partial_trace_qubit(previous, q_in, n_qubits)
    vector = ry_state(theta)
    encoded = np.outer(vector, vector.conj())
    return np.asarray(np.kron(encoded, reduced), dtype=complex)


def input_projection(virtual_nodes: int, input_size: int, seed: int) -> NDArray[np.float64]:
    """Create deterministic unit-row-norm random projection B independent of labels."""

    if virtual_nodes <= 0 or input_size <= 0:
        raise ValueError("virtual_nodes and input_size must be positive")
    seed_sequence = np.random.SeedSequence([seed, 0x51524342])
    rng = np.random.default_rng(seed_sequence)
    projection = np.asarray(rng.normal(size=(virtual_nodes, input_size)), dtype=float)
    norms = np.linalg.norm(projection, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("could not normalize QRC input projection")
    return np.asarray(projection / norms[:, None], dtype=float)


def array_checksum(value: NDArray[np.generic]) -> str:
    """Return a stable checksum over shape, dtype, and contiguous array bytes."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()
