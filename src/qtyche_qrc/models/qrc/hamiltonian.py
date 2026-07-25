"""Sparse disordered transverse-field Ising Hamiltonian construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

ComplexMatrix = NDArray[np.complex128]
Edge = tuple[int, int]

IDENTITY = np.eye(2, dtype=complex)
PAULI_X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
PAULI_Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def pauli_operators() -> dict[str, ComplexMatrix]:
    """Return independent copies of the one-qubit Pauli operators."""

    return {
        "I": np.asarray(IDENTITY.copy(), dtype=complex),
        "X": np.asarray(PAULI_X.copy(), dtype=complex),
        "Y": np.asarray(PAULI_Y.copy(), dtype=complex),
        "Z": np.asarray(PAULI_Z.copy(), dtype=complex),
    }


def ring_edges(n_qubits: int) -> tuple[Edge, ...]:
    """Return the deterministic undirected nearest-neighbour ring ordering."""

    if n_qubits < 2:
        raise ValueError("a ring QRC requires at least two qubits")
    if n_qubits == 2:
        return ((0, 1),)
    return tuple((index, (index + 1) % n_qubits) for index in range(n_qubits))


def graph_edges(n_qubits: int, graph: str, chords: tuple[Edge, ...] = ()) -> tuple[Edge, ...]:
    """Build ring or ring-plus-chords edges with stable canonical ordering."""

    base = list(ring_edges(n_qubits))
    if graph == "ring":
        if chords:
            raise ValueError("chords may only be supplied for ring_plus_chords")
        return tuple(base)
    if graph != "ring_plus_chords":
        raise ValueError("graph must be ring or ring_plus_chords")
    canonical = {tuple(sorted(edge)) for edge in base}
    for first, second in chords:
        if first == second or min(first, second) < 0 or max(first, second) >= n_qubits:
            raise ValueError(f"invalid graph chord: {(first, second)}")
        edge: Edge = (min(first, second), max(first, second))
        if edge not in canonical:
            base.append(edge)
            canonical.add(edge)
    return tuple(base)


def operator_on_qubit(operator: ComplexMatrix, qubit: int, n_qubits: int) -> ComplexMatrix:
    """Embed an operator using q0 as the most-significant Kronecker factor."""

    if operator.shape != (2, 2) or not 0 <= qubit < n_qubits:
        raise ValueError("operator embedding arguments are invalid")
    result = np.asarray([[1.0]], dtype=complex)
    for index in range(n_qubits):
        result = np.kron(result, operator if index == qubit else IDENTITY)
    return np.asarray(result, dtype=complex)


def zz_operator(first: int, second: int, n_qubits: int) -> ComplexMatrix:
    """Return Z_first Z_second in the documented q0-most-significant ordering."""

    if first == second or not 0 <= first < n_qubits or not 0 <= second < n_qubits:
        raise ValueError("ZZ operator qubits are invalid")
    result = np.asarray([[1.0]], dtype=complex)
    for index in range(n_qubits):
        result = np.kron(result, PAULI_Z if index in {first, second} else IDENTITY)
    return np.asarray(result, dtype=complex)


@dataclass(frozen=True)
class HamiltonianDefinition:
    """Frozen Hamiltonian matrix and its sampled disordered parameters."""

    matrix: ComplexMatrix
    edges: tuple[Edge, ...]
    couplings: NDArray[np.float64]
    fields: NDArray[np.float64]
    seed: int
    graph: str
    j_strength: float
    h_strength: float
    h_min_factor: float
    h_max_factor: float

    @property
    def checksum(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.matrix).view(np.uint8).tobytes())
        return digest.hexdigest()

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe sampled-parameter provenance."""

        return {
            "definition": "sum_(i,j in E) J_ij Z_i Z_j + sum_i h_i X_i",
            "qubit_ordering": "q0 is the most-significant Kronecker factor",
            "graph": self.graph,
            "edges": [list(edge) for edge in self.edges],
            "couplings": [float(value) for value in self.couplings],
            "fields": [float(value) for value in self.fields],
            "seed": self.seed,
            "j_strength": self.j_strength,
            "h_strength": self.h_strength,
            "h_min_factor": self.h_min_factor,
            "h_max_factor": self.h_max_factor,
            "matrix_shape": list(self.matrix.shape),
            "matrix_checksum": self.checksum,
        }

    @property
    def parameter_checksum(self) -> str:
        payload = json.dumps(self.metadata(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_hamiltonian(
    n_qubits: int,
    *,
    seed: int,
    graph: str = "ring",
    chords: tuple[Edge, ...] = (),
    j_strength: float = 1.0,
    h_strength: float = 1.0,
    h_min_factor: float = 0.5,
    h_max_factor: float = 1.5,
) -> HamiltonianDefinition:
    """Sample once and freeze a sparse disordered Ising Hamiltonian."""

    if n_qubits > 6:
        raise ValueError("exact density-matrix QRC safety limit is n_qubits <= 6")
    if j_strength < 0 or h_strength <= 0:
        raise ValueError("j_strength must be non-negative and h_strength positive")
    if h_min_factor <= 0 or h_max_factor <= h_min_factor:
        raise ValueError("field-factor interval must be positive and increasing")
    edges = graph_edges(n_qubits, graph, chords)
    rng = np.random.default_rng(seed)
    couplings = np.asarray(rng.uniform(-1.0, 1.0, len(edges)) * j_strength, dtype=float)
    fields = np.asarray(
        rng.uniform(h_min_factor, h_max_factor, n_qubits) * h_strength,
        dtype=float,
    )
    dimension = 2**n_qubits
    matrix = np.zeros((dimension, dimension), dtype=complex)
    for coupling, (first, second) in zip(couplings, edges):
        matrix += coupling * zz_operator(first, second, n_qubits)
    for qubit, field in enumerate(fields):
        matrix += field * operator_on_qubit(PAULI_X, qubit, n_qubits)
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12):
        raise ValueError("generated Hamiltonian is not Hermitian")
    return HamiltonianDefinition(
        np.asarray(matrix, dtype=complex),
        edges,
        couplings,
        fields,
        seed,
        graph,
        j_strength,
        h_strength,
        h_min_factor,
        h_max_factor,
    )
