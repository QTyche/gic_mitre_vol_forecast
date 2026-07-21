"""Deterministic QRC observables and temporal-multiplexed feature ordering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qtyche_qrc.models.qrc.hamiltonian import PAULI_Z, Edge, operator_on_qubit, zz_operator


@dataclass(frozen=True)
class ObservableSet:
    """All single-Z and interaction-edge ZZ observables in fixed order."""

    n_qubits: int
    edges: tuple[Edge, ...]
    virtual_nodes: int
    names: tuple[str, ...]
    matrices: tuple[NDArray[np.complex128], ...]

    @classmethod
    def build(cls, n_qubits: int, edges: tuple[Edge, ...], virtual_nodes: int) -> ObservableSet:
        if virtual_nodes <= 0:
            raise ValueError("virtual_nodes must be positive")
        names = tuple([f"Z_{index}" for index in range(n_qubits)]) + tuple(
            f"Z_{first}Z_{second}" for first, second in edges
        )
        matrices = tuple(
            [operator_on_qubit(PAULI_Z, index, n_qubits) for index in range(n_qubits)]
            + [zz_operator(first, second, n_qubits) for first, second in edges]
        )
        return cls(n_qubits, edges, virtual_nodes, names, matrices)

    @property
    def raw_feature_dimension(self) -> int:
        return self.virtual_nodes * len(self.names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(
            f"slice_{slice_index}:{name}"
            for slice_index in range(self.virtual_nodes)
            for name in self.names
        )

    @property
    def checksum(self) -> str:
        encoded = json.dumps(self.metadata(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def metadata(self) -> dict[str, Any]:
        """Return definitions without recursively including their checksum."""

        return {
            "qubit_ordering": "q0 is the most-significant Kronecker factor",
            "within_slice_ordering": list(self.names),
            "slice_ordering": list(range(self.virtual_nodes)),
            "feature_ordering": list(self.feature_names),
            "raw_feature_dimension": self.raw_feature_dimension,
            "connected_correlations_in_readout": False,
        }

    def expectations(
        self, state: NDArray[np.complex128]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Measure exact observables and diagnostic-only connected correlations."""

        values = np.asarray(
            [float(np.real(np.einsum("ij,ji->", state, matrix))) for matrix in self.matrices],
            dtype=float,
        )
        if not np.isfinite(values).all() or np.any(np.abs(values) > 1.0 + 1e-9):
            raise ValueError("QRC observable expectation is outside its physical range")
        z_values = values[: self.n_qubits]
        zz_values = values[self.n_qubits :]
        connected = np.asarray(
            [
                value - z_values[first] * z_values[second]
                for value, (first, second) in zip(zz_values, self.edges)
            ],
            dtype=float,
        )
        return values, connected
