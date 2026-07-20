"""Quantum reservoir interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class QuantumReservoir(ABC):
    """Transform a causal input sequence into fixed-reservoir observables."""

    @abstractmethod
    def transform(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return reservoir features without training the reservoir dynamics."""

    @abstractmethod
    def resource_metadata(self) -> dict[str, Any]:
        """Return qubits, circuit, shot, backend, seed, and runtime metadata."""
