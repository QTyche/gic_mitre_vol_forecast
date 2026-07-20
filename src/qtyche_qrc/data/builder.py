"""Dataset builder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class DatasetBuilder(ABC):
    """Build a versioned dataset without leaking future observations."""

    @abstractmethod
    def build(self, output_dir: Path) -> Path:
        """Build, validate, and persist the dataset, returning its path."""
