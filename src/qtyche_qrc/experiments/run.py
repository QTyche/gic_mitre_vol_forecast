"""Experiment runner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from qtyche_qrc.config import ProjectConfig


class ExperimentRunner(ABC):
    """Execute one saved configuration and persist every output it produces."""

    @abstractmethod
    def run(self, config: ProjectConfig) -> list[Path]:
        """Execute the experiment and return all persisted output paths."""
