"""Small, explicit YAML configuration loader and validator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration cannot satisfy the project contract."""


@dataclass(frozen=True)
class ExperimentSettings:
    """Validated settings shared by every executable experiment."""

    name: str
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class ProjectConfig:
    """A validated configuration together with its source and raw content."""

    source: Path
    schema_version: int
    experiment: ExperimentSettings
    raw: Mapping[str, Any]


def _required_mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be a YAML mapping")
    return value


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate the common experiment fields from a YAML file."""

    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"configuration file does not exist: {source}")

    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not read configuration {source}: {exc}") from exc

    root = _required_mapping(loaded, "configuration root")
    schema_version = root.get("schema_version")
    if schema_version != 1:
        raise ConfigError("schema_version must be the integer 1")

    experiment = _required_mapping(root.get("experiment"), "experiment")
    name = experiment.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("experiment.name must be a non-empty string")

    seed = experiment.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ConfigError("experiment.seed must be an integer in [0, 2**32)")

    output_dir = experiment.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigError("experiment.output_dir must be a non-empty path string")

    return ProjectConfig(
        source=source.resolve(),
        schema_version=schema_version,
        experiment=ExperimentSettings(
            name=name.strip(),
            seed=seed,
            output_dir=Path(output_dir),
        ),
        raw=root,
    )
