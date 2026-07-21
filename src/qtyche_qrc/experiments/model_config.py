"""Configuration contract for classical baseline experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True)
class ModelExperimentConfig:
    """Validated model experiment configuration."""

    source: Path
    project_root: Path
    name: str
    seed: int
    output_root: Path
    processed_dir: Path
    model_type: str
    task: str
    parameters: dict[str, Any]
    search_enabled: bool
    selection_metric: str
    maximum_trials: int
    search_space: dict[str, list[Any]]
    transition_threshold: float
    variance_floor: float
    raw: dict[str, Any]


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a YAML mapping")
    return cast(Mapping[str, Any], value)


def _text(mapping: Mapping[str, Any], key: str, location: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location}.{key} must be a non-empty string")
    return value


def load_model_config(path: Path) -> ModelExperimentConfig:
    """Load a model configuration without duplicating the frozen data contract."""

    source = path.resolve()
    if not source.is_file():
        raise ValueError(f"model configuration does not exist: {source}")
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(value, "configuration")
    if root.get("schema_version") != 1:
        raise ValueError("model configuration schema_version must be 1")
    experiment = _mapping(root.get("experiment"), "experiment")
    project_root_setting = _text(experiment, "project_root", "experiment")
    project_root = (source.parent / project_root_setting).resolve()
    name = _text(experiment, "name", "experiment")
    seed = experiment.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("experiment.seed must be an integer")
    output_root = (project_root / _text(experiment, "output_root", "experiment")).resolve()

    data = _mapping(root.get("data"), "data")
    processed_dir = (project_root / _text(data, "processed_dir", "data")).resolve()
    manifest_setting = _text(data, "manifest", "data")
    expected_manifest = (project_root / manifest_setting).resolve()
    if expected_manifest != processed_dir / "data_manifest.json":
        raise ValueError("data.manifest must reference processed_dir/data_manifest.json")

    model = _mapping(root.get("model"), "model")
    model_type = _text(model, "type", "model")
    task = _text(model, "task", "model")
    allowed_models = {
        "majority_classifier",
        "regime_persistence",
        "logistic_regression",
        "rv_persistence",
        "esn_classifier",
        "esn_regressor",
        "qrc_classifier",
        "qrc_regressor",
    }
    if model_type not in allowed_models:
        raise ValueError(f"unsupported model.type: {model_type}")
    if task not in {"regime_classification", "rv_regression"}:
        raise ValueError("model.task must be regime_classification or rv_regression")
    parameters = dict(_mapping(model.get("parameters", {}), "model.parameters"))

    search = _mapping(root.get("search", {}), "search")
    search_enabled = bool(search.get("enabled", False))
    default_metric = "macro_f1" if task == "regime_classification" else "qlike"
    selection_metric = str(search.get("selection_metric", default_metric))
    if selection_metric != default_metric:
        raise ValueError(f"default task selection metric must be {default_metric}")
    maximum_trials = search.get("maximum_trials", 1)
    if (
        isinstance(maximum_trials, bool)
        or not isinstance(maximum_trials, int)
        or maximum_trials <= 0
    ):
        raise ValueError("search.maximum_trials must be a positive integer")
    space_value = _mapping(search.get("space", {}), "search.space")
    search_space: dict[str, list[Any]] = {}
    for key, choices in space_value.items():
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"search.space.{key} must be a non-empty list")
        search_space[key] = choices

    evaluation = _mapping(root.get("evaluation", {}), "evaluation")
    transition_threshold = float(evaluation.get("transition_threshold", 0.5))
    variance_floor = float(evaluation.get("variance_floor", 1e-12))
    if not 0 <= transition_threshold <= 1 or variance_floor <= 0:
        raise ValueError("evaluation thresholds are invalid")

    return ModelExperimentConfig(
        source=source,
        project_root=project_root,
        name=name,
        seed=seed,
        output_root=output_root,
        processed_dir=processed_dir,
        model_type=model_type,
        task=task,
        parameters=parameters,
        search_enabled=search_enabled,
        selection_metric=selection_metric,
        maximum_trials=maximum_trials,
        search_space=search_space,
        transition_threshold=transition_threshold,
        variance_floor=variance_floor,
        raw=dict(root),
    )
