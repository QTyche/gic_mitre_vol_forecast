"""Validated configuration for the versioned market-data contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import yaml

from qtyche_qrc.config import ConfigError, load_config

CANONICAL_COLUMNS = (
    "date",
    "spy_open",
    "spy_high",
    "spy_low",
    "spy_close",
    "spy_adjusted_close",
    "spy_volume",
    "vix_close",
    "qqq_close",
    "qqq_volume",
)


@dataclass(frozen=True)
class DateRange:
    """Inclusive calendar-date range."""

    start: date
    end: date


@dataclass(frozen=True)
class SplitBoundary:
    """Inclusive observation boundaries for one chronological split."""

    name: str
    start: date
    end: date


@dataclass(frozen=True)
class DataPreparationConfig:
    """Complete, validated configuration for deterministic data preparation."""

    source: Path
    split_source: Path
    project_root: Path
    mode: str
    raw_paths: Mapping[str, Path]
    processed_path: Path
    symbols: Mapping[str, str]
    source_dates: DateRange
    observation_dates: DateRange
    required_columns: tuple[str, ...]
    feature_names: tuple[str, ...]
    target_definition_version: str
    target_horizon: int
    annualization_factor: float
    regime_quantiles: tuple[float, float]
    split_boundaries: tuple[SplitBoundary, ...]
    purge_trading_days: int
    missing_data_policy: str
    data_source_type: str
    is_synthetic: bool
    snapshot_id: str | None
    snapshot_manifest_path: Path | None
    provider: str | None
    large_move_threshold: float
    fatal_audit_conditions: tuple[str, ...]


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be a YAML mapping")
    return cast(Mapping[str, Any], value)


def _text(mapping: Mapping[str, Any], key: str, location: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _date_value(value: object, location: str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ConfigError(f"{location} must use YYYY-MM-DD format") from exc
    raise ConfigError(f"{location} must be a date")


def _date_range(mapping: Mapping[str, Any], location: str) -> DateRange:
    start = _date_value(mapping.get("start"), f"{location}.start")
    end = _date_value(mapping.get("end"), f"{location}.end")
    if start > end:
        raise ConfigError(f"{location}.start must not be after its end")
    return DateRange(start, end)


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not read configuration {path}: {exc}") from exc
    return _mapping(value, str(path))


def _string_sequence(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{location} must be a non-empty YAML list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ConfigError(f"every entry in {location} must be a non-empty string")
    return tuple(cast(list[str], value))


def load_data_config(path: str | Path) -> DataPreparationConfig:
    """Load the data and split contracts, rejecting leakage-prone settings."""

    source = Path(path).resolve()
    load_config(source)
    root = _load_yaml_mapping(source)
    data_config = _mapping(root.get("data"), "data")

    project_root_setting = _text(data_config, "project_root", "data")
    project_root = (source.parent / project_root_setting).resolve()
    mode = _text(data_config, "mode", "data")
    if mode not in {"cached_csv", "download"}:
        raise ConfigError("data.mode must be cached_csv or download")

    raw_path_values = _mapping(data_config.get("raw_paths"), "data.raw_paths")
    raw_paths = {
        name: (project_root / _text(raw_path_values, name, "data.raw_paths")).resolve()
        for name in ("spy", "vix", "qqq")
    }
    processed_path = (project_root / _text(data_config, "processed_path", "data")).resolve()

    symbols_value = _mapping(data_config.get("symbols"), "data.symbols")
    symbols = {name: _text(symbols_value, name, "data.symbols") for name in ("spy", "vix", "qqq")}

    source_dates = _date_range(
        _mapping(data_config.get("source_date_range"), "data.source_date_range"),
        "data.source_date_range",
    )
    observation_dates = _date_range(
        _mapping(data_config.get("observation_date_range"), "data.observation_date_range"),
        "data.observation_date_range",
    )
    if not (
        source_dates.start <= observation_dates.start and observation_dates.end <= source_dates.end
    ):
        raise ConfigError("data.observation_date_range must lie within source_date_range")

    required_columns = _string_sequence(
        data_config.get("required_columns"), "data.required_columns"
    )
    missing_columns = sorted(set(CANONICAL_COLUMNS) - set(required_columns))
    if missing_columns:
        raise ConfigError(f"data.required_columns omits canonical columns: {missing_columns}")
    feature_names = _string_sequence(data_config.get("features"), "data.features")

    targets = _mapping(data_config.get("targets"), "data.targets")
    definition_version = _text(targets, "definition_version", "data.targets")
    if definition_version != "qtyche_volatility_regime_v1":
        raise ConfigError(
            "unsupported target definition_version; expected qtyche_volatility_regime_v1"
        )
    horizon = targets.get("horizon_trading_days")
    if horizon != 5:
        raise ConfigError("data.targets.horizon_trading_days must be 5 for definition v1")
    annualization = targets.get("annualization_factor")
    if isinstance(annualization, bool) or not isinstance(annualization, (int, float)):
        raise ConfigError("data.targets.annualization_factor must be numeric")
    if float(annualization) <= 0:
        raise ConfigError("data.targets.annualization_factor must be positive")
    quantiles_value = targets.get("regime_quantiles")
    if not isinstance(quantiles_value, list) or len(quantiles_value) != 2:
        raise ConfigError("data.targets.regime_quantiles must contain exactly two values")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in quantiles_value
    ):
        raise ConfigError("data.targets.regime_quantiles must be numeric")
    quantiles = tuple(float(item) for item in quantiles_value)
    if not 0 < quantiles[0] < quantiles[1] < 1:
        raise ConfigError("regime quantiles must satisfy 0 < q_low < q_high < 1")

    split_setting = _text(data_config, "split_config", "data")
    split_source = Path(split_setting)
    if not split_source.is_absolute():
        split_source = (source.parent / split_source).resolve()
    split_root = _load_yaml_mapping(split_source)
    if split_root.get("schema_version") != 1:
        raise ConfigError("split configuration schema_version must be 1")
    split = _mapping(split_root.get("split"), "split")
    if split.get("strategy") != "chronological" or split.get("shuffle") is not False:
        raise ConfigError("splits must be chronological with shuffle: false")
    if split.get("require_forward_window_containment") is not True:
        raise ConfigError("splits must require complete forward-window containment")
    purge = split.get("purge_trading_days")
    if purge != horizon:
        raise ConfigError("split purge_trading_days must equal the five-day target horizon")
    boundaries_value = _mapping(split.get("boundaries"), "split.boundaries")
    boundaries: list[SplitBoundary] = []
    previous_end: date | None = None
    for name in ("train", "validation", "test"):
        dates = _date_range(_mapping(boundaries_value.get(name), f"split.boundaries.{name}"), name)
        if previous_end is not None and dates.start <= previous_end:
            raise ConfigError("split boundaries must be strictly chronological and non-overlapping")
        if dates.start < observation_dates.start or dates.end > observation_dates.end:
            raise ConfigError(f"{name} split lies outside the observation date range")
        boundaries.append(SplitBoundary(name, dates.start, dates.end))
        previous_end = dates.end

    missing_policy = _text(data_config, "missing_data_policy", "data")
    if missing_policy not in {"drop_secondary_and_report", "error"}:
        raise ConfigError("unsupported data.missing_data_policy")

    source_type_value = data_config.get("data_source_type")
    if source_type_value is None:
        data_source_type = "fixture"
    elif isinstance(source_type_value, str):
        data_source_type = source_type_value.strip()
    else:
        raise ConfigError("data.data_source_type must be fixture or public_market")
    if data_source_type not in {"fixture", "public_market"}:
        raise ConfigError("data.data_source_type must be fixture or public_market")
    synthetic_value = data_config.get("is_synthetic", data_source_type == "fixture")
    if not isinstance(synthetic_value, bool):
        raise ConfigError("data.is_synthetic must be a boolean")
    if synthetic_value != (data_source_type == "fixture"):
        raise ConfigError("fixture data must be synthetic and public_market data non-synthetic")

    snapshot_id: str | None = None
    snapshot_manifest_path: Path | None = None
    provider: str | None = None
    snapshot_value = data_config.get("snapshot")
    if data_source_type == "public_market":
        snapshot = _mapping(snapshot_value, "data.snapshot")
        snapshot_id = _text(snapshot, "id", "data.snapshot")
        provider = _text(snapshot, "provider", "data.snapshot")
        if provider != "yahoo_chart":
            raise ConfigError("data.snapshot.provider must currently be yahoo_chart")
        manifest_setting = _text(snapshot, "manifest", "data.snapshot")
        snapshot_manifest_path = (project_root / manifest_setting).resolve()
        snapshot_dir = snapshot_manifest_path.parent
        if any(path.parent != snapshot_dir for path in raw_paths.values()):
            raise ConfigError("public raw_paths and snapshot manifest must share one directory")
        if processed_path == (project_root / "data/processed").resolve():
            raise ConfigError("public data must not overwrite the fixture processed directory")

    audit = _mapping(data_config.get("audit", {}), "data.audit")
    large_move_threshold = float(audit.get("large_move_threshold", 0.20))
    if not 0 < large_move_threshold < 10:
        raise ConfigError("data.audit.large_move_threshold must lie in (0, 10)")
    fatal_value = audit.get(
        "fatal_conditions",
        [
            "duplicate_dates",
            "non_increasing_dates",
            "missing_required_values",
            "non_positive_prices",
            "ohlc_violations",
        ],
    )
    if not isinstance(fatal_value, list) or any(not isinstance(item, str) for item in fatal_value):
        raise ConfigError("data.audit.fatal_conditions must be a list of strings")

    return DataPreparationConfig(
        source=source,
        split_source=split_source,
        project_root=project_root,
        mode=mode,
        raw_paths=raw_paths,
        processed_path=processed_path,
        symbols=symbols,
        source_dates=source_dates,
        observation_dates=observation_dates,
        required_columns=required_columns,
        feature_names=feature_names,
        target_definition_version=definition_version,
        target_horizon=horizon,
        annualization_factor=float(annualization),
        regime_quantiles=cast(tuple[float, float], quantiles),
        split_boundaries=tuple(boundaries),
        purge_trading_days=purge,
        missing_data_policy=missing_policy,
        data_source_type=data_source_type,
        is_synthetic=synthetic_value,
        snapshot_id=snapshot_id,
        snapshot_manifest_path=snapshot_manifest_path,
        provider=provider,
        large_move_threshold=large_move_threshold,
        fatal_audit_conditions=tuple(fatal_value),
    )
