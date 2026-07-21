"""End-to-end deterministic preparation of the Phase 3 data contract."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtyche_qrc.data.config import DataPreparationConfig
from qtyche_qrc.data.download import load_raw_frames, sha256_file, verify_public_snapshot
from qtyche_qrc.data.features import build_features, ensure_sufficient_rows
from qtyche_qrc.data.preprocessing import TrainStandardizer
from qtyche_qrc.data.splits import (
    assign_chronological_splits,
    validate_forward_window_containment,
)
from qtyche_qrc.data.targets import (
    TARGET_NAMES,
    RegimeThresholds,
    add_continuous_targets,
    add_regime_and_transition_targets,
    fit_regime_thresholds,
)
from qtyche_qrc.data.validation import (
    DataValidationError,
    align_on_spy_calendar,
    audit_raw_market_frames,
)


@dataclass(frozen=True)
class PreparationResult:
    """Paths and quality summary produced by one data-preparation run."""

    processed_dir: Path
    output_paths: dict[str, Path]
    quality_report: dict[str, Any]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.12g",
        lineterminator="\n",
    )


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_dirty(project_root: Path) -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _manifest_path(path: Path, project_root: Path, *, relative: bool) -> str:
    if relative:
        return path.relative_to(project_root).as_posix()
    return str(path)


def _date_summary(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty:
        return {"start": None, "end": None}
    return {
        "start": frame["date"].min().date().isoformat(),
        "end": frame["date"].max().date().isoformat(),
    }


def _threshold_document(thresholds: RegimeThresholds) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "target_definition_version": "qtyche_volatility_regime_v1",
        "fit_split": "train",
        "fit_column": "target_rv_5d",
        "quantiles": list(thresholds.quantiles),
        "low_medium": thresholds.low_medium,
        "medium_high": thresholds.medium_high,
        "training_rows": thresholds.training_rows,
        "label_rule": {
            "0": "value <= low_medium",
            "1": "low_medium < value <= medium_high",
            "2": "value > medium_high",
        },
    }


def _scaled_output(
    rows: pd.DataFrame,
    feature_names: tuple[str, ...],
    standardizer: TrainStandardizer,
) -> pd.DataFrame:
    features = standardizer.transform(rows.loc[:, list(feature_names)])
    output = rows.loc[:, ["date", "split"]].copy()
    for feature in feature_names:
        output[feature] = features[feature]
    for target in TARGET_NAMES:
        output[target] = rows[target]
    return output


def _split_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    statistics: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        rows = frame.loc[frame["split"].eq(split_name)]
        regimes = rows["target_regime_5d"].value_counts().sort_index()
        target = rows["target_rv_5d"]
        statistics[split_name] = {
            "regime_counts": {str(label): int(regimes.get(label, 0)) for label in (0, 1, 2)},
            "regime_percentages": {
                str(label): float(regimes.get(label, 0) / len(rows)) for label in (0, 1, 2)
            },
            "transition_count": int(rows["target_transition"].sum()),
            "transition_rate": float(rows["target_transition"].mean()),
            "upward_transition_count": int(rows["target_upward_transition"].sum()),
            "upward_transition_rate": float(rows["target_upward_transition"].mean()),
            "downward_transition_count": int(rows["target_downward_transition"].sum()),
            "downward_transition_rate": float(rows["target_downward_transition"].mean()),
            "target_rv_5d": {
                "mean": float(target.mean()),
                "median": float(target.median()),
                "standard_deviation": float(target.std(ddof=0)),
                "minimum": float(target.min()),
                "maximum": float(target.max()),
                "quantiles": {
                    str(value): float(target.quantile(value))
                    for value in (0.01, 0.05, 0.25, 0.75, 0.95, 0.99)
                },
            },
        }
    return statistics


def prepare_data(config: DataPreparationConfig) -> PreparationResult:
    """Build all unscaled/scaled splits, fit artifacts, manifests, and quality reports."""

    generated_at = datetime.now(timezone.utc).isoformat()
    raw_frames = load_raw_frames(config)
    raw_market_audit = audit_raw_market_frames(
        raw_frames,
        large_move_threshold=config.large_move_threshold,
        fatal_conditions=config.fatal_audit_conditions,
    )
    snapshot_manifest: dict[str, Any] | None = None
    snapshot_manifest_checksum: str | None = None
    if config.data_source_type == "public_market":
        snapshot_manifest = verify_public_snapshot(config)
        if config.snapshot_manifest_path is None:
            raise DataValidationError("public data configuration has no snapshot manifest")
        snapshot_manifest_checksum = sha256_file(config.snapshot_manifest_path)
    raw_source_metadata: dict[str, Any] = {}
    for name, frame in raw_frames.items():
        parsed_dates = pd.to_datetime(frame["date"], errors="raise")
        raw_source_metadata[name] = {
            "path": _manifest_path(
                config.raw_paths[name],
                config.project_root,
                relative=config.data_source_type == "public_market",
            ),
            "sha256": sha256_file(config.raw_paths[name]),
            "rows": len(frame),
            "date_range": {
                "start": parsed_dates.min().date().isoformat(),
                "end": parsed_dates.max().date().isoformat(),
            },
            "symbol": config.symbols[name],
        }

    canonical, alignment_report = align_on_spy_calendar(
        raw_frames,
        config.missing_data_policy,
    )
    source_date_mask = canonical["date"].between(
        pd.Timestamp(config.source_dates.start),
        pd.Timestamp(config.source_dates.end),
        inclusive="both",
    )
    rows_outside_source_range = int((~source_date_mask).sum())
    canonical = canonical.loc[source_date_mask].reset_index(drop=True)
    ensure_sufficient_rows(canonical, history=20, horizon=config.target_horizon)

    constructed = build_features(
        canonical,
        requested_features=config.feature_names,
        annualization=config.annualization_factor,
    )
    constructed = add_continuous_targets(
        constructed,
        horizon=config.target_horizon,
        annualization=config.annualization_factor,
    )
    constructed.loc[:, list(config.feature_names)] = constructed.loc[
        :, list(config.feature_names)
    ].replace([np.inf, -np.inf], np.nan)
    complete_feature_mask = constructed.loc[:, list(config.feature_names)].notna().all(axis=1)
    complete_target_mask = constructed[["target_rv_5d", "target_window_end"]].notna().all(axis=1)
    earliest_valid_feature_date = (
        constructed.loc[complete_feature_mask, "date"].min().date().isoformat()
    )
    latest_valid_target_date = (
        constructed.loc[complete_target_mask, "date"].max().date().isoformat()
    )
    latest_target_window_end = (
        constructed.loc[complete_target_mask, "target_window_end"].max().date().isoformat()
    )

    source_feature_warmup = int(
        constructed.loc[:, list(config.feature_names)].isna().any(axis=1).sum()
    )
    source_target_tail = int(
        constructed[["target_rv_5d", "target_window_end"]].isna().any(axis=1).sum()
    )
    observation_mask = constructed["date"].between(
        pd.Timestamp(config.observation_dates.start),
        pd.Timestamp(config.observation_dates.end),
        inclusive="both",
    )
    observations = constructed.loc[observation_mask].copy()
    missing_before_filter = {
        column: int(count)
        for column, count in observations[
            [*config.feature_names, "target_rv_5d", "target_window_end"]
        ]
        .isna()
        .sum()
        .items()
        if count
    }
    missing_features = observations.loc[:, list(config.feature_names)].isna().any(axis=1)
    missing_targets = observations[["target_rv_5d", "target_window_end"]].isna().any(axis=1)
    rows_removed_missing_features = int(missing_features.sum())
    rows_removed_missing_targets = int((missing_targets & ~missing_features).sum())
    complete = observations.loc[~(missing_features | missing_targets)].copy()
    if complete.empty:
        raise DataValidationError("no complete rows remain after feature and target construction")

    split_frame, split_report = assign_chronological_splits(
        complete,
        config.split_boundaries,
    )
    if split_frame.empty:
        raise DataValidationError("no rows remain after chronological splitting and purging")
    for split_name in ("train", "validation", "test"):
        if not split_frame["split"].eq(split_name).any():
            raise DataValidationError(f"{split_name} is empty after chronological purging")
    validate_forward_window_containment(split_frame, config.split_boundaries)

    thresholds = fit_regime_thresholds(split_frame, config.regime_quantiles)
    labeled = add_regime_and_transition_targets(split_frame, thresholds)
    for regime_column in ("target_regime_5d", "current_regime"):
        if not set(labeled[regime_column].unique()).issubset({0, 1, 2}):
            raise DataValidationError(f"invalid labels in {regime_column}")

    training = labeled.loc[labeled["split"].eq("train")]
    standardizer = TrainStandardizer.fit(training.loc[:, list(config.feature_names)])

    processed_dir = config.processed_path
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "features_unscaled": processed_dir / "features_unscaled.csv",
        "train": processed_dir / "train.csv",
        "validation": processed_dir / "validation.csv",
        "test": processed_dir / "test.csv",
        "preprocessing": processed_dir / "preprocessing.json",
        "regime_thresholds": processed_dir / "regime_thresholds.json",
        "data_manifest": processed_dir / "data_manifest.json",
        "data_quality_report": processed_dir / "data_quality_report.json",
    }

    unscaled_columns = ["date", "split", *config.feature_names, *TARGET_NAMES]
    unscaled = labeled.loc[:, unscaled_columns].copy()
    _write_csv(unscaled, output_paths["features_unscaled"])
    scaled_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        rows = labeled.loc[labeled["split"].eq(split_name)].copy()
        scaled = _scaled_output(rows, config.feature_names, standardizer)
        _write_csv(scaled, output_paths[split_name])
        scaled_frames[split_name] = scaled
    standardizer.save(output_paths["preprocessing"])
    threshold_document = _threshold_document(thresholds)
    _write_json(output_paths["regime_thresholds"], threshold_document)

    output_missing_counts = {
        name: int(frame.isna().sum().sum()) for name, frame in scaled_frames.items()
    }
    quality_report: dict[str, Any] = {
        "schema_version": 1,
        "generation_timestamp": generated_at,
        "status": "passed",
        "alignment": alignment_report,
        "raw_market_audit": raw_market_audit,
        "construction": {
            "canonical_rows_in_source_range": len(canonical),
            "rows_outside_configured_source_range": rows_outside_source_range,
            "source_rows_with_feature_warmup_missingness": source_feature_warmup,
            "source_rows_without_complete_forward_target": source_target_tail,
            "observation_rows_before_filter": len(observations),
            "rows_removed_missing_features": rows_removed_missing_features,
            "rows_removed_missing_targets": rows_removed_missing_targets,
            "missing_value_counts_before_filter": missing_before_filter,
            "earliest_valid_feature_date": earliest_valid_feature_date,
            "latest_valid_forward_target_observation_date": latest_valid_target_date,
            "latest_valid_forward_target_window_end": latest_target_window_end,
        },
        "splitting": split_report,
        "output_missing_value_counts": output_missing_counts,
        "zero_variance_features": list(standardizer.zero_variance_features),
    }

    split_boundaries = {
        boundary.name: {"start": boundary.start.isoformat(), "end": boundary.end.isoformat()}
        for boundary in config.split_boundaries
    }
    date_ranges = {
        "canonical_source": _date_summary(canonical),
        **{name: _date_summary(frame) for name, frame in scaled_frames.items()},
    }
    row_counts = {
        "canonical_source": len(canonical),
        "features_unscaled": len(unscaled),
        **{name: len(frame) for name, frame in scaled_frames.items()},
    }
    split_statistics = _split_statistics(labeled)
    core_processed_names = (
        "features_unscaled",
        "train",
        "validation",
        "test",
        "preprocessing",
        "regime_thresholds",
    )
    processed_checksums = {
        output_paths[name].name: sha256_file(output_paths[name]) for name in core_processed_names
    }
    relative_paths = config.data_source_type == "public_market"
    data_manifest: dict[str, Any] = {
        "schema_version": 1,
        "generation_timestamp": generated_at,
        "git_commit_hash": _git_commit(config.project_root),
        "git_dirty": _git_dirty(config.project_root),
        "data_mode": config.mode,
        "data_source_type": config.data_source_type,
        "is_synthetic": config.is_synthetic,
        "source_snapshot_id": config.snapshot_id,
        "source_snapshot_manifest_checksum": snapshot_manifest_checksum,
        "source_snapshot": snapshot_manifest,
        "requested_date_range": {
            "start": config.source_dates.start.isoformat(),
            "end": config.source_dates.end.isoformat(),
        },
        "source_files": raw_source_metadata,
        "source_file_paths": {
            name: _manifest_path(path, config.project_root, relative=relative_paths)
            for name, path in config.raw_paths.items()
        },
        "source_checksums": {
            name: metadata["sha256"] for name, metadata in raw_source_metadata.items()
        },
        "date_ranges": date_ranges,
        "row_counts": row_counts,
        "split_row_counts": {name: len(frame) for name, frame in scaled_frames.items()},
        "feature_names": list(config.feature_names),
        "target_names": list(TARGET_NAMES),
        "target_definition_version": config.target_definition_version,
        "split_boundaries": split_boundaries,
        "purge_trading_days": config.purge_trading_days,
        "purged_dates": split_report["purged_forward_window_dates"],
        "regime_thresholds": threshold_document,
        "split_statistics": split_statistics,
        "preprocessing_parameters_checksum": processed_checksums["preprocessing.json"],
        "processed_checksums": processed_checksums,
        "missing_value_counts": {
            "before_filter": missing_before_filter,
            "outputs": output_missing_counts,
        },
        "configuration_file_used": {
            "data": {
                "path": _manifest_path(config.source, config.project_root, relative=relative_paths),
                "sha256": sha256_file(config.source),
            },
            "splits": {
                "path": _manifest_path(
                    config.split_source, config.project_root, relative=relative_paths
                ),
                "sha256": sha256_file(config.split_source),
            },
        },
        "output_paths": {
            name: _manifest_path(path, config.project_root, relative=relative_paths)
            for name, path in output_paths.items()
        },
    }
    _write_json(output_paths["data_quality_report"], quality_report)
    _write_json(output_paths["data_manifest"], data_manifest)
    return PreparationResult(processed_dir, output_paths, quality_report)


def inspect_processed_targets(processed_dir: Path) -> dict[str, Any]:
    """Summarize saved target balance without calculating predictive performance."""

    thresholds_path = processed_dir / "regime_thresholds.json"
    if not thresholds_path.is_file():
        raise DataValidationError(f"missing thresholds file: {thresholds_path}")
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    splits: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        path = processed_dir / f"{split_name}.csv"
        if not path.is_file():
            raise DataValidationError(f"missing processed split: {path}")
        frame = pd.read_csv(path)
        counts = frame["target_regime_5d"].value_counts().sort_index()
        splits[split_name] = {
            "rows": len(frame),
            "target_regime_counts": {str(label): int(counts.get(label, 0)) for label in (0, 1, 2)},
            "transition_count": int(frame["target_transition"].sum()),
            "upward_transition_count": int(frame["target_upward_transition"].sum()),
            "downward_transition_count": int(frame["target_downward_transition"].sum()),
        }
    return {
        "thresholds": {
            "low_medium": thresholds["low_medium"],
            "medium_high": thresholds["medium_high"],
            "fit_split": thresholds["fit_split"],
        },
        "splits": splits,
    }
