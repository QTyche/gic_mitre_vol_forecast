"""Formal paired statistical validation of frozen financial and MNIST predictions."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from qtyche_qrc.experiments.run import _git_metadata, _write_json
from qtyche_qrc.runtime import runtime_metadata
from qtyche_qrc.statistics.bootstrap import (
    bootstrap_interval,
    circular_block_bootstrap_indices,
    indices_to_counts,
    stratified_bootstrap_indices,
)
from qtyche_qrc.statistics.hac import hac_mean_test
from qtyche_qrc.statistics.pairwise import (
    ClassificationMetric,
    classification_metric,
    classification_metric_distribution,
    holm_adjust,
    loss_values,
    mcnemar_exact,
    mincer_zarnowitz,
)

STUDY_ID = "formal_benchmark_statistical_validation_v1"
FINANCIAL_SEEDS = (2026, 2027, 2028)
FINANCIAL_LOSSES = ("qlike", "squared_error", "absolute_error")
FINANCIAL_CLASSIFICATION_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "transition_pr_auc",
)
MNIST_METRICS = ("accuracy", "macro_f1", "balanced_accuracy", "macro_roc_auc")


@dataclass(frozen=True)
class FrozenSource:
    """One immutable prediction input pinned by SHA-256."""

    source_id: str
    display_name: str
    path: Path
    sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StatisticalValidationConfig:
    """Validated Stage 2A controls and frozen source registry."""

    source: Path
    project_root: Path
    output_root: Path
    confidence_level: float
    hac_lags: tuple[int, ...]
    primary_hac_lag: int
    forecast_horizon: int
    apply_hln: bool
    bootstrap_repetitions: int
    smoke_repetitions: int
    bootstrap_seed: int
    block_lengths: tuple[int, ...]
    primary_block_length: int
    financial_prediction_count: int
    mnist_prediction_count: int
    financial_architecture_manifest: FrozenSource
    mnist_subset_checksum: str
    financial_qrc_classifier: dict[int, FrozenSource]
    financial_qrc_regressor: dict[int, FrozenSource]
    financial_classification_baselines: dict[str, FrozenSource]
    financial_regression_baselines: dict[str, FrozenSource]
    mnist_qrc: dict[int, FrozenSource]
    mnist_baselines: dict[str, FrozenSource]
    raw: dict[str, Any]


@dataclass(frozen=True)
class FinancialRegressionData:
    """Exactly aligned frozen variance forecasts."""

    dates: NDArray[np.str_]
    truth: NDArray[np.float64]
    qrc: dict[int, NDArray[np.float64]]
    baselines: dict[str, NDArray[np.float64]]


@dataclass(frozen=True)
class FrozenClassificationModel:
    """Frozen labels and optional supplied scores for one classifier."""

    predictions: NDArray[np.int64]
    transition_scores: NDArray[np.float64] | None
    probabilities: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class FinancialClassificationData:
    """Exactly aligned frozen financial classification outputs."""

    dates: NDArray[np.str_]
    truth: NDArray[np.int64]
    transition_truth: NDArray[np.int64]
    qrc: dict[int, FrozenClassificationModel]
    baselines: dict[str, FrozenClassificationModel]


@dataclass(frozen=True)
class MNISTData:
    """Exactly aligned official-test identities and frozen ten-class outputs."""

    identities: tuple[tuple[str, int], ...]
    truth: NDArray[np.int64]
    qrc: dict[int, FrozenClassificationModel]
    baselines: dict[str, FrozenClassificationModel]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    return dict(value)


def _integer_list(value: object, location: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{location} must be a non-empty integer list")
    result = tuple(int(item) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{location} contains duplicates")
    return result


def _resolved(project_root: Path, value: object, location: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty path")
    return (project_root / value).resolve()


def _frozen_source(
    project_root: Path,
    source_id: str,
    raw: object,
    *,
    default_display: str,
) -> FrozenSource:
    record = _mapping(raw, source_id)
    path = _resolved(project_root, record.get("path"), f"{source_id}.path")
    checksum = record.get("sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError(f"{source_id}.sha256 must be a SHA-256 digest")
    return FrozenSource(
        source_id,
        str(record.get("display_name", default_display)),
        path,
        checksum,
        record,
    )


def load_statistical_validation_config(path: Path) -> StatisticalValidationConfig:
    """Load and strictly validate the frozen Stage 2A contract."""

    source = path.resolve()
    root = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "configuration")
    if root.get("schema_version") != 1:
        raise ValueError("statistical-validation schema_version must be 1")
    study = _mapping(root.get("study"), "study")
    if study.get("id") != STUDY_ID:
        raise ValueError(f"study.id must remain {STUDY_ID}")
    project_setting = study.get("project_root")
    if not isinstance(project_setting, str):
        raise ValueError("study.project_root must be a path")
    project_root = (source.parent / project_setting).resolve()
    inference = _mapping(root.get("inference"), "inference")
    hac = _mapping(inference.get("hac"), "inference.hac")
    bootstrap = _mapping(inference.get("bootstrap"), "inference.bootstrap")
    primary_hac_lag = int(hac.get("primary_lag", -1))
    sensitivity_hac = _integer_list(hac.get("sensitivity_lags"), "hac.sensitivity_lags")
    hac_lags = tuple(sorted({primary_hac_lag, *sensitivity_hac}))
    primary_block_length = int(bootstrap.get("primary_block_length", -1))
    sensitivity_blocks = _integer_list(
        bootstrap.get("sensitivity_block_lengths"),
        "bootstrap.sensitivity_block_lengths",
    )
    block_lengths = tuple(sorted({primary_block_length, *sensitivity_blocks}))
    financial = _mapping(root.get("financial"), "financial")
    financial_qrc = _mapping(financial.get("qrc"), "financial.qrc")
    if tuple(sorted(int(seed) for seed in financial_qrc)) != FINANCIAL_SEEDS:
        raise ValueError("financial QRC seed set changed")
    qrc_classifier: dict[int, FrozenSource] = {}
    qrc_regressor: dict[int, FrozenSource] = {}
    for seed in FINANCIAL_SEEDS:
        record = _mapping(
            cast(dict[Any, Any], financial_qrc).get(seed),
            f"financial.qrc.{seed}",
        )
        qrc_classifier[seed] = _frozen_source(
            project_root,
            f"financial_qrc_classifier_{seed}",
            record.get("classifier"),
            default_display=f"QRC {seed}",
        )
        qrc_regressor[seed] = _frozen_source(
            project_root,
            f"financial_qrc_regressor_{seed}",
            record.get("regressor"),
            default_display=f"QRC {seed}",
        )

    def source_mapping(
        raw: object,
        location: str,
    ) -> dict[str, FrozenSource]:
        records = _mapping(raw, location)
        return {
            str(source_id): _frozen_source(
                project_root,
                f"{location}.{source_id}",
                record,
                default_display=str(source_id).replace("_", " "),
            )
            for source_id, record in records.items()
        }

    mnist = _mapping(root.get("mnist"), "mnist")
    mnist_qrc_raw = _mapping(mnist.get("qrc"), "mnist.qrc")
    if tuple(sorted(int(seed) for seed in mnist_qrc_raw)) != FINANCIAL_SEEDS:
        raise ValueError("MNIST QRC seed set changed")
    mnist_qrc = {
        seed: _frozen_source(
            project_root,
            f"mnist.qrc.{seed}",
            cast(dict[Any, Any], mnist_qrc_raw).get(seed),
            default_display=f"QRC {seed}",
        )
        for seed in FINANCIAL_SEEDS
    }
    architecture = _frozen_source(
        project_root,
        "financial_architecture_manifest",
        study.get("financial_architecture_manifest"),
        default_display="Frozen financial architecture manifest",
    )
    config = StatisticalValidationConfig(
        source=source,
        project_root=project_root,
        output_root=_resolved(project_root, study.get("output_root"), "study.output_root"),
        confidence_level=float(inference.get("confidence_level", -1.0)),
        hac_lags=hac_lags,
        primary_hac_lag=primary_hac_lag,
        forecast_horizon=int(hac.get("forecast_horizon", -1)),
        apply_hln=bool(hac.get("apply_harvey_leybourne_newbold")),
        bootstrap_repetitions=int(bootstrap.get("repetitions", -1)),
        smoke_repetitions=int(bootstrap.get("smoke_repetitions", -1)),
        bootstrap_seed=int(bootstrap.get("seed", -1)),
        block_lengths=block_lengths,
        primary_block_length=primary_block_length,
        financial_prediction_count=int(study.get("financial_prediction_count", -1)),
        mnist_prediction_count=int(study.get("mnist_prediction_count", -1)),
        financial_architecture_manifest=architecture,
        mnist_subset_checksum=str(study.get("mnist_subset_checksum", "")),
        financial_qrc_classifier=qrc_classifier,
        financial_qrc_regressor=qrc_regressor,
        financial_classification_baselines=source_mapping(
            financial.get("classification_baselines"),
            "financial.classification_baselines",
        ),
        financial_regression_baselines=source_mapping(
            financial.get("regression_baselines"),
            "financial.regression_baselines",
        ),
        mnist_qrc=mnist_qrc,
        mnist_baselines=source_mapping(mnist.get("baselines"), "mnist.baselines"),
        raw=root,
    )
    if (
        not 0.0 < config.confidence_level < 1.0
        or config.hac_lags != (0, 4, 10, 20)
        or config.primary_hac_lag != 4
        or config.forecast_horizon != 5
        or not config.apply_hln
        or config.bootstrap_repetitions != 10_000
        or config.smoke_repetitions <= 0
        or config.bootstrap_seed != 2026
        or config.block_lengths != (5, 10, 20)
        or config.primary_block_length != 10
        or config.financial_prediction_count != 497
        or config.mnist_prediction_count != 1000
        or len(config.mnist_subset_checksum) != 64
    ):
        raise ValueError("Stage 2A inferential controls changed")
    if set(config.financial_classification_baselines) != {
        "logistic_regression",
        "esn_classifier",
        "regime_persistence",
        "majority_classifier",
    }:
        raise ValueError("financial classification comparator set changed")
    if not {
        "garch_1_1",
        "esn_regressor",
        "rv_persistence",
    }.issubset(config.financial_regression_baselines):
        raise ValueError("required financial regression comparator is missing")
    if set(config.mnist_baselines) != {"flattened_logistic", "esn"}:
        raise ValueError("MNIST comparator set changed")
    return config


def _all_sources(config: StatisticalValidationConfig) -> tuple[FrozenSource, ...]:
    return (
        config.financial_architecture_manifest,
        *config.financial_qrc_classifier.values(),
        *config.financial_qrc_regressor.values(),
        *config.financial_classification_baselines.values(),
        *config.financial_regression_baselines.values(),
        *config.mnist_qrc.values(),
        *config.mnist_baselines.values(),
    )


def verify_frozen_sources(config: StatisticalValidationConfig) -> list[dict[str, Any]]:
    """Verify every frozen input before statistical computation."""

    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for source in _all_sources(config):
        if source.path in seen:
            raise ValueError(f"duplicate frozen source path: {source.path}")
        seen.add(source.path)
        if not source.path.is_file():
            raise FileNotFoundError(f"missing frozen statistical input: {source.path}")
        actual = sha256_path(source.path)
        if actual != source.sha256:
            raise ValueError(
                f"frozen statistical input checksum mismatch: {source.source_id}; "
                f"expected {source.sha256}, got {actual}"
            )
        if source.path == config.output_root or config.output_root in source.path.parents:
            raise ValueError("a frozen input points inside the statistical output tree")
        records.append(
            {
                "source_id": source.source_id,
                "display_name": source.display_name,
                "path": source.path.relative_to(config.project_root).as_posix(),
                "sha256": actual,
                "bytes": source.path.stat().st_size,
            }
        )
    return records


def require_exact_alignment(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    identity_columns: tuple[str, ...],
    truth_columns: tuple[str, ...],
    candidate_name: str,
) -> None:
    """Fail unless frozen predictions have identical ordered identities and truth."""

    required = (*identity_columns, *truth_columns)
    missing = [
        column
        for column in required
        if column not in reference.columns or column not in candidate.columns
    ]
    if missing:
        raise ValueError(f"{candidate_name} lacks alignment columns: {missing}")
    if len(reference) != len(candidate):
        raise ValueError(
            f"incomplete alignment for {candidate_name}: "
            f"{len(candidate)} rows versus {len(reference)}"
        )
    for column in identity_columns:
        left = reference[column].astype(str).to_numpy()
        right = candidate[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"prediction identity/order mismatch for {candidate_name}: {column}")
    for column in truth_columns:
        left = reference[column].to_numpy()
        right = candidate[column].to_numpy()
        if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
            aligned = np.array_equal(
                np.asarray(left, dtype=float),
                np.asarray(right, dtype=float),
            )
        else:
            aligned = np.array_equal(left.astype(str), right.astype(str))
        if not aligned:
            raise ValueError(f"frozen truth mismatch for {candidate_name}: {column}")


def _read_csv(source: FrozenSource, expected_count: int) -> pd.DataFrame:
    frame = pd.read_csv(source.path)
    if len(frame) != expected_count:
        raise ValueError(
            f"{source.source_id} row count changed: expected {expected_count}, got {len(frame)}"
        )
    return frame


def _load_financial_data(
    config: StatisticalValidationConfig,
) -> tuple[FinancialRegressionData, FinancialClassificationData]:
    count = config.financial_prediction_count
    regression_frames = {
        seed: _read_csv(source, count) for seed, source in config.financial_qrc_regressor.items()
    }
    regression_baselines = {
        name: _read_csv(source, count)
        for name, source in config.financial_regression_baselines.items()
    }
    regression_reference = regression_frames[2026]
    for seed, frame in regression_frames.items():
        require_exact_alignment(
            regression_reference,
            frame,
            identity_columns=("date",),
            truth_columns=("true_rv_5d",),
            candidate_name=f"financial QRC regressor {seed}",
        )
    for name, frame in regression_baselines.items():
        require_exact_alignment(
            regression_reference,
            frame,
            identity_columns=("date",),
            truth_columns=("true_rv_5d",),
            candidate_name=name,
        )
    regression = FinancialRegressionData(
        dates=regression_reference["date"].astype(str).to_numpy(dtype=str),
        truth=regression_reference["true_rv_5d"].to_numpy(dtype=float),
        qrc={
            seed: frame["predicted_rv_5d"].to_numpy(dtype=float)
            for seed, frame in regression_frames.items()
        },
        baselines={
            name: frame["predicted_rv_5d"].to_numpy(dtype=float)
            for name, frame in regression_baselines.items()
        },
    )
    if any(
        not np.isfinite(values).all() or np.any(values <= 0.0)
        for values in (*regression.qrc.values(), *regression.baselines.values())
    ):
        raise ValueError("frozen variance predictions must remain finite and positive")

    classification_frames = {
        seed: _read_csv(source, count) for seed, source in config.financial_qrc_classifier.items()
    }
    classification_baselines = {
        name: _read_csv(source, count)
        for name, source in config.financial_classification_baselines.items()
    }
    classification_reference = classification_frames[2026]
    truth_columns = ("current_regime", "true_regime", "true_transition")
    for seed, frame in classification_frames.items():
        require_exact_alignment(
            classification_reference,
            frame,
            identity_columns=("date",),
            truth_columns=truth_columns,
            candidate_name=f"financial QRC classifier {seed}",
        )
    for name, frame in classification_baselines.items():
        require_exact_alignment(
            classification_reference,
            frame,
            identity_columns=("date",),
            truth_columns=truth_columns,
            candidate_name=name,
        )
    if not np.array_equal(
        regression.dates,
        classification_reference["date"].astype(str).to_numpy(dtype=str),
    ):
        raise ValueError("frozen financial classification and regression dates differ")

    def financial_model(frame: pd.DataFrame) -> FrozenClassificationModel:
        required = ("predicted_regime", "predicted_transition_probability")
        if any(column not in frame for column in required):
            raise ValueError("financial classifier lacks frozen labels or transition scores")
        return FrozenClassificationModel(
            predictions=frame["predicted_regime"].to_numpy(dtype=np.int64),
            transition_scores=frame["predicted_transition_probability"].to_numpy(dtype=float),
        )

    classification = FinancialClassificationData(
        dates=classification_reference["date"].astype(str).to_numpy(dtype=str),
        truth=classification_reference["true_regime"].to_numpy(dtype=np.int64),
        transition_truth=classification_reference["true_transition"].to_numpy(dtype=np.int64),
        qrc={seed: financial_model(frame) for seed, frame in classification_frames.items()},
        baselines={
            name: financial_model(frame) for name, frame in classification_baselines.items()
        },
    )
    return regression, classification


def _load_mnist_data(config: StatisticalValidationConfig) -> MNISTData:
    count = config.mnist_prediction_count
    qrc_frames = {seed: _read_csv(source, count) for seed, source in config.mnist_qrc.items()}
    baseline_frames = {
        name: _read_csv(source, count) for name, source in config.mnist_baselines.items()
    }
    reference = qrc_frames[2026]
    for seed, frame in qrc_frames.items():
        require_exact_alignment(
            reference,
            frame,
            identity_columns=("official_partition", "official_index"),
            truth_columns=("true_digit",),
            candidate_name=f"MNIST QRC {seed}",
        )
    for name, frame in baseline_frames.items():
        require_exact_alignment(
            reference,
            frame,
            identity_columns=("official_partition", "official_index"),
            truth_columns=("true_digit",),
            candidate_name=f"MNIST {name}",
        )
    probability_columns = [f"probability_{digit}" for digit in range(10)]

    def mnist_model(frame: pd.DataFrame) -> FrozenClassificationModel:
        probabilities = frame[probability_columns].to_numpy(dtype=float)
        predictions = frame["predicted_digit"].to_numpy(dtype=np.int64)
        if (
            probabilities.shape != (count, 10)
            or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
            or not np.array_equal(np.argmax(probabilities, axis=1), predictions)
        ):
            raise ValueError("frozen MNIST probabilities or predictions are invalid")
        return FrozenClassificationModel(predictions, None, probabilities)

    identities = tuple(
        zip(
            reference["official_partition"].astype(str),
            reference["official_index"].astype(int),
        )
    )
    if set(reference["official_partition"].astype(str)) != {"official_test"}:
        raise ValueError("MNIST inference must use only the frozen official test partition")
    return MNISTData(
        identities=identities,
        truth=reference["true_digit"].to_numpy(dtype=np.int64),
        qrc={seed: mnist_model(frame) for seed, frame in qrc_frames.items()},
        baselines={name: mnist_model(frame) for name, frame in baseline_frames.items()},
    )


def architecture_level_differences(
    per_seed_differences: dict[int, NDArray[np.floating[Any]]],
) -> NDArray[np.float64]:
    """Average per-seed differences only; never average forecasts or probabilities."""

    if tuple(sorted(per_seed_differences)) != FINANCIAL_SEEDS:
        raise ValueError("architecture inference requires all three frozen seeds")
    values = [np.asarray(per_seed_differences[seed], dtype=float) for seed in FINANCIAL_SEEDS]
    missing_mask = np.isnan(values[0])
    if (
        any(value.shape != values[0].shape for value in values)
        or any(not np.array_equal(np.isnan(value), missing_mask) for value in values)
        or any(not np.isfinite(value[~missing_mask]).all() for value in values)
    ):
        raise ValueError("per-seed difference arrays do not align")
    return np.asarray(np.mean(np.vstack(values), axis=0), dtype=float)


def _direction(value: float, *, positive_favours_qrc: bool) -> str:
    if value == 0.0:
        return "neither"
    qrc = value > 0.0 if positive_favours_qrc else value < 0.0
    return "QRC" if qrc else "classical baseline"


def _bootstrap_summary(
    draws: NDArray[np.float64],
    *,
    observed: float,
    repetitions: int,
    block_length: int | None,
    confidence_level: float,
) -> dict[str, Any]:
    return {
        "observed_difference": observed,
        "bootstrap_repetitions": repetitions,
        "block_length": block_length,
        **bootstrap_interval(draws, confidence_level=confidence_level),
    }


def _financial_regression_inference(
    config: StatisticalValidationConfig,
    data: FinancialRegressionData,
    *,
    repetitions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    differentials: dict[tuple[int | None, str, str], NDArray[np.float64]] = {}
    for baseline, baseline_predictions in data.baselines.items():
        for metric in FINANCIAL_LOSSES:
            baseline_loss = loss_values(data.truth, baseline_predictions, cast(Any, metric))
            seed_values: dict[int, NDArray[np.float64]] = {}
            for seed, predictions in data.qrc.items():
                differential = (
                    loss_values(data.truth, predictions, cast(Any, metric)) - baseline_loss
                )
                differentials[(seed, baseline, metric)] = differential
                seed_values[seed] = differential
            differentials[(None, baseline, metric)] = architecture_level_differences(seed_values)
    bootstrap_draws: dict[
        tuple[int | None, str, str, int],
        NDArray[np.float64],
    ] = {}
    for block_length in config.block_lengths:
        indices = circular_block_bootstrap_indices(
            len(data.truth),
            repetitions,
            block_length,
            config.bootstrap_seed,
        )
        counts = indices_to_counts(indices, len(data.truth))
        for key, differential in differentials.items():
            bootstrap_draws[(*key, block_length)] = np.asarray(
                np.einsum("ij,j->i", counts, differential) / counts.sum(axis=1),
                dtype=float,
            )
    per_seed_rows: list[dict[str, Any]] = []
    architecture_rows: list[dict[str, Any]] = []
    for (qrc_seed, baseline, metric), differential in differentials.items():
        hac_sensitivity = [
            hac_mean_test(
                differential,
                lag=lag,
                confidence_level=config.confidence_level,
                forecast_horizon=config.forecast_horizon,
                apply_hln=config.apply_hln,
            )
            for lag in config.hac_lags
        ]
        hac_primary = next(
            value for value in hac_sensitivity if value["hac_lag"] == config.primary_hac_lag
        )
        bootstrap_sensitivity = [
            _bootstrap_summary(
                bootstrap_draws[(qrc_seed, baseline, metric, block)],
                observed=float(differential.mean()),
                repetitions=repetitions,
                block_length=block,
                confidence_level=config.confidence_level,
            )
            for block in config.block_lengths
        ]
        bootstrap_primary = next(
            value
            for value in bootstrap_sensitivity
            if value["block_length"] == config.primary_block_length
        )
        row = {
            "analysis": "financial_regression",
            "inference_level": "per_seed" if qrc_seed is not None else "architecture",
            "qrc_seed": qrc_seed,
            "qrc_seeds": list(FINANCIAL_SEEDS) if qrc_seed is None else [qrc_seed],
            "baseline": baseline,
            "baseline_display_name": config.financial_regression_baselines[baseline].display_name,
            "baseline_primary": bool(
                config.financial_regression_baselines[baseline].metadata.get(
                    "primary",
                    False,
                )
            ),
            "loss_metric": metric,
            "difference_definition": "loss_QRC_minus_loss_baseline",
            "negative_favours": "QRC",
            "positive_favours": "classical baseline",
            "mean_loss_differential": float(differential.mean()),
            "direction_favoured": _direction(
                float(differential.mean()),
                positive_favours_qrc=False,
            ),
            "aligned_observation_count": len(differential),
            "primary_hac_lag": config.primary_hac_lag,
            "standard_error": hac_primary["reported_standard_error"],
            "test_statistic": hac_primary["test_statistic"],
            "raw_p_value": hac_primary["raw_p_value"],
            "adjusted_p_value": None,
            "confidence_interval_lower": hac_primary["confidence_interval_lower"],
            "confidence_interval_upper": hac_primary["confidence_interval_upper"],
            "harvey_leybourne_newbold_applied": hac_primary["harvey_leybourne_newbold_applied"],
            "harvey_leybourne_newbold_factor": hac_primary["harvey_leybourne_newbold_factor"],
            "hac_sensitivity": hac_sensitivity,
            "primary_block_length": config.primary_block_length,
            "bootstrap_confidence_interval_lower": bootstrap_primary["confidence_interval_lower"],
            "bootstrap_confidence_interval_upper": bootstrap_primary["confidence_interval_upper"],
            "bootstrap_proportion_below_zero": bootstrap_primary["probability_below_zero"],
            "bootstrap_proportion_above_zero": bootstrap_primary["probability_above_zero"],
            "bootstrap_valid_count": bootstrap_primary["valid_bootstrap_count"],
            "bootstrap_invalid_count": bootstrap_primary["invalid_bootstrap_count"],
            "bootstrap_sensitivity": bootstrap_sensitivity,
            "architecture_method": (
                "datewise mean of three per-seed loss differentials"
                if qrc_seed is None
                else "individual frozen QRC reservoir seed"
            ),
            "forecast_averaging_used": False,
        }
        (per_seed_rows if qrc_seed is not None else architecture_rows).append(row)
    return per_seed_rows, architecture_rows


def _classification_distributions(
    truth: NDArray[np.int64],
    transition_truth: NDArray[np.int64],
    models: dict[str, FrozenClassificationModel],
    counts: NDArray[np.int32],
    metrics: tuple[str, ...],
) -> dict[tuple[str, str], NDArray[np.float64]]:
    distributions: dict[tuple[str, str], NDArray[np.float64]] = {}
    for model_name, model in models.items():
        for metric in metrics:
            metric_truth = transition_truth if metric == "transition_pr_auc" else truth
            metric_predictions = (
                (cast(NDArray[np.float64], model.transition_scores) >= 0.5).astype(np.int64)
                if metric == "transition_pr_auc"
                else model.predictions
            )
            distributions[(model_name, metric)] = classification_metric_distribution(
                metric_truth,
                metric_predictions,
                counts,
                cast(ClassificationMetric, metric),
                class_labels=(0, 1) if metric == "transition_pr_auc" else (0, 1, 2),
                scores=model.transition_scores,
            )
    return distributions


def _financial_classification_inference(
    config: StatisticalValidationConfig,
    data: FinancialClassificationData,
    *,
    repetitions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    named_models = {
        **{f"qrc_{seed}": model for seed, model in data.qrc.items()},
        **data.baselines,
    }
    observed: dict[tuple[str, str], float] = {}
    for model_name, model in named_models.items():
        for metric in FINANCIAL_CLASSIFICATION_METRICS:
            truth = data.transition_truth if metric == "transition_pr_auc" else data.truth
            predictions = (
                (cast(NDArray[np.float64], model.transition_scores) >= 0.5).astype(np.int64)
                if metric == "transition_pr_auc"
                else model.predictions
            )
            observed[(model_name, metric)] = classification_metric(
                truth,
                predictions,
                cast(ClassificationMetric, metric),
                class_labels=(0, 1) if metric == "transition_pr_auc" else (0, 1, 2),
                scores=model.transition_scores,
            )
    bootstrap_draws: dict[
        tuple[int | None, str, str, int],
        NDArray[np.float64],
    ] = {}
    for block_length in config.block_lengths:
        indices = circular_block_bootstrap_indices(
            len(data.truth),
            repetitions,
            block_length,
            config.bootstrap_seed,
        )
        counts = indices_to_counts(indices, len(data.truth))
        distributions = _classification_distributions(
            data.truth,
            data.transition_truth,
            named_models,
            counts,
            FINANCIAL_CLASSIFICATION_METRICS,
        )
        for baseline in data.baselines:
            for metric in FINANCIAL_CLASSIFICATION_METRICS:
                seed_differences: dict[int, NDArray[np.float64]] = {}
                for seed in FINANCIAL_SEEDS:
                    difference = (
                        distributions[(f"qrc_{seed}", metric)] - distributions[(baseline, metric)]
                    )
                    bootstrap_draws[(seed, baseline, metric, block_length)] = difference
                    seed_differences[seed] = difference
                bootstrap_draws[(None, baseline, metric, block_length)] = (
                    architecture_level_differences(seed_differences)
                )
    per_seed_rows: list[dict[str, Any]] = []
    architecture_rows: list[dict[str, Any]] = []
    for baseline in data.baselines:
        for metric in FINANCIAL_CLASSIFICATION_METRICS:
            observed_seed = {
                seed: observed[(f"qrc_{seed}", metric)] - observed[(baseline, metric)]
                for seed in FINANCIAL_SEEDS
            }
            for qrc_seed in (*FINANCIAL_SEEDS, None):
                observed_difference = (
                    observed_seed[qrc_seed]
                    if qrc_seed is not None
                    else float(np.mean(list(observed_seed.values())))
                )
                sensitivity = [
                    _bootstrap_summary(
                        bootstrap_draws[(qrc_seed, baseline, metric, block)],
                        observed=observed_difference,
                        repetitions=repetitions,
                        block_length=block,
                        confidence_level=config.confidence_level,
                    )
                    for block in config.block_lengths
                ]
                primary = next(
                    value
                    for value in sensitivity
                    if value["block_length"] == config.primary_block_length
                )
                row = {
                    "analysis": "financial_classification",
                    "inference_level": ("per_seed" if qrc_seed is not None else "architecture"),
                    "qrc_seed": qrc_seed,
                    "qrc_seeds": (list(FINANCIAL_SEEDS) if qrc_seed is None else [qrc_seed]),
                    "baseline": baseline,
                    "baseline_display_name": config.financial_classification_baselines[
                        baseline
                    ].display_name,
                    "baseline_informative": bool(
                        config.financial_classification_baselines[baseline].metadata.get(
                            "informative",
                            True,
                        )
                    ),
                    "metric": metric,
                    "difference_definition": "metric_QRC_minus_metric_baseline",
                    "positive_favours": "QRC",
                    "negative_favours": "classical baseline",
                    "observed_metric_difference": observed_difference,
                    "direction_favoured": _direction(
                        observed_difference,
                        positive_favours_qrc=True,
                    ),
                    "primary_block_length": config.primary_block_length,
                    "confidence_interval_lower": primary["confidence_interval_lower"],
                    "confidence_interval_upper": primary["confidence_interval_upper"],
                    "probability_difference_above_zero": primary["probability_above_zero"],
                    "probability_difference_below_zero": primary["probability_below_zero"],
                    "valid_bootstrap_count": primary["valid_bootstrap_count"],
                    "invalid_bootstrap_count": primary["invalid_bootstrap_count"],
                    "raw_p_value": primary["two_sided_p_value"],
                    "adjusted_p_value": None,
                    "bootstrap_sensitivity": sensitivity,
                    "transition_scores": (
                        "supplied frozen predicted_transition_probability"
                        if metric == "transition_pr_auc"
                        else None
                    ),
                    "architecture_method": (
                        "replicate-wise mean of three per-seed metric differences"
                        if qrc_seed is None
                        else "individual frozen QRC reservoir seed"
                    ),
                    "forecast_or_probability_averaging_used": False,
                }
                (per_seed_rows if qrc_seed is not None else architecture_rows).append(row)
    return per_seed_rows, architecture_rows


def _mincer_zarnowitz_rows(
    config: StatisticalValidationConfig,
    data: FinancialRegressionData,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, predictions, display_name, seed in (
        *[
            (
                baseline,
                values,
                config.financial_regression_baselines[baseline].display_name,
                None,
            )
            for baseline, values in data.baselines.items()
        ],
        *[(f"qrc_{seed}", data.qrc[seed], f"Frozen QRC {seed}", seed) for seed in FINANCIAL_SEEDS],
    ):
        rows.append(
            {
                "analysis": "mincer_zarnowitz",
                "model": model_id,
                "model_display_name": display_name,
                "qrc_seed": seed,
                **mincer_zarnowitz(
                    data.truth,
                    predictions,
                    hac_lag=config.primary_hac_lag,
                ),
            }
        )
    return rows


def _mnist_distributions(
    data: MNISTData,
    counts: NDArray[np.int32],
) -> dict[tuple[str, str], NDArray[np.float64]]:
    models = {
        **{f"qrc_{seed}": model for seed, model in data.qrc.items()},
        **data.baselines,
    }
    result: dict[tuple[str, str], NDArray[np.float64]] = {}
    for name, model in models.items():
        for metric in MNIST_METRICS:
            result[(name, metric)] = classification_metric_distribution(
                data.truth,
                model.predictions,
                counts,
                cast(ClassificationMetric, metric),
                class_labels=tuple(range(10)),
                scores=model.probabilities,
            )
    return result


def _mnist_inference(
    config: StatisticalValidationConfig,
    data: MNISTData,
    *,
    repetitions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indices = stratified_bootstrap_indices(
        data.truth,
        repetitions,
        config.bootstrap_seed,
    )
    counts = indices_to_counts(indices, len(data.truth))
    distributions = _mnist_distributions(data, counts)
    models = {
        **{f"qrc_{seed}": model for seed, model in data.qrc.items()},
        **data.baselines,
    }
    observed = {
        (name, metric): classification_metric(
            data.truth,
            model.predictions,
            cast(ClassificationMetric, metric),
            class_labels=tuple(range(10)),
            scores=model.probabilities,
        )
        for name, model in models.items()
        for metric in MNIST_METRICS
    }
    per_seed_rows: list[dict[str, Any]] = []
    architecture_rows: list[dict[str, Any]] = []
    for baseline in data.baselines:
        for metric in MNIST_METRICS:
            seed_differences = {
                seed: distributions[(f"qrc_{seed}", metric)] - distributions[(baseline, metric)]
                for seed in FINANCIAL_SEEDS
            }
            architecture = architecture_level_differences(seed_differences)
            observed_seed = {
                seed: observed[(f"qrc_{seed}", metric)] - observed[(baseline, metric)]
                for seed in FINANCIAL_SEEDS
            }
            for qrc_seed in (*FINANCIAL_SEEDS, None):
                difference = seed_differences[qrc_seed] if qrc_seed is not None else architecture
                observed_difference = (
                    observed_seed[qrc_seed]
                    if qrc_seed is not None
                    else float(np.mean(list(observed_seed.values())))
                )
                summary = _bootstrap_summary(
                    difference,
                    observed=observed_difference,
                    repetitions=repetitions,
                    block_length=None,
                    confidence_level=config.confidence_level,
                )
                mcnemar = None
                if metric == "accuracy" and qrc_seed is not None:
                    mcnemar = mcnemar_exact(
                        data.truth,
                        data.qrc[qrc_seed].predictions,
                        data.baselines[baseline].predictions,
                    )
                row = {
                    "analysis": "mnist",
                    "inference_level": ("per_seed" if qrc_seed is not None else "architecture"),
                    "qrc_seed": qrc_seed,
                    "qrc_seeds": (list(FINANCIAL_SEEDS) if qrc_seed is None else [qrc_seed]),
                    "baseline": baseline,
                    "baseline_display_name": config.mnist_baselines[baseline].display_name,
                    "metric": metric,
                    "difference_definition": "metric_QRC_minus_metric_baseline",
                    "positive_favours": "QRC",
                    "negative_favours": "classical baseline",
                    "observed_metric_difference": observed_difference,
                    "direction_favoured": _direction(
                        observed_difference,
                        positive_favours_qrc=True,
                    ),
                    "confidence_interval_lower": summary["confidence_interval_lower"],
                    "confidence_interval_upper": summary["confidence_interval_upper"],
                    "probability_difference_above_zero": summary["probability_above_zero"],
                    "probability_difference_below_zero": summary["probability_below_zero"],
                    "valid_bootstrap_count": summary["valid_bootstrap_count"],
                    "invalid_bootstrap_count": summary["invalid_bootstrap_count"],
                    "bootstrap_raw_p_value": summary["two_sided_p_value"],
                    "bootstrap_adjusted_p_value": None,
                    "bootstrap_method": (
                        "paired class-stratified resampling independently within true digit"
                    ),
                    "mcnemar": mcnemar,
                    "mcnemar_raw_p_value": (
                        mcnemar["raw_p_value"] if mcnemar is not None else None
                    ),
                    "mcnemar_adjusted_p_value": None,
                    "architecture_method": (
                        "replicate-wise mean of three per-seed metric differences"
                        if qrc_seed is None
                        else "individual frozen QRC reservoir seed"
                    ),
                    "probability_averaging_used": False,
                }
                (per_seed_rows if qrc_seed is not None else architecture_rows).append(row)
    return per_seed_rows, architecture_rows


def _apply_multiple_testing(
    regression_per_seed: list[dict[str, Any]],
    regression_architecture: list[dict[str, Any]],
    classification_per_seed: list[dict[str, Any]],
    classification_architecture: list[dict[str, Any]],
    mnist_per_seed: list[dict[str, Any]],
    mnist_architecture: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    adjustments: list[dict[str, Any]] = []

    def adjust(
        rows: list[dict[str, Any]],
        *,
        metric_key: str,
        raw_key: str,
        adjusted_key: str,
        test_method: str,
        analysis: str,
        level: str,
    ) -> None:
        for metric in sorted({str(row[metric_key]) for row in rows}):
            selected = [
                row
                for row in rows
                if str(row[metric_key]) == metric and row.get(raw_key) is not None
            ]
            if not selected:
                continue
            adjusted = holm_adjust(
                np.asarray([float(row[raw_key]) for row in selected], dtype=float)
            )
            family = f"{analysis}:{level}:{metric}:{test_method}"
            for row, value in zip(selected, adjusted):
                row[adjusted_key] = float(value)
                adjustments.append(
                    {
                        "analysis": analysis,
                        "inference_level": level,
                        "metric_family": metric,
                        "test_method": test_method,
                        "family_id": family,
                        "hypothesis": (
                            f"qrc_seed={row.get('qrc_seed')};baseline={row['baseline']}"
                        ),
                        "raw_p_value": row[raw_key],
                        "holm_adjusted_p_value": float(value),
                        "family_size": len(selected),
                    }
                )

    for rows, level in (
        (regression_per_seed, "per_seed"),
        (regression_architecture, "architecture"),
    ):
        adjust(
            rows,
            metric_key="loss_metric",
            raw_key="raw_p_value",
            adjusted_key="adjusted_p_value",
            test_method="DM-style HAC with HLN",
            analysis="financial_regression",
            level=level,
        )
    for rows, level in (
        (classification_per_seed, "per_seed"),
        (classification_architecture, "architecture"),
    ):
        adjust(
            rows,
            metric_key="metric",
            raw_key="raw_p_value",
            adjusted_key="adjusted_p_value",
            test_method="paired circular-block bootstrap",
            analysis="financial_classification",
            level=level,
        )
    for rows, level in (
        (mnist_per_seed, "per_seed"),
        (mnist_architecture, "architecture"),
    ):
        adjust(
            rows,
            metric_key="metric",
            raw_key="bootstrap_raw_p_value",
            adjusted_key="bootstrap_adjusted_p_value",
            test_method="paired class-stratified bootstrap",
            analysis="mnist",
            level=level,
        )
    adjust(
        [row for row in mnist_per_seed if row["metric"] == "accuracy"],
        metric_key="metric",
        raw_key="mcnemar_raw_p_value",
        adjusted_key="mcnemar_adjusted_p_value",
        test_method="exact McNemar",
        analysis="mnist",
        level="per_seed",
    )
    return adjustments


def _csv_safe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        for key, value in tuple(record.items()):
            if isinstance(value, (dict, list, tuple)):
                record[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        safe.append(record)
    return safe


def _write_table_pair(
    output_dir: Path,
    name: str,
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    json_path = output_dir / f"{name}.json"
    csv_path = output_dir / f"{name}.csv"
    _write_json(json_path, {"schema_version": 1, "rows": rows})
    frame = pd.DataFrame(_csv_safe(rows))
    frame = frame.reindex(sorted(frame.columns), axis=1)
    frame.to_csv(csv_path, index=False)
    return {
        f"{name}_json": json_path.as_posix(),
        f"{name}_csv": csv_path.as_posix(),
    }


def _save_figure(figure: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=240)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def _forest_plot(
    rows: list[dict[str, Any]],
    *,
    difference_key: str,
    lower_key: str,
    upper_key: str,
    destination: Path,
    title: str,
    negative_label: str,
    positive_label: str,
) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["baseline"]),
            row["qrc_seed"] is None,
            -1 if row["qrc_seed"] is None else int(row["qrc_seed"]),
        ),
    )
    labels = [
        (
            f"architecture vs {row['baseline_display_name']}"
            if row["qrc_seed"] is None
            else f"QRC {row['qrc_seed']} vs {row['baseline_display_name']}"
        )
        for row in ordered
    ]
    values = np.asarray([float(row[difference_key]) for row in ordered], dtype=float)
    lower = np.asarray([float(row[lower_key]) for row in ordered], dtype=float)
    upper = np.asarray([float(row[upper_key]) for row in ordered], dtype=float)
    positions = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(10.0, max(5.0, 0.38 * len(ordered) + 1.8)))
    colors = ["#4c78a8" if row["qrc_seed"] is not None else "#f58518" for row in ordered]
    axis.errorbar(
        values,
        positions,
        xerr=np.maximum(np.vstack((values - lower, upper - values)), 0.0),
        fmt="none",
        ecolor="#4c78a8",
        capsize=3,
        linewidth=1.4,
    )
    axis.scatter(values, positions, c=colors, s=28, zorder=3)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel(f"{negative_label}  \u2190  difference  \u2192  {positive_label}")
    axis.grid(axis="x", alpha=0.2)
    _save_figure(figure, destination)


def _mincer_zarnowitz_plot(rows: list[dict[str, Any]], destination: Path) -> None:
    labels = [str(row["model_display_name"]) for row in rows]
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(12.0, max(4.6, 0.45 * len(rows) + 1.6)))
    for axis, estimate, standard_error, reference, title in (
        (
            axes[0],
            "alpha",
            "alpha_hac_standard_error",
            0.0,
            "Mincer\u2013Zarnowitz intercept",
        ),
        (
            axes[1],
            "beta",
            "beta_hac_standard_error",
            1.0,
            "Mincer\u2013Zarnowitz slope",
        ),
    ):
        values = np.asarray([float(row[estimate]) for row in rows])
        errors = 1.96 * np.asarray([float(row[standard_error]) for row in rows])
        axis.errorbar(values, positions, xerr=errors, fmt="o", capsize=3, color="#4c78a8")
        axis.axvline(reference, color="black", linestyle="--", linewidth=1.0)
        axis.set_yticks(positions, labels if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("estimate with 95% HAC interval")
        axis.grid(axis="x", alpha=0.2)
    _save_figure(figure, destination)


def _figures(
    config: StatisticalValidationConfig,
    regression_per_seed: list[dict[str, Any]],
    regression_architecture: list[dict[str, Any]],
    classification_per_seed: list[dict[str, Any]],
    classification_architecture: list[dict[str, Any]],
    mincer_rows: list[dict[str, Any]],
    mnist_per_seed: list[dict[str, Any]],
    mnist_architecture: list[dict[str, Any]],
) -> dict[str, str]:
    figures = config.output_root / "figures"
    regression = regression_per_seed + regression_architecture
    classification = classification_per_seed + classification_architecture
    mnist = mnist_per_seed + mnist_architecture
    destinations = {
        "financial_qlike_forest": figures / "financial_qlike_loss_difference_forest",
        "financial_squared_error_forest": (figures / "financial_squared_error_difference_forest"),
        "financial_macro_f1_forest": figures / "financial_macro_f1_difference_forest",
        "financial_transition_pr_auc_forest": (
            figures / "financial_transition_pr_auc_difference_forest"
        ),
        "mincer_zarnowitz": figures / "mincer_zarnowitz_alpha_beta",
        "mnist_accuracy_forest": figures / "mnist_accuracy_difference_forest",
        "mnist_macro_f1_forest": figures / "mnist_macro_f1_difference_forest",
    }
    for metric, key, title in (
        ("qlike", "financial_qlike_forest", "Financial QLIKE loss differentials"),
        (
            "squared_error",
            "financial_squared_error_forest",
            "Financial squared-error loss differentials",
        ),
    ):
        _forest_plot(
            [row for row in regression if row["loss_metric"] == metric],
            difference_key="mean_loss_differential",
            lower_key="bootstrap_confidence_interval_lower",
            upper_key="bootstrap_confidence_interval_upper",
            destination=destinations[key],
            title=title,
            negative_label="favours QRC",
            positive_label="favours baseline",
        )
    for metric, key, title in (
        ("macro_f1", "financial_macro_f1_forest", "Financial macro-F1 differences"),
        (
            "transition_pr_auc",
            "financial_transition_pr_auc_forest",
            "Financial transition PR-AUC differences",
        ),
    ):
        _forest_plot(
            [row for row in classification if row["metric"] == metric],
            difference_key="observed_metric_difference",
            lower_key="confidence_interval_lower",
            upper_key="confidence_interval_upper",
            destination=destinations[key],
            title=title,
            negative_label="favours baseline",
            positive_label="favours QRC",
        )
    _mincer_zarnowitz_plot(mincer_rows, destinations["mincer_zarnowitz"])
    for metric, key, title in (
        ("accuracy", "mnist_accuracy_forest", "MNIST accuracy differences"),
        ("macro_f1", "mnist_macro_f1_forest", "MNIST macro-F1 differences"),
    ):
        _forest_plot(
            [row for row in mnist if row["metric"] == metric],
            difference_key="observed_metric_difference",
            lower_key="confidence_interval_lower",
            upper_key="confidence_interval_upper",
            destination=destinations[key],
            title=title,
            negative_label="favours baseline",
            positive_label="favours QRC",
        )
    return {
        f"{key}_{extension}": path.with_suffix(f".{extension}")
        .relative_to(config.project_root)
        .as_posix()
        for key, path in destinations.items()
        for extension in ("png", "pdf")
    }


def _conclusion_counts(rows: list[dict[str, Any]], adjusted_key: str) -> dict[str, int]:
    significant = [
        row for row in rows if row.get(adjusted_key) is not None and float(row[adjusted_key]) < 0.05
    ]
    return {
        "comparisons": len(rows),
        "holm_significant_at_0_05": len(significant),
        "holm_significant_favouring_qrc": sum(
            row["direction_favoured"] == "QRC" for row in significant
        ),
        "holm_significant_favouring_baseline": sum(
            row["direction_favoured"] == "classical baseline" for row in significant
        ),
    }


def run_statistical_validation(config_path: Path, *, smoke: bool = False) -> Path:
    """Run deterministic inference using frozen predictions only."""

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config = load_statistical_validation_config(config_path)
    source_records_before = verify_frozen_sources(config)
    financial_regression, financial_classification = _load_financial_data(config)
    mnist = _load_mnist_data(config)
    repetitions = config.smoke_repetitions if smoke else config.bootstrap_repetitions
    mode = "smoke" if smoke else "full"
    regression_per_seed, regression_architecture = _financial_regression_inference(
        config,
        financial_regression,
        repetitions=repetitions,
    )
    classification_per_seed, classification_architecture = _financial_classification_inference(
        config,
        financial_classification,
        repetitions=repetitions,
    )
    mincer_rows = _mincer_zarnowitz_rows(config, financial_regression)
    mnist_per_seed, mnist_architecture = _mnist_inference(
        config,
        mnist,
        repetitions=repetitions,
    )
    adjustment_rows = _apply_multiple_testing(
        regression_per_seed,
        regression_architecture,
        classification_per_seed,
        classification_architecture,
        mnist_per_seed,
        mnist_architecture,
    )
    config.output_root.mkdir(parents=True, exist_ok=True)
    table_dir = config.output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for name, rows in (
        ("financial_regression_pairwise_per_seed", regression_per_seed),
        ("financial_regression_architecture_level", regression_architecture),
        ("financial_classification_pairwise_per_seed", classification_per_seed),
        (
            "financial_classification_architecture_level",
            classification_architecture,
        ),
        ("mincer_zarnowitz", mincer_rows),
        ("mnist_pairwise_per_seed", mnist_per_seed),
        ("mnist_architecture_level", mnist_architecture),
        ("multiple_testing_adjustments", adjustment_rows),
    ):
        outputs = _write_table_pair(table_dir, name, rows)
        output_paths.update(
            {
                key: Path(value).relative_to(config.project_root).as_posix()
                for key, value in outputs.items()
            }
        )
    figure_paths = _figures(
        config,
        regression_per_seed,
        regression_architecture,
        classification_per_seed,
        classification_architecture,
        mincer_rows,
        mnist_per_seed,
        mnist_architecture,
    )
    source_records_after = verify_frozen_sources(config)
    if source_records_before != source_records_after:
        raise RuntimeError("a frozen prediction input changed during statistical validation")
    environment_path = config.output_root / "environment_manifest.json"
    _write_json(
        environment_path,
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "mode": mode,
            "configuration": {
                "path": config.source.relative_to(config.project_root).as_posix(),
                "sha256": sha256_path(config.source),
            },
            "frozen_inputs": source_records_after,
            "frozen_inputs_unchanged_during_run": True,
            "financial_architecture_manifest_checksum": (
                config.financial_architecture_manifest.sha256
            ),
            "mnist_subset_checksum": config.mnist_subset_checksum,
            "git": _git_metadata(config.project_root),
            **runtime_metadata(),
            "models_refit": False,
            "forecasts_modified": False,
            "forecast_or_probability_ensembles_created": False,
            "physical_qpu_execution": False,
            "quantum_advantage_claim": False,
        },
    )
    summary_path = config.output_root / "statistical_validation_summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "status": "success",
            "mode": mode,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.perf_counter() - started,
            "bootstrap": {
                "repetitions": repetitions,
                "seed": config.bootstrap_seed,
                "primary_block_length": config.primary_block_length,
                "sensitivity_block_lengths": [
                    value for value in config.block_lengths if value != config.primary_block_length
                ],
                "paired_observations_preserved": True,
            },
            "hac": {
                "primary_lag": config.primary_hac_lag,
                "sensitivity_lags": [
                    value for value in config.hac_lags if value != config.primary_hac_lag
                ],
                "forecast_horizon": config.forecast_horizon,
                "harvey_leybourne_newbold_applied": config.apply_hln,
            },
            "sample_counts": {
                "financial": len(financial_regression.truth),
                "mnist": len(mnist.truth),
            },
            "conclusions": {
                "financial_regression_per_seed": _conclusion_counts(
                    regression_per_seed,
                    "adjusted_p_value",
                ),
                "financial_regression_architecture": _conclusion_counts(
                    regression_architecture,
                    "adjusted_p_value",
                ),
                "financial_classification_per_seed": _conclusion_counts(
                    classification_per_seed,
                    "adjusted_p_value",
                ),
                "financial_classification_architecture": _conclusion_counts(
                    classification_architecture,
                    "adjusted_p_value",
                ),
                "mnist_per_seed_bootstrap": _conclusion_counts(
                    mnist_per_seed,
                    "bootstrap_adjusted_p_value",
                ),
                "mnist_architecture_bootstrap": _conclusion_counts(
                    mnist_architecture,
                    "bootstrap_adjusted_p_value",
                ),
            },
            "outputs": {
                **output_paths,
                **figure_paths,
                "environment_manifest": environment_path.relative_to(
                    config.project_root
                ).as_posix(),
            },
            "scientific_constraints": {
                "models_refit": False,
                "architecture_or_hyperparameters_changed": False,
                "test_results_used_for_selection": False,
                "frozen_predictions_only": True,
                "architecture_loss_method": ("mean per-seed loss differential at each date"),
                "architecture_classification_method": (
                    "mean per-seed metric difference inside each bootstrap replicate"
                ),
                "forecast_or_probability_averaging_used": False,
                "physical_qpu_execution": False,
                "quantum_advantage_claim": False,
            },
        },
    )
    return summary_path
