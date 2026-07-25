"""Frozen-model calibration, regime, temporal, numerical, and MNIST diagnostics."""

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

from qtyche_qrc.diagnostics.calibration import (
    classwise_calibration_summary,
    multiclass_calibration_summary,
    probability_entropy,
    qlike_values,
    variance_bootstrap_intervals,
    variance_calibration_deciles,
    variance_point_metrics,
)
from qtyche_qrc.diagnostics.regimes import (
    REGIME_NAMES,
    assess_lead_time_identifiability,
    classification_diagnostics,
    confusion_counts,
    fixed_regime_labels,
    per_class_diagnostics,
    transition_subset_diagnostics,
    transition_type_diagnostics,
    variance_regime_diagnostics,
    variance_tail_diagnostics,
)
from qtyche_qrc.diagnostics.temporal import (
    cumulative_classification_error_difference,
    cumulative_loss_difference,
    rolling_variance_diagnostics,
)
from qtyche_qrc.experiments.run import _git_metadata, _write_json
from qtyche_qrc.experiments.statistical_validation import require_exact_alignment
from qtyche_qrc.runtime import runtime_metadata
from qtyche_qrc.statistics.bootstrap import (
    circular_block_bootstrap_indices,
    indices_to_counts,
    stratified_bootstrap_indices,
)

STUDY_ID = "frozen_benchmark_diagnostics_v1"
SPLITS = ("validation", "test")
QRC_SEEDS = (2026, 2027, 2028)
CALIBRATION_BINS = (5, 10, 15, 20)
REGRESSION_MODELS = (
    "rv_persistence",
    "esn_regressor",
    "garch_1_1",
    "qrc_2026",
    "qrc_2027",
    "qrc_2028",
)
CLASSIFICATION_MODELS = (
    "majority_classifier",
    "regime_persistence",
    "logistic_regression",
    "esn_classifier",
    "qrc_2026",
    "qrc_2027",
    "qrc_2028",
)
PROBABILITY_MODELS = (
    "logistic_regression",
    "esn_classifier",
    "qrc_2026",
    "qrc_2027",
    "qrc_2028",
)
MNIST_MODELS = (
    "flattened_logistic",
    "esn",
    "qrc_2026",
    "qrc_2027",
    "qrc_2028",
)
MNIST_ROBUSTNESS = (
    "analytic",
    "shots_2048",
    "depolarizing_0_01",
    "measurement_flip_0_02",
)


@dataclass(frozen=True)
class FrozenSource:
    """One immutable diagnostic input."""

    source_id: str
    display_name: str
    path: Path
    sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SplitModelSource:
    """Frozen validation and test predictions for one financial model."""

    model_id: str
    display_name: str
    reservoir_seed: int | None
    probability_model: bool
    validation: FrozenSource
    test: FrozenSource


@dataclass(frozen=True)
class BenchmarkDiagnosticsConfig:
    """Strict Stage 2B source and calculation contract."""

    source: Path
    project_root: Path
    output_root: Path
    data_snapshot_id: str
    primary_calibration_bins: int
    calibration_bins: tuple[int, ...]
    rolling_window: int
    bootstrap_repetitions: int
    smoke_repetitions: int
    bootstrap_block_length: int
    bootstrap_seed: int
    financial_architecture_manifest: FrozenSource
    qrc_exact_table: FrozenSource
    financial_manifest: FrozenSource
    financial_training: FrozenSource
    financial_splits: dict[str, FrozenSource]
    regime_thresholds: FrozenSource
    regression_models: dict[str, SplitModelSource]
    classification_models: dict[str, SplitModelSource]
    qrc_numerical: dict[int, FrozenSource]
    mnist_selected_indices: FrozenSource
    mnist_models: dict[str, FrozenSource]
    mnist_robustness: dict[str, FrozenSource]
    raw: dict[str, Any]


@dataclass(frozen=True)
class RegressionPrediction:
    """One frozen variance forecast vector."""

    model_id: str
    display_name: str
    reservoir_seed: int | None
    forecasts: NDArray[np.float64]


@dataclass(frozen=True)
class ClassificationPrediction:
    """Frozen regime labels, probabilities, and supplied transition scores."""

    model_id: str
    display_name: str
    reservoir_seed: int | None
    probability_model: bool
    predictions: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    transition_scores: NDArray[np.float64]


@dataclass(frozen=True)
class FinancialSplit:
    """Aligned frozen financial observations and all predictions for one split."""

    split: str
    dates: NDArray[np.str_]
    realised_variance: NDArray[np.float64]
    regimes: NDArray[np.int64]
    current_regimes: NDArray[np.int64]
    transitions: NDArray[np.int64]
    regression: dict[str, RegressionPrediction]
    classification: dict[str, ClassificationPrediction]


@dataclass(frozen=True)
class MNISTPrediction:
    """Aligned frozen MNIST labels and probabilities."""

    model_id: str
    display_name: str
    predictions: NDArray[np.int64]
    probabilities: NDArray[np.float64]


@dataclass(frozen=True)
class MNISTData:
    """The immutable 1,000-image official-test subset."""

    identities: tuple[tuple[str, int], ...]
    truth: NDArray[np.int64]
    models: dict[str, MNISTPrediction]
    robustness: dict[str, MNISTPrediction]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, location: str) -> dict[Any, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    return dict(value)


def _source(
    project_root: Path,
    source_id: str,
    raw: object,
    *,
    default_display: str,
) -> FrozenSource:
    record = _mapping(raw, source_id)
    path_value = record.get("path")
    checksum = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError(f"{source_id} must contain a path and SHA-256")
    return FrozenSource(
        source_id=source_id,
        display_name=str(record.get("display_name", default_display)),
        path=(project_root / path_value).resolve(),
        sha256=checksum,
        metadata=cast(dict[str, Any], record),
    )


def _split_model_sources(
    project_root: Path,
    raw: object,
    location: str,
) -> dict[str, SplitModelSource]:
    records = _mapping(raw, location)
    result: dict[str, SplitModelSource] = {}
    for raw_model_id, raw_record in records.items():
        model_id = str(raw_model_id)
        record = _mapping(raw_record, f"{location}.{model_id}")
        display_name = str(record.get("display_name", model_id))
        seed_value = record.get("reservoir_seed")
        reservoir_seed = int(seed_value) if seed_value is not None else None
        result[model_id] = SplitModelSource(
            model_id=model_id,
            display_name=display_name,
            reservoir_seed=reservoir_seed,
            probability_model=bool(record.get("probability_model", False)),
            validation=_source(
                project_root,
                f"{location}.{model_id}.validation",
                record.get("validation"),
                default_display=display_name,
            ),
            test=_source(
                project_root,
                f"{location}.{model_id}.test",
                record.get("test"),
                default_display=display_name,
            ),
        )
    return result


def load_benchmark_diagnostics_config(path: Path) -> BenchmarkDiagnosticsConfig:
    """Load and validate the complete frozen Stage 2B source contract."""

    source_path = path.resolve()
    root = _mapping(yaml.safe_load(source_path.read_text(encoding="utf-8")), "configuration")
    if root.get("schema_version") != 1:
        raise ValueError("benchmark-diagnostics schema_version must be 1")
    study = _mapping(root.get("study"), "study")
    if study.get("id") != STUDY_ID:
        raise ValueError(f"study.id must remain {STUDY_ID}")
    project_setting = study.get("project_root")
    output_setting = study.get("output_root")
    if not isinstance(project_setting, str) or not isinstance(output_setting, str):
        raise ValueError("study paths must be strings")
    project_root = (source_path.parent / project_setting).resolve()
    diagnostics = _mapping(root.get("diagnostics"), "diagnostics")
    bootstrap = _mapping(diagnostics.get("bootstrap"), "diagnostics.bootstrap")
    sensitivity = diagnostics.get("sensitivity_calibration_bins")
    if not isinstance(sensitivity, list) or any(
        not isinstance(value, int) for value in sensitivity
    ):
        raise ValueError("calibration sensitivity bins must be integers")
    primary_bins = int(diagnostics.get("primary_calibration_bins", -1))
    calibration_bins = tuple(sorted({primary_bins, *(int(value) for value in sensitivity)}))
    financial_data = _mapping(root.get("financial_data"), "financial_data")
    financial_splits = {
        split: _source(
            project_root,
            f"financial_data.{split}",
            financial_data.get(split),
            default_display=f"Public financial {split}",
        )
        for split in SPLITS
    }
    qrc_numerical_raw = _mapping(root.get("qrc_numerical"), "qrc_numerical")
    qrc_numerical = {
        seed: _source(
            project_root,
            f"qrc_numerical.{seed}",
            qrc_numerical_raw.get(seed),
            default_display=f"QRC {seed} numerical diagnostics",
        )
        for seed in QRC_SEEDS
    }
    mnist = _mapping(root.get("mnist"), "mnist")

    def flat_sources(raw: object, location: str) -> dict[str, FrozenSource]:
        records = _mapping(raw, location)
        return {
            str(source_id): _source(
                project_root,
                f"{location}.{source_id}",
                record,
                default_display=str(source_id),
            )
            for source_id, record in records.items()
        }

    config = BenchmarkDiagnosticsConfig(
        source=source_path,
        project_root=project_root,
        output_root=(project_root / output_setting).resolve(),
        data_snapshot_id=str(study.get("data_snapshot_id", "")),
        primary_calibration_bins=primary_bins,
        calibration_bins=calibration_bins,
        rolling_window=int(diagnostics.get("rolling_window", -1)),
        bootstrap_repetitions=int(bootstrap.get("repetitions", -1)),
        smoke_repetitions=int(bootstrap.get("smoke_repetitions", -1)),
        bootstrap_block_length=int(bootstrap.get("block_length", -1)),
        bootstrap_seed=int(bootstrap.get("seed", -1)),
        financial_architecture_manifest=_source(
            project_root,
            "financial_architecture_manifest",
            study.get("financial_architecture_manifest"),
            default_display="Frozen financial architecture manifest",
        ),
        qrc_exact_table=_source(
            project_root,
            "qrc_exact_table",
            study.get("qrc_exact_table"),
            default_display="Frozen QRC exact table",
        ),
        financial_manifest=_source(
            project_root,
            "financial_data.manifest",
            financial_data.get("manifest"),
            default_display="Public financial data manifest",
        ),
        financial_training=_source(
            project_root,
            "financial_data.training",
            financial_data.get("training"),
            default_display="Public financial training split",
        ),
        financial_splits=financial_splits,
        regime_thresholds=_source(
            project_root,
            "financial_data.regime_thresholds",
            financial_data.get("regime_thresholds"),
            default_display="Frozen training regime thresholds",
        ),
        regression_models=_split_model_sources(
            project_root,
            root.get("financial_regression"),
            "financial_regression",
        ),
        classification_models=_split_model_sources(
            project_root,
            root.get("financial_classification"),
            "financial_classification",
        ),
        qrc_numerical=qrc_numerical,
        mnist_selected_indices=_source(
            project_root,
            "mnist.selected_indices",
            mnist.get("selected_indices"),
            default_display="Frozen MNIST selected identities",
        ),
        mnist_models=flat_sources(mnist.get("models"), "mnist.models"),
        mnist_robustness=flat_sources(mnist.get("robustness"), "mnist.robustness"),
        raw=cast(dict[str, Any], root),
    )
    if (
        config.data_snapshot_id != "yahoo_chart_20100101_20251231_v1"
        or config.primary_calibration_bins != 10
        or config.calibration_bins != CALIBRATION_BINS
        or config.rolling_window != 60
        or config.bootstrap_repetitions != 5_000
        or config.smoke_repetitions <= 0
        or config.bootstrap_block_length != 10
        or config.bootstrap_seed != 2026
        or tuple(config.regression_models) != REGRESSION_MODELS
        or tuple(config.classification_models) != CLASSIFICATION_MODELS
        or tuple(config.mnist_models) != MNIST_MODELS
        or tuple(config.mnist_robustness) != MNIST_ROBUSTNESS
        or tuple(sorted(config.qrc_numerical)) != QRC_SEEDS
    ):
        raise ValueError("Stage 2B frozen sources or diagnostic controls changed")
    actual_probability = tuple(
        model_id
        for model_id, model in config.classification_models.items()
        if model.probability_model
    )
    if actual_probability != PROBABILITY_MODELS:
        raise ValueError("financial probability-calibration model set changed")
    return config


def _all_sources(config: BenchmarkDiagnosticsConfig) -> tuple[FrozenSource, ...]:
    return (
        config.financial_architecture_manifest,
        config.qrc_exact_table,
        config.financial_manifest,
        config.financial_training,
        *config.financial_splits.values(),
        config.regime_thresholds,
        *[
            source
            for model in config.regression_models.values()
            for source in (model.validation, model.test)
        ],
        *[
            source
            for model in config.classification_models.values()
            for source in (model.validation, model.test)
        ],
        *config.qrc_numerical.values(),
        config.mnist_selected_indices,
        *config.mnist_models.values(),
        *config.mnist_robustness.values(),
    )


def verify_frozen_diagnostic_sources(
    config: BenchmarkDiagnosticsConfig,
) -> list[dict[str, Any]]:
    """Verify every unique frozen input and reject output-tree aliasing."""

    records: list[dict[str, Any]] = []
    verified: dict[Path, str] = {}
    for source in _all_sources(config):
        prior = verified.get(source.path)
        if prior is not None:
            if prior != source.sha256:
                raise ValueError(f"conflicting checksums for duplicate source {source.path}")
            continue
        if not source.path.is_file():
            raise FileNotFoundError(f"missing frozen diagnostic input: {source.path}")
        actual = sha256_path(source.path)
        if actual != source.sha256:
            raise ValueError(
                f"frozen diagnostic checksum mismatch for {source.source_id}: "
                f"expected {source.sha256}, got {actual}"
            )
        if source.path == config.output_root or config.output_root in source.path.parents:
            raise ValueError("frozen diagnostic input points inside the Stage 2B output tree")
        verified[source.path] = actual
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


def _probability_columns(class_count: int, *, prefix: str) -> list[str]:
    return [f"{prefix}{index}" for index in range(class_count)]


def _load_financial_splits(
    config: BenchmarkDiagnosticsConfig,
) -> tuple[dict[str, FinancialSplit], dict[str, float], dict[str, Any]]:
    manifest = json.loads(config.financial_manifest.path.read_text(encoding="utf-8"))
    if (
        manifest.get("data_source_type") != "public_market"
        or manifest.get("is_synthetic") is not False
        or manifest.get("row_counts")
        != {
            "canonical_source": 4024,
            "features_unscaled": 3989,
            "test": 497,
            "train": 2744,
            "validation": 748,
        }
    ):
        raise ValueError("Stage 2B requires the frozen non-synthetic public-market dataset")
    training = pd.read_csv(config.financial_training.path)
    thresholds_payload = json.loads(config.regime_thresholds.path.read_text(encoding="utf-8"))
    if (
        thresholds_payload.get("fit_split") != "train"
        or thresholds_payload.get("fit_column") != "target_rv_5d"
        or thresholds_payload.get("training_rows") != len(training)
    ):
        raise ValueError("regime thresholds are not the frozen training-derived thresholds")
    low_medium = float(thresholds_payload["low_medium"])
    medium_high = float(thresholds_payload["medium_high"])
    training_truth = training["target_rv_5d"].to_numpy(dtype=float)
    if not np.array_equal(
        fixed_regime_labels(
            training_truth,
            low_medium=low_medium,
            medium_high=medium_high,
        ),
        training["target_regime_5d"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("frozen training regimes do not match their stored thresholds")
    tail_thresholds = {
        "training_p90": float(np.quantile(training_truth, 0.90)),
        "training_p95": float(np.quantile(training_truth, 0.95)),
    }
    splits: dict[str, FinancialSplit] = {}
    for split in SPLITS:
        processed = pd.read_csv(config.financial_splits[split].path)
        expected_count = 748 if split == "validation" else 497
        if len(processed) != expected_count or set(processed["split"]) != {split}:
            raise ValueError(f"frozen public financial {split} split changed")
        dates = processed["date"].astype(str).to_numpy(dtype=str)
        realised = processed["target_rv_5d"].to_numpy(dtype=float)
        regimes = processed["target_regime_5d"].to_numpy(dtype=np.int64)
        current = processed["current_regime"].to_numpy(dtype=np.int64)
        transitions = processed["target_transition"].to_numpy(dtype=np.int64)
        if not np.array_equal(
            fixed_regime_labels(
                realised,
                low_medium=low_medium,
                medium_high=medium_high,
            ),
            regimes,
        ):
            raise ValueError(f"{split} regimes do not use the frozen training thresholds")
        regression: dict[str, RegressionPrediction] = {}
        regression_reference: pd.DataFrame | None = None
        for model_id, model in config.regression_models.items():
            source = model.validation if split == "validation" else model.test
            frame = pd.read_csv(source.path)
            if regression_reference is None:
                regression_reference = frame
            else:
                require_exact_alignment(
                    regression_reference,
                    frame,
                    identity_columns=("date",),
                    truth_columns=("true_rv_5d",),
                    candidate_name=f"{model_id} {split}",
                )
            if (
                len(frame) != expected_count
                or not np.array_equal(frame["date"].astype(str).to_numpy(dtype=str), dates)
                or not np.array_equal(frame["true_rv_5d"].to_numpy(dtype=float), realised)
            ):
                raise ValueError(f"{model_id} {split} variance predictions do not align")
            forecasts = frame["predicted_rv_5d"].to_numpy(dtype=float)
            if not np.isfinite(forecasts).all() or np.any(forecasts <= 0.0):
                raise ValueError(f"{model_id} {split} has invalid frozen variance forecasts")
            regression[model_id] = RegressionPrediction(
                model_id,
                model.display_name,
                model.reservoir_seed,
                forecasts,
            )
        classification: dict[str, ClassificationPrediction] = {}
        classification_reference: pd.DataFrame | None = None
        probability_columns = [
            "probability_low",
            "probability_medium",
            "probability_high",
        ]
        for model_id, model in config.classification_models.items():
            source = model.validation if split == "validation" else model.test
            frame = pd.read_csv(source.path)
            if classification_reference is None:
                classification_reference = frame
            else:
                require_exact_alignment(
                    classification_reference,
                    frame,
                    identity_columns=("date",),
                    truth_columns=("current_regime", "true_regime", "true_transition"),
                    candidate_name=f"{model_id} {split}",
                )
            if (
                len(frame) != expected_count
                or not np.array_equal(frame["date"].astype(str).to_numpy(dtype=str), dates)
                or not np.array_equal(frame["true_regime"].to_numpy(dtype=np.int64), regimes)
                or not np.array_equal(
                    frame["current_regime"].to_numpy(dtype=np.int64),
                    current,
                )
                or not np.array_equal(
                    frame["true_transition"].to_numpy(dtype=np.int64),
                    transitions,
                )
            ):
                raise ValueError(f"{model_id} {split} classification predictions do not align")
            probabilities = frame[probability_columns].to_numpy(dtype=float)
            predictions = frame["predicted_regime"].to_numpy(dtype=np.int64)
            transition_scores = frame["predicted_transition_probability"].to_numpy(dtype=float)
            if (
                not np.isfinite(probabilities).all()
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
                or not np.array_equal(np.argmax(probabilities, axis=1), predictions)
                or not np.isfinite(transition_scores).all()
            ):
                raise ValueError(f"{model_id} {split} has invalid frozen probabilities")
            classification[model_id] = ClassificationPrediction(
                model_id,
                model.display_name,
                model.reservoir_seed,
                model.probability_model,
                predictions,
                probabilities,
                transition_scores,
            )
        splits[split] = FinancialSplit(
            split,
            dates,
            realised,
            regimes,
            current,
            transitions,
            regression,
            classification,
        )
    return splits, tail_thresholds, thresholds_payload


def _load_mnist(config: BenchmarkDiagnosticsConfig) -> MNISTData:
    selected = json.loads(config.mnist_selected_indices.path.read_text(encoding="utf-8"))
    test_selection = selected.get("splits", {}).get("test", {})
    if (
        selected.get("selection_seed") != 2026
        or test_selection.get("source_partition") != "official_test"
        or test_selection.get("rows") != 1000
        or test_selection.get("class_counts") != {str(index): 100 for index in range(10)}
    ):
        raise ValueError("Stage 2B requires the frozen genuine-MNIST official-test subset")
    probability_columns = _probability_columns(10, prefix="probability_")
    reference: pd.DataFrame | None = None

    def load_prediction(model_id: str, source: FrozenSource) -> MNISTPrediction:
        nonlocal reference
        frame = pd.read_csv(source.path)
        if reference is None:
            reference = frame
        else:
            require_exact_alignment(
                reference,
                frame,
                identity_columns=("official_partition", "official_index"),
                truth_columns=("true_digit",),
                candidate_name=f"MNIST {model_id}",
            )
        if len(frame) != 1000 or set(frame["official_partition"].astype(str)) != {"official_test"}:
            raise ValueError(f"MNIST {model_id} does not use the frozen official-test subset")
        probabilities = frame[probability_columns].to_numpy(dtype=float)
        predictions = frame["predicted_digit"].to_numpy(dtype=np.int64)
        if (
            not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
            or not np.array_equal(np.argmax(probabilities, axis=1), predictions)
        ):
            raise ValueError(f"MNIST {model_id} probabilities are invalid")
        return MNISTPrediction(
            model_id,
            source.display_name,
            predictions,
            probabilities,
        )

    models = {
        model_id: load_prediction(model_id, source)
        for model_id, source in config.mnist_models.items()
    }
    robustness = {
        condition: load_prediction(condition, source)
        for condition, source in config.mnist_robustness.items()
    }
    assert reference is not None
    identities = tuple(
        zip(
            reference["official_partition"].astype(str),
            reference["official_index"].astype(int),
        )
    )
    return MNISTData(
        identities,
        reference["true_digit"].to_numpy(dtype=np.int64),
        models,
        robustness,
    )


def _financial_tables(
    config: BenchmarkDiagnosticsConfig,
    splits: dict[str, FinancialSplit],
    *,
    tail_thresholds: dict[str, float],
    repetitions: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, NDArray[np.int32]]]:
    counts_by_split: dict[str, NDArray[np.int32]] = {}
    for split, data in splits.items():
        indices = circular_block_bootstrap_indices(
            len(data.dates),
            repetitions,
            config.bootstrap_block_length,
            config.bootstrap_seed,
        )
        counts_by_split[split] = indices_to_counts(indices, len(data.dates))

    overall_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    transition_type_rows: list[dict[str, Any]] = []
    transition_subset_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []

    for split, data in splits.items():
        bootstrap_counts = counts_by_split[split]
        for model_id, model in data.regression.items():
            common = {
                "analysis": "variance_calibration",
                "model": model_id,
                "model_display_name": model.display_name,
                "reservoir_seed": model.reservoir_seed,
                "split": split,
                "data_snapshot_id": config.data_snapshot_id,
                "frozen_predictions": True,
                "diagnostic_only": True,
            }
            overall_rows.append(
                {
                    **common,
                    **variance_point_metrics(data.realised_variance, model.forecasts),
                    **variance_bootstrap_intervals(
                        data.realised_variance,
                        model.forecasts,
                        bootstrap_counts,
                    ),
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_block_length": config.bootstrap_block_length,
                    "bootstrap_seed": config.bootstrap_seed,
                }
            )
            for row in variance_calibration_deciles(
                data.realised_variance,
                model.forecasts,
            ):
                decile_rows.append({**common, **row})
            for row in variance_regime_diagnostics(
                data.realised_variance,
                model.forecasts,
                data.regimes,
                bootstrap_counts=bootstrap_counts,
            ):
                regime_rows.append(
                    {
                        **common,
                        **row,
                        "regime_threshold_source": "frozen training-derived thresholds",
                        "bootstrap_repetitions": repetitions,
                        "bootstrap_block_length": config.bootstrap_block_length,
                        "bootstrap_seed": config.bootstrap_seed,
                    }
                )
            for row in variance_tail_diagnostics(
                data.realised_variance,
                model.forecasts,
                thresholds=tail_thresholds,
                bootstrap_counts=bootstrap_counts,
            ):
                tail_rows.append(
                    {
                        **common,
                        **row,
                        "bootstrap_repetitions": repetitions,
                        "bootstrap_block_length": config.bootstrap_block_length,
                        "bootstrap_seed": config.bootstrap_seed,
                    }
                )
            for row in rolling_variance_diagnostics(
                data.dates,
                data.realised_variance,
                model.forecasts,
                window=config.rolling_window,
            ):
                temporal_rows.append(
                    {
                        "analysis": "temporal_error",
                        "diagnostic_type": "rolling_variance",
                        "model": model_id,
                        "model_display_name": model.display_name,
                        "reference_model": None,
                        "split": split,
                        **row,
                    }
                )

        qlike_by_model = {
            model_id: qlike_values(data.realised_variance, model.forecasts)
            for model_id, model in data.regression.items()
        }
        for reference_id in ("garch_1_1", "esn_regressor"):
            for model_id, losses in qlike_by_model.items():
                values = cumulative_loss_difference(
                    losses,
                    qlike_by_model[reference_id],
                )
                for index, value in enumerate(values):
                    temporal_rows.append(
                        {
                            "analysis": "temporal_error",
                            "diagnostic_type": "cumulative_qlike_difference",
                            "model": model_id,
                            "model_display_name": data.regression[model_id].display_name,
                            "reference_model": reference_id,
                            "reference_model_display_name": data.regression[
                                reference_id
                            ].display_name,
                            "split": split,
                            "date": str(data.dates[index]),
                            "cumulative_difference": float(value),
                            "difference_definition": (
                                "cumulative_QLIKE_model_minus_QLIKE_reference"
                            ),
                            "negative_favours": "model",
                            "positive_favours": "reference",
                            "used_for_selection": False,
                        }
                    )

        for model_id, classification_model in data.classification.items():
            common_classification = {
                "analysis": "regime_classification",
                "model": model_id,
                "model_display_name": classification_model.display_name,
                "reservoir_seed": classification_model.reservoir_seed,
                "split": split,
                "data_snapshot_id": config.data_snapshot_id,
                "frozen_predictions": True,
                "diagnostic_only": True,
            }
            classification_rows.append(
                {
                    **common_classification,
                    **classification_diagnostics(
                        data.regimes,
                        classification_model.predictions,
                    ),
                }
            )
            for row in transition_type_diagnostics(
                data.current_regimes,
                data.regimes,
                classification_model.predictions,
                classification_model.transition_scores,
            ):
                transition_type_rows.append({**common_classification, **row})
            for row in transition_subset_diagnostics(
                data.regimes,
                classification_model.predictions,
                data.transitions,
                classification_model.transition_scores,
                probabilities=classification_model.probabilities,
            ):
                transition_subset_rows.append({**common_classification, **row})
            if classification_model.probability_model:
                for bin_count in config.calibration_bins:
                    probability_rows.append(
                        {
                            "analysis": "probability_calibration",
                            "model": model_id,
                            "model_display_name": classification_model.display_name,
                            "reservoir_seed": classification_model.reservoir_seed,
                            "split": split,
                            "primary_bin_count": (bin_count == config.primary_calibration_bins),
                            "probabilities_recalibrated": False,
                            "diagnostic_only": True,
                            **multiclass_calibration_summary(
                                data.regimes,
                                classification_model.probabilities,
                                bin_count=bin_count,
                            ),
                        }
                    )
                    for row in classwise_calibration_summary(
                        data.regimes,
                        classification_model.probabilities,
                        bin_count=bin_count,
                    ):
                        classwise_rows.append(
                            {
                                "analysis": "classwise_probability_calibration",
                                "model": model_id,
                                "model_display_name": classification_model.display_name,
                                "reservoir_seed": classification_model.reservoir_seed,
                                "split": split,
                                "class_name": REGIME_NAMES[int(row["class_label"])],
                                "primary_bin_count": (bin_count == config.primary_calibration_bins),
                                "probabilities_recalibrated": False,
                                "diagnostic_only": True,
                                **row,
                            }
                        )

        logistic = data.classification["logistic_regression"]
        for model_id, classification_model in data.classification.items():
            values = cumulative_classification_error_difference(
                data.regimes,
                classification_model.predictions,
                logistic.predictions,
            )
            for index, value in enumerate(values):
                temporal_rows.append(
                    {
                        "analysis": "temporal_error",
                        "diagnostic_type": ("cumulative_classification_error_difference"),
                        "model": model_id,
                        "model_display_name": classification_model.display_name,
                        "reference_model": "logistic_regression",
                        "reference_model_display_name": logistic.display_name,
                        "split": split,
                        "date": str(data.dates[index]),
                        "cumulative_difference": float(value),
                        "difference_definition": (
                            "cumulative_error_model_minus_error_logistic_regression"
                        ),
                        "negative_favours": "model",
                        "positive_favours": "logistic_regression",
                        "used_for_selection": False,
                    }
                )

    test = splits["test"]
    largest_indices = np.argsort(-test.realised_variance, kind="stable")[:5]
    for rank, index in enumerate(largest_indices, start=1):
        temporal_rows.append(
            {
                "analysis": "temporal_error",
                "diagnostic_type": "largest_realised_variance_observation",
                "rank": rank,
                "split": "test",
                "date": str(test.dates[index]),
                "realised_variance": float(test.realised_variance[index]),
                "all_model_forecasts": {
                    model_id: float(model.forecasts[index])
                    for model_id, model in test.regression.items()
                },
                **{
                    f"forecast_{model_id}": float(model.forecasts[index])
                    for model_id, model in test.regression.items()
                },
                "all_financial_variance_models_included": True,
                "used_for_selection": False,
            }
        )
    return (
        {
            "variance_overall_calibration": overall_rows,
            "variance_decile_calibration": decile_rows,
            "variance_regime_diagnostics": regime_rows,
            "variance_tail_diagnostics": tail_rows,
            "probability_calibration": probability_rows,
            "classwise_probability_calibration": classwise_rows,
            "regime_classification_diagnostics": classification_rows,
            "transition_type_diagnostics": transition_type_rows,
            "transition_vs_nontransition": transition_subset_rows,
            "temporal_error_diagnostics": temporal_rows,
        },
        counts_by_split,
    )


def _per_digit_bootstrap_distributions(
    truth: NDArray[np.int64],
    predictions: NDArray[np.int64],
    counts: NDArray[np.int32],
) -> dict[int, dict[str, NDArray[np.float64]]]:
    result: dict[int, dict[str, NDArray[np.float64]]] = {}
    weights = np.asarray(counts, dtype=float)
    for digit in range(10):
        true_indicator = (truth == digit).astype(float)
        predicted_indicator = (predictions == digit).astype(float)
        true_count = np.einsum("ij,j->i", weights, true_indicator)
        predicted_count = np.einsum("ij,j->i", weights, predicted_indicator)
        true_positive = np.einsum(
            "ij,j->i",
            weights,
            true_indicator * predicted_indicator,
        )
        accuracy = np.divide(
            true_positive,
            true_count,
            out=np.full(len(weights), np.nan, dtype=float),
            where=true_count > 0.0,
        )
        denominator = true_count + predicted_count
        f1 = np.divide(
            2.0 * true_positive,
            denominator,
            out=np.full(len(weights), np.nan, dtype=float),
            where=denominator > 0.0,
        )
        result[digit] = {"accuracy": accuracy, "f1": f1}
    return result


def _interval(values: NDArray[np.float64]) -> tuple[float, float, int, int]:
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    if not len(finite):
        raise ValueError("diagnostic bootstrap produced no valid values")
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return float(lower), float(upper), int(finite_mask.sum()), int((~finite_mask).sum())


def _mnist_tables(
    config: BenchmarkDiagnosticsConfig,
    data: MNISTData,
    *,
    repetitions: int,
) -> dict[str, list[dict[str, Any]]]:
    indices = stratified_bootstrap_indices(
        data.truth,
        repetitions,
        config.bootstrap_seed,
    )
    counts = indices_to_counts(indices, len(data.truth))
    per_digit_rows: list[dict[str, Any]] = []
    model_distributions: dict[str, dict[int, dict[str, NDArray[np.float64]]]] = {}
    for model_id, model in data.models.items():
        diagnostics = per_class_diagnostics(
            data.truth,
            model.predictions,
            labels=tuple(range(10)),
        )
        distributions = _per_digit_bootstrap_distributions(
            data.truth,
            model.predictions,
            counts,
        )
        model_distributions[model_id] = distributions
        confidence = np.max(model.probabilities, axis=1)
        entropy = probability_entropy(model.probabilities)
        for row in diagnostics:
            digit = int(row["class_label"])
            accuracy_interval = _interval(distributions[digit]["accuracy"])
            f1_interval = _interval(distributions[digit]["f1"])
            digit_mask = data.truth == digit
            per_digit_rows.append(
                {
                    "analysis": "mnist_per_digit",
                    "model": model_id,
                    "model_display_name": model.display_name,
                    "digit": digit,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "accuracy_within_true_digit": row["recall"],
                    "support": row["support"],
                    "accuracy_ci_lower": accuracy_interval[0],
                    "accuracy_ci_upper": accuracy_interval[1],
                    "f1_ci_lower": f1_interval[0],
                    "f1_ci_upper": f1_interval[1],
                    "bootstrap_valid_count": min(accuracy_interval[2], f1_interval[2]),
                    "bootstrap_invalid_count": max(accuracy_interval[3], f1_interval[3]),
                    "mean_confidence": float(confidence[digit_mask].mean()),
                    "mean_probability_entropy": float(entropy[digit_mask].mean()),
                    "confusion_row": confusion_counts(
                        data.truth,
                        model.predictions,
                        labels=tuple(range(10)),
                    )[digit]
                    .astype(int)
                    .tolist(),
                    "bootstrap_method": "paired class-stratified",
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_seed": config.bootstrap_seed,
                    "frozen_predictions": True,
                }
            )

    overlap_rows: list[dict[str, Any]] = []
    totals = counts.sum(axis=1)
    for seed in QRC_SEEDS:
        qrc_id = f"qrc_{seed}"
        qrc = data.models[qrc_id]
        qrc_correct = qrc.predictions == data.truth
        for baseline_id in ("flattened_logistic", "esn"):
            baseline = data.models[baseline_id]
            baseline_correct = baseline.predictions == data.truth
            per_digit_accuracy_difference = {
                digit: float(
                    qrc_correct[data.truth == digit].mean()
                    - baseline_correct[data.truth == digit].mean()
                )
                for digit in range(10)
            }
            largest_gain_digit = max(
                per_digit_accuracy_difference,
                key=lambda digit: (per_digit_accuracy_difference[digit], -digit),
            )
            largest_loss_digit = min(
                per_digit_accuracy_difference,
                key=lambda digit: (per_digit_accuracy_difference[digit], digit),
            )
            difference = (
                np.einsum(
                    "ij,j->i",
                    counts,
                    qrc_correct.astype(float) - baseline_correct.astype(float),
                )
                / totals
            )
            lower, upper, valid_count, invalid_count = _interval(difference)
            overlap_rows.append(
                {
                    "analysis": "mnist_paired_error_overlap",
                    "qrc_seed": seed,
                    "qrc_model": qrc_id,
                    "baseline": baseline_id,
                    "baseline_display_name": baseline.display_name,
                    "both_correct": int(np.sum(qrc_correct & baseline_correct)),
                    "qrc_only_correct": int(np.sum(qrc_correct & ~baseline_correct)),
                    "baseline_only_correct": int(np.sum(~qrc_correct & baseline_correct)),
                    "both_wrong": int(np.sum(~qrc_correct & ~baseline_correct)),
                    "observed_accuracy_difference": float(
                        qrc_correct.mean() - baseline_correct.mean()
                    ),
                    "per_digit_accuracy_difference": {
                        str(digit): difference
                        for digit, difference in per_digit_accuracy_difference.items()
                    },
                    "largest_qrc_accuracy_gain_digit": largest_gain_digit,
                    "largest_qrc_accuracy_gain": per_digit_accuracy_difference[largest_gain_digit],
                    "largest_qrc_accuracy_loss_digit": largest_loss_digit,
                    "largest_qrc_accuracy_loss": per_digit_accuracy_difference[largest_loss_digit],
                    "accuracy_difference_ci_lower": lower,
                    "accuracy_difference_ci_upper": upper,
                    "bootstrap_valid_count": valid_count,
                    "bootstrap_invalid_count": invalid_count,
                    "bootstrap_method": "paired class-stratified",
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_seed": config.bootstrap_seed,
                    "probability_or_prediction_averaging_used": False,
                }
            )

    robustness_rows: list[dict[str, Any]] = []
    robustness_distributions = {
        condition: _per_digit_bootstrap_distributions(
            data.truth,
            model.predictions,
            counts,
        )
        for condition, model in data.robustness.items()
    }
    robustness_points = {
        condition: {
            int(row["class_label"]): row
            for row in per_class_diagnostics(
                data.truth,
                model.predictions,
                labels=tuple(range(10)),
            )
        }
        for condition, model in data.robustness.items()
    }
    analytic = data.robustness["analytic"]
    analytic_matrix = confusion_counts(
        data.truth,
        analytic.predictions,
        labels=tuple(range(10)),
    ).astype(int)
    for condition, model in data.robustness.items():
        matrix = confusion_counts(
            data.truth,
            model.predictions,
            labels=tuple(range(10)),
        ).astype(int)
        for digit in range(10):
            point = robustness_points[condition][digit]
            analytic_point = robustness_points["analytic"][digit]
            accuracy_interval = _interval(robustness_distributions[condition][digit]["accuracy"])
            f1_interval = _interval(robustness_distributions[condition][digit]["f1"])
            digit_mask = data.truth == digit
            accuracy = float(point["recall"])
            f1 = float(point["f1"])
            analytic_accuracy = float(analytic_point["recall"])
            analytic_f1 = float(analytic_point["f1"])
            confusion_delta = matrix[digit] - analytic_matrix[digit]
            robustness_rows.append(
                {
                    "analysis": "mnist_robustness_by_digit",
                    "condition": condition,
                    "condition_display_name": config.mnist_robustness[condition].display_name,
                    "reservoir_seed": 2026,
                    "digit": digit,
                    "accuracy": accuracy,
                    "f1": f1,
                    "accuracy_ci_lower": accuracy_interval[0],
                    "accuracy_ci_upper": accuracy_interval[1],
                    "f1_ci_lower": f1_interval[0],
                    "f1_ci_upper": f1_interval[1],
                    "accuracy_degradation_from_analytic": (analytic_accuracy - accuracy),
                    "f1_degradation_from_analytic": analytic_f1 - f1,
                    "absolute_accuracy_change_from_analytic": abs(analytic_accuracy - accuracy),
                    "absolute_f1_change_from_analytic": abs(analytic_f1 - f1),
                    "changed_predictions_from_analytic": int(
                        np.sum(model.predictions[digit_mask] != analytic.predictions[digit_mask])
                    ),
                    "confusion_pattern_change_by_predicted_digit": {
                        str(predicted_digit): int(confusion_delta[predicted_digit])
                        for predicted_digit in range(10)
                    },
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_seed": config.bootstrap_seed,
                    "physical_qpu_execution": False,
                    "frozen_predictions": True,
                }
            )
    return {
        "mnist_per_digit_diagnostics": per_digit_rows,
        "mnist_pairwise_error_overlap": overlap_rows,
        "mnist_robustness_by_digit": robustness_rows,
    }


def _qrc_seed_numerical_rows(
    config: BenchmarkDiagnosticsConfig,
    splits: dict[str, FinancialSplit],
    financial_tables: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    exact_payload = json.loads(config.qrc_exact_table.path.read_text(encoding="utf-8"))
    exact_rows = exact_payload.get("rows")
    if not isinstance(exact_rows, list) or len(exact_rows) != 12:
        raise ValueError("frozen final-QRC exact table changed")

    def exact_row(seed: int, task: str) -> dict[str, Any]:
        matches = [
            cast(dict[str, Any], row)
            for row in exact_rows
            if row.get("reservoir_seed") == seed
            and row.get("task") == task
            and row.get("split") == "validation"
        ]
        if len(matches) != 1:
            raise ValueError(f"missing frozen QRC diagnostics for {seed} {task}")
        return matches[0]

    probability_rows = financial_tables["probability_calibration"]
    regime_rows = financial_tables["variance_regime_diagnostics"]
    tail_rows = financial_tables["variance_tail_diagnostics"]
    rows: list[dict[str, Any]] = []
    for seed in QRC_SEEDS:
        model_id = f"qrc_{seed}"
        classifier = exact_row(seed, "regime_classification")
        regressor = exact_row(seed, "rv_regression")
        numerical = json.loads(config.qrc_numerical[seed].path.read_text(encoding="utf-8"))
        test_regression = splits["test"].regression[model_id].forecasts
        validation_regression = splits["validation"].regression[model_id].forecasts
        test_classification = splits["test"].classification[model_id]
        validation_classification = splits["validation"].classification[model_id]
        test_probability = next(
            row
            for row in probability_rows
            if row["model"] == model_id and row["split"] == "test" and row["primary_bin_count"]
        )
        validation_probability = next(
            row
            for row in probability_rows
            if row["model"] == model_id
            and row["split"] == "validation"
            and row["primary_bin_count"]
        )
        high_regime = next(
            row
            for row in regime_rows
            if row["model"] == model_id and row["split"] == "test" and row["regime"] == "high"
        )
        p95_tail = next(
            row
            for row in tail_rows
            if row["model"] == model_id
            and row["split"] == "test"
            and row["tail_threshold_id"] == "training_p95"
        )
        arrays_finite = all(
            np.isfinite(values).all()
            for values in (
                test_regression,
                validation_regression,
                test_classification.predictions.astype(float),
                validation_classification.predictions.astype(float),
                test_classification.probabilities,
                validation_classification.probabilities,
                test_classification.transition_scores,
                validation_classification.transition_scores,
            )
        )
        rows.append(
            {
                "analysis": "qrc_seed_numerical_diagnostics",
                "reservoir_seed": seed,
                "near_singular_seed_2027": seed == 2027,
                "condition_number": classifier["condition_number"],
                "effective_feature_rank": classifier["effective_rank"],
                "numerical_rank": classifier["numerical_rank"],
                "largest_singular_value": classifier["largest_singular_value"],
                "smallest_retained_singular_value": classifier["smallest_retained_singular_value"],
                "classifier_ridge_alpha": classifier["selected_ridge_alpha"],
                "regressor_ridge_alpha": regressor["selected_ridge_alpha"],
                "classifier_coefficient_l2_norm": classifier["readout_coefficient_l2_norm"],
                "regressor_coefficient_l2_norm": regressor["readout_coefficient_l2_norm"],
                "classifier_maximum_absolute_coefficient": classifier[
                    "maximum_absolute_readout_coefficient"
                ],
                "regressor_maximum_absolute_coefficient": regressor[
                    "maximum_absolute_readout_coefficient"
                ],
                "validation_prediction_mean": float(validation_regression.mean()),
                "validation_prediction_standard_deviation": float(
                    validation_regression.std(ddof=0)
                ),
                "test_prediction_mean": float(test_regression.mean()),
                "test_prediction_standard_deviation": float(test_regression.std(ddof=0)),
                "validation_probability_entropy": validation_probability[
                    "mean_probability_entropy"
                ],
                "test_probability_entropy": test_probability["mean_probability_entropy"],
                "validation_mean_confidence": validation_probability["mean_confidence"],
                "test_mean_confidence": test_probability["mean_confidence"],
                "validation_top_label_ece": validation_probability[
                    "top_label_expected_calibration_error"
                ],
                "test_top_label_ece": test_probability["top_label_expected_calibration_error"],
                "test_high_regime_bias": high_regime["bias"],
                "test_high_regime_mae": high_regime["mae"],
                "test_p95_tail_relative_underprediction": p95_tail["mean_relative_underprediction"],
                "test_p95_tail_detection_rate": p95_tail["detection_rate"],
                "classifier_coefficients_finite": classifier["finite_coefficients"],
                "regressor_coefficients_finite": regressor["finite_coefficients"],
                "classifier_predictions_finite": classifier["finite_predictions"],
                "regressor_predictions_finite": regressor["finite_predictions"],
                "all_loaded_predictions_probabilities_scores_finite": arrays_finite,
                "non_finite_state_count": int(numerical["non_finite_states"]),
                "states_validated": int(numerical["states_validated"]),
                "maximum_hermiticity_error": numerical["maximum_hermiticity_error"],
                "maximum_trace_error_before_correction": numerical[
                    "maximum_trace_error_before_correction"
                ],
                "minimum_eigenvalue": numerical["minimum_eigenvalue"],
                "all_finite_checks_passed": bool(
                    classifier["finite_coefficients"]
                    and regressor["finite_coefficients"]
                    and classifier["finite_predictions"]
                    and regressor["finite_predictions"]
                    and arrays_finite
                    and int(numerical["non_finite_states"]) == 0
                ),
                "seed_excluded": False,
                "causal_claim_made": False,
            }
        )
    return rows


def _save_figure(
    figure: Any,
    destination: Path,
    *,
    apply_tight_layout: bool = True,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if apply_tight_layout:
        figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=240)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def _variance_calibration_figure(
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 6.2))
    selected = [row for row in rows if row["split"] == "test"]
    for model_id in REGRESSION_MODELS:
        model_rows = sorted(
            (row for row in selected if row["model"] == model_id),
            key=lambda row: int(row["decile"]),
        )
        axis.plot(
            [float(row["mean_forecast"]) for row in model_rows],
            [float(row["mean_realised_variance"]) for row in model_rows],
            marker="o",
            linewidth=1.4,
            label=str(model_rows[0]["model_display_name"]),
        )
    bounds = axis.get_xlim()
    lower = max(min(bounds[0], axis.get_ylim()[0]), 1e-6)
    upper = max(bounds[1], axis.get_ylim()[1])
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="black", label="ideal")
    axis.set(
        xlabel="mean forecast within post-hoc forecast decile",
        ylabel="mean realised variance",
        title="Test variance calibration by frozen model",
    )
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.2)
    _save_figure(figure, destination)


def _forecast_scatter_figure(
    data: FinancialSplit,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), sharex=True, sharey=True)
    positive = data.realised_variance[data.realised_variance > 0.0]
    lower = float(
        min(
            positive.min(),
            *(model.forecasts.min() for model in data.regression.values()),
        )
    )
    upper = float(
        max(
            data.realised_variance.max(),
            *(model.forecasts.max() for model in data.regression.values()),
        )
    )
    for axis, model_id in zip(axes.ravel(), REGRESSION_MODELS):
        model = data.regression[model_id]
        axis.scatter(
            data.realised_variance,
            model.forecasts,
            s=9,
            alpha=0.35,
            edgecolors="none",
        )
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="black")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(model.display_name)
        axis.grid(alpha=0.2)
    figure.supxlabel("realised five-day variance")
    figure.supylabel("frozen forecast")
    figure.suptitle("Frozen test forecasts versus realised variance")
    _save_figure(figure, destination)


def _grouped_bar_figure(
    rows: list[dict[str, Any]],
    *,
    category_key: str,
    value_key: str,
    categories: list[str],
    models: tuple[str, ...],
    destination: Path,
    title: str,
    ylabel: str,
    horizontal_zero: bool = True,
) -> None:
    figure, axis = plt.subplots(figsize=(11.0, 6.0))
    x_positions = np.arange(len(categories), dtype=float)
    width = 0.8 / len(models)
    for model_index, model_id in enumerate(models):
        model_rows = {
            str(row[category_key]): row
            for row in rows
            if row["model"] == model_id and row["split"] == "test"
        }
        values = [float(model_rows[category][value_key]) for category in categories]
        axis.bar(
            x_positions - 0.4 + width / 2.0 + model_index * width,
            values,
            width,
            label=str(next(iter(model_rows.values()))["model_display_name"]),
        )
    if horizontal_zero:
        axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(x_positions, categories)
    axis.set(title=title, ylabel=ylabel)
    axis.legend(fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    _save_figure(figure, destination)


def _reliability_figure(
    rows: list[dict[str, Any]],
    *,
    class_label: int | None,
    destination: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 6.2))
    selected = [
        row
        for row in rows
        if row["split"] == "test"
        and row["primary_bin_count"]
        and (class_label is None or row.get("class_label") == class_label)
    ]
    for row in selected:
        bins = row["top_label_reliability"] if class_label is None else row["reliability_bins"]
        populated = [value for value in bins if not value["empty_bin"]]
        axis.plot(
            [float(value["mean_probability"]) for value in populated],
            [float(value["event_rate"]) for value in populated],
            marker="o",
            linewidth=1.3,
            label=str(row["model_display_name"]),
        )
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", label="ideal")
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="mean predicted probability",
        ylabel="observed frequency",
        title=title,
    )
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.2)
    _save_figure(figure, destination)


def _confusion_figure(
    splits: dict[str, FinancialSplit],
    destination: Path,
) -> None:
    test = splits["test"]
    figure, axes = plt.subplots(3, 3, figsize=(12.0, 11.0))
    for axis, model_id in zip(axes.ravel(), CLASSIFICATION_MODELS):
        model = test.classification[model_id]
        matrix = confusion_counts(
            test.regimes,
            model.predictions,
            labels=(0, 1, 2),
        )
        normalized = np.divide(
            matrix,
            matrix.sum(axis=1, keepdims=True),
            out=np.zeros_like(matrix),
            where=matrix.sum(axis=1, keepdims=True) > 0.0,
        )
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        for row_index in range(3):
            for column_index in range(3):
                axis.text(
                    column_index,
                    row_index,
                    f"{int(matrix[row_index, column_index])}\n"
                    f"{normalized[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        axis.set_xticks(range(3), ["low", "medium", "high"])
        axis.set_yticks(range(3), ["low", "medium", "high"])
        axis.set_title(model.display_name)
    for axis in axes.ravel()[len(CLASSIFICATION_MODELS) :]:
        axis.axis("off")
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7, label="row-normalized rate")
    figure.supxlabel("predicted regime")
    figure.supylabel("true regime")
    figure.suptitle("Frozen test regime confusion matrices")
    _save_figure(figure, destination, apply_tight_layout=False)


def _transition_figure(
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    selected = [row for row in rows if row["split"] == "test"]
    transition_types = sorted({str(row["transition_type"]) for row in selected})
    figure, axis = plt.subplots(figsize=(12.0, 6.3))
    x_positions = np.arange(len(transition_types), dtype=float)
    width = 0.8 / len(CLASSIFICATION_MODELS)
    for model_index, model_id in enumerate(CLASSIFICATION_MODELS):
        model_rows = {
            str(row["transition_type"]): row for row in selected if row["model"] == model_id
        }
        values = [
            float(model_rows[transition_type]["correctly_predicted_destination_rate"])
            for transition_type in transition_types
        ]
        axis.bar(
            x_positions - 0.4 + width / 2.0 + model_index * width,
            values,
            width,
            label=str(next(iter(model_rows.values()))["model_display_name"]),
        )
    axis.set_xticks(
        x_positions,
        [value.replace("_", " ") for value in transition_types],
        rotation=25,
        ha="right",
    )
    axis.set(
        ylim=(0.0, 1.0),
        ylabel="correct destination-regime rate",
        title="Test performance by observed transition type",
    )
    axis.legend(fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    _save_figure(figure, destination)


def _rolling_qlike_figure(
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    selected = [
        row
        for row in rows
        if row["split"] == "test" and row["diagnostic_type"] == "rolling_variance"
    ]
    figure, axis = plt.subplots(figsize=(11.0, 6.0))
    for model_id in REGRESSION_MODELS:
        model_rows = [row for row in selected if row["model"] == model_id]
        axis.plot(
            pd.to_datetime([row["date"] for row in model_rows]),
            [float(row["rolling_qlike"]) for row in model_rows],
            linewidth=1.2,
            label=str(model_rows[0]["model_display_name"]),
        )
    axis.set(
        xlabel="window end date",
        ylabel="trailing 60-observation QLIKE",
        title="Descriptive rolling test QLIKE",
    )
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.2)
    _save_figure(figure, destination)


def _cumulative_garch_figure(
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    selected = [
        row
        for row in rows
        if row["split"] == "test"
        and row["diagnostic_type"] == "cumulative_qlike_difference"
        and row["reference_model"] == "garch_1_1"
        and row["model"] != "garch_1_1"
    ]
    figure, axis = plt.subplots(figsize=(11.0, 6.0))
    for model_id in [value for value in REGRESSION_MODELS if value != "garch_1_1"]:
        model_rows = [row for row in selected if row["model"] == model_id]
        axis.plot(
            pd.to_datetime([row["date"] for row in model_rows]),
            [float(row["cumulative_difference"]) for row in model_rows],
            linewidth=1.3,
            label=str(model_rows[0]["model_display_name"]),
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set(
        xlabel="date",
        ylabel="cumulative QLIKE(model - GARCH)",
        title="Descriptive cumulative test QLIKE difference versus GARCH",
    )
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.2)
    _save_figure(figure, destination)


def _numerical_figure(
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["reservoir_seed"]))
    seeds = [int(row["reservoir_seed"]) for row in ordered]
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.8))
    axes[0].bar(seeds, [float(row["condition_number"]) for row in ordered])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("training-feature condition number (log)")
    axes[1].bar(
        np.asarray(seeds) - 0.15,
        [float(row["classifier_coefficient_l2_norm"]) for row in ordered],
        width=0.3,
        label="classifier",
    )
    axes[1].bar(
        np.asarray(seeds) + 0.15,
        [float(row["regressor_coefficient_l2_norm"]) for row in ordered],
        width=0.3,
        label="regressor",
    )
    axes[1].set_ylabel("coefficient L2 norm")
    axes[1].legend()
    axes[2].bar(
        np.asarray(seeds) - 0.15,
        [float(row["test_top_label_ece"]) for row in ordered],
        width=0.3,
        label="top-label ECE",
    )
    axes[2].bar(
        np.asarray(seeds) + 0.15,
        [float(row["test_p95_tail_relative_underprediction"]) for row in ordered],
        width=0.3,
        label="P95 relative underprediction",
    )
    axes[2].axhline(0.0, color="black", linewidth=1.0)
    axes[2].set_ylabel("test diagnostic value")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(seeds)
        axis.set_xlabel("reservoir seed")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("QRC seed conditioning and observed diagnostic behaviour")
    _save_figure(figure, destination)


def _mnist_per_digit_figure(
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11.0, 6.0))
    for model_id in MNIST_MODELS:
        model_rows = sorted(
            (row for row in rows if row["model"] == model_id),
            key=lambda row: int(row["digit"]),
        )
        axis.plot(
            [int(row["digit"]) for row in model_rows],
            [float(row["f1"]) for row in model_rows],
            marker="o",
            linewidth=1.3,
            label=str(model_rows[0]["model_display_name"]),
        )
    axis.set(
        xticks=range(10),
        xlabel="true digit",
        ylabel="F1",
        ylim=(0.0, 1.0),
        title="Frozen MNIST test F1 by digit",
    )
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.2)
    _save_figure(figure, destination)


def _mnist_robustness_figure(
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    conditions = [value for value in MNIST_ROBUSTNESS if value != "analytic"]
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 8.2), sharex=True)
    for condition in conditions:
        condition_rows = sorted(
            (row for row in rows if row["condition"] == condition),
            key=lambda row: int(row["digit"]),
        )
        axes[0].plot(
            range(10),
            [float(row["accuracy_degradation_from_analytic"]) for row in condition_rows],
            marker="o",
            label=str(condition_rows[0]["condition_display_name"]),
        )
        axes[1].plot(
            range(10),
            [float(row["f1_degradation_from_analytic"]) for row in condition_rows],
            marker="o",
            label=str(condition_rows[0]["condition_display_name"]),
        )
    axes[0].set_ylabel("accuracy degradation")
    axes[1].set_ylabel("F1 degradation")
    axes[1].set_xlabel("true digit")
    axes[1].set_xticks(range(10))
    for axis in axes:
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Seed-2026 MNIST degradation from frozen analytic predictions")
    _save_figure(figure, destination)


def _figures(
    config: BenchmarkDiagnosticsConfig,
    splits: dict[str, FinancialSplit],
    tables: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    figure_root = config.output_root / "figures"
    destinations = {
        "variance_calibration": figure_root / "variance_calibration_by_model",
        "forecast_vs_realised": figure_root / "forecast_versus_realised_variance",
        "regime_bias": figure_root / "forecast_bias_by_realised_regime",
        "tail_underprediction": figure_root / "high_volatility_tail_underprediction",
        "top_label_reliability": figure_root / "top_label_reliability",
        "high_regime_reliability": figure_root / "high_regime_probability_reliability",
        "regime_confusion": figure_root / "regime_confusion_matrices",
        "transition_type": figure_root / "transition_type_performance",
        "rolling_qlike": figure_root / "rolling_qlike_by_model",
        "cumulative_garch": figure_root / "cumulative_qlike_difference_vs_garch",
        "qrc_conditioning": figure_root / "qrc_seed_numerical_conditioning",
        "mnist_per_digit": figure_root / "mnist_per_digit_f1",
        "mnist_robustness": figure_root / "mnist_analytic_vs_shot_noise_by_digit",
    }
    _variance_calibration_figure(
        tables["variance_decile_calibration"],
        destinations["variance_calibration"],
    )
    _forecast_scatter_figure(splits["test"], destinations["forecast_vs_realised"])
    _grouped_bar_figure(
        tables["variance_regime_diagnostics"],
        category_key="regime",
        value_key="bias",
        categories=["low", "medium", "high"],
        models=REGRESSION_MODELS,
        destination=destinations["regime_bias"],
        title="Test forecast bias by frozen realised regime",
        ylabel="mean forecast error",
    )
    _grouped_bar_figure(
        tables["variance_tail_diagnostics"],
        category_key="tail_threshold_id",
        value_key="mean_relative_underprediction",
        categories=["training_p90", "training_p95"],
        models=REGRESSION_MODELS,
        destination=destinations["tail_underprediction"],
        title="Test high-volatility-tail relative underprediction",
        ylabel="mean (realised - forecast) / realised",
    )
    _reliability_figure(
        tables["probability_calibration"],
        class_label=None,
        destination=destinations["top_label_reliability"],
        title="Frozen test top-label reliability",
    )
    _reliability_figure(
        tables["classwise_probability_calibration"],
        class_label=2,
        destination=destinations["high_regime_reliability"],
        title="Frozen test high-regime probability reliability",
    )
    _confusion_figure(splits, destinations["regime_confusion"])
    _transition_figure(
        tables["transition_type_diagnostics"],
        destinations["transition_type"],
    )
    _rolling_qlike_figure(
        tables["temporal_error_diagnostics"],
        destinations["rolling_qlike"],
    )
    _cumulative_garch_figure(
        tables["temporal_error_diagnostics"],
        destinations["cumulative_garch"],
    )
    _numerical_figure(
        tables["qrc_seed_numerical_diagnostics"],
        destinations["qrc_conditioning"],
    )
    _mnist_per_digit_figure(
        tables["mnist_per_digit_diagnostics"],
        destinations["mnist_per_digit"],
    )
    _mnist_robustness_figure(
        tables["mnist_robustness_by_digit"],
        destinations["mnist_robustness"],
    )
    return {
        f"{name}_{extension}": path.with_suffix(f".{extension}")
        .relative_to(config.project_root)
        .as_posix()
        for name, path in destinations.items()
        for extension in ("png", "pdf")
    }


def _csv_safe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            converted[key] = (
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                if isinstance(value, (dict, list))
                else value
            )
        result.append(converted)
    return result


def _write_table_pair(
    directory: Path,
    name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{name}.json"
    csv_path = directory / f"{name}.csv"
    _write_json(json_path, {"schema_version": 1, "rows": rows})
    frame = pd.DataFrame(_csv_safe(rows))
    frame = frame.reindex(sorted(frame.columns), axis=1)
    frame.to_csv(csv_path, index=False)
    return {f"{name}_json": json_path, f"{name}_csv": csv_path}


def _generated_output_manifest(
    project_root: Path,
    output_paths: dict[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for output_id, relative_path in sorted(output_paths.items()):
        path = project_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"generated diagnostic output is missing: {path}")
        records.append(
            {
                "output_id": output_id,
                "path": relative_path,
                "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def _descriptive_findings(
    tables: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    overall_test = [row for row in tables["variance_overall_calibration"] if row["split"] == "test"]
    probability_test = [
        row
        for row in tables["probability_calibration"]
        if row["split"] == "test" and row["primary_bin_count"]
    ]
    transition_test = [
        row
        for row in tables["transition_vs_nontransition"]
        if row["split"] == "test" and row["subset"] == "transition"
    ]
    numerical = tables["qrc_seed_numerical_diagnostics"]
    seed_2027 = next(row for row in numerical if row["reservoir_seed"] == 2027)
    other_seeds = [row for row in numerical if row["reservoir_seed"] != 2027]
    return {
        "test_variance_by_model": [
            {
                "model": row["model"],
                "mean_error": row["mean_error"],
                "qlike": row["qlike"],
                "rmse": row["rmse"],
                "mae": row["mae"],
                "correlation": row["correlation"],
            }
            for row in overall_test
        ],
        "test_probability_calibration_by_model": [
            {
                "model": row["model"],
                "brier": row["multiclass_brier_score"],
                "log_loss": row["multiclass_log_loss"],
                "top_label_ece": row["top_label_expected_calibration_error"],
                "mean_confidence": row["mean_confidence"],
                "accuracy": row["accuracy"],
            }
            for row in probability_test
        ],
        "test_transition_subset_by_model": [
            {
                "model": row["model"],
                "sample_count": row["sample_count"],
                "destination_accuracy": row["destination_regime_accuracy"],
                "classification_error_rate": row["classification_error_rate"],
                "mean_transition_score": row["mean_transition_score"],
            }
            for row in transition_test
        ],
        "seed_2027_factual_comparison": {
            "condition_number": seed_2027["condition_number"],
            "condition_number_ratio_to_largest_other_seed": float(
                float(seed_2027["condition_number"])
                / max(float(row["condition_number"]) for row in other_seeds)
            ),
            "test_prediction_standard_deviation": seed_2027["test_prediction_standard_deviation"],
            "other_seed_test_prediction_standard_deviations": [
                row["test_prediction_standard_deviation"] for row in other_seeds
            ],
            "test_top_label_ece": seed_2027["test_top_label_ece"],
            "other_seed_test_top_label_ece": [row["test_top_label_ece"] for row in other_seeds],
            "test_high_regime_mae": seed_2027["test_high_regime_mae"],
            "other_seed_test_high_regime_mae": [row["test_high_regime_mae"] for row in other_seeds],
            "test_p95_tail_relative_underprediction": seed_2027[
                "test_p95_tail_relative_underprediction"
            ],
            "other_seed_test_p95_tail_relative_underprediction": [
                row["test_p95_tail_relative_underprediction"] for row in other_seeds
            ],
            "causal_interpretation": False,
            "seed_excluded": False,
        },
    }


def run_benchmark_diagnostics(
    config_path: Path,
    *,
    smoke: bool = False,
) -> Path:
    """Generate Stage 2B diagnostics from checksum-verified frozen predictions only."""

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config = load_benchmark_diagnostics_config(config_path)
    frozen_before = verify_frozen_diagnostic_sources(config)
    splits, tail_thresholds, regime_thresholds = _load_financial_splits(config)
    mnist = _load_mnist(config)
    repetitions = config.smoke_repetitions if smoke else config.bootstrap_repetitions
    mode = "smoke" if smoke else "full"
    financial_tables, _ = _financial_tables(
        config,
        splits,
        tail_thresholds=tail_thresholds,
        repetitions=repetitions,
    )
    financial_tables["qrc_seed_numerical_diagnostics"] = _qrc_seed_numerical_rows(
        config,
        splits,
        financial_tables,
    )
    tables = {
        **financial_tables,
        **_mnist_tables(
            config,
            mnist,
            repetitions=repetitions,
        ),
    }
    first_classifier_source = config.classification_models["logistic_regression"].test
    prediction_columns = set(pd.read_csv(first_classifier_source.path, nrows=0).columns)
    lead_time = assess_lead_time_identifiability(
        prediction_columns=prediction_columns,
        target_horizon=5,
        target_definition="realised-variance regime target",
    )
    table_counts = {name: len(rows) for name, rows in tables.items()}
    tables["benchmark_diagnostics_summary"] = [
        {
            "study_id": STUDY_ID,
            "mode": mode,
            "bootstrap_repetitions": repetitions,
            "bootstrap_block_length": config.bootstrap_block_length,
            "bootstrap_seed": config.bootstrap_seed,
            "financial_validation_observations": len(splits["validation"].dates),
            "financial_test_observations": len(splits["test"].dates),
            "mnist_test_observations": len(mnist.truth),
            "regime_threshold_low_medium": float(regime_thresholds["low_medium"]),
            "regime_threshold_medium_high": float(regime_thresholds["medium_high"]),
            "tail_threshold_training_p90": tail_thresholds["training_p90"],
            "tail_threshold_training_p95": tail_thresholds["training_p95"],
            "lead_time_analysis_identifiable": lead_time["identifiable"],
            "lead_time_analysis_performed": lead_time["performed"],
            "lead_time_reason": lead_time["reason"],
            "models_fitted": False,
            "probabilities_recalibrated": False,
            "thresholds_changed": False,
            "ensembles_created": False,
            "table_row_counts": table_counts,
        }
    ]
    config.output_root.mkdir(parents=True, exist_ok=True)
    table_root = config.output_root / "tables"
    output_paths: dict[str, str] = {}
    for name, rows in tables.items():
        for output_id, output_path in _write_table_pair(table_root, name, rows).items():
            output_paths[output_id] = output_path.relative_to(config.project_root).as_posix()
    figure_paths = _figures(config, splits, tables)
    frozen_after = verify_frozen_diagnostic_sources(config)
    if frozen_before != frozen_after:
        raise RuntimeError("a frozen diagnostic source changed during Stage 2B")
    findings = _descriptive_findings(tables)
    provenance_path = config.output_root / "provenance_manifest.json"
    summary_path = config.output_root / "benchmark_diagnostics_summary.json"
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
                "financial_method": "paired circular moving-block bootstrap",
                "mnist_method": "paired class-stratified bootstrap",
                "repetitions": repetitions,
                "block_length": config.bootstrap_block_length,
                "seed": config.bootstrap_seed,
            },
            "sample_counts": {
                "financial_validation": len(splits["validation"].dates),
                "financial_test": len(splits["test"].dates),
                "mnist_test": len(mnist.truth),
            },
            "frozen_thresholds": {
                "regime_low_medium": regime_thresholds["low_medium"],
                "regime_medium_high": regime_thresholds["medium_high"],
                **tail_thresholds,
            },
            "lead_time_analysis": lead_time,
            "descriptive_findings": findings,
            "table_row_counts": {name: len(rows) for name, rows in tables.items()},
            "outputs": {
                **output_paths,
                **figure_paths,
                "provenance_manifest": provenance_path.relative_to(config.project_root).as_posix(),
            },
            "scientific_constraints": {
                "models_fitted": False,
                "probabilities_recalibrated": False,
                "variance_forecasts_recalibrated": False,
                "architectures_or_hyperparameters_changed": False,
                "thresholds_changed": False,
                "ensembles_created": False,
                "diagnostics_used_for_selection": False,
                "existing_frozen_predictions_only": True,
                "physical_qpu_execution": False,
                "quantum_advantage_claim": False,
            },
        },
    )
    generated_output_paths = {
        **output_paths,
        **figure_paths,
        "benchmark_diagnostics_summary_root": summary_path.relative_to(
            config.project_root
        ).as_posix(),
    }
    _write_json(
        provenance_path,
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "mode": mode,
            "configuration": {
                "path": config.source.relative_to(config.project_root).as_posix(),
                "sha256": sha256_path(config.source),
            },
            "frozen_inputs": frozen_after,
            "frozen_inputs_unchanged_during_run": True,
            "generated_outputs": _generated_output_manifest(
                config.project_root,
                generated_output_paths,
            ),
            "checksum_manifest_self_excluded": True,
            "data_snapshot_id": config.data_snapshot_id,
            "financial_architecture_manifest_checksum": (
                config.financial_architecture_manifest.sha256
            ),
            "mnist_official_test_identity_checksum": (config.mnist_selected_indices.sha256),
            "regime_thresholds": {
                "source": "frozen training split",
                "fit_split": regime_thresholds["fit_split"],
                "low_medium": regime_thresholds["low_medium"],
                "medium_high": regime_thresholds["medium_high"],
                "changed": False,
            },
            "tail_thresholds": {
                "source": "frozen training target_rv_5d only",
                "quantile_method": "NumPy linear empirical quantile",
                **tail_thresholds,
                "optimised": False,
            },
            "lead_time_analysis": lead_time,
            "git": _git_metadata(config.project_root),
            **runtime_metadata(),
            "models_fitted": False,
            "probabilities_recalibrated": False,
            "variance_forecasts_recalibrated": False,
            "architectures_or_hyperparameters_changed": False,
            "thresholds_changed": False,
            "ensembles_created": False,
            "diagnostics_used_for_selection": False,
            "physical_qpu_execution": False,
            "quantum_advantage_claim": False,
        },
    )
    return summary_path
