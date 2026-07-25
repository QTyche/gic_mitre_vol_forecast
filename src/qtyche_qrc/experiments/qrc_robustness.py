"""One-factor-at-a-time QRC finite-shot and simulated-noise robustness study."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from qtyche_qrc.data.config import load_data_config
from qtyche_qrc.data.download import sha256_file, verify_public_snapshot
from qtyche_qrc.experiments.model_config import ModelExperimentConfig, load_model_config
from qtyche_qrc.experiments.qrc_run import (
    QRCSelectionResult,
    qrc_config_from_model,
    qrc_readout_config,
    select_qrc_readout,
)
from qtyche_qrc.experiments.run import (
    SyntheticResultsError,
    _classification_predictions,
    _experiment_directory,
    _git_metadata,
    _regression_predictions,
    _sha256,
    _write_json,
)
from qtyche_qrc.models.base import ForecastModel
from qtyche_qrc.models.dataset import ModelDataset, load_model_dataset
from qtyche_qrc.models.qrc.features import QRCFeatureBundle
from qtyche_qrc.models.qrc.noise import QRCMeasurementConfig
from qtyche_qrc.models.qrc.readout import QRCClassifier, QRCRegressor
from qtyche_qrc.models.qrc.reservoir import QRCConfig
from qtyche_qrc.models.qrc.robust_features import (
    RobustFeatureCacheKey,
    generate_or_load_robust_features,
    make_robust_feature_cache_key,
)
from qtyche_qrc.runtime import runtime_metadata

STUDY_ID = "qrc_shot_noise_robustness_v1"
FINAL_STUDY_ID = "final_financial_qrc_robustness_v1"
STUDY_TYPES = (
    "analytic_reference",
    "finite_shot",
    "depolarizing_noise",
    "measurement_noise",
)
SELECTABLE_STUDIES = STUDY_TYPES[1:]
TASKS = ("regime_classification", "rv_regression")
MODEL_TYPES = {
    "regime_classification": "qrc_classifier",
    "rv_regression": "qrc_regressor",
}
CLASSIFICATION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "log_loss",
    "multiclass_brier_score",
    "transition_accuracy",
    "transition_balanced_accuracy",
    "transition_f1",
    "transition_roc_auc",
    "transition_pr_auc",
    "transition_brier_score",
)
REGRESSION_METRICS = (
    "rmse",
    "mae",
    "qlike",
    "r_squared",
    "prediction_mean",
    "prediction_median",
    "prediction_minimum",
    "prediction_maximum",
    "non_finite_prediction_count",
    "floored_prediction_count",
)
STABILITY_METRICS = (
    "prediction_mae_vs_analytic",
    "prediction_rmse_vs_analytic",
    "prediction_max_abs_vs_analytic",
    "prediction_correlation_vs_analytic",
    "prediction_label_agreement_vs_analytic",
    "transition_probability_rmse_vs_analytic",
)
RESOURCE_METRICS = (
    "state_generation_seconds",
    "sampling_seconds",
    "readout_fitting_seconds",
    "selected_ridge_alpha",
    "raw_feature_dimension",
    "trainable_readout_parameters",
)


@dataclass(frozen=True, order=True)
class RobustnessPoint:
    """One reservoir, measurement RNG, budget, and one-factor noise condition."""

    study_type: str
    n_qubits: int
    reservoir_seed: int
    measurement_seed: int | None
    shot_count: int | None
    depolarizing_probability: float
    measurement_bit_flip_probability: float

    def validate(self) -> None:
        if self.study_type not in STUDY_TYPES:
            raise ValueError(f"unsupported robustness study type: {self.study_type}")
        measurement = self.measurement_config
        measurement.validate()
        if not 2 <= self.n_qubits <= 6:
            raise ValueError("robustness n_qubits must lie in [2, 6]")
        if self.study_type == "analytic_reference":
            if (
                self.shot_count is not None
                or self.measurement_seed is not None
                or self.depolarizing_probability != 0.0
                or self.measurement_bit_flip_probability != 0.0
            ):
                raise ValueError("analytic reference must have no shots, RNG seed, or noise")
        elif self.study_type == "finite_shot":
            if self.depolarizing_probability != 0.0 or self.measurement_bit_flip_probability != 0.0:
                raise ValueError("finite-shot study must keep both noise probabilities zero")
        elif self.study_type == "depolarizing_noise":
            if self.shot_count != 2048 or self.measurement_bit_flip_probability != 0.0:
                raise ValueError("depolarizing study fixes 2048 shots and zero readout noise")
        elif self.study_type == "measurement_noise" and (
            self.shot_count != 2048 or self.depolarizing_probability != 0.0
        ):
            raise ValueError("measurement-noise study fixes 2048 shots and zero depolarisation")

    @property
    def measurement_config(self) -> QRCMeasurementConfig:
        return QRCMeasurementConfig(
            shots=self.shot_count,
            measurement_seed=self.measurement_seed,
            depolarizing_probability=self.depolarizing_probability,
            measurement_bit_flip_probability=self.measurement_bit_flip_probability,
        )

    @property
    def checksum(self) -> str:
        import hashlib

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def key(self) -> str:
        return f"{self.study_type}_{self.checksum[:16]}"


@dataclass(frozen=True)
class RobustnessStudyConfig:
    """Validated grids, references, paths, and fixed QRC controls."""

    source: Path
    project_root: Path
    study_id: str
    output_root: Path
    classifier_reference: Path
    classifier_reference_sha256: str
    regressor_reference: Path
    regressor_reference_sha256: str
    data_snapshot_id: str
    selected_n_qubits: int
    reservoir_seeds: tuple[int, ...]
    measurement_seeds: tuple[int, ...]
    shots: tuple[int, ...]
    depolarizing_probabilities: tuple[float, ...]
    measurement_noise_probabilities: tuple[float, ...]
    smoke_reservoir_seeds: tuple[int, ...]
    smoke_measurement_seeds: tuple[int, ...]
    smoke_shots: tuple[int, ...]
    smoke_depolarizing_probabilities: tuple[float, ...]
    smoke_measurement_noise_probabilities: tuple[float, ...]
    fixed_qrc: dict[str, Any]
    raw: dict[str, Any]


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a YAML mapping")
    return dict(value)


def _text(mapping: dict[str, Any], key: str, location: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location}.{key} must be a non-empty string")
    return value


def _integer(mapping: dict[str, Any], key: str, location: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location}.{key} must be an integer")
    return value


def _integer_tuple(mapping: dict[str, Any], key: str, location: str) -> tuple[int, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}.{key} must be a non-empty integer list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{location}.{key} must be a non-empty integer list")
    result = tuple(int(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{location}.{key} contains duplicates")
    return result


def _float_tuple(mapping: dict[str, Any], key: str, location: str) -> tuple[float, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}.{key} must be a non-empty numeric list")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{location}.{key} must be a non-empty numeric list")
    result = tuple(float(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{location}.{key} contains duplicates")
    return result


def _validate_grids(config: RobustnessStudyConfig) -> None:
    if config.study_id not in {STUDY_ID, FINAL_STUDY_ID}:
        raise ValueError(f"robustness study ID must be {STUDY_ID} or {FINAL_STUDY_ID}")
    if not 2 <= config.selected_n_qubits <= 6:
        raise ValueError("selected_n_qubits must lie in [2, 6]")
    if config.reservoir_seeds != (2026, 2027, 2028):
        raise ValueError("full reservoir seed grid must remain 2026, 2027, 2028")
    if config.measurement_seeds != (0, 1, 2):
        raise ValueError("full measurement seed grid must remain 0, 1, 2")
    if config.shots != (128, 512, 2048, 8192):
        raise ValueError("finite-shot grid must remain 128, 512, 2048, 8192")
    if config.depolarizing_probabilities != (0.0, 0.0001, 0.001, 0.01):
        raise ValueError("depolarizing probability grid changed")
    if config.measurement_noise_probabilities != (0.0, 0.005, 0.01, 0.02):
        raise ValueError("measurement-noise probability grid changed")
    expected_policy = "reset_each_input" if config.study_id == FINAL_STUDY_ID else "carry_inputs"
    if config.fixed_qrc.get("state_policy") != expected_policy:
        raise ValueError(f"{config.study_id} must use state_policy={expected_policy}")
    for collection in (
        config.depolarizing_probabilities,
        config.measurement_noise_probabilities,
        config.smoke_depolarizing_probabilities,
        config.smoke_measurement_noise_probabilities,
    ):
        if any(not 0.0 <= value <= 1.0 for value in collection):
            raise ValueError("noise probability grids must lie in [0, 1]")
    if set(config.smoke_reservoir_seeds) - set(config.reservoir_seeds):
        raise ValueError("smoke reservoir seeds must be a subset of the full grid")
    if set(config.smoke_measurement_seeds) - set(config.measurement_seeds):
        raise ValueError("smoke measurement seeds must be a subset of the full grid")
    if set(config.smoke_shots) - set(config.shots):
        raise ValueError("smoke shots must be a subset of the full grid")
    if set(config.smoke_depolarizing_probabilities) - set(config.depolarizing_probabilities):
        raise ValueError("smoke depolarizing probabilities must be a subset")
    if set(config.smoke_measurement_noise_probabilities) - set(
        config.measurement_noise_probabilities
    ):
        raise ValueError("smoke measurement-noise probabilities must be a subset")


def _assert_reference_contracts(
    config: RobustnessStudyConfig,
) -> tuple[ModelExperimentConfig, ModelExperimentConfig]:
    for path, expected in (
        (config.classifier_reference, config.classifier_reference_sha256),
        (config.regressor_reference, config.regressor_reference_sha256),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen robustness reference checksum mismatch for {path}: {actual}")
    classifier = load_model_config(config.classifier_reference)
    regressor = load_model_config(config.regressor_reference)
    if classifier.model_type != "qrc_classifier" or classifier.task != TASKS[0]:
        raise ValueError("classifier reference is not the frozen public QRC classifier")
    if regressor.model_type != "qrc_regressor" or regressor.task != TASKS[1]:
        raise ValueError("regressor reference is not the frozen public QRC regressor")
    classifier_qrc = qrc_config_from_model(classifier)
    regressor_qrc = qrc_config_from_model(regressor)
    if classifier_qrc != regressor_qrc:
        raise ValueError("reference readouts disagree on frozen reservoir dynamics")
    if classifier.processed_dir != regressor.processed_dir:
        raise ValueError("reference readouts disagree on the frozen processed dataset")
    for name in (
        "virtual_nodes",
        "graph",
        "j_strength",
        "h_strength",
        "tau",
        "input_scaling",
        "state_policy",
        "backend",
    ):
        if getattr(classifier_qrc, name) != config.fixed_qrc.get(name):
            raise ValueError(f"frozen QRC reference disagrees with fixed_qrc.{name}")
    return classifier, regressor


def load_robustness_study_config(path: Path) -> RobustnessStudyConfig:
    """Load and verify the complete shot/noise study contract."""

    source = path.resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "configuration")
    if root.get("schema_version") != 1:
        raise ValueError("QRC robustness configuration schema_version must be 1")
    study = _mapping(root.get("study"), "study")
    studies = _mapping(root.get("studies"), "studies")
    finite = _mapping(studies.get("finite_shot"), "studies.finite_shot")
    depolarizing = _mapping(
        studies.get("depolarizing_noise"),
        "studies.depolarizing_noise",
    )
    measurement = _mapping(studies.get("measurement_noise"), "studies.measurement_noise")
    smoke = _mapping(root.get("smoke"), "smoke")
    fixed_qrc = _mapping(root.get("fixed_qrc"), "fixed_qrc")
    project_root = (source.parent / _text(study, "project_root", "study")).resolve()
    config = RobustnessStudyConfig(
        source=source,
        project_root=project_root,
        study_id=_text(study, "id", "study"),
        output_root=(project_root / _text(study, "output_root", "study")).resolve(),
        classifier_reference=(
            project_root / _text(study, "classifier_reference", "study")
        ).resolve(),
        classifier_reference_sha256=_text(study, "classifier_reference_sha256", "study"),
        regressor_reference=(project_root / _text(study, "regressor_reference", "study")).resolve(),
        regressor_reference_sha256=_text(study, "regressor_reference_sha256", "study"),
        data_snapshot_id=_text(study, "data_snapshot_id", "study"),
        selected_n_qubits=_integer(study, "selected_n_qubits", "study"),
        reservoir_seeds=_integer_tuple(study, "reservoir_seeds", "study"),
        measurement_seeds=_integer_tuple(study, "measurement_seeds", "study"),
        shots=_integer_tuple(finite, "shots", "studies.finite_shot"),
        depolarizing_probabilities=_float_tuple(
            depolarizing,
            "probabilities",
            "studies.depolarizing_noise",
        ),
        measurement_noise_probabilities=_float_tuple(
            measurement,
            "probabilities",
            "studies.measurement_noise",
        ),
        smoke_reservoir_seeds=_integer_tuple(smoke, "reservoir_seeds", "smoke"),
        smoke_measurement_seeds=_integer_tuple(smoke, "measurement_seeds", "smoke"),
        smoke_shots=_integer_tuple(smoke, "finite_shot_shots", "smoke"),
        smoke_depolarizing_probabilities=_float_tuple(
            smoke,
            "depolarizing_probabilities",
            "smoke",
        ),
        smoke_measurement_noise_probabilities=_float_tuple(
            smoke,
            "measurement_noise_probabilities",
            "smoke",
        ),
        fixed_qrc=fixed_qrc,
        raw=root,
    )
    if _integer(depolarizing, "shots", "studies.depolarizing_noise") != 2048:
        raise ValueError("depolarizing study must remain fixed at 2048 shots")
    if _integer(measurement, "shots", "studies.measurement_noise") != 2048:
        raise ValueError("measurement-noise study must remain fixed at 2048 shots")
    _validate_grids(config)
    _assert_reference_contracts(config)
    return config


def build_robustness_grid(
    *,
    n_qubits: int,
    reservoir_seeds: tuple[int, ...],
    measurement_seeds: tuple[int, ...],
    shots: tuple[int, ...],
    depolarizing_probabilities: tuple[float, ...],
    measurement_noise_probabilities: tuple[float, ...],
    studies: tuple[str, ...] = SELECTABLE_STUDIES,
) -> tuple[RobustnessPoint, ...]:
    """Build a deterministic analytic-reference plus one-factor-at-a-time grid."""

    if not reservoir_seeds or not measurement_seeds:
        raise ValueError("both reservoir and measurement seed grids must be non-empty")
    if len(set(reservoir_seeds)) != len(reservoir_seeds) or len(set(measurement_seeds)) != len(
        measurement_seeds
    ):
        raise ValueError("robustness seed grids contain duplicates")
    if len(set(studies)) != len(studies) or set(studies) - set(SELECTABLE_STUDIES):
        raise ValueError("studies must be unique selectable robustness study names")
    points: list[RobustnessPoint] = [
        RobustnessPoint(
            "analytic_reference",
            n_qubits,
            reservoir_seed,
            None,
            None,
            0.0,
            0.0,
        )
        for reservoir_seed in sorted(reservoir_seeds)
    ]
    if "finite_shot" in studies:
        points.extend(
            RobustnessPoint(
                "finite_shot",
                n_qubits,
                reservoir_seed,
                measurement_seed,
                shot_count,
                0.0,
                0.0,
            )
            for shot_count in sorted(shots)
            for reservoir_seed in sorted(reservoir_seeds)
            for measurement_seed in sorted(measurement_seeds)
        )
    if "depolarizing_noise" in studies:
        points.extend(
            RobustnessPoint(
                "depolarizing_noise",
                n_qubits,
                reservoir_seed,
                measurement_seed,
                2048,
                probability,
                0.0,
            )
            for probability in sorted(depolarizing_probabilities)
            for reservoir_seed in sorted(reservoir_seeds)
            for measurement_seed in sorted(measurement_seeds)
        )
    if "measurement_noise" in studies:
        points.extend(
            RobustnessPoint(
                "measurement_noise",
                n_qubits,
                reservoir_seed,
                measurement_seed,
                2048,
                0.0,
                probability,
            )
            for probability in sorted(measurement_noise_probabilities)
            for reservoir_seed in sorted(reservoir_seeds)
            for measurement_seed in sorted(measurement_seeds)
        )
    for point in points:
        point.validate()
    checksums = [point.checksum for point in points]
    if len(checksums) != len(set(checksums)):
        raise ValueError("robustness grid contains duplicate experimental points")
    return tuple(points)


def verify_robustness_public_data(
    config: RobustnessStudyConfig,
    classifier_reference: ModelExperimentConfig,
) -> tuple[ModelDataset, dict[str, Any]]:
    """Verify frozen raw and processed checksums and reject fixture data."""

    data_config = load_data_config(config.project_root / "configs/data_public_market.yaml")
    snapshot = verify_public_snapshot(data_config)
    if snapshot.get("snapshot_id") != config.data_snapshot_id:
        raise ValueError("public snapshot ID disagrees with robustness configuration")
    if data_config.snapshot_manifest_path is None:
        raise FileNotFoundError("public snapshot configuration has no manifest")
    dataset = load_model_dataset(classifier_reference.processed_dir)
    if dataset.is_synthetic or dataset.data_source_type != "public_market":
        raise SyntheticResultsError(
            "QRC robustness study requires verified non-synthetic public-market data"
        )
    if dataset.manifest.get("source_snapshot_id") != config.data_snapshot_id:
        raise ValueError("processed data manifest disagrees with robustness snapshot ID")
    return dataset, {
        "snapshot_id": config.data_snapshot_id,
        "snapshot_manifest_sha256": sha256_file(data_config.snapshot_manifest_path),
        "raw_file_checksums": {
            name: str(record["sha256"]) for name, record in sorted(snapshot["files"].items())
        },
        "processed_manifest_sha256": dataset.processed_checksums["data_manifest.json"],
        "processed_checksums": dataset.processed_checksums,
        "split_row_counts": {
            "train": len(dataset.train.X),
            "validation": len(dataset.validation.X),
            "test": len(dataset.test.X),
        },
    }


def _reference_for_task(config: RobustnessStudyConfig, task: str) -> tuple[Path, str]:
    if task == TASKS[0]:
        return config.classifier_reference, config.classifier_reference_sha256
    if task == TASKS[1]:
        return config.regressor_reference, config.regressor_reference_sha256
    raise ValueError(f"unsupported robustness task: {task}")


def write_robustness_model_config(
    config: RobustnessStudyConfig,
    point: RobustnessPoint,
    task: str,
) -> Path:
    """Derive a readout config while changing no frozen dynamics except N and seed."""

    reference_path, reference_checksum = _reference_for_task(config, task)
    raw = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("frozen QRC reference config must be a mapping")
    config_dir = config.output_root / "generated_configs"
    project_root_setting = os.path.relpath(config.project_root, config_dir)
    output_root_setting = os.path.relpath(config.output_root / "runs", config.project_root)
    feature_cache_setting = os.path.relpath(
        config.output_root / "feature_cache",
        config.project_root,
    )
    raw["experiment"]["name"] = f"qrc_robustness_{point.key}_{MODEL_TYPES[task]}"
    raw["experiment"]["project_root"] = Path(project_root_setting).as_posix()
    raw["experiment"]["seed"] = point.reservoir_seed
    raw["experiment"]["output_root"] = Path(output_root_setting).as_posix()
    raw["model"]["parameters"]["n_qubits"] = point.n_qubits
    raw["model"]["parameters"]["reservoir_seed"] = point.reservoir_seed
    raw["qrc"]["feature_cache"] = Path(feature_cache_setting).as_posix()
    raw["qrc"]["reservoir_seeds"] = list(config.reservoir_seeds)
    raw["robustness_study"] = {
        "id": config.study_id,
        "point": asdict(point),
        "point_checksum": point.checksum,
        "reference_config": reference_path.relative_to(config.project_root).as_posix(),
        "reference_config_sha256": reference_checksum,
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    destination = config_dir / f"{point.key}_{MODEL_TYPES[task]}.yaml"
    destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    derived = load_model_config(destination)
    reference = load_model_config(reference_path)
    derived_qrc = qrc_config_from_model(derived)
    reference_qrc = qrc_config_from_model(reference)
    for name, value in asdict(reference_qrc).items():
        if name in {"n_qubits", "reservoir_seed"}:
            continue
        if asdict(derived_qrc)[name] != value:
            raise ValueError(f"robustness config unexpectedly changed QRC field {name}")
    if derived_qrc.n_qubits != point.n_qubits or derived_qrc.reservoir_seed != point.reservoir_seed:
        raise ValueError("derived robustness config has the wrong architecture identity")
    return destination


def _qrc_config_for_point(model_config: ModelExperimentConfig, point: RobustnessPoint) -> QRCConfig:
    qrc_config = qrc_config_from_model(model_config, point.reservoir_seed)
    if qrc_config.n_qubits != point.n_qubits:
        raise ValueError("robustness model config and point disagree on n_qubits")
    return qrc_config


def prepare_robustness_features(
    *,
    config: RobustnessStudyConfig,
    point: RobustnessPoint,
    classifier_config_path: Path,
    dataset: ModelDataset,
) -> tuple[QRCFeatureBundle, RobustFeatureCacheKey]:
    """Generate/load a point once for shared classifier and regressor use."""

    classifier = load_model_config(classifier_config_path)
    if dataset.is_synthetic or dataset.data_source_type != "public_market":
        raise SyntheticResultsError("robustness feature preparation rejects synthetic data")
    qrc_config = _qrc_config_for_point(classifier, point)
    key = make_robust_feature_cache_key(
        processed_data_manifest_checksum=dataset.processed_checksums["data_manifest.json"],
        feature_names=dataset.feature_names,
        qrc_config=qrc_config,
        measurement_config=point.measurement_config,
    )
    bundle = generate_or_load_robust_features(
        cache_root=config.output_root / "feature_cache",
        key=key,
        feature_names=dataset.feature_names,
        qrc_config=qrc_config,
        measurement_config=point.measurement_config,
        X_train=dataset.train.X,
        X_validation=dataset.validation.X,
        X_test=dataset.test.X,
    )
    return bundle, key


def _run_robustness_readout(
    *,
    study_config: RobustnessStudyConfig,
    point: RobustnessPoint,
    model_config_path: Path,
    dataset: ModelDataset,
    bundle: QRCFeatureBundle,
    cache_key: RobustFeatureCacheKey,
    cache_hit: bool,
) -> Path:
    """Fit one validation-selected readout and persist a compact complete run."""

    config = load_model_config(model_config_path)
    qrc_config = _qrc_config_for_point(config, point)
    experiment_dir = _experiment_directory(config)
    shutil.copyfile(config.source, experiment_dir / "config.yaml")
    selected_alpha: float | None = None
    trials: list[QRCSelectionResult] = []
    timing: dict[str, float] = {}
    status = "success"
    test_evaluated_after_freeze = False
    try:
        selected_alpha, trials = select_qrc_readout(
            config=config,
            data=dataset.for_selection(),
            train_features=bundle.train,
            validation_features=bundle.validation,
        )
        feature_names = tuple(bundle.metadata["observable_metadata"]["feature_ordering"])
        head_config = qrc_readout_config(config, selected_alpha)
        started = time.perf_counter()
        model: ForecastModel
        if config.task == TASKS[0]:
            classifier = QRCClassifier(feature_names, head_config)
            classifier.fit(bundle.train, dataset.train.y_regime)
            model = classifier
            timing["readout_fitting_seconds"] = time.perf_counter() - started
            validation_raw = classifier.predict_proba(bundle.validation)
            readout_frozen = True
            test_raw = classifier.predict_proba(bundle.test)
            test_evaluated_after_freeze = readout_frozen
            validation_metrics, validation_predictions = _classification_predictions(
                dataset.validation,
                validation_raw,
                config.transition_threshold,
            )
            test_metrics, test_predictions = _classification_predictions(
                dataset.test,
                test_raw,
                config.transition_threshold,
            )
        else:
            regressor = QRCRegressor(feature_names, head_config)
            regressor.fit(bundle.train, dataset.train.y_rv)
            model = regressor
            timing["readout_fitting_seconds"] = time.perf_counter() - started
            validation_raw = regressor.predict(bundle.validation)
            readout_frozen = True
            test_raw = regressor.predict(bundle.test)
            test_evaluated_after_freeze = readout_frozen
            validation_metrics, validation_predictions = _regression_predictions(
                dataset.validation,
                validation_raw,
                config.variance_floor,
            )
            test_metrics, test_predictions = _regression_predictions(
                dataset.test,
                test_raw,
                config.variance_floor,
            )
        pd.DataFrame([asdict(result) for result in trials]).to_csv(
            experiment_dir / "selection_results.csv",
            index=False,
        )
        validation_predictions.to_csv(experiment_dir / "validation_predictions.csv", index=False)
        test_predictions.to_csv(experiment_dir / "test_predictions.csv", index=False)
        _write_json(experiment_dir / "validation_metrics.json", validation_metrics)
        _write_json(experiment_dir / "test_metrics.json", test_metrics)
        resource = bundle.metadata["resource_metadata"]
        timing["state_generation_seconds"] = float(resource["state_generation_seconds"])
        timing["sampling_seconds"] = float(resource["sampling_seconds"])
        _write_json(experiment_dir / "timing.json", timing)
        model.save(experiment_dir / "model")
        _write_json(experiment_dir / "model_metadata.json", model.get_model_metadata())
        _write_json(experiment_dir / "qrc_feature_metadata.json", bundle.metadata)
    except Exception:
        status = "failure"
        raise
    finally:
        resource = bundle.metadata["resource_metadata"]
        raw_dimension = int(bundle.train.shape[1])
        outputs = 3 if config.task == TASKS[0] else 1
        manifest = {
            "schema_version": 1,
            **runtime_metadata(),
            "experiment_id": experiment_dir.name,
            "git": _git_metadata(config.project_root),
            "study_id": study_config.study_id,
            "study_configuration_checksum": sha256_file(study_config.source),
            "robustness_point": asdict(point),
            "robustness_point_checksum": point.checksum,
            "model_type": config.model_type,
            "task": config.task,
            "configuration_checksum": _sha256(experiment_dir / "config.yaml"),
            "processed_data_checksums": dataset.processed_checksums,
            "data_manifest_checksum": dataset.processed_checksums["data_manifest.json"],
            "data_snapshot_id": dataset.manifest.get("source_snapshot_id"),
            "source_snapshot_manifest_checksum": dataset.manifest.get(
                "source_snapshot_manifest_checksum"
            ),
            "data_source_type": dataset.data_source_type,
            "is_synthetic": dataset.is_synthetic,
            "split_row_counts": {
                "train": len(dataset.train.X),
                "validation": len(dataset.validation.X),
                "test": len(dataset.test.X),
            },
            "reservoir_seed": point.reservoir_seed,
            "measurement_seed": point.measurement_seed,
            "model_selection_metric": config.selection_metric,
            "model_selection_trial_count": len(trials),
            "model_selection_data": "validation only",
            "test_evaluated_after_readout_freeze": test_evaluated_after_freeze,
            "selected_hyperparameters": {"ridge_alpha": selected_alpha},
            "qrc_configuration": asdict(qrc_config),
            "qrc_configuration_checksum": qrc_config.checksum,
            "measurement_configuration": point.measurement_config.metadata(),
            "measurement_configuration_checksum": point.measurement_config.checksum,
            "qrc_feature_cache_key_checksum": cache_key.checksum,
            "qrc_feature_cache_hit": cache_hit,
            "qrc_features_generated_without_labels": True,
            "qrc_raw_feature_dimension": raw_dimension,
            "readout_shape": [raw_dimension + 1, outputs],
            "trainable_readout_parameters": (raw_dimension + 1) * outputs,
            "state_generation_time": timing.get("state_generation_seconds"),
            "sampling_time": timing.get("sampling_seconds"),
            "readout_fitting_time": timing.get("readout_fitting_seconds"),
            "backend": resource["backend"],
            "exact_state_evolution": resource["exact_state_evolution"],
            "exact_noiseless": resource["exact_noiseless"],
            "physical_qpu_execution": False,
            "hardware_calibrated_noise": False,
            "status": status,
        }
        _write_json(experiment_dir / "manifest.json", manifest)
    return experiment_dir


RunIdentity = tuple[str, str]


def discover_completed_robustness_runs(
    runs_root: Path,
    *,
    study_id: str,
    snapshot_id: str,
    study_configuration_checksum: str | None = None,
) -> dict[RunIdentity, Path]:
    """Find latest complete provenance-valid robustness tasks for resumption."""

    completed: dict[RunIdentity, Path] = {}
    if not runs_root.is_dir():
        return completed
    for manifest_path in sorted(runs_root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("study_id") != study_id:
            continue
        if (
            study_configuration_checksum is not None
            and manifest.get("study_configuration_checksum") != study_configuration_checksum
        ):
            raise ValueError("completed robustness run uses a different study configuration")
        experiment_dir = manifest_path.parent
        if manifest.get("status") != "success":
            continue
        required = (
            experiment_dir / "config.yaml",
            experiment_dir / "validation_metrics.json",
            experiment_dir / "test_metrics.json",
            experiment_dir / "validation_predictions.csv",
            experiment_dir / "test_predictions.csv",
        )
        if not all(path.is_file() for path in required):
            continue
        if manifest.get("is_synthetic") or manifest.get("data_source_type") != "public_market":
            raise SyntheticResultsError("completed robustness run contains synthetic data")
        if manifest.get("data_snapshot_id") != snapshot_id:
            raise ValueError("completed robustness run uses a different public snapshot")
        if manifest.get("model_selection_data") != "validation only":
            raise ValueError("completed robustness run violates validation-only selection")
        if manifest.get("test_evaluated_after_readout_freeze") is not True:
            raise ValueError("completed robustness run evaluated test before readout freeze")
        if manifest.get("physical_qpu_execution") is not False:
            raise ValueError("completed robustness run is not a controlled simulation")
        point_checksum = manifest.get("robustness_point_checksum")
        task = manifest.get("task")
        if not isinstance(point_checksum, str) or task not in TASKS:
            raise ValueError("completed robustness run has an invalid identity")
        identity = (point_checksum, str(task))
        previous = completed.get(identity)
        if previous is None or experiment_dir.name > previous.name:
            completed[identity] = experiment_dir
    return completed


def pending_robustness_runs(
    points: tuple[RobustnessPoint, ...],
    completed: dict[RunIdentity, Path],
) -> tuple[tuple[RobustnessPoint, str], ...]:
    return tuple(
        (point, task)
        for point in points
        for task in TASKS
        if (point.checksum, task) not in completed
    )


def _prediction_stability(
    prediction_path: Path,
    reference_path: Path,
    *,
    task: str,
) -> dict[str, float | None]:
    predictions = pd.read_csv(prediction_path)
    reference = pd.read_csv(reference_path)
    if not predictions["date"].equals(reference["date"]):
        raise ValueError("robustness and analytic prediction dates do not align")
    if task == TASKS[0]:
        columns = ("probability_low", "probability_medium", "probability_high")
        values = predictions.loc[:, list(columns)].to_numpy(dtype=float)
        exact = reference.loc[:, list(columns)].to_numpy(dtype=float)
        label_agreement = float(
            np.mean(
                predictions["predicted_regime"].to_numpy(dtype=int)
                == reference["predicted_regime"].to_numpy(dtype=int)
            )
        )
        transition_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        predictions["predicted_transition_probability"].to_numpy(dtype=float)
                        - reference["predicted_transition_probability"].to_numpy(dtype=float)
                    )
                )
            )
        )
    else:
        values = predictions["predicted_rv_5d"].to_numpy(dtype=float)[:, None]
        exact = reference["predicted_rv_5d"].to_numpy(dtype=float)[:, None]
        label_agreement = None
        transition_rmse = None
    difference = values - exact
    flat_values = values.reshape(-1)
    flat_exact = exact.reshape(-1)
    if np.std(flat_values) > 0.0 and np.std(flat_exact) > 0.0:
        candidate = float(np.corrcoef(flat_values, flat_exact)[0, 1])
        correlation: float | None = candidate if np.isfinite(candidate) else None
    elif np.array_equal(flat_values, flat_exact):
        correlation = 1.0
    else:
        correlation = None
    return {
        "prediction_mae_vs_analytic": float(np.mean(np.abs(difference))),
        "prediction_rmse_vs_analytic": float(np.sqrt(np.mean(np.square(difference)))),
        "prediction_max_abs_vs_analytic": float(np.max(np.abs(difference))),
        "prediction_correlation_vs_analytic": correlation,
        "prediction_label_agreement_vs_analytic": label_agreement,
        "transition_probability_rmse_vs_analytic": transition_rmse,
    }


def collect_robustness_rows(
    run_dirs: dict[RunIdentity, Path],
    points: tuple[RobustnessPoint, ...],
    *,
    repository_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Collect complete split-level provenance, metrics, and stability diagnostics."""

    point_by_checksum = {point.checksum: point for point in points}
    analytic: dict[tuple[int, str], Path] = {}
    for (point_checksum, task), directory in run_dirs.items():
        point = point_by_checksum[point_checksum]
        if point.study_type == "analytic_reference":
            analytic[(point.reservoir_seed, task)] = directory
    expected_analytic = {
        (point.reservoir_seed, task)
        for point in points
        if point.study_type == "analytic_reference"
        for task in TASKS
    }
    if set(analytic) != expected_analytic:
        raise ValueError("robustness result set omits an analytic readout reference")

    rows: list[dict[str, Any]] = []
    for identity, experiment_dir in sorted(run_dirs.items()):
        point_checksum, task = identity
        point = point_by_checksum[point_checksum]
        manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
        required = {
            "python_version",
            "operating_system",
            "package_versions",
            "git",
            "configuration_checksum",
            "study_configuration_checksum",
            "data_manifest_checksum",
            "qrc_configuration_checksum",
            "measurement_configuration_checksum",
            "qrc_feature_cache_key_checksum",
        }
        missing = sorted(required - set(manifest))
        if missing:
            raise ValueError(f"robustness run manifest omits provenance fields: {missing}")
        directory_value = experiment_dir
        if repository_root is not None:
            directory_value = experiment_dir.relative_to(repository_root)
        common: dict[str, Any] = {
            "experiment_id": manifest["experiment_id"],
            "experiment_directory": directory_value.as_posix(),
            "study_type": point.study_type,
            "analytic_reference": point.study_type == "analytic_reference",
            "n_qubits": point.n_qubits,
            "virtual_nodes": manifest["qrc_configuration"]["virtual_nodes"],
            "reservoir_seed": point.reservoir_seed,
            "measurement_seed": point.measurement_seed,
            "shot_count": point.shot_count,
            "depolarizing_probability": point.depolarizing_probability,
            "measurement_bit_flip_probability": point.measurement_bit_flip_probability,
            "task": task,
            "selected_ridge_alpha": manifest["selected_hyperparameters"]["ridge_alpha"],
            "raw_feature_dimension": manifest["qrc_raw_feature_dimension"],
            "readout_shape": manifest["readout_shape"],
            "trainable_readout_parameters": manifest["trainable_readout_parameters"],
            "state_generation_seconds": manifest["state_generation_time"],
            "sampling_seconds": manifest["sampling_time"],
            "readout_fitting_seconds": manifest["readout_fitting_time"],
            "feature_cache_hit": manifest["qrc_feature_cache_hit"],
            "feature_cache_key_checksum": manifest["qrc_feature_cache_key_checksum"],
            "data_snapshot_id": manifest["data_snapshot_id"],
            "data_manifest_checksum": manifest["data_manifest_checksum"],
            "source_snapshot_manifest_checksum": manifest["source_snapshot_manifest_checksum"],
            "study_configuration_checksum": manifest["study_configuration_checksum"],
            "run_configuration_checksum": manifest["configuration_checksum"],
            "qrc_configuration_checksum": manifest["qrc_configuration_checksum"],
            "measurement_configuration_checksum": manifest["measurement_configuration_checksum"],
            "backend": manifest["backend"],
            "exact_state_evolution": manifest["exact_state_evolution"],
            "exact_noiseless": manifest["exact_noiseless"],
            "physical_qpu_execution": manifest["physical_qpu_execution"],
            "model_selection_data": manifest["model_selection_data"],
            "test_evaluated_after_readout_freeze": manifest["test_evaluated_after_readout_freeze"],
            "git_commit": manifest["git"]["commit"],
            "git_dirty": manifest["git"]["dirty"],
            "python_version": manifest["python_version"],
            "operating_system": manifest["operating_system"],
            "execution_platform": manifest["execution_platform"],
            "package_versions": manifest["package_versions"],
        }
        reference_dir = analytic[(point.reservoir_seed, task)]
        for split in ("validation", "test"):
            metrics = json.loads(
                (experiment_dir / f"{split}_metrics.json").read_text(encoding="utf-8")
            )
            stability = _prediction_stability(
                experiment_dir / f"{split}_predictions.csv",
                reference_dir / f"{split}_predictions.csv",
                task=task,
            )
            numeric_metrics = {
                name: value
                for name, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            rows.append(
                {
                    **common,
                    "split": split,
                    "metrics": metrics,
                    **numeric_metrics,
                    **stability,
                }
            )
    return rows


def _summary_statistics(values: NDArray[np.float64]) -> dict[str, float | int]:
    return {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "repetition_count": len(values),
    }


def _plain_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def aggregate_robustness_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate measurement seeds, reservoir-level means, and all repetitions."""

    condition_fields = (
        "study_type",
        "analytic_reference",
        "n_qubits",
        "virtual_nodes",
        "shot_count",
        "depolarizing_probability",
        "measurement_bit_flip_probability",
        "task",
        "split",
    )
    table = pd.DataFrame(rows)
    aggregated: list[dict[str, Any]] = []
    condition_groups = table.groupby(list(condition_fields), dropna=False, sort=True)
    for condition, condition_table in condition_groups:
        common = {name: _plain_scalar(value) for name, value in zip(condition_fields, condition)}
        task = str(common["task"])
        metric_names = (
            (CLASSIFICATION_METRICS if task == TASKS[0] else REGRESSION_METRICS)
            + STABILITY_METRICS
            + RESOURCE_METRICS
        )
        package_versions = condition_table.iloc[0]["package_versions"]
        if any(value != package_versions for value in condition_table["package_versions"]):
            package_versions = {"note": "multiple environments; inspect per-run table"}
        provenance = {
            "data_snapshot_id": str(condition_table.iloc[0]["data_snapshot_id"]),
            "backend": str(condition_table.iloc[0]["backend"]),
            "git_commits": sorted(str(value) for value in condition_table["git_commit"].unique()),
            "git_dirty_values": sorted(
                bool(value) for value in condition_table["git_dirty"].unique()
            ),
            "python_versions": sorted(
                str(value) for value in condition_table["python_version"].unique()
            ),
            "operating_systems": sorted(
                str(value) for value in condition_table["operating_system"].unique()
            ),
            "execution_platforms": sorted(
                str(value) for value in condition_table["execution_platform"].unique()
            ),
            "package_versions": package_versions,
            "data_manifest_checksums": sorted(
                str(value) for value in condition_table["data_manifest_checksum"].unique()
            ),
            "source_snapshot_manifest_checksums": sorted(
                str(value)
                for value in condition_table["source_snapshot_manifest_checksum"].unique()
            ),
            "study_configuration_checksums": sorted(
                str(value) for value in condition_table["study_configuration_checksum"].unique()
            ),
            "run_configuration_checksums": sorted(
                str(value) for value in condition_table["run_configuration_checksum"].unique()
            ),
            "qrc_configuration_checksums": sorted(
                str(value) for value in condition_table["qrc_configuration_checksum"].unique()
            ),
            "measurement_configuration_checksums": sorted(
                str(value)
                for value in condition_table["measurement_configuration_checksum"].unique()
            ),
            "feature_cache_key_checksums": sorted(
                str(value) for value in condition_table["feature_cache_key_checksum"].unique()
            ),
        }

        for reservoir_seed, reservoir_table in condition_table.groupby(
            "reservoir_seed",
            sort=True,
        ):
            reservoir_seed_integer = int(cast(Any, reservoir_seed))
            for metric in metric_names:
                if metric not in reservoir_table:
                    continue
                values = reservoir_table[metric].dropna().to_numpy(dtype=float)
                if not len(values):
                    continue
                aggregated.append(
                    {
                        **common,
                        "aggregation_level": "measurement_seeds_within_reservoir",
                        "reservoir_seed": reservoir_seed_integer,
                        "measurement_seeds": sorted(
                            int(value)
                            for value in reservoir_table["measurement_seed"].dropna().unique()
                        ),
                        "reservoir_seeds": [reservoir_seed_integer],
                        "metric": metric,
                        **_summary_statistics(values),
                        **provenance,
                    }
                )

        for metric in metric_names:
            if metric not in condition_table:
                continue
            raw_values = condition_table[metric].dropna().to_numpy(dtype=float)
            if not len(raw_values):
                continue
            reservoir_means = (
                condition_table.groupby("reservoir_seed", sort=True)[metric]
                .mean()
                .dropna()
                .to_numpy(dtype=float)
            )
            aggregated.append(
                {
                    **common,
                    "aggregation_level": "reservoir_seeds",
                    "reservoir_seed": None,
                    "measurement_seeds": sorted(
                        int(value)
                        for value in condition_table["measurement_seed"].dropna().unique()
                    ),
                    "reservoir_seeds": sorted(
                        int(value) for value in condition_table["reservoir_seed"].unique()
                    ),
                    "metric": metric,
                    **_summary_statistics(reservoir_means),
                    **provenance,
                }
            )
            aggregated.append(
                {
                    **common,
                    "aggregation_level": "all_repetitions",
                    "reservoir_seed": None,
                    "measurement_seeds": sorted(
                        int(value)
                        for value in condition_table["measurement_seed"].dropna().unique()
                    ),
                    "reservoir_seeds": sorted(
                        int(value) for value in condition_table["reservoir_seed"].unique()
                    ),
                    "metric": metric,
                    **_summary_statistics(raw_values),
                    **provenance,
                }
            )
    return aggregated


def _csv_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for row in rows:
        values = dict(row)
        for name, value in tuple(values.items()):
            if isinstance(value, (dict, list, tuple)):
                values[name] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        safe.append(values)
    return safe


def _reference_statistics(table: pd.DataFrame, metric: str, task: str) -> tuple[float, float]:
    reference = table.loc[
        table["analytic_reference"].eq(True) & table["task"].eq(task) & table["split"].eq("test"),
        metric,
    ].to_numpy(dtype=float)
    if not len(reference):
        raise ValueError(f"analytic reference omits {metric}")
    return float(reference.mean()), float(reference.std(ddof=0))


def _plot_condition_metric(
    table: pd.DataFrame,
    *,
    study_type: str,
    x_column: str,
    metric: str,
    task: str,
    xlabel: str,
    ylabel: str,
    destination: Path,
    log_x: bool = False,
    symlog_x: bool = False,
) -> None:
    subset = table.loc[
        table["study_type"].eq(study_type) & table["task"].eq(task) & table["split"].eq("test")
    ].copy()
    if subset.empty or metric not in subset:
        raise ValueError(f"robustness results omit {study_type} test metric {metric}")
    figure, axis = plt.subplots(figsize=(5.9, 3.9))
    axis.scatter(
        subset[x_column],
        subset[metric],
        color="#4c78a8",
        alpha=0.28,
        s=18,
        label="reservoir and measurement repetitions",
    )
    grouped = subset.groupby(x_column)[metric]
    mean = grouped.mean()
    standard_deviation = grouped.std(ddof=0).fillna(0.0)
    axis.errorbar(
        mean.index,
        mean.to_numpy(dtype=float),
        yerr=standard_deviation.to_numpy(dtype=float),
        color="#1f1f1f",
        linewidth=1.8,
        marker="o",
        capsize=3,
        label="mean ± population SD",
        zorder=4,
    )
    reference_mean, reference_sd = _reference_statistics(table, metric, task)
    axis.axhline(
        reference_mean,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label="analytic-expectation mean",
    )
    if reference_sd > 0.0:
        axis.axhspan(
            reference_mean - reference_sd,
            reference_mean + reference_sd,
            color="#d62728",
            alpha=0.08,
        )
    if log_x:
        axis.set_xscale("log", base=2)
    if symlog_x:
        positive = [float(value) for value in subset[x_column].unique() if float(value) > 0.0]
        linthresh = min(positive) if positive else 1e-4
        axis.set_xscale("symlog", linthresh=linthresh)
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=220)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def _plot_runtime_vs_shots(table: pd.DataFrame, destination: Path) -> None:
    subset = table.loc[
        table["study_type"].eq("finite_shot")
        & table["task"].eq(TASKS[0])
        & table["split"].eq("test")
    ].copy()
    subset["feature_runtime_seconds"] = (
        subset["state_generation_seconds"] + subset["sampling_seconds"]
    )
    figure, axis = plt.subplots(figsize=(5.9, 3.9))
    axis.scatter(
        subset["shot_count"],
        subset["feature_runtime_seconds"],
        alpha=0.28,
        s=18,
        color="#4c78a8",
        label="reservoir and measurement repetitions",
    )
    grouped = subset.groupby("shot_count")["feature_runtime_seconds"]
    mean = grouped.mean()
    deviation = grouped.std(ddof=0).fillna(0.0)
    axis.errorbar(
        mean.index,
        mean.to_numpy(dtype=float),
        yerr=deviation.to_numpy(dtype=float),
        color="black",
        marker="o",
        linewidth=1.8,
        capsize=3,
        label="mean ± population SD",
    )
    analytic = table.loc[
        table["analytic_reference"].eq(True)
        & table["task"].eq(TASKS[0])
        & table["split"].eq("test"),
        "state_generation_seconds",
    ].to_numpy(dtype=float)
    axis.axhline(
        float(analytic.mean()),
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label="analytic-expectation mean",
    )
    axis.set_xscale("log", base=2)
    axis.set(xlabel="shots per virtual-node state", ylabel="feature runtime (seconds)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=220)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def _plot_measurement_seed_variance(table: pd.DataFrame, destination: Path) -> None:
    specifications = (
        ("macro_f1", TASKS[0], "macro-F1 variance"),
        ("transition_pr_auc", TASKS[0], "transition PR-AUC variance"),
        ("qlike", TASKS[1], "QLIKE variance"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    for axis, (metric, task, ylabel) in zip(axes, specifications):
        subset = table.loc[
            table["study_type"].eq("finite_shot")
            & table["task"].eq(task)
            & table["split"].eq("test")
        ]
        variance = (
            subset.groupby(["shot_count", "reservoir_seed"])[metric]
            .var(ddof=0)
            .reset_index(name="variance")
        )
        axis.scatter(
            variance["shot_count"],
            variance["variance"],
            alpha=0.35,
            s=18,
            label="reservoir seeds",
        )
        mean = variance.groupby("shot_count")["variance"].mean()
        axis.plot(mean.index, mean, color="black", marker="o", label="reservoir mean")
        axis.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.2)
        axis.set_xscale("log", base=2)
        axis.set(xlabel="shots", ylabel=ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=220)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def plot_robustness_figures(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    """Write all required seed-level plus aggregate/reference figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(_csv_safe_rows(rows))
    outputs: dict[str, Path] = {}
    specifications = (
        (
            "test_macro_f1_vs_shots",
            "finite_shot",
            "shot_count",
            "macro_f1",
            TASKS[0],
            "shots per virtual-node state",
            "test macro F1",
            True,
            False,
        ),
        (
            "test_transition_pr_auc_vs_shots",
            "finite_shot",
            "shot_count",
            "transition_pr_auc",
            TASKS[0],
            "shots per virtual-node state",
            "test transition PR-AUC",
            True,
            False,
        ),
        (
            "test_qlike_vs_shots",
            "finite_shot",
            "shot_count",
            "qlike",
            TASKS[1],
            "shots per virtual-node state",
            "test QLIKE",
            True,
            False,
        ),
    )
    probability_specs = (
        ("macro_f1", TASKS[0], "test macro F1"),
        ("transition_pr_auc", TASKS[0], "test transition PR-AUC"),
        ("qlike", TASKS[1], "test QLIKE"),
    )
    expanded = list(specifications)
    for metric, task, ylabel in probability_specs:
        expanded.append(
            (
                f"test_{metric}_vs_depolarizing_probability",
                "depolarizing_noise",
                "depolarizing_probability",
                metric,
                task,
                "local depolarizing probability",
                ylabel,
                False,
                True,
            )
        )
        expanded.append(
            (
                f"test_{metric}_vs_measurement_noise_probability",
                "measurement_noise",
                "measurement_bit_flip_probability",
                metric,
                task,
                "measurement bit-flip probability",
                ylabel,
                False,
                False,
            )
        )
    for (
        name,
        study_type,
        x_column,
        metric,
        task,
        xlabel,
        ylabel,
        log_x,
        symlog_x,
    ) in expanded:
        destination = output_dir / name
        _plot_condition_metric(
            table,
            study_type=study_type,
            x_column=x_column,
            metric=metric,
            task=task,
            xlabel=xlabel,
            ylabel=ylabel,
            destination=destination,
            log_x=log_x,
            symlog_x=symlog_x,
        )
        outputs[f"{name}_png"] = destination.with_suffix(".png")
        outputs[f"{name}_pdf"] = destination.with_suffix(".pdf")
    runtime_destination = output_dir / "runtime_vs_shots"
    _plot_runtime_vs_shots(table, runtime_destination)
    outputs["runtime_vs_shots_png"] = runtime_destination.with_suffix(".png")
    outputs["runtime_vs_shots_pdf"] = runtime_destination.with_suffix(".pdf")
    variance_destination = output_dir / "measurement_seed_variance_vs_shots"
    _plot_measurement_seed_variance(table, variance_destination)
    outputs["measurement_seed_variance_vs_shots_png"] = variance_destination.with_suffix(".png")
    outputs["measurement_seed_variance_vs_shots_pdf"] = variance_destination.with_suffix(".pdf")
    return outputs


def robustness_resource_estimate(
    points: tuple[RobustnessPoint, ...],
    *,
    split_rows: int,
    virtual_nodes: int,
) -> dict[str, Any]:
    """Estimate unique cache, dense-state, and sampled-bitstring requirements."""

    measurement_identities = {
        (
            point.n_qubits,
            point.reservoir_seed,
            point.measurement_config.checksum,
        ): point
        for point in points
    }
    unique_points = tuple(measurement_identities.values())
    n_qubits = unique_points[0].n_qubits
    edges = 1 if n_qubits == 2 else n_qubits
    feature_dimension = virtual_nodes * (n_qubits + edges)
    bytes_per_feature_cache = split_rows * feature_dimension * 8
    virtual_node_states = split_rows * virtual_nodes
    sampled_bitstrings = 0
    for point in unique_points:
        if point.shot_count is not None:
            sampled_bitstrings += virtual_node_states * point.shot_count
    maximum_shots = max(
        (point.shot_count or 0 for point in unique_points),
        default=0,
    )
    return {
        "n_qubits": n_qubits,
        "hilbert_dimension": 2**n_qubits,
        "raw_feature_dimension": feature_dimension,
        "split_rows": split_rows,
        "virtual_nodes": virtual_nodes,
        "requested_experimental_points": len(points),
        "unique_feature_cache_points": len(unique_points),
        "readout_task_runs": len(points) * len(TASKS),
        "bytes_per_uncompressed_feature_cache": bytes_per_feature_cache,
        "estimated_total_uncompressed_feature_cache_bytes": (
            bytes_per_feature_cache * len(unique_points)
        ),
        "estimated_peak_density_matrix_bytes": 3 * (2**n_qubits) ** 2 * 16,
        "maximum_sampling_batch_bytes": maximum_shots * n_qubits,
        "virtual_node_state_evolutions": virtual_node_states * len(unique_points),
        "sampled_bitstrings": sampled_bitstrings,
        "prediction_artifacts_and_library_overhead_included": False,
    }


def _write_tables(
    output_root: Path,
    rows: list[dict[str, Any]],
    aggregated: list[dict[str, Any]],
    selected_resources: dict[str, Any],
    full_resources: dict[str, Any],
) -> dict[str, Path]:
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_run_json": tables / "qrc_noise_robustness_per_run.json",
        "per_run_csv": tables / "qrc_noise_robustness_per_run.csv",
        "aggregate_json": tables / "qrc_noise_robustness_aggregate.json",
        "aggregate_csv": tables / "qrc_noise_robustness_aggregate.csv",
        "resources_json": tables / "qrc_noise_robustness_resources.json",
        "resources_csv": tables / "qrc_noise_robustness_resources.csv",
    }
    _write_json(outputs["per_run_json"], {"schema_version": 1, "rows": rows})
    pd.DataFrame(_csv_safe_rows(rows)).to_csv(outputs["per_run_csv"], index=False)
    _write_json(outputs["aggregate_json"], {"schema_version": 1, "rows": aggregated})
    pd.DataFrame(_csv_safe_rows(aggregated)).to_csv(outputs["aggregate_csv"], index=False)
    resources = [
        {"scope": "selected_grid", **selected_resources},
        {"scope": "configured_full_grid", **full_resources},
    ]
    _write_json(outputs["resources_json"], {"schema_version": 1, "rows": resources})
    pd.DataFrame(_csv_safe_rows(resources)).to_csv(outputs["resources_csv"], index=False)
    return outputs


def run_qrc_noise_robustness(
    config_path: Path,
    *,
    n_qubits: int | None = None,
    smoke: bool = False,
    resume: bool = True,
    studies: tuple[str, ...] = SELECTABLE_STUDIES,
) -> Path:
    """Run/resume all requested one-factor points and publish aggregate artifacts."""

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config = load_robustness_study_config(config_path)
    selected_n_qubits = n_qubits if n_qubits is not None else config.selected_n_qubits
    selected_points = build_robustness_grid(
        n_qubits=selected_n_qubits,
        reservoir_seeds=(config.smoke_reservoir_seeds if smoke else config.reservoir_seeds),
        measurement_seeds=(config.smoke_measurement_seeds if smoke else config.measurement_seeds),
        shots=config.smoke_shots if smoke else config.shots,
        depolarizing_probabilities=(
            config.smoke_depolarizing_probabilities if smoke else config.depolarizing_probabilities
        ),
        measurement_noise_probabilities=(
            config.smoke_measurement_noise_probabilities
            if smoke
            else config.measurement_noise_probabilities
        ),
        studies=studies,
    )
    full_points = build_robustness_grid(
        n_qubits=selected_n_qubits,
        reservoir_seeds=config.reservoir_seeds,
        measurement_seeds=config.measurement_seeds,
        shots=config.shots,
        depolarizing_probabilities=config.depolarizing_probabilities,
        measurement_noise_probabilities=config.measurement_noise_probabilities,
    )
    classifier_reference, _ = _assert_reference_contracts(config)
    dataset, data_provenance = verify_robustness_public_data(config, classifier_reference)
    config.output_root.mkdir(parents=True, exist_ok=True)
    runs_root = config.output_root / "runs"
    completed = discover_completed_robustness_runs(
        runs_root,
        study_id=config.study_id,
        snapshot_id=config.data_snapshot_id,
        study_configuration_checksum=sha256_file(config.source),
    )
    requested_identities = {(point.checksum, task) for point in selected_points for task in TASKS}
    completed_before = len(requested_identities & set(completed))
    executed: list[dict[str, Any]] = []
    resumed: list[dict[str, Any]] = []
    state_path = config.output_root / "robustness_state.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "study_id": config.study_id,
        "points": {},
    }
    if state_path.is_file():
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            not isinstance(loaded, dict)
            or loaded.get("study_id") != config.study_id
            or not isinstance(loaded.get("points"), dict)
        ):
            raise ValueError("robustness state file is invalid")
        state = loaded
    point_state = cast(dict[str, Any], state["points"])

    for point in selected_points:
        classifier_path = write_robustness_model_config(config, point, TASKS[0])
        regressor_path = write_robustness_model_config(config, point, TASKS[1])
        bundle, key = prepare_robustness_features(
            config=config,
            point=point,
            classifier_config_path=classifier_path,
            dataset=dataset,
        )
        state_record = {
            "point": asdict(point),
            "point_checksum": point.checksum,
            "feature_cache_key_checksum": key.checksum,
            "feature_cache_directory": bundle.cache_dir.relative_to(config.project_root).as_posix(),
            "feature_preparation_cache_hit": bundle.cache_hit,
            "completed_tasks": [],
        }
        previous_state = point_state.get(point.checksum)
        if isinstance(previous_state, dict):
            if previous_state.get("feature_cache_key_checksum") != key.checksum:
                raise ValueError("resumed robustness point cache key changed")
            state_record["completed_tasks"] = list(previous_state.get("completed_tasks", []))
        point_state[point.checksum] = state_record
        _write_json(state_path, state)

        for task, model_path in (
            (TASKS[0], classifier_path),
            (TASKS[1], regressor_path),
        ):
            identity = (point.checksum, task)
            if resume and identity in completed:
                experiment_dir = completed[identity]
                resumed.append(
                    {
                        "point_checksum": point.checksum,
                        "study_type": point.study_type,
                        "task": task,
                        "experiment_directory": experiment_dir.relative_to(
                            config.project_root
                        ).as_posix(),
                    }
                )
            else:
                task_cache_hit = bundle.cache_hit or task == TASKS[1]
                experiment_dir = _run_robustness_readout(
                    study_config=config,
                    point=point,
                    model_config_path=model_path,
                    dataset=dataset,
                    bundle=bundle,
                    cache_key=key,
                    cache_hit=task_cache_hit,
                )
                completed[identity] = experiment_dir
                executed.append(
                    {
                        "point_checksum": point.checksum,
                        "study_type": point.study_type,
                        "task": task,
                        "experiment_directory": experiment_dir.relative_to(
                            config.project_root
                        ).as_posix(),
                    }
                )
            manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
            if manifest["qrc_feature_cache_key_checksum"] != key.checksum:
                raise ValueError("classifier/regressor did not reuse the prepared feature cache")
            completed_tasks = cast(list[str], state_record["completed_tasks"])
            if task not in completed_tasks:
                completed_tasks.append(task)
                completed_tasks.sort()
                _write_json(state_path, state)

    requested_runs = {
        identity: completed[identity]
        for identity in sorted(requested_identities)
        if identity in completed
    }
    if len(requested_runs) != len(requested_identities):
        raise RuntimeError(
            f"robustness grid incomplete: {len(requested_runs)} "
            f"of {len(requested_identities)} task runs"
        )
    rows = collect_robustness_rows(
        requested_runs,
        selected_points,
        repository_root=config.project_root,
    )
    aggregated = aggregate_robustness_rows(rows)
    split_rows = len(dataset.train.X) + len(dataset.validation.X) + len(dataset.test.X)
    selected_resources = robustness_resource_estimate(
        selected_points,
        split_rows=split_rows,
        virtual_nodes=int(config.fixed_qrc["virtual_nodes"]),
    )
    full_resources = robustness_resource_estimate(
        full_points,
        split_rows=split_rows,
        virtual_nodes=int(config.fixed_qrc["virtual_nodes"]),
    )
    outputs = _write_tables(
        config.output_root,
        rows,
        aggregated,
        selected_resources,
        full_resources,
    )
    figures = plot_robustness_figures(rows, config.output_root / "figures")
    summary_path = config.output_root / "robustness_run_summary.json"
    summary = {
        "schema_version": 1,
        "study_id": config.study_id,
        "status": "success",
        "mode": "smoke" if smoke else "full",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        **runtime_metadata(),
        "git": _git_metadata(config.project_root),
        "configuration": {
            "path": config.source.relative_to(config.project_root).as_posix(),
            "sha256": sha256_file(config.source),
        },
        "n_qubits": selected_n_qubits,
        "selected_architecture_default": config.selected_n_qubits,
        "one_factor_at_a_time": True,
        "studies": list(studies),
        "grid": [asdict(point) for point in selected_points],
        "data_provenance": data_provenance,
        "resume_enabled": resume,
        "completed_before_run": completed_before,
        "executed_runs": executed,
        "resumed_runs": resumed,
        "per_run_row_count": len(rows),
        "aggregate_row_count": len(aggregated),
        "selected_grid_resources": selected_resources,
        "configured_full_grid_resources": full_resources,
        "outputs": {
            **{
                name: path.relative_to(config.project_root).as_posix()
                for name, path in outputs.items()
            },
            **{
                f"figure_{name}": path.relative_to(config.project_root).as_posix()
                for name, path in figures.items()
            },
        },
        "interpretation": (
            "controlled classical density-matrix and measurement simulation; "
            "not physical-QPU execution or a quantum-advantage claim"
        ),
    }
    _write_json(summary_path, summary)
    return summary_path
