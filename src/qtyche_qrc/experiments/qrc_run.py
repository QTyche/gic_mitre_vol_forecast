"""Feature generation, validation-only selection, and exact QRC experiments."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from qtyche_qrc.evaluation.metrics import classification_metrics, regression_metrics
from qtyche_qrc.evaluation.plots import (
    SYNTHETIC_WARNING,
    plot_confusion_matrix,
    plot_regression_diagnostics,
    plot_rv_series,
    plot_transition_calibration,
    plot_transition_series,
)
from qtyche_qrc.experiments.model_config import ModelExperimentConfig, load_model_config
from qtyche_qrc.experiments.run import (
    SyntheticResultsError,
    _classification_predictions,
    _experiment_directory,
    _git_metadata,
    _package_versions,
    _regression_predictions,
    _sha256,
    _write_json,
)
from qtyche_qrc.models.base import ForecastModel
from qtyche_qrc.models.dataset import ModelDataset, SelectionDataset, load_model_dataset
from qtyche_qrc.models.qrc.features import (
    FeatureCacheKey,
    QRCFeatureBundle,
    generate_or_load_features,
    make_feature_cache_key,
)
from qtyche_qrc.models.qrc.readout import QRCClassifier, QRCReadoutConfig, QRCRegressor
from qtyche_qrc.models.qrc.reservoir import QRCConfig, QuantumReservoir


@dataclass(frozen=True)
class QRCSelectionResult:
    """One validation-only ridge-head candidate."""

    trial: int
    ridge_alpha: float
    selection_metric: str
    validation_score: float | None
    status: str
    training_seconds: float | None = None
    prediction_seconds: float | None = None
    error: str | None = None


def qrc_config_from_model(
    config: ModelExperimentConfig, reservoir_seed: int | None = None
) -> QRCConfig:
    """Extract only frozen reservoir dynamics from the model parameter mapping."""

    parameters = dict(config.parameters)
    for readout_key in ("ridge_alpha", "transform_epsilon"):
        parameters.pop(readout_key, None)
    parameters["reservoir_seed"] = int(
        reservoir_seed
        if reservoir_seed is not None
        else parameters.get("reservoir_seed", config.seed)
    )
    chords = parameters.get("chords")
    if chords is not None:
        parameters["chords"] = tuple(tuple(int(item) for item in edge) for edge in chords)
    return QRCConfig(**parameters)


def qrc_readout_config(config: ModelExperimentConfig, ridge_alpha: float) -> QRCReadoutConfig:
    return QRCReadoutConfig(
        ridge_alpha=ridge_alpha,
        transform_epsilon=float(config.parameters.get("transform_epsilon", 1e-12)),
    )


def _cache_root(config: ModelExperimentConfig) -> Path:
    qrc_section = config.raw.get("qrc", {})
    setting = qrc_section.get("feature_cache", "results/qrc_cache")
    if not isinstance(setting, str) or not setting:
        raise ValueError("qrc.feature_cache must be a non-empty path string")
    return (config.project_root / setting).resolve()


def _validated_qrc_inputs(
    config_path: Path,
    *,
    allow_synthetic_results: bool,
    reservoir_seed: int | None,
) -> tuple[ModelExperimentConfig, ModelDataset, QRCConfig, FeatureCacheKey]:
    config = load_model_config(config_path)
    if config.model_type not in {"qrc_classifier", "qrc_regressor"}:
        raise ValueError("QRC command requires qrc_classifier or qrc_regressor model.type")
    data = load_model_dataset(config.processed_dir)
    if data.is_synthetic and not allow_synthetic_results:
        raise SyntheticResultsError(
            "public QRC pilot rejects synthetic data; pass --allow-synthetic-results only "
            "for the offline fixture smoke experiment"
        )
    qrc_config = qrc_config_from_model(config, reservoir_seed)
    manifest_checksum = data.processed_checksums["data_manifest.json"]
    key = make_feature_cache_key(
        processed_data_manifest_checksum=manifest_checksum,
        feature_names=data.feature_names,
        config=qrc_config,
    )
    return config, data, qrc_config, key


def generate_qrc_features(
    config_path: Path,
    *,
    allow_synthetic_results: bool = False,
    reservoir_seed: int | None = None,
) -> QRCFeatureBundle:
    """Generate or checksum-load QRC states from inputs and never pass targets."""

    config, data, qrc_config, key = _validated_qrc_inputs(
        config_path,
        allow_synthetic_results=allow_synthetic_results,
        reservoir_seed=reservoir_seed,
    )
    return generate_or_load_features(
        cache_root=_cache_root(config),
        key=key,
        feature_names=data.feature_names,
        config=qrc_config,
        X_train=data.train.X,
        X_validation=data.validation.X,
        X_test=data.test.X,
    )


def select_qrc_readout(
    *,
    config: ModelExperimentConfig,
    data: SelectionDataset,
    train_features: NDArray[np.float64],
    validation_features: NDArray[np.float64],
) -> tuple[float, list[QRCSelectionResult]]:
    """Select ridge alpha using only the structurally test-free dataset view."""

    if len(train_features) != len(data.train.X) or len(validation_features) != len(
        data.validation.X
    ):
        raise ValueError("QRC selection features do not align with train/validation inputs")
    candidates = (
        config.search_space.get("ridge_alpha", [config.parameters.get("ridge_alpha", 0.1)])
        if config.search_enabled
        else [config.parameters.get("ridge_alpha", 0.1)]
    )
    candidates = candidates[: config.maximum_trials]
    results: list[QRCSelectionResult] = []
    for trial, candidate in enumerate(candidates, start=1):
        alpha = float(candidate)
        try:
            head_config = qrc_readout_config(config, alpha)
            started = time.perf_counter()
            if config.task == "regime_classification":
                model = QRCClassifier(tuple(), head_config)
                model.fit(train_features, data.train.y_regime)
                training_seconds = time.perf_counter() - started
                started = time.perf_counter()
                prediction = model.predict_proba(validation_features)
                prediction_seconds = time.perf_counter() - started
                score = float(
                    classification_metrics(data.validation.y_regime, prediction)["macro_f1"]
                )
                metric = "macro_f1"
            else:
                regressor = QRCRegressor(tuple(), head_config)
                regressor.fit(train_features, data.train.y_rv)
                training_seconds = time.perf_counter() - started
                started = time.perf_counter()
                prediction = regressor.predict(validation_features)
                prediction_seconds = time.perf_counter() - started
                score = float(
                    regression_metrics(
                        data.validation.y_rv, prediction, config.variance_floor
                    ).metrics["qlike"]
                )
                metric = "qlike"
            results.append(
                QRCSelectionResult(
                    trial,
                    alpha,
                    metric,
                    score,
                    "success",
                    training_seconds,
                    prediction_seconds,
                )
            )
        except Exception as exc:
            results.append(
                QRCSelectionResult(
                    trial,
                    alpha,
                    config.selection_metric,
                    None,
                    "failure",
                    error=str(exc),
                )
            )
    successful = [result for result in results if result.status == "success"]
    if not successful:
        raise ValueError("all QRC ridge candidates failed")
    if any(item.validation_score is None for item in successful):
        raise ValueError("successful QRC candidate unexpectedly omitted its validation score")
    selected = (
        max(successful, key=lambda item: item.validation_score or float("-inf"))
        if config.task == "regime_classification"
        else min(successful, key=lambda item: item.validation_score or float("inf"))
    )
    return selected.ridge_alpha, results


def _save_reservoir_artifacts(reservoir: QuantumReservoir, model_dir: Path) -> None:
    np.savez_compressed(
        model_dir / "qrc_hamiltonian.npz",
        matrix=reservoir.hamiltonian.matrix,
        edges=np.asarray(reservoir.hamiltonian.edges, dtype=int),
        couplings=reservoir.hamiltonian.couplings,
        fields=reservoir.hamiltonian.fields,
    )
    np.save(model_dir / "input_projection.npy", reservoir.input_projection)
    _write_json(
        model_dir / "observables.json",
        {
            **reservoir.observables.metadata(),
            "checksum": reservoir.observables.checksum,
        },
    )
    _write_json(model_dir / "qrc_hamiltonian.json", reservoir.hamiltonian.metadata())


def _write_effective_configuration(source: Path, destination: Path, reservoir_seed: int) -> None:
    """Persist the exact config after applying the explicit CLI seed override."""

    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("QRC model configuration must be a mapping")
    raw["experiment"]["seed"] = reservoir_seed
    raw["model"]["parameters"]["reservoir_seed"] = reservoir_seed
    destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def run_qrc_experiment(
    config_path: Path,
    *,
    allow_synthetic_results: bool = False,
    reservoir_seed: int | None = None,
) -> Path:
    """Select on validation, freeze the ridge head, and only then evaluate test."""

    config, data, qrc_config, key = _validated_qrc_inputs(
        config_path,
        allow_synthetic_results=allow_synthetic_results,
        reservoir_seed=reservoir_seed,
    )
    if reservoir_seed is not None:
        updated_parameters = dict(config.parameters)
        updated_parameters["reservoir_seed"] = reservoir_seed
        config = replace(config, seed=reservoir_seed, parameters=updated_parameters)
    bundle = generate_or_load_features(
        cache_root=_cache_root(config),
        key=key,
        feature_names=data.feature_names,
        config=qrc_config,
        X_train=data.train.X,
        X_validation=data.validation.X,
        X_test=data.test.X,
    )
    experiment_dir = _experiment_directory(config)
    _write_effective_configuration(
        config.source, experiment_dir / "config.yaml", qrc_config.reservoir_seed
    )
    warning = SYNTHETIC_WARNING if data.is_synthetic else None
    warnings_list = [warning] if warning else []
    status = "success"
    test_evaluated_after_freeze = False
    selected_alpha: float | None = None
    timing: dict[str, float] = {}
    trials: list[QRCSelectionResult] = []
    try:
        selected_alpha, trials = select_qrc_readout(
            config=config,
            data=data.for_selection(),
            train_features=bundle.train,
            validation_features=bundle.validation,
        )
        feature_names = tuple(bundle.metadata["observable_metadata"]["feature_ordering"])
        head_config = qrc_readout_config(config, selected_alpha)
        started = time.perf_counter()
        model: ForecastModel
        if config.task == "regime_classification":
            classifier = QRCClassifier(feature_names, head_config)
            classifier.fit(bundle.train, data.train.y_regime)
            model = classifier
            timing["readout_fitting_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            validation_raw = classifier.predict_proba(bundle.validation)
            timing["validation_prediction_seconds"] = time.perf_counter() - started
            readout_frozen = True
            started = time.perf_counter()
            test_raw = classifier.predict_proba(bundle.test)
            timing["test_prediction_seconds"] = time.perf_counter() - started
            test_evaluated_after_freeze = readout_frozen
            validation_metrics, validation_predictions = _classification_predictions(
                data.validation, validation_raw, config.transition_threshold
            )
            test_metrics, test_predictions = _classification_predictions(
                data.test, test_raw, config.transition_threshold
            )
            plot_confusion_matrix(
                test_metrics["confusion_matrix"],
                experiment_dir / "figures" / "confusion_matrix.png",
                data.is_synthetic,
            )
            plot_transition_series(
                test_predictions,
                experiment_dir / "figures" / "transition_probability.png",
                data.is_synthetic,
            )
            plot_transition_calibration(
                data.test.y_transition,
                test_predictions["predicted_transition_probability"].to_numpy(dtype=float),
                experiment_dir / "figures" / "transition_calibration.png",
                data.is_synthetic,
            )
        else:
            regressor = QRCRegressor(feature_names, head_config)
            regressor.fit(bundle.train, data.train.y_rv)
            model = regressor
            timing["readout_fitting_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            validation_raw = regressor.predict(bundle.validation)
            timing["validation_prediction_seconds"] = time.perf_counter() - started
            readout_frozen = True
            started = time.perf_counter()
            test_raw = regressor.predict(bundle.test)
            timing["test_prediction_seconds"] = time.perf_counter() - started
            test_evaluated_after_freeze = readout_frozen
            validation_metrics, validation_predictions = _regression_predictions(
                data.validation, validation_raw, config.variance_floor
            )
            test_metrics, test_predictions = _regression_predictions(
                data.test, test_raw, config.variance_floor
            )
            plot_rv_series(
                test_predictions,
                experiment_dir / "figures" / "realized_variance.png",
                data.is_synthetic,
            )
            plot_regression_diagnostics(
                test_predictions, experiment_dir / "figures", data.is_synthetic
            )

        trial_rows = [asdict(result) for result in trials]
        trial_table = pd.DataFrame(trial_rows)
        trial_table.insert(0, "data_source_type", data.data_source_type)
        trial_table.insert(1, "is_synthetic", data.is_synthetic)
        trial_table.insert(2, "data_warning", warning)
        trial_table.to_csv(experiment_dir / "selection_results.csv", index=False)
        validation_predictions.to_csv(experiment_dir / "validation_predictions.csv", index=False)
        test_predictions.to_csv(experiment_dir / "test_predictions.csv", index=False)
        _write_json(
            experiment_dir / "validation_metrics.json",
            {
                "data_source_type": data.data_source_type,
                "is_synthetic": data.is_synthetic,
                "data_warning": warning,
                **validation_metrics,
            },
        )
        _write_json(
            experiment_dir / "test_metrics.json",
            {
                "data_source_type": data.data_source_type,
                "is_synthetic": data.is_synthetic,
                "data_warning": warning,
                **test_metrics,
            },
        )
        timing["state_generation_seconds"] = float(
            bundle.metadata["resource_metadata"]["state_generation_seconds"]
        )
        _write_json(experiment_dir / "timing.json", timing)
        model.save(experiment_dir / "model")
        reservoir = QuantumReservoir(len(data.feature_names), qrc_config)
        _save_reservoir_artifacts(reservoir, experiment_dir / "model")
        model_metadata = model.get_model_metadata()
        _write_json(experiment_dir / "model_metadata.json", model_metadata)
        _write_json(
            experiment_dir / "qrc_backend_metadata.json",
            bundle.metadata["resource_metadata"],
        )
        _write_json(
            experiment_dir / "qrc_numerical_diagnostics.json",
            bundle.metadata["numerical_diagnostics"],
        )
        _write_json(
            experiment_dir / "qrc_feature_metadata.json",
            {
                **bundle.metadata,
                "cache_directory": str(bundle.cache_dir),
                "cache_hit_for_experiment": bundle.cache_hit,
            },
        )
    except Exception as exc:
        status = "failure"
        warnings_list.append(str(exc))
        (experiment_dir / "logs" / "failure.txt").write_text(str(exc) + "\n", encoding="utf-8")
        raise
    finally:
        raw_dimension = int(bundle.train.shape[1])
        outputs = 3 if config.task == "regime_classification" else 1
        qrc_section = config.raw.get("qrc", {})
        configured_seeds = qrc_section.get("reservoir_seeds", [qrc_config.reservoir_seed])
        if not isinstance(configured_seeds, list) or not configured_seeds:
            configured_seeds = [qrc_config.reservoir_seed]
        manifest = {
            "schema_version": 1,
            "experiment_id": experiment_dir.name,
            "git": _git_metadata(config.project_root),
            "model_type": config.model_type,
            "task": config.task,
            "configuration_checksum": _sha256(experiment_dir / "config.yaml"),
            "processed_data_checksums": data.processed_checksums,
            "data_manifest_checksum": data.processed_checksums.get("data_manifest.json"),
            "data_snapshot_id": data.manifest.get("source_snapshot_id"),
            "source_snapshot_manifest_checksum": data.manifest.get(
                "source_snapshot_manifest_checksum"
            ),
            "data_source_type": data.data_source_type,
            "is_synthetic": data.is_synthetic,
            "data_warning": warning,
            "split_row_counts": {
                "train": len(data.train.X),
                "validation": len(data.validation.X),
                "test": len(data.test.X),
            },
            "selected_features": list(data.feature_names),
            "target": "target_regime_5d"
            if config.task == "regime_classification"
            else "target_rv_5d",
            "seed": qrc_config.reservoir_seed,
            "reservoir_seed": qrc_config.reservoir_seed,
            "number_of_reservoir_seeds": len(configured_seeds),
            "configured_reservoir_seeds": configured_seeds,
            "selected_hyperparameters": {"ridge_alpha": selected_alpha},
            "model_selection_metric": config.selection_metric,
            "model_selection_trial_count": len(trials),
            "model_selection_data": "validation only",
            "test_evaluated_after_readout_freeze": test_evaluated_after_freeze,
            "qrc_configuration": asdict(qrc_config),
            "qrc_configuration_checksum": qrc_config.checksum,
            "qrc_feature_cache_key_checksum": key.checksum,
            "qrc_feature_cache_hit": bundle.cache_hit,
            "qrc_raw_feature_dimension": raw_dimension,
            "readout_shape": [raw_dimension + 1, outputs],
            "trainable_readout_parameters": (raw_dimension + 1) * outputs,
            "state_generation_time": timing.get("state_generation_seconds"),
            "readout_fitting_time": timing.get("readout_fitting_seconds"),
            "backend": bundle.metadata["resource_metadata"]["backend"],
            "exact_noiseless": True,
            "estimated_peak_density_matrix_bytes": bundle.metadata["resource_metadata"][
                "estimated_peak_density_matrix_bytes"
            ],
            "qrc_features_generated_without_labels": True,
            "package_versions": _package_versions(),
            "hostname": socket.gethostname(),
            "status": status,
            "warnings": warnings_list,
            "synthetic_override_used": allow_synthetic_results and data.is_synthetic,
        }
        _write_json(experiment_dir / "manifest.json", manifest)
    return experiment_dir


def inspect_qrc_experiment(experiment_dir: Path) -> dict[str, Any]:
    """Inspect standard metrics plus QRC backend, feature, and numerical evidence."""

    return {
        "manifest": json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8")),
        "validation_metrics": json.loads(
            (experiment_dir / "validation_metrics.json").read_text(encoding="utf-8")
        ),
        "test_metrics": json.loads(
            (experiment_dir / "test_metrics.json").read_text(encoding="utf-8")
        ),
        "backend": json.loads(
            (experiment_dir / "qrc_backend_metadata.json").read_text(encoding="utf-8")
        ),
        "numerical_diagnostics": json.loads(
            (experiment_dir / "qrc_numerical_diagnostics.json").read_text(encoding="utf-8")
        ),
        "feature_metadata": json.loads(
            (experiment_dir / "qrc_feature_metadata.json").read_text(encoding="utf-8")
        ),
    }


def _latest_qrc_experiments(results_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    latest: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for path in results_dir.rglob("manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("model_type") not in {"qrc_classifier", "qrc_regressor"}:
            continue
        if manifest.get("status") != "success":
            continue
        if manifest.get("data_source_type") != "public_market" or manifest.get("is_synthetic"):
            continue
        identity = (str(manifest["task"]), int(manifest["reservoir_seed"]))
        if identity not in latest or path.parent.name > latest[identity][0].name:
            latest[identity] = (path.parent, manifest)
    if not latest:
        raise ValueError(f"no completed public-market QRC experiments found below {results_dir}")
    return sorted(latest.values(), key=lambda item: (item[1]["task"], item[1]["reservoir_seed"]))


def compare_qrc_seeds(results_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Write separate pilot validation/test rows and across-seed summaries."""

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for experiment_dir, manifest in _latest_qrc_experiments(results_dir):
        common = {
            "experiment_id": manifest["experiment_id"],
            "task": manifest["task"],
            "model": manifest["model_type"],
            "reservoir_seed": manifest["reservoir_seed"],
            "selected_ridge_alpha": manifest["selected_hyperparameters"]["ridge_alpha"],
            "raw_feature_dimension": manifest["qrc_raw_feature_dimension"],
            "readout_shape": json.dumps(manifest["readout_shape"]),
            "trainable_readout_parameters": manifest["trainable_readout_parameters"],
            "state_generation_seconds": manifest["state_generation_time"],
            "readout_fitting_seconds": manifest["readout_fitting_time"],
            "cache_key_checksum": manifest["qrc_feature_cache_key_checksum"],
            "data_snapshot_id": manifest["data_snapshot_id"],
        }
        for split, rows in (("validation", validation_rows), ("test", test_rows)):
            metrics = json.loads(
                (experiment_dir / f"{split}_metrics.json").read_text(encoding="utf-8")
            )
            rows.append(
                {
                    **common,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    },
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "qrc_pilot_validation_by_seed.csv"
    test_path = output_dir / "qrc_pilot_test_by_seed.csv"
    summary_path = output_dir / "qrc_pilot_seed_summary.csv"
    validation = pd.DataFrame(validation_rows)
    test = pd.DataFrame(test_rows)
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)
    summary_rows: list[dict[str, Any]] = []
    metric_names = {
        "regime_classification": (
            "macro_f1",
            "balanced_accuracy",
            "transition_pr_auc",
            "log_loss",
            "multiclass_brier_score",
        ),
        "rv_regression": ("qlike", "rmse", "mae", "r_squared", "floored_prediction_count"),
    }
    for split_name, table in (("validation", validation), ("test", test)):
        for task, names in metric_names.items():
            subset = table.loc[table["task"].eq(task)]
            for name in names:
                values = subset[name].to_numpy(dtype=float)
                if not len(values):
                    continue
                summary_rows.append(
                    {
                        "split": split_name,
                        "task": task,
                        "metric": name,
                        "mean": float(values.mean()),
                        "standard_deviation": float(values.std(ddof=0)),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                        "seed_count": len(values),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    figure_path = output_dir / "qrc_pilot_performance_by_seed.png"
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    classification = test.loc[test["task"].eq("regime_classification")]
    regression = test.loc[test["task"].eq("rv_regression")]
    axes[0].plot(classification["reservoir_seed"], classification["macro_f1"], marker="o")
    axes[0].set(xlabel="reservoir seed", ylabel="test macro F1")
    axes[1].plot(regression["reservoir_seed"], regression["qlike"], marker="o")
    axes[1].set(xlabel="reservoir seed", ylabel="test QLIKE")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return {
        "validation": validation_path,
        "test": test_path,
        "summary": summary_path,
        "figure": figure_path,
    }
