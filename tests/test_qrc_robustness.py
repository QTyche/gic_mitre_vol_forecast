import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import yaml

from qtyche_qrc.experiments.model_config import load_model_config
from qtyche_qrc.experiments.qrc_robustness import (
    TASKS,
    RobustnessPoint,
    _robustness_run_uses_cache,
    aggregate_robustness_rows,
    build_robustness_grid,
    collect_robustness_rows,
    discover_completed_robustness_runs,
    load_robustness_study_config,
    pending_robustness_runs,
    robustness_resource_estimate,
    verify_robustness_public_data,
    write_robustness_model_config,
)
from qtyche_qrc.experiments.qrc_run import qrc_config_from_model
from qtyche_qrc.experiments.run import SyntheticResultsError
from qtyche_qrc.models.qrc.noise import QRCMeasurementConfig
from qtyche_qrc.models.qrc.reservoir import QRCConfig
from qtyche_qrc.models.qrc.robust_features import make_robust_feature_cache_key


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _study_path() -> Path:
    return _root() / "configs/qrc_noise_robustness.yaml"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fake_predictions(path: Path, task: str, offset: float) -> None:
    if task == TASKS[0]:
        frame = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-02"],
                "predicted_regime": [0, 1],
                "probability_low": [0.7 - offset, 0.2 + offset],
                "probability_medium": [0.2 + offset, 0.6 - offset],
                "probability_high": [0.1, 0.2],
                "predicted_transition_probability": [0.3 + offset, 0.4 - offset],
            }
        )
    else:
        frame = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-02"],
                "predicted_rv_5d": [0.01 + offset, 0.02 + offset],
            }
        )
    frame.to_csv(path, index=False)


def _fake_run(
    root: Path,
    point: RobustnessPoint,
    task: str,
    *,
    complete: bool = True,
    synthetic: bool = False,
    prediction_offset: float = 0.0,
) -> Path:
    directory = root / f"{point.key}_{task}"
    directory.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "experiment_id": directory.name,
        "status": "success",
        "study_id": "qrc_shot_noise_robustness_v1",
        "study_configuration_checksum": "study-config",
        "robustness_point": {
            **point.__dict__,
        },
        "robustness_point_checksum": point.checksum,
        "model_type": "qrc_classifier" if task == TASKS[0] else "qrc_regressor",
        "task": task,
        "configuration_checksum": "run-config",
        "processed_data_checksums": {"data_manifest.json": "data"},
        "data_manifest_checksum": "data",
        "data_snapshot_id": "yahoo_chart_20100101_20251231_v1",
        "source_snapshot_manifest_checksum": "snapshot",
        "data_source_type": "fixture" if synthetic else "public_market",
        "is_synthetic": synthetic,
        "reservoir_seed": point.reservoir_seed,
        "measurement_seed": point.measurement_seed,
        "model_selection_data": "validation only",
        "test_evaluated_after_readout_freeze": True,
        "selected_hyperparameters": {"ridge_alpha": 0.1},
        "qrc_configuration": {"n_qubits": point.n_qubits, "virtual_nodes": 2},
        "qrc_configuration_checksum": "qrc-config",
        "measurement_configuration_checksum": point.measurement_config.checksum,
        "qrc_feature_cache_key_checksum": f"cache-{point.measurement_config.checksum}",
        "qrc_feature_cache_hit": True,
        "qrc_raw_feature_dimension": 6,
        "readout_shape": [7, 3 if task == TASKS[0] else 1],
        "trainable_readout_parameters": 21 if task == TASKS[0] else 7,
        "state_generation_time": 0.1,
        "sampling_time": 0.2 if point.shot_count is not None else 0.0,
        "readout_fitting_time": 0.01,
        "backend": "numpy_density_matrix_exact_controlled_measurement",
        "exact_state_evolution": True,
        "exact_noiseless": point.study_type == "analytic_reference",
        "physical_qpu_execution": False,
        "python_version": "3.11",
        "operating_system": "test",
        "execution_platform": "local",
        "package_versions": {"numpy": "2"},
        "git": {"commit": "deadbeef", "dirty": False},
    }
    _write_json(directory / "manifest.json", manifest)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "robustness_study": {
                    "id": "qrc_shot_noise_robustness_v1",
                    "point_checksum": point.checksum,
                },
            }
        ),
        encoding="utf-8",
    )
    metrics = (
        {"macro_f1": 0.5, "transition_pr_auc": 0.6}
        if task == TASKS[0]
        else {"qlike": -2.0, "rmse": 0.1}
    )
    _write_json(directory / "validation_metrics.json", metrics)
    _fake_predictions(
        directory / "validation_predictions.csv",
        task,
        prediction_offset,
    )
    if complete:
        _write_json(directory / "test_metrics.json", metrics)
        _fake_predictions(
            directory / "test_predictions.csv",
            task,
            prediction_offset,
        )
    return directory


def test_full_grid_is_one_factor_at_a_time_and_has_analytic_references() -> None:
    config = load_robustness_study_config(_study_path())
    points = build_robustness_grid(
        n_qubits=config.selected_n_qubits,
        reservoir_seeds=config.reservoir_seeds,
        measurement_seeds=config.measurement_seeds,
        shots=config.shots,
        depolarizing_probabilities=config.depolarizing_probabilities,
        measurement_noise_probabilities=config.measurement_noise_probabilities,
    )

    assert len(points) == 111
    assert sum(point.study_type == "analytic_reference" for point in points) == 3
    assert all(point.n_qubits == 2 for point in points)
    assert all(
        point.depolarizing_probability == 0.0 and point.measurement_bit_flip_probability == 0.0
        for point in points
        if point.study_type == "finite_shot"
    )
    assert all(
        point.shot_count == 2048 and point.measurement_bit_flip_probability == 0.0
        for point in points
        if point.study_type == "depolarizing_noise"
    )
    assert all(
        point.shot_count == 2048 and point.depolarizing_probability == 0.0
        for point in points
        if point.study_type == "measurement_noise"
    )


def test_n_qubits_defaults_to_two_but_generated_configs_accept_an_override(
    tmp_path: Path,
) -> None:
    base = load_robustness_study_config(_study_path())
    config = replace(base, output_root=tmp_path / "qrc_noise_robustness")
    point = RobustnessPoint("analytic_reference", 4, 2026, None, None, 0.0, 0.0)

    classifier_path = write_robustness_model_config(config, point, TASKS[0])
    classifier = load_model_config(classifier_path)

    assert base.selected_n_qubits == 2
    assert qrc_config_from_model(classifier).n_qubits == 4
    assert classifier.output_root == config.output_root / "runs"
    assert (
        classifier.project_root / classifier.raw["qrc"]["feature_cache"]
    ).resolve() == config.output_root / "feature_cache"
    assert config.output_root.name == "qrc_noise_robustness"
    assert config.output_root.name not in {"qrc_public_pilot", "qrc_qubit_scaling"}


def test_classifier_and_regressor_configs_share_robust_feature_identity(
    tmp_path: Path,
) -> None:
    base = load_robustness_study_config(_study_path())
    config = replace(base, output_root=tmp_path / "qrc_noise_robustness")
    point = RobustnessPoint("finite_shot", 2, 2027, 1, 512, 0.0, 0.0)
    classifier = load_model_config(write_robustness_model_config(config, point, TASKS[0]))
    regressor = load_model_config(write_robustness_model_config(config, point, TASKS[1]))
    classifier_qrc = qrc_config_from_model(classifier)
    regressor_qrc = qrc_config_from_model(regressor)

    assert classifier_qrc == regressor_qrc
    assert classifier.raw["qrc"]["feature_cache"] == regressor.raw["qrc"]["feature_cache"]
    classifier_key = make_robust_feature_cache_key(
        processed_data_manifest_checksum="data",
        feature_names=("a", "b"),
        qrc_config=classifier_qrc,
        measurement_config=point.measurement_config,
    )
    regressor_key = make_robust_feature_cache_key(
        processed_data_manifest_checksum="data",
        feature_names=("a", "b"),
        qrc_config=regressor_qrc,
        measurement_config=point.measurement_config,
    )
    assert classifier_key.checksum == regressor_key.checksum


def test_robust_cache_key_is_deterministic_and_measurement_seed_sensitive() -> None:
    qrc = QRCConfig(n_qubits=2, reservoir_seed=2026)
    measurement = QRCMeasurementConfig(shots=128, measurement_seed=0)

    first = make_robust_feature_cache_key(
        processed_data_manifest_checksum="data",
        feature_names=("a",),
        qrc_config=qrc,
        measurement_config=measurement,
    )
    repeated = make_robust_feature_cache_key(
        processed_data_manifest_checksum="data",
        feature_names=("a",),
        qrc_config=qrc,
        measurement_config=measurement,
    )
    changed = make_robust_feature_cache_key(
        processed_data_manifest_checksum="data",
        feature_names=("a",),
        qrc_config=qrc,
        measurement_config=replace(measurement, measurement_seed=1),
    )

    assert first.checksum == repeated.checksum
    assert first.checksum != changed.checksum


def test_aggregation_separates_measurement_reservoir_and_all_repetitions() -> None:
    rows: list[dict[str, Any]] = []
    for reservoir_seed, baseline in ((2026, 0.4), (2027, 0.6)):
        for measurement_seed, increment in ((0, 0.0), (1, 0.1)):
            rows.append(
                {
                    "study_type": "finite_shot",
                    "analytic_reference": False,
                    "n_qubits": 2,
                    "virtual_nodes": 2,
                    "shot_count": 128,
                    "depolarizing_probability": 0.0,
                    "measurement_bit_flip_probability": 0.0,
                    "task": TASKS[0],
                    "split": "test",
                    "reservoir_seed": reservoir_seed,
                    "measurement_seed": measurement_seed,
                    "macro_f1": baseline + increment,
                    "prediction_mae_vs_analytic": increment,
                    "state_generation_seconds": 0.1,
                    "sampling_seconds": 0.2,
                    "readout_fitting_seconds": 0.01,
                    "selected_ridge_alpha": 0.1,
                    "raw_feature_dimension": 6,
                    "trainable_readout_parameters": 21,
                    "data_snapshot_id": "snapshot",
                    "data_manifest_checksum": "data",
                    "source_snapshot_manifest_checksum": "snapshot-manifest",
                    "study_configuration_checksum": "study",
                    "run_configuration_checksum": "run",
                    "qrc_configuration_checksum": "qrc",
                    "measurement_configuration_checksum": "measurement",
                    "feature_cache_key_checksum": "cache",
                    "backend": "backend",
                    "git_commit": "commit",
                    "git_dirty": False,
                    "python_version": "3.11",
                    "operating_system": "test",
                    "execution_platform": "local",
                    "package_versions": {"numpy": "2"},
                }
            )

    aggregate = aggregate_robustness_rows(rows)

    within = next(
        row
        for row in aggregate
        if row["aggregation_level"] == "measurement_seeds_within_reservoir"
        and row["reservoir_seed"] == 2026
        and row["metric"] == "macro_f1"
    )
    reservoirs = next(
        row
        for row in aggregate
        if row["aggregation_level"] == "reservoir_seeds" and row["metric"] == "macro_f1"
    )
    all_repetitions = next(
        row
        for row in aggregate
        if row["aggregation_level"] == "all_repetitions" and row["metric"] == "macro_f1"
    )
    assert within["mean"] == pytest.approx(0.45)
    assert within["standard_deviation"] == pytest.approx(0.05)
    assert reservoirs["mean"] == pytest.approx(0.55)
    assert reservoirs["repetition_count"] == 2
    assert all_repetitions["mean"] == pytest.approx(0.55)
    assert all_repetitions["repetition_count"] == 4


def test_collection_adds_prediction_stability_and_required_provenance(tmp_path: Path) -> None:
    analytic = RobustnessPoint("analytic_reference", 2, 2026, None, None, 0.0, 0.0)
    finite = RobustnessPoint("finite_shot", 2, 2026, 0, 128, 0.0, 0.0)
    run_dirs = {}
    for point, offset in ((analytic, 0.0), (finite, 0.02)):
        for task in TASKS:
            run_dirs[(point.checksum, task)] = _fake_run(
                tmp_path,
                point,
                task,
                prediction_offset=offset,
            )

    rows = collect_robustness_rows(run_dirs, (analytic, finite))

    finite_classification = next(
        row
        for row in rows
        if row["study_type"] == "finite_shot" and row["task"] == TASKS[0] and row["split"] == "test"
    )
    analytic_regression = next(
        row
        for row in rows
        if row["study_type"] == "analytic_reference"
        and row["task"] == TASKS[1]
        and row["split"] == "validation"
    )
    required = {
        "study_type",
        "n_qubits",
        "reservoir_seed",
        "measurement_seed",
        "shot_count",
        "depolarizing_probability",
        "measurement_bit_flip_probability",
        "state_generation_seconds",
        "sampling_seconds",
        "selected_ridge_alpha",
        "prediction_rmse_vs_analytic",
        "data_manifest_checksum",
        "study_configuration_checksum",
        "git_commit",
        "python_version",
        "package_versions",
    }
    assert required.issubset(finite_classification)
    assert finite_classification["prediction_rmse_vs_analytic"] > 0.0
    assert analytic_regression["prediction_rmse_vs_analytic"] == 0.0


def test_synthetic_data_is_rejected_before_robustness_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_robustness_study_config(_study_path())
    classifier = load_model_config(config.classifier_reference)
    synthetic: Any = SimpleNamespace(is_synthetic=True, data_source_type="fixture")
    monkeypatch.setattr(
        "qtyche_qrc.experiments.qrc_robustness.verify_public_snapshot",
        lambda _config: {
            "snapshot_id": "yahoo_chart_20100101_20251231_v1",
            "files": {},
        },
    )
    monkeypatch.setattr(
        "qtyche_qrc.experiments.qrc_robustness.load_model_dataset",
        lambda _path: synthetic,
    )

    with pytest.raises(SyntheticResultsError, match="non-synthetic"):
        verify_robustness_public_data(config, classifier)


def test_partial_completion_resumes_only_the_missing_readout(tmp_path: Path) -> None:
    point = RobustnessPoint("finite_shot", 2, 2026, 0, 128, 0.0, 0.0)
    classifier = _fake_run(tmp_path, point, TASKS[0])
    _fake_run(tmp_path, point, TASKS[1], complete=False)

    completed = discover_completed_robustness_runs(
        tmp_path,
        study_id="qrc_shot_noise_robustness_v1",
        snapshot_id="yahoo_chart_20100101_20251231_v1",
    )
    pending = pending_robustness_runs((point,), completed)

    assert completed == {(point.checksum, TASKS[0]): classifier}
    assert pending == ((point, TASKS[1]),)


def test_partial_completion_recomputes_a_stale_feature_cache(tmp_path: Path) -> None:
    point = RobustnessPoint("finite_shot", 2, 2026, 0, 128, 0.0, 0.0)
    directory = _fake_run(tmp_path, point, TASKS[0])
    current = f"cache-{point.measurement_config.checksum}"

    assert _robustness_run_uses_cache(directory, current)
    assert not _robustness_run_uses_cache(directory, "stale-cache")


def test_full_resource_estimate_accounts_for_cache_reuse_across_zero_noise_points() -> None:
    config = load_robustness_study_config(_study_path())
    points = build_robustness_grid(
        n_qubits=2,
        reservoir_seeds=config.reservoir_seeds,
        measurement_seeds=config.measurement_seeds,
        shots=config.shots,
        depolarizing_probabilities=config.depolarizing_probabilities,
        measurement_noise_probabilities=config.measurement_noise_probabilities,
    )

    estimate = robustness_resource_estimate(points, split_rows=3989, virtual_nodes=2)

    assert estimate["requested_experimental_points"] == 111
    assert estimate["unique_feature_cache_points"] == 93
    assert estimate["readout_task_runs"] == 222
    assert estimate["raw_feature_dimension"] == 6
    assert estimate["estimated_peak_density_matrix_bytes"] == 768
    assert estimate["sampled_bitstrings"] == 1_663_508_736
