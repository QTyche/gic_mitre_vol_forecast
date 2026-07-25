import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qtyche_qrc.experiments.model_config import load_model_config
from qtyche_qrc.experiments.qrc_run import qrc_config_from_model
from qtyche_qrc.experiments.qrc_scaling import (
    TASKS,
    ScalingPoint,
    aggregate_scaling_rows,
    build_scaling_grid,
    collect_scaling_rows,
    discover_completed_scaling_runs,
    load_scaling_study_config,
    pending_scaling_runs,
    scaling_resource_estimates,
    verify_scaling_public_data,
    write_scaling_model_config,
)
from qtyche_qrc.experiments.run import SyntheticResultsError
from qtyche_qrc.models.qrc.features import make_feature_cache_key


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fake_scaling_run(
    root: Path,
    *,
    n_qubits: int,
    seed: int,
    task: str,
    synthetic: bool = False,
    frozen_test: bool = True,
    complete: bool = True,
    metric_value: float = 0.5,
) -> Path:
    model_type = "qrc_classifier" if task == "regime_classification" else "qrc_regressor"
    directory = root / f"20260101T000000.000000Z_{model_type}_{task}_seed{seed}_q{n_qubits}"
    directory.mkdir(parents=True)
    _write_json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "experiment_id": directory.name,
            "status": "success",
            "model_type": model_type,
            "task": task,
            "reservoir_seed": seed,
            "selected_hyperparameters": {"ridge_alpha": 0.1},
            "qrc_raw_feature_dimension": 6 if n_qubits == 2 else 4 * n_qubits,
            "readout_shape": [
                7 if n_qubits == 2 else 4 * n_qubits + 1,
                3 if task == TASKS[0] else 1,
            ],
            "trainable_readout_parameters": (
                (7 if n_qubits == 2 else 4 * n_qubits + 1) * (3 if task == TASKS[0] else 1)
            ),
            "state_generation_time": float(n_qubits),
            "readout_fitting_time": 0.01,
            "qrc_feature_cache_hit": True,
            "qrc_feature_cache_key_checksum": f"cache-q{n_qubits}-seed{seed}",
            "data_snapshot_id": "yahoo_chart_20100101_20251231_v1",
            "data_manifest_checksum": "manifest-checksum",
            "data_source_type": "fixture" if synthetic else "public_market",
            "is_synthetic": synthetic,
            "backend": "numpy_density_matrix_exact",
            "exact_noiseless": True,
            "git": {"commit": "deadbeef", "dirty": False},
            "package_versions": {"numpy": "2.0"},
            "qrc_configuration": {"n_qubits": n_qubits, "virtual_nodes": 2},
            "model_selection_data": "validation only",
            "test_evaluated_after_readout_freeze": frozen_test,
        },
    )
    (directory / "config.yaml").write_text(
        "schema_version: 1\nscaling_study:\n  id: exact_qrc_qubit_scaling_v1\n",
        encoding="utf-8",
    )
    metrics = (
        {
            "macro_f1": metric_value,
            "transition_pr_auc": metric_value + 0.1,
            "balanced_accuracy": metric_value,
            "accuracy": metric_value,
            "weighted_f1": metric_value,
            "log_loss": 1.0,
            "multiclass_brier_score": 0.5,
            "transition_accuracy": 0.5,
            "transition_balanced_accuracy": 0.5,
            "transition_f1": 0.5,
            "transition_roc_auc": 0.6,
            "transition_brier_score": 0.2,
        }
        if task == "regime_classification"
        else {
            "qlike": metric_value,
            "rmse": 0.1,
            "mae": 0.08,
            "r_squared": 0.2,
            "prediction_mean": 0.01,
            "prediction_median": 0.009,
            "prediction_minimum": 0.001,
            "prediction_maximum": 0.02,
            "non_finite_prediction_count": 0,
            "floored_prediction_count": 0,
        }
    )
    _write_json(directory / "validation_metrics.json", metrics)
    if complete:
        _write_json(directory / "test_metrics.json", metrics)
    return directory


def test_scaling_grid_is_qubit_major_and_rejects_duplicates() -> None:
    grid = build_scaling_grid((6, 2, 4), (2028, 2026))

    assert grid == (
        ScalingPoint(2, 2026),
        ScalingPoint(2, 2028),
        ScalingPoint(4, 2026),
        ScalingPoint(4, 2028),
        ScalingPoint(6, 2026),
        ScalingPoint(6, 2028),
    )
    with pytest.raises(ValueError, match="duplicates"):
        build_scaling_grid((2, 2), (2026,))


def test_scaling_cache_keys_are_deterministic_and_change_only_with_qubits() -> None:
    study = load_scaling_study_config(_root() / "configs/qrc_qubit_scaling.yaml")
    classifier = load_model_config(study.classifier_reference)
    reference = qrc_config_from_model(classifier)
    two_qubits = replace(reference, n_qubits=2)
    four_qubits = replace(reference, n_qubits=4)

    first = make_feature_cache_key(
        processed_data_manifest_checksum="data",
        feature_names=("a", "b"),
        config=two_qubits,
    )
    repeated = make_feature_cache_key(
        processed_data_manifest_checksum="data",
        feature_names=("a", "b"),
        config=two_qubits,
    )
    changed = make_feature_cache_key(
        processed_data_manifest_checksum="data",
        feature_names=("a", "b"),
        config=four_qubits,
    )

    assert first.checksum == repeated.checksum
    assert first.checksum != changed.checksum


def test_classifier_and_regressor_derived_configs_share_feature_cache(
    tmp_path: Path,
) -> None:
    base = load_scaling_study_config(_root() / "configs/qrc_qubit_scaling.yaml")
    study = replace(base, output_root=tmp_path / "qrc_qubit_scaling")
    point = ScalingPoint(4, 2027)
    classifier_path = write_scaling_model_config(study, point, TASKS[0])
    regressor_path = write_scaling_model_config(study, point, TASKS[1])
    classifier = load_model_config(classifier_path)
    regressor = load_model_config(regressor_path)
    classifier_qrc = qrc_config_from_model(classifier)
    regressor_qrc = qrc_config_from_model(regressor)

    assert classifier.raw["qrc"]["feature_cache"] == regressor.raw["qrc"]["feature_cache"]
    assert (
        classifier.project_root / classifier.raw["qrc"]["feature_cache"]
    ).resolve() == study.output_root / "feature_cache"
    assert classifier_qrc == regressor_qrc
    classifier_key = make_feature_cache_key(
        processed_data_manifest_checksum="data",
        feature_names=("a",),
        config=classifier_qrc,
    )
    regressor_key = make_feature_cache_key(
        processed_data_manifest_checksum="data",
        feature_names=("a",),
        config=regressor_qrc,
    )
    assert classifier_key.checksum == regressor_key.checksum

    reference_classifier = load_model_config(study.classifier_reference)
    reference = qrc_config_from_model(reference_classifier)
    differences = {
        name for name, value in asdict(classifier_qrc).items() if value != asdict(reference)[name]
    }
    assert differences == {"n_qubits", "reservoir_seed"}
    assert classifier.raw["data"] == reference_classifier.raw["data"]
    assert classifier.raw["search"] == reference_classifier.raw["search"]
    assert classifier.raw["evaluation"] == reference_classifier.raw["evaluation"]


def test_scaling_aggregation_groups_each_qubit_count_over_seeds(tmp_path: Path) -> None:
    run_dirs = {}
    for n_qubits in (2, 4):
        for seed, value in ((2026, 0.4), (2027, 0.6)):
            for task in TASKS:
                identity = (n_qubits, seed, task)
                run_dirs[identity] = _fake_scaling_run(
                    tmp_path,
                    n_qubits=n_qubits,
                    seed=seed,
                    task=task,
                    metric_value=value,
                )
    rows = collect_scaling_rows(run_dirs)
    aggregate = aggregate_scaling_rows(rows)

    macro = next(
        row
        for row in aggregate
        if row["n_qubits"] == 2
        and row["task"] == "regime_classification"
        and row["split"] == "test"
        and row["metric"] == "macro_f1"
    )
    assert macro["mean"] == pytest.approx(0.5)
    assert macro["standard_deviation"] == pytest.approx(0.1)
    assert macro["minimum"] == pytest.approx(0.4)
    assert macro["maximum"] == pytest.approx(0.6)
    assert macro["reservoir_seeds"] == [2026, 2027]


def test_scaling_rows_include_required_provenance_and_temporal_fields(
    tmp_path: Path,
) -> None:
    identity = (2, 2026, "regime_classification")
    directory = _fake_scaling_run(
        tmp_path,
        n_qubits=2,
        seed=2026,
        task="regime_classification",
    )

    rows = collect_scaling_rows({identity: directory})

    required = {
        "n_qubits",
        "virtual_nodes",
        "reservoir_seed",
        "task",
        "split",
        "selected_ridge_alpha",
        "raw_feature_dimension",
        "readout_shape",
        "trainable_readout_parameters",
        "state_generation_seconds",
        "readout_fitting_seconds",
        "cache_hit",
        "cache_key_checksum",
        "data_snapshot_id",
        "backend",
        "exact_noiseless",
        "git_commit",
        "package_versions",
        "model_selection_data",
        "test_evaluated_after_readout_freeze",
    }
    assert required.issubset(rows[0])
    assert rows[0]["model_selection_data"] == "validation only"
    assert rows[0]["test_evaluated_after_readout_freeze"] is True


def test_scaling_discovery_rejects_synthetic_runs(tmp_path: Path) -> None:
    _fake_scaling_run(
        tmp_path,
        n_qubits=2,
        seed=2026,
        task="regime_classification",
        synthetic=True,
    )

    with pytest.raises(SyntheticResultsError, match="synthetic"):
        discover_completed_scaling_runs(
            tmp_path,
            study_id="exact_qrc_qubit_scaling_v1",
            snapshot_id="yahoo_chart_20100101_20251231_v1",
        )


def test_scaling_public_data_guard_rejects_synthetic_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = load_scaling_study_config(_root() / "configs/qrc_qubit_scaling.yaml")
    classifier = load_model_config(study.classifier_reference)
    fake_dataset: Any = SimpleNamespace(is_synthetic=True, data_source_type="fixture")
    monkeypatch.setattr(
        "qtyche_qrc.experiments.qrc_scaling.verify_public_snapshot",
        lambda _config: {
            "snapshot_id": "yahoo_chart_20100101_20251231_v1",
            "files": {},
        },
    )
    monkeypatch.setattr(
        "qtyche_qrc.experiments.qrc_scaling.load_model_dataset",
        lambda _processed_dir: fake_dataset,
    )

    with pytest.raises(SyntheticResultsError, match="non-synthetic"):
        verify_scaling_public_data(study, classifier)


def test_scaling_rows_reject_non_temporal_test_evaluation(tmp_path: Path) -> None:
    identity = (2, 2026, "regime_classification")
    directory = _fake_scaling_run(
        tmp_path,
        n_qubits=2,
        seed=2026,
        task="regime_classification",
        frozen_test=False,
    )

    with pytest.raises(ValueError, match="frozen-readout"):
        collect_scaling_rows({identity: directory})


def test_partial_scaling_completion_resumes_only_missing_task(tmp_path: Path) -> None:
    classifier = _fake_scaling_run(
        tmp_path,
        n_qubits=4,
        seed=2026,
        task="regime_classification",
    )
    _fake_scaling_run(
        tmp_path,
        n_qubits=4,
        seed=2026,
        task="rv_regression",
        complete=False,
    )

    completed = discover_completed_scaling_runs(
        tmp_path,
        study_id="exact_qrc_qubit_scaling_v1",
        snapshot_id="yahoo_chart_20100101_20251231_v1",
    )
    pending = pending_scaling_runs((ScalingPoint(4, 2026),), completed)

    assert completed == {(4, 2026, "regime_classification"): classifier}
    assert pending == ((ScalingPoint(4, 2026), "rv_regression"),)


def test_scaling_resource_estimates_cover_two_to_six_qubits() -> None:
    rows = scaling_resource_estimates(
        (2, 3, 4, 5, 6),
        virtual_nodes=2,
        split_rows=3989,
    )

    assert [row["raw_feature_dimension"] for row in rows] == [6, 12, 16, 20, 24]
    assert [row["estimated_peak_density_matrix_bytes"] for row in rows] == [
        768,
        3072,
        12288,
        49152,
        196608,
    ]
