import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qtyche_qrc.data.download import sha256_file
from qtyche_qrc.experiments.model_config import load_model_config
from qtyche_qrc.experiments.qrc_encoding_density import (
    STUDY_ID,
    TASKS,
    EncodingDensityPoint,
    aggregate_encoding_density_rows,
    build_encoding_density_grid,
    build_validation_candidate_table,
    collect_encoding_density_rows,
    compare_v2_with_qubit_scaling_reference,
    discover_completed_encoding_density_runs,
    encoding_density_resource_estimates,
    expected_raw_feature_dimension,
    load_encoding_density_study_config,
    pending_encoding_density_runs,
    temporal_sampling_times,
    verify_encoding_density_public_data,
    write_encoding_density_model_config,
)
from qtyche_qrc.experiments.qrc_run import qrc_config_from_model
from qtyche_qrc.experiments.qrc_scaling import (
    ScalingPoint,
    load_scaling_study_config,
    write_scaling_model_config,
)
from qtyche_qrc.experiments.run import SyntheticResultsError
from qtyche_qrc.models.qrc.features import make_feature_cache_key
from qtyche_qrc.models.qrc.reservoir import QuantumReservoir


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _point_metadata(virtual_nodes: int, seed: int) -> dict[str, Any]:
    raw_dimension = 3 * virtual_nodes
    return {
        "state_generation_seconds": float(virtual_nodes),
        "feature_preparation_wall_seconds": 0.01,
        "array_checksums": {"train": "train", "validation": "validation", "test": "test"},
        "study_configuration_checksum": "study-config",
        "condition_diagnostics": {
            "effective_rank": float(raw_dimension - 1),
            "numerical_rank": raw_dimension,
            "largest_singular_value": 10.0,
            "smallest_retained_singular_value": 0.1,
            "condition_number": 100.0,
            "rank_tolerance": 1e-12,
        },
        "numerical_diagnostics": {"maximum_trace_error": 1e-15},
        "v2_reference_agreement": None,
        "reservoir_seed": seed,
    }


def _fake_run(
    root: Path,
    *,
    virtual_nodes: int,
    seed: int,
    task: str,
    synthetic: bool = False,
    frozen_test: bool = True,
    complete: bool = True,
    metric_value: float = 0.5,
) -> Path:
    model_type = "qrc_classifier" if task == TASKS[0] else "qrc_regressor"
    directory = root / f"20260101T000000.000000Z_{model_type}_{task}_seed{seed}_v{virtual_nodes}"
    directory.mkdir(parents=True)
    raw_dimension = expected_raw_feature_dimension(2, virtual_nodes)
    outputs = 3 if task == TASKS[0] else 1
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
            "qrc_raw_feature_dimension": raw_dimension,
            "readout_shape": [raw_dimension + 1, outputs],
            "trainable_readout_parameters": (raw_dimension + 1) * outputs,
            "state_generation_time": float(virtual_nodes),
            "readout_fitting_time": 0.02,
            "qrc_feature_cache_hit": True,
            "qrc_feature_cache_key_checksum": f"cache-v{virtual_nodes}-seed{seed}",
            "configuration_checksum": "run-config",
            "processed_data_checksums": {"data_manifest.json": "manifest-checksum"},
            "data_snapshot_id": "yahoo_chart_20100101_20251231_v1",
            "data_manifest_checksum": "manifest-checksum",
            "data_source_type": "fixture" if synthetic else "public_market",
            "is_synthetic": synthetic,
            "backend": "numpy_density_matrix_exact",
            "exact_noiseless": True,
            "git": {"commit": "deadbeef", "dirty": False},
            "python_version": "3.11",
            "operating_system": "test-os",
            "execution_platform": "local",
            "package_versions": {"numpy": "2.0"},
            "qrc_configuration": {
                "n_qubits": 2,
                "virtual_nodes": virtual_nodes,
                "graph": "ring",
                "j_strength": 1.0,
                "h_strength": 1.0,
                "h_min_factor": 0.5,
                "h_max_factor": 1.5,
                "tau": 1.0,
                "input_scaling": 0.5,
                "state_policy": "carry_inputs",
                "reservoir_seed": seed,
                "backend": "numpy_density_matrix_exact",
                "chords": [],
            },
            "model_selection_data": "validation only",
            "test_evaluated_after_readout_freeze": frozen_test,
        },
    )
    (directory / "config.yaml").write_text(
        f"schema_version: 1\nencoding_density_study:\n  id: {STUDY_ID}\n",
        encoding="utf-8",
    )
    _write_json(
        directory / "timing.json",
        {
            "state_generation_seconds": float(virtual_nodes),
            "readout_fitting_seconds": 0.02,
            "validation_prediction_seconds": 0.001,
            "test_prediction_seconds": 0.001,
        },
    )
    metrics = (
        {
            "accuracy": metric_value,
            "balanced_accuracy": metric_value,
            "macro_f1": metric_value,
            "weighted_f1": metric_value,
            "log_loss": 1.0,
            "multiclass_brier_score": 0.5,
            "transition_accuracy": 0.5,
            "transition_balanced_accuracy": 0.5,
            "transition_f1": 0.5,
            "transition_roc_auc": 0.6,
            "transition_pr_auc": metric_value + 0.1,
            "transition_brier_score": 0.2,
        }
        if task == TASKS[0]
        else {
            "rmse": 0.1,
            "mae": 0.08,
            "qlike": metric_value,
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


def test_grid_and_expected_feature_dimensions() -> None:
    grid = build_encoding_density_grid((8, 1, 4, 2), (2028, 2026))

    assert grid[0] == EncodingDensityPoint(1, 2026)
    assert grid[-1] == EncodingDensityPoint(8, 2028)
    assert [expected_raw_feature_dimension(2, value) for value in (1, 2, 4, 8)] == [
        3,
        6,
        12,
        24,
    ]
    with pytest.raises(ValueError, match="duplicates"):
        build_encoding_density_grid((1, 1), (2026,))


def test_total_interval_is_fixed_and_sampling_times_are_equal() -> None:
    study = load_encoding_density_study_config(_root() / "configs/qrc_encoding_density.yaml")
    tau = float(study.fixed_qrc["tau"])

    for virtual_nodes in (1, 2, 4, 8):
        point = EncodingDensityPoint(virtual_nodes, 2026)
        generated = write_encoding_density_model_config(study, point, TASKS[0])
        qrc = qrc_config_from_model(load_model_config(generated))
        times = temporal_sampling_times(qrc.tau, virtual_nodes)

        assert qrc.delta_tau == pytest.approx(tau / virtual_nodes)
        assert qrc.delta_tau * virtual_nodes == pytest.approx(tau)
        assert times[-1] == pytest.approx(tau)
        assert np.diff((0.0, *times)) == pytest.approx(np.full(virtual_nodes, tau / virtual_nodes))


def test_features_are_seed_deterministic_and_change_between_seeds() -> None:
    study = load_encoding_density_study_config(_root() / "configs/qrc_encoding_density.yaml")
    reference = qrc_config_from_model(load_model_config(study.classifier_reference))
    inputs = np.asarray([[0.1, -0.3], [0.2, 0.4], [-0.5, 0.6], [0.7, -0.2]], dtype=float)
    first_config = replace(reference, n_qubits=2, virtual_nodes=4, reservoir_seed=2026)
    second_config = replace(first_config, reservoir_seed=2027)

    first = QuantumReservoir(2, first_config).transform(inputs, reset=True)
    repeated = QuantumReservoir(2, first_config).transform(inputs, reset=True)
    changed = QuantumReservoir(2, second_config).transform(inputs, reset=True)

    assert np.array_equal(first, repeated)
    assert not np.allclose(first, changed)


def test_cache_keys_are_deterministic_and_change_with_virtual_nodes() -> None:
    study = load_encoding_density_study_config(_root() / "configs/qrc_encoding_density.yaml")
    reference = qrc_config_from_model(load_model_config(study.classifier_reference))
    v1 = replace(reference, n_qubits=2, virtual_nodes=1)
    v2 = replace(reference, n_qubits=2, virtual_nodes=2)

    first = make_feature_cache_key(
        processed_data_manifest_checksum="data",
        feature_names=("a", "b"),
        config=v1,
    )
    repeated = make_feature_cache_key(
        processed_data_manifest_checksum="data",
        feature_names=("a", "b"),
        config=v1,
    )
    changed = make_feature_cache_key(
        processed_data_manifest_checksum="data",
        feature_names=("a", "b"),
        config=v2,
    )

    assert first.checksum == repeated.checksum
    assert first.checksum != changed.checksum


def test_classifier_and_regressor_share_isolated_feature_cache(tmp_path: Path) -> None:
    base = load_encoding_density_study_config(_root() / "configs/qrc_encoding_density.yaml")
    study = replace(base, output_root=tmp_path / "qrc_encoding_density")
    point = EncodingDensityPoint(4, 2027)
    classifier = load_model_config(write_encoding_density_model_config(study, point, TASKS[0]))
    regressor = load_model_config(write_encoding_density_model_config(study, point, TASKS[1]))
    classifier_qrc = qrc_config_from_model(classifier)
    regressor_qrc = qrc_config_from_model(regressor)

    assert classifier_qrc == regressor_qrc
    assert classifier.raw["qrc"]["feature_cache"] == regressor.raw["qrc"]["feature_cache"]
    assert (
        classifier.project_root / classifier.raw["qrc"]["feature_cache"]
    ).resolve() == study.output_root / "feature_cache"
    assert classifier.output_root == regressor.output_root == study.output_root / "runs"
    reference = load_model_config(study.classifier_reference)
    differences = {
        name
        for name, value in asdict(classifier_qrc).items()
        if value != asdict(qrc_config_from_model(reference))[name]
    }
    assert differences == {"n_qubits", "virtual_nodes", "reservoir_seed"}
    assert classifier.raw["data"] == reference.raw["data"]
    assert classifier.raw["search"] == reference.raw["search"]
    assert classifier.raw["evaluation"] == reference.raw["evaluation"]


def test_v2_numerically_matches_existing_two_qubit_exact_configuration(
    tmp_path: Path,
) -> None:
    encoding_base = load_encoding_density_study_config(
        _root() / "configs/qrc_encoding_density.yaml"
    )
    scaling_base = load_scaling_study_config(_root() / "configs/qrc_qubit_scaling.yaml")
    encoding = replace(encoding_base, output_root=tmp_path / "encoding")
    scaling = replace(scaling_base, output_root=tmp_path / "scaling")
    encoding_qrc = qrc_config_from_model(
        load_model_config(
            write_encoding_density_model_config(encoding, EncodingDensityPoint(2, 2026), TASKS[0])
        )
    )
    scaling_qrc = qrc_config_from_model(
        load_model_config(
            write_scaling_model_config(scaling, ScalingPoint(2, 2026), "regime_classification")
        )
    )
    inputs = np.asarray([[0.2, -0.1], [0.5, 0.4], [-0.3, 0.7]], dtype=float)

    encoding_features = QuantumReservoir(2, encoding_qrc).transform(inputs, reset=True)
    scaling_features = QuantumReservoir(2, scaling_qrc).transform(inputs, reset=True)

    assert encoding_qrc == scaling_qrc
    assert np.array_equal(encoding_features, scaling_features)


def test_v2_reference_cache_comparison_is_array_level(tmp_path: Path) -> None:
    checksum = "cache-checksum"
    reference = tmp_path / "results/qrc_qubit_scaling/feature_cache" / checksum
    encoding = tmp_path / "results/qrc_encoding_density/feature_cache" / checksum
    reference.mkdir(parents=True)
    encoding.mkdir(parents=True)
    arrays = {
        "train": np.arange(12, dtype=float).reshape(2, 6),
        "validation": np.arange(6, dtype=float).reshape(1, 6),
        "test": np.ones((1, 6), dtype=float),
    }
    np.savez_compressed(reference / "qrc_features.npz", **arrays)
    np.savez_compressed(encoding / "qrc_features.npz", **arrays)

    result = compare_v2_with_qubit_scaling_reference(
        project_root=tmp_path,
        cache_key_checksum=checksum,
        encoding_cache_dir=encoding,
    )

    assert result["compared"] is True
    assert result["within_tolerance"] is True
    assert result["maximum_absolute_difference"] == 0.0
    assert result["array_checksums_match"] is True


def test_aggregation_and_validation_candidate_table_use_seeds(tmp_path: Path) -> None:
    run_dirs = {}
    metadata = {}
    for virtual_nodes in (1, 2):
        for seed, value in ((2026, 0.4), (2027, 0.6)):
            metadata[(virtual_nodes, seed)] = _point_metadata(virtual_nodes, seed)
            for task in TASKS:
                identity = (virtual_nodes, seed, task)
                run_dirs[identity] = _fake_run(
                    tmp_path,
                    virtual_nodes=virtual_nodes,
                    seed=seed,
                    task=task,
                    metric_value=value,
                )
    rows = collect_encoding_density_rows(run_dirs, point_metadata=metadata)
    aggregate = aggregate_encoding_density_rows(rows)
    candidates = build_validation_candidate_table(rows)

    macro = next(
        row
        for row in aggregate
        if row["virtual_nodes"] == 1
        and row["task"] == TASKS[0]
        and row["split"] == "validation"
        and row["metric"] == "macro_f1"
    )
    assert macro["mean"] == pytest.approx(0.5)
    assert macro["standard_deviation"] == pytest.approx(0.1)
    assert macro["minimum"] == pytest.approx(0.4)
    assert macro["maximum"] == pytest.approx(0.6)
    assert macro["seed_count"] == 2
    assert candidates[0]["validation_macro_f1_mean"] == pytest.approx(0.5)
    assert candidates[0]["validation_qlike_mean"] == pytest.approx(0.5)
    assert candidates[0]["selection_basis"] == "validation only"
    assert candidates[0]["test_metrics_used"] is False
    assert candidates[0]["architecture_frozen"] is False


def test_rows_include_required_provenance_and_temporal_fields(tmp_path: Path) -> None:
    identity = (2, 2026, TASKS[0])
    directory = _fake_run(
        tmp_path,
        virtual_nodes=2,
        seed=2026,
        task=TASKS[0],
    )

    rows = collect_encoding_density_rows(
        {identity: directory},
        point_metadata={(2, 2026): _point_metadata(2, 2026)},
    )

    required = {
        "virtual_nodes",
        "n_qubits",
        "reservoir_seed",
        "raw_feature_dimension",
        "task",
        "split",
        "selected_ridge_alpha",
        "state_generation_seconds",
        "feature_generation_seconds",
        "readout_fitting_seconds",
        "total_runtime_seconds",
        "condition_number",
        "cache_key_checksum",
        "cache_array_checksums",
        "study_configuration_checksum",
        "run_configuration_checksum",
        "data_snapshot_id",
        "data_manifest_checksum",
        "git_commit",
        "python_version",
        "platform",
        "package_versions",
        "model_selection_data",
        "test_evaluated_after_readout_freeze",
        "total_evolution_time_per_input",
        "substep_evolution_time",
        "sampling_times",
    }
    assert required.issubset(rows[0])
    assert rows[0]["model_selection_data"] == "validation only"
    assert rows[0]["test_evaluated_after_readout_freeze"] is True
    assert rows[0]["total_evolution_time_per_input"] == 1.0


def test_discovery_rejects_synthetic_and_non_temporal_runs(tmp_path: Path) -> None:
    _fake_run(
        tmp_path,
        virtual_nodes=2,
        seed=2026,
        task=TASKS[0],
        synthetic=True,
    )
    with pytest.raises(SyntheticResultsError, match="synthetic"):
        discover_completed_encoding_density_runs(
            tmp_path,
            study_id=STUDY_ID,
            snapshot_id="yahoo_chart_20100101_20251231_v1",
        )

    clean_root = tmp_path / "non_temporal"
    identity = (2, 2026, TASKS[0])
    directory = _fake_run(
        clean_root,
        virtual_nodes=2,
        seed=2026,
        task=TASKS[0],
        frozen_test=False,
    )
    with pytest.raises(ValueError, match="frozen-readout"):
        collect_encoding_density_rows(
            {identity: directory},
            point_metadata={(2, 2026): _point_metadata(2, 2026)},
        )


def test_public_data_guard_rejects_synthetic_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = load_encoding_density_study_config(_root() / "configs/qrc_encoding_density.yaml")
    classifier = load_model_config(study.classifier_reference)
    fake_dataset: Any = SimpleNamespace(is_synthetic=True, data_source_type="fixture")
    monkeypatch.setattr(
        "qtyche_qrc.experiments.qrc_encoding_density.verify_public_snapshot",
        lambda _config: {
            "snapshot_id": "yahoo_chart_20100101_20251231_v1",
            "files": {},
        },
    )
    monkeypatch.setattr(
        "qtyche_qrc.experiments.qrc_encoding_density.load_model_dataset",
        lambda _processed_dir: fake_dataset,
    )

    with pytest.raises(SyntheticResultsError, match="non-synthetic"):
        verify_encoding_density_public_data(study, classifier)


def test_partial_completion_resumes_only_missing_task(tmp_path: Path) -> None:
    classifier = _fake_run(
        tmp_path,
        virtual_nodes=4,
        seed=2026,
        task=TASKS[0],
    )
    _fake_run(
        tmp_path,
        virtual_nodes=4,
        seed=2026,
        task=TASKS[1],
        complete=False,
    )

    completed = discover_completed_encoding_density_runs(
        tmp_path,
        study_id=STUDY_ID,
        snapshot_id="yahoo_chart_20100101_20251231_v1",
    )
    pending = pending_encoding_density_runs((EncodingDensityPoint(4, 2026),), completed)

    assert completed == {(4, 2026, TASKS[0]): classifier}
    assert pending == ((EncodingDensityPoint(4, 2026), TASKS[1]),)


def test_output_namespace_and_existing_contracts_remain_isolated() -> None:
    root = _root()
    study = load_encoding_density_study_config(root / "configs/qrc_encoding_density.yaml")

    assert study.output_root == root / "results/qrc_encoding_density"
    assert study.output_root not in {
        root / "results/qrc_public_pilot",
        root / "results/qrc_qubit_scaling",
        root / "results/qrc_noise_robustness",
    }
    assert sha256_file(root / "configs/models/qrc_classifier_pilot.yaml") == (
        "65a8b098aa466b2ab0b336cb8f5f6cabe412c43a2e8b27673d427304c24e0002"
    )
    assert sha256_file(root / "configs/models/qrc_regressor_pilot.yaml") == (
        "3a3c1d75521c6e0ac027686fade31d319c16108a699efbde1eeb147bc4158bb3"
    )
    assert all(
        not str(path).startswith(str(study.output_root))
        for path in (
            root / "results/qrc_public_pilot",
            root / "results/qrc_qubit_scaling",
            root / "results/qrc_noise_robustness",
        )
    )


def test_resource_estimates_scale_features_but_not_exact_state_memory() -> None:
    rows = encoding_density_resource_estimates(
        (1, 2, 4, 8),
        split_rows=3989,
        train_rows=2500,
    )

    assert [row["raw_feature_dimension"] for row in rows] == [3, 6, 12, 24]
    assert {row["estimated_peak_density_matrix_bytes"] for row in rows} == {768}
    assert [row["estimated_cached_feature_bytes"] for row in rows] == [
        95_736,
        191_472,
        382_944,
        765_888,
    ]
