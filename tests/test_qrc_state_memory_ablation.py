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
    EncodingDensityPoint,
    load_encoding_density_study_config,
    write_encoding_density_model_config,
)
from qtyche_qrc.experiments.qrc_run import qrc_config_from_model
from qtyche_qrc.experiments.qrc_state_memory import (
    STUDY_ID,
    TASKS,
    StateMemoryPoint,
    aggregate_state_memory_rows,
    build_state_memory_grid,
    collect_state_memory_rows,
    compare_carry_with_encoding_density_reference,
    discover_completed_state_memory_runs,
    lagged_input_feature_correlations,
    load_state_memory_study_config,
    paired_state_policy_differences,
    pending_state_memory_runs,
    perturbation_decay_diagnostics,
    select_state_policy_from_validation,
    state_memory_resource_estimates,
    verify_state_memory_public_data,
    write_state_memory_model_config,
)
from qtyche_qrc.experiments.run import SyntheticResultsError
from qtyche_qrc.models.qrc.features import make_feature_cache_key
from qtyche_qrc.models.qrc.reservoir import (
    QRCConfig,
    QuantumReservoir,
    split_qrc_features,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _point_metadata(policy: str, seed: int) -> dict[str, Any]:
    return {
        "feature_generation_seconds": 0.2,
        "feature_preparation_wall_seconds": 0.01,
        "array_checksums": {"train": "train", "validation": "validation", "test": "test"},
        "study_configuration_checksum": "study-config",
        "condition_diagnostics": {
            "effective_rank": 5.0,
            "numerical_rank": 6,
            "largest_singular_value": 10.0,
            "smallest_retained_singular_value": 0.5,
            "condition_number": 20.0,
            "rank_tolerance": 1e-12,
        },
        "feature_autocorrelation": [
            {
                "lag": 1,
                "mean_absolute_autocorrelation": 0.4,
                "nonconstant_feature_count": 6,
            }
        ],
        "lagged_input_feature_correlations": [
            {
                "lag": 1,
                "mean_absolute_correlation": 0.2,
                "maximum_absolute_correlation": 0.5,
                "correlation_pair_count": 12,
            }
        ],
        "perturbation_summary": {
            "final_trace_distance": 0.0 if policy == "reset_each_input" else 0.1,
            "trace_distance_auc": 0.0 if policy == "reset_each_input" else 1.0,
        },
        "numerical_diagnostics": {"maximum_trace_error": 1e-15},
        "carry_reference_agreement": None,
        "reservoir_seed": seed,
    }


def _fake_run(
    root: Path,
    *,
    policy: str,
    seed: int,
    task: str,
    synthetic: bool = False,
    frozen_test: bool = True,
    complete: bool = True,
    metric_value: float = 0.5,
) -> Path:
    model_type = "qrc_classifier" if task == TASKS[0] else "qrc_regressor"
    directory = root / f"20260101T000000.000000Z_{model_type}_{task}_{policy}_seed{seed}"
    directory.mkdir(parents=True)
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
            "qrc_raw_feature_dimension": 6,
            "readout_shape": [7, outputs],
            "trainable_readout_parameters": 7 * outputs,
            "state_generation_time": 0.2,
            "readout_fitting_time": 0.02,
            "qrc_feature_cache_hit": True,
            "qrc_feature_cache_key_checksum": f"cache-{policy}-seed{seed}",
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
                "virtual_nodes": 2,
                "graph": "ring",
                "j_strength": 1.0,
                "h_strength": 1.0,
                "h_min_factor": 0.5,
                "h_max_factor": 1.5,
                "tau": 1.0,
                "input_scaling": 0.5,
                "state_policy": policy,
                "reservoir_seed": seed,
                "backend": "numpy_density_matrix_exact",
                "chords": [],
            },
            "model_selection_data": "validation only",
            "test_evaluated_after_readout_freeze": frozen_test,
            "qrc_features_generated_without_labels": True,
        },
    )
    (directory / "config.yaml").write_text(
        f"schema_version: 1\nstate_memory_study:\n  id: {STUDY_ID}\n",
        encoding="utf-8",
    )
    _write_json(
        directory / "timing.json",
        {
            "state_generation_seconds": 0.2,
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
            "rmse": metric_value + 0.2,
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


def _small_qrc_config(policy: str, seed: int = 2026) -> QRCConfig:
    return QRCConfig(
        n_qubits=2,
        virtual_nodes=2,
        state_policy=policy,
        reservoir_seed=seed,
    )


def test_full_reservoir_is_reset_before_every_input() -> None:
    inputs = np.asarray([[0.1, -0.3], [0.4, 0.2], [-0.2, 0.5]], dtype=float)
    config = _small_qrc_config("reset_each_input")
    reservoir = QuantumReservoir(2, config)

    reset_features = reservoir.transform(inputs, reset_each_input=True)
    independent = np.vstack(
        [QuantumReservoir(2, config).transform(row[None, :], reset=True)[0] for row in inputs]
    )
    final_reference = QuantumReservoir(2, config)
    final_reference.step(inputs[-1])

    assert np.array_equal(reset_features, independent)
    assert np.array_equal(reservoir.get_state(), final_reference.get_state())


def test_carry_inputs_persists_state_between_observations() -> None:
    inputs = np.asarray([[0.1, -0.3], [0.4, 0.2]], dtype=float)
    config = _small_qrc_config("carry_inputs")
    carried = QuantumReservoir(2, config).transform(inputs, reset=True)
    fresh_second = QuantumReservoir(2, config).transform(inputs[1:], reset=True)

    assert not np.allclose(carried[1], fresh_second[0])


def test_both_policies_treat_the_first_input_identically() -> None:
    inputs = np.asarray([[0.1, -0.3], [0.4, 0.2]], dtype=float)
    carry = QuantumReservoir(2, _small_qrc_config("carry_inputs"))
    reset = QuantumReservoir(2, _small_qrc_config("reset_each_input"))

    carry_features, _, _ = split_qrc_features(carry, inputs, inputs[:1], None, "carry_inputs")
    reset_features, _, _ = split_qrc_features(reset, inputs, inputs[:1], None, "reset_each_input")

    assert np.array_equal(carry_features[0], reset_features[0])
    assert not np.allclose(carry_features[1], reset_features[1])


def test_state_policy_features_are_deterministic_and_distinct() -> None:
    rng = np.random.default_rng(4)
    inputs = rng.normal(size=(8, 3))
    carry_config = _small_qrc_config("carry_inputs")
    reset_config = _small_qrc_config("reset_each_input")

    first = QuantumReservoir(3, carry_config).transform(inputs, reset=True)
    repeated = QuantumReservoir(3, carry_config).transform(inputs, reset=True)
    reset = QuantumReservoir(3, reset_config).transform(inputs, reset_each_input=True)

    assert np.array_equal(first, repeated)
    assert not np.allclose(first[1:], reset[1:])


def test_grid_is_policy_major_and_validates_inputs() -> None:
    grid = build_state_memory_grid(("reset_each_input", "carry_inputs"), (2028, 2026))

    assert grid == (
        StateMemoryPoint("carry_inputs", 2026),
        StateMemoryPoint("carry_inputs", 2028),
        StateMemoryPoint("reset_each_input", 2026),
        StateMemoryPoint("reset_each_input", 2028),
    )
    with pytest.raises(ValueError, match="duplicates"):
        build_state_memory_grid(("carry_inputs", "carry_inputs"), (2026,))


def test_selected_v2_evidence_and_fixed_contract_are_verified() -> None:
    study = load_state_memory_study_config(_root() / "configs/qrc_state_memory_ablation.yaml")

    assert study.selected_virtual_nodes == 2
    assert study.fixed_qrc["virtual_nodes"] == 2
    assert study.state_policies == ("carry_inputs", "reset_each_input")


def test_classifier_and_regressor_share_isolated_policy_cache(
    tmp_path: Path,
) -> None:
    base = load_state_memory_study_config(_root() / "configs/qrc_state_memory_ablation.yaml")
    study = replace(base, output_root=tmp_path / "qrc_state_memory_ablation")
    point = StateMemoryPoint("reset_each_input", 2027)
    classifier = load_model_config(write_state_memory_model_config(study, point, TASKS[0]))
    regressor = load_model_config(write_state_memory_model_config(study, point, TASKS[1]))
    classifier_qrc = qrc_config_from_model(classifier)
    regressor_qrc = qrc_config_from_model(regressor)

    assert classifier_qrc == regressor_qrc
    assert classifier.raw["qrc"]["feature_cache"] == regressor.raw["qrc"]["feature_cache"]
    assert (
        classifier.project_root / classifier.raw["qrc"]["feature_cache"]
    ).resolve() == study.output_root / "feature_cache"
    reference = load_model_config(study.classifier_reference)
    differences = {
        name
        for name, value in asdict(classifier_qrc).items()
        if value != asdict(qrc_config_from_model(reference))[name]
    }
    assert differences == {"n_qubits", "reservoir_seed", "state_policy"}
    assert classifier.raw["data"] == reference.raw["data"]
    assert classifier.raw["search"] == reference.raw["search"]
    assert classifier.raw["evaluation"] == reference.raw["evaluation"]

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


def test_carry_inputs_matches_selected_encoding_density_reference(
    tmp_path: Path,
) -> None:
    memory_base = load_state_memory_study_config(_root() / "configs/qrc_state_memory_ablation.yaml")
    encoding_base = load_encoding_density_study_config(
        _root() / "configs/qrc_encoding_density.yaml"
    )
    memory = replace(memory_base, output_root=tmp_path / "memory")
    encoding = replace(encoding_base, output_root=tmp_path / "encoding")
    memory_qrc = qrc_config_from_model(
        load_model_config(
            write_state_memory_model_config(
                memory, StateMemoryPoint("carry_inputs", 2026), TASKS[0]
            )
        )
    )
    encoding_qrc = qrc_config_from_model(
        load_model_config(
            write_encoding_density_model_config(encoding, EncodingDensityPoint(2, 2026), TASKS[0])
        )
    )
    inputs = np.asarray([[0.2, -0.1], [0.5, 0.4], [-0.3, 0.7]], dtype=float)

    memory_features = QuantumReservoir(2, memory_qrc).transform(inputs, reset=True)
    reference_features = QuantumReservoir(2, encoding_qrc).transform(inputs, reset=True)

    assert memory_qrc == encoding_qrc
    assert np.array_equal(memory_features, reference_features)


def test_carry_reference_cache_comparison_is_array_level(tmp_path: Path) -> None:
    checksum = "cache-checksum"
    reference = tmp_path / "results/qrc_encoding_density/feature_cache" / checksum
    memory = tmp_path / "results/qrc_state_memory_ablation/feature_cache" / checksum
    reference.mkdir(parents=True)
    memory.mkdir(parents=True)
    arrays = {
        "train": np.arange(12, dtype=float).reshape(2, 6),
        "validation": np.arange(6, dtype=float).reshape(1, 6),
        "test": np.ones((1, 6), dtype=float),
    }
    np.savez_compressed(reference / "qrc_features.npz", **arrays)
    np.savez_compressed(memory / "qrc_features.npz", **arrays)

    result = compare_carry_with_encoding_density_reference(
        project_root=tmp_path,
        cache_key_checksum=checksum,
        state_memory_cache_dir=memory,
    )

    assert result["compared"] is True
    assert result["within_tolerance"] is True
    assert result["maximum_absolute_difference"] == 0.0
    assert result["array_checksums_match"] is True


def test_memory_diagnostics_distinguish_reset_perturbation() -> None:
    rng = np.random.default_rng(8)
    inputs = rng.normal(size=(12, 2))
    carry_curve, carry = perturbation_decay_diagnostics(
        _small_qrc_config("carry_inputs"), inputs, steps=8
    )
    reset_curve, reset = perturbation_decay_diagnostics(
        _small_qrc_config("reset_each_input"), inputs, steps=8
    )
    features = rng.normal(size=(12, 6))
    lagged = lagged_input_feature_correlations(features, inputs, maximum_lag=3)

    assert carry_curve[0]["trace_distance"] == pytest.approx(1.0)
    assert reset_curve[1]["trace_distance"] == pytest.approx(0.0, abs=1e-12)
    assert reset["final_trace_distance"] == pytest.approx(0.0, abs=1e-12)
    assert carry["trace_distance_auc"] > reset["trace_distance_auc"]
    assert [row["lag"] for row in lagged] == [1, 2, 3]
    assert all(row["correlation_pair_count"] == 12 for row in lagged)


def test_aggregation_and_paired_differences_are_seed_aligned(
    tmp_path: Path,
) -> None:
    run_dirs = {}
    metadata = {}
    values = {
        ("carry_inputs", 2026): 0.6,
        ("carry_inputs", 2027): 0.7,
        ("reset_each_input", 2026): 0.4,
        ("reset_each_input", 2027): 0.5,
    }
    for (policy, seed), value in values.items():
        metadata[(policy, seed)] = _point_metadata(policy, seed)
        for task in TASKS:
            identity = (policy, seed, task)
            run_dirs[identity] = _fake_run(
                tmp_path,
                policy=policy,
                seed=seed,
                task=task,
                metric_value=value,
            )
    rows = collect_state_memory_rows(run_dirs, point_metadata=metadata)
    aggregate = aggregate_state_memory_rows(rows)
    paired, summary = paired_state_policy_differences(rows)

    macro = next(
        row
        for row in aggregate
        if row["state_policy"] == "carry_inputs"
        and row["task"] == TASKS[0]
        and row["split"] == "validation"
        and row["metric"] == "macro_f1"
    )
    qlike = next(
        row
        for row in paired
        if row["split"] == "validation"
        and row["metric"] == "qlike"
        and row["reservoir_seed"] == 2026
    )
    macro_summary = next(
        row for row in summary if row["split"] == "test" and row["metric"] == "macro_f1"
    )
    assert macro["mean"] == pytest.approx(0.65)
    assert macro["standard_deviation"] == pytest.approx(0.05)
    assert qlike["delta_carry_minus_reset"] == pytest.approx(0.2)
    assert qlike["directional_improvement_for_carry"] == pytest.approx(-0.2)
    assert qlike["favored_policy"] == "reset_each_input"
    assert "negative" in qlike["qlike_interpretation"]
    assert macro_summary["mean_delta_carry_minus_reset"] == pytest.approx(0.2)


def test_validation_policy_selection_never_reads_test_metrics(tmp_path: Path) -> None:
    run_dirs = {}
    for policy, value in (("carry_inputs", 0.7), ("reset_each_input", 0.5)):
        for task in TASKS:
            identity = (policy, 2026, task)
            directory = _fake_run(
                tmp_path,
                policy=policy,
                seed=2026,
                task=task,
                metric_value=value,
                complete=False,
            )
            run_dirs[identity] = directory

    selection = select_state_policy_from_validation(run_dirs)

    assert selection["selected_state_policy"] == "carry_inputs"
    assert selection["decisive_criterion"] == "validation_macro_f1_mean"
    assert selection["test_metrics_read"] is False
    assert selection["selected_before_test_reporting"] is True


def test_rows_include_required_provenance_and_temporal_fields(tmp_path: Path) -> None:
    identity = ("carry_inputs", 2026, TASKS[0])
    directory = _fake_run(
        tmp_path,
        policy="carry_inputs",
        seed=2026,
        task=TASKS[0],
    )
    rows = collect_state_memory_rows(
        {identity: directory},
        point_metadata={("carry_inputs", 2026): _point_metadata("carry_inputs", 2026)},
    )

    required = {
        "state_policy",
        "n_qubits",
        "virtual_nodes",
        "reservoir_seed",
        "task",
        "split",
        "selected_ridge_alpha",
        "raw_feature_dimension",
        "feature_generation_seconds",
        "readout_fitting_seconds",
        "total_runtime_seconds",
        "condition_number",
        "effective_rank",
        "lag1_mean_absolute_feature_autocorrelation",
        "lag1_mean_absolute_input_feature_correlation",
        "perturbation_final_trace_distance",
        "cache_key_checksum",
        "data_manifest_checksum",
        "git_commit",
        "python_version",
        "platform",
        "package_versions",
        "model_selection_data",
        "test_evaluated_after_readout_freeze",
    }
    assert required.issubset(rows[0])
    assert rows[0]["model_selection_data"] == "validation only"
    assert rows[0]["test_evaluated_after_readout_freeze"] is True


def test_discovery_rejects_synthetic_and_non_temporal_runs(tmp_path: Path) -> None:
    _fake_run(
        tmp_path,
        policy="carry_inputs",
        seed=2026,
        task=TASKS[0],
        synthetic=True,
    )
    with pytest.raises(SyntheticResultsError, match="synthetic"):
        discover_completed_state_memory_runs(
            tmp_path,
            study_id=STUDY_ID,
            snapshot_id="yahoo_chart_20100101_20251231_v1",
        )

    clean_root = tmp_path / "non_temporal"
    identity = ("carry_inputs", 2026, TASKS[0])
    directory = _fake_run(
        clean_root,
        policy="carry_inputs",
        seed=2026,
        task=TASKS[0],
        frozen_test=False,
    )
    with pytest.raises(ValueError, match="frozen-readout"):
        collect_state_memory_rows(
            {identity: directory},
            point_metadata={("carry_inputs", 2026): _point_metadata("carry_inputs", 2026)},
        )


def test_public_data_guard_rejects_synthetic_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = load_state_memory_study_config(_root() / "configs/qrc_state_memory_ablation.yaml")
    classifier = load_model_config(study.classifier_reference)
    fake_dataset: Any = SimpleNamespace(is_synthetic=True, data_source_type="fixture")
    monkeypatch.setattr(
        "qtyche_qrc.experiments.qrc_state_memory.verify_public_snapshot",
        lambda _config: {
            "snapshot_id": "yahoo_chart_20100101_20251231_v1",
            "files": {},
        },
    )
    monkeypatch.setattr(
        "qtyche_qrc.experiments.qrc_state_memory.load_model_dataset",
        lambda _processed_dir: fake_dataset,
    )

    with pytest.raises(SyntheticResultsError, match="non-synthetic"):
        verify_state_memory_public_data(study, classifier)


def test_partial_completion_resumes_only_missing_task(tmp_path: Path) -> None:
    classifier = _fake_run(
        tmp_path,
        policy="reset_each_input",
        seed=2026,
        task=TASKS[0],
    )
    _fake_run(
        tmp_path,
        policy="reset_each_input",
        seed=2026,
        task=TASKS[1],
        complete=False,
    )

    completed = discover_completed_state_memory_runs(
        tmp_path,
        study_id=STUDY_ID,
        snapshot_id="yahoo_chart_20100101_20251231_v1",
    )
    pending = pending_state_memory_runs((StateMemoryPoint("reset_each_input", 2026),), completed)

    assert completed == {("reset_each_input", 2026, TASKS[0]): classifier}
    assert pending == ((StateMemoryPoint("reset_each_input", 2026), TASKS[1]),)


def test_output_namespace_and_existing_contracts_remain_isolated() -> None:
    root = _root()
    study = load_state_memory_study_config(root / "configs/qrc_state_memory_ablation.yaml")

    assert study.output_root == root / "results/qrc_state_memory_ablation"
    assert study.output_root not in {
        root / "results/qrc_public_pilot",
        root / "results/qrc_qubit_scaling",
        root / "results/qrc_noise_robustness",
        root / "results/qrc_encoding_density",
    }
    assert sha256_file(root / "configs/models/qrc_classifier_pilot.yaml") == (
        "65a8b098aa466b2ab0b336cb8f5f6cabe412c43a2e8b27673d427304c24e0002"
    )
    assert sha256_file(root / "configs/models/qrc_regressor_pilot.yaml") == (
        "3a3c1d75521c6e0ac027686fade31d319c16108a699efbde1eeb147bc4158bb3"
    )


def test_resource_estimates_are_equal_across_policies() -> None:
    rows = state_memory_resource_estimates(split_rows=3989, train_rows=2744)

    assert [row["state_policy"] for row in rows] == [
        "carry_inputs",
        "reset_each_input",
    ]
    assert {row["raw_feature_dimension"] for row in rows} == {6}
    assert {row["estimated_peak_density_matrix_bytes"] for row in rows} == {768}
    assert {row["estimated_cached_feature_bytes"] for row in rows} == {191_472}
    assert {row["estimated_train_condition_matrix_bytes"] for row in rows} == {131_712}
