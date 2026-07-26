import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

import qtyche_qrc.experiments.final_financial_qrc as final_module
from qtyche_qrc.experiments.final_financial_qrc import (
    RIDGE_GRID,
    SEEDS,
    STUDY_ID,
    _completed_run_uses_cache,
    _run_row,
    _validation_selection,
    aggregate_exact_rows,
    discover_completed_final_runs,
    load_final_qrc_config,
)
from qtyche_qrc.experiments.model_config import load_model_config
from qtyche_qrc.experiments.qrc_robustness import (
    FINAL_STUDY_ID,
    TASKS,
    build_robustness_grid,
    load_robustness_study_config,
    write_robustness_model_config,
)
from qtyche_qrc.experiments.qrc_run import qrc_config_from_model
from qtyche_qrc.experiments.qrc_state_memory import _feature_condition_diagnostics
from qtyche_qrc.experiments.run import SyntheticResultsError
from qtyche_qrc.models.qrc.features import make_feature_cache_key


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _final_path() -> Path:
    return _root() / "configs/final_financial_qrc.yaml"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_final_configuration_matches_validation_selected_architecture() -> None:
    config = load_final_qrc_config(_final_path())
    architecture = config.raw["architecture"]
    freeze = config.raw["freeze"]

    assert config.seeds == SEEDS
    assert architecture == {
        "n_qubits": 2,
        "virtual_nodes": 2,
        "state_policy": "reset_each_input",
        "tau": 1.0,
        "graph": "ring",
        "j_strength": 1.0,
        "h_strength": 1.0,
        "input_scaling": 0.5,
        "backend": "numpy_density_matrix_exact",
        "exact_noiseless": True,
        "input_reinjection": "partial_input_qubit_reinjection",
        "observables": {
            "single_qubit": "analytic_Z_i",
            "two_qubit": "analytic_unique_Z_i_Z_j",
        },
        "raw_feature_dimension": 6,
    }
    assert tuple(config.raw["readout"]["ridge_alpha_grid"]) == RIDGE_GRID
    assert config.raw["readout"]["preprocessing_fit_split"] == "train"
    assert config.raw["readout"]["selection_split"] == "validation"
    assert config.raw["readout"]["test_evaluated_after_freeze"] is True
    assert freeze["architecture_frozen"] is True
    assert freeze["no_further_test_tuning"] is True
    assert freeze["test_metrics_used_for_selection"] is False


def test_validation_selection_artifact_excludes_held_out_metrics() -> None:
    config = load_final_qrc_config(_final_path())
    payload = _validation_selection(config)
    serialized = json.dumps(payload, sort_keys=True).lower()

    assert payload["selection_basis"] == "validation only"
    assert payload["held_out_metrics_excluded"] is True
    assert '"split": "validation"' in serialized
    assert '"split": "test"' not in serialized
    assert "test_metrics" not in serialized


def test_final_robustness_is_reset_only_complete_and_isolated(tmp_path: Path) -> None:
    source = _root() / "configs/final_financial_qrc_robustness.yaml"
    config = load_robustness_study_config(source)
    assert config.study_id == FINAL_STUDY_ID
    assert config.fixed_qrc["state_policy"] == "reset_each_input"
    assert config.output_root == _root() / "results/final_financial_qrc/robustness"
    assert config.output_root != _root() / "results/qrc_noise_robustness"
    previous_trees = (
        "public_market",
        "garch_baseline",
        "qrc_public_pilot",
        "qrc_qubit_scaling",
        "qrc_encoding_density",
        "qrc_state_memory_ablation",
        "qrc_noise_robustness",
    )
    assert all(config.output_root != _root() / "results" / tree for tree in previous_trees)

    points = build_robustness_grid(
        n_qubits=2,
        reservoir_seeds=config.reservoir_seeds,
        measurement_seeds=config.measurement_seeds,
        shots=config.shots,
        depolarizing_probabilities=config.depolarizing_probabilities,
        measurement_noise_probabilities=config.measurement_noise_probabilities,
    )
    assert len(points) == 111
    assert {point.shot_count for point in points if point.study_type == "finite_shot"} == {
        128,
        512,
        2048,
        8192,
    }
    assert {
        point.depolarizing_probability
        for point in points
        if point.study_type == "depolarizing_noise"
    } == {0.0, 0.0001, 0.001, 0.01}
    assert {
        point.measurement_bit_flip_probability
        for point in points
        if point.study_type == "measurement_noise"
    } == {0.0, 0.005, 0.01, 0.02}

    from dataclasses import replace

    isolated = replace(config, output_root=tmp_path / "robustness")
    for task in TASKS:
        generated = load_model_config(write_robustness_model_config(isolated, points[0], task))
        assert qrc_config_from_model(generated).state_policy == "reset_each_input"


def test_final_data_verification_rejects_synthetic_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_final_qrc_config(_final_path())

    class FakeDataConfig:
        snapshot_manifest_path = config.project_root / "snapshot.json"

    class FakeModelConfig:
        processed_dir = config.project_root / "fixture"

    class FakeDataset:
        is_synthetic = True
        data_source_type = "fixture"

    monkeypatch.setattr(final_module, "load_data_config", lambda _path: FakeDataConfig())
    monkeypatch.setattr(
        final_module,
        "verify_public_snapshot",
        lambda _config: {"snapshot_id": config.snapshot_id, "files": {}},
    )
    monkeypatch.setattr(final_module, "load_model_config", lambda _path: FakeModelConfig())
    monkeypatch.setattr(final_module, "load_model_dataset", lambda _path: FakeDataset())

    with pytest.raises(SyntheticResultsError, match="non-synthetic public-market"):
        final_module.verify_final_public_data(config)


def test_classifier_and_regressor_share_deterministic_exact_cache_key() -> None:
    classifier = load_model_config(_root() / "configs/models/final_financial_qrc_classifier.yaml")
    regressor = load_model_config(_root() / "configs/models/final_financial_qrc_regressor.yaml")
    classifier_qrc = qrc_config_from_model(classifier, reservoir_seed=2027)
    regressor_qrc = qrc_config_from_model(regressor, reservoir_seed=2027)
    first = make_feature_cache_key(
        processed_data_manifest_checksum="frozen-data",
        feature_names=("x", "y"),
        config=classifier_qrc,
    )
    repeated = make_feature_cache_key(
        processed_data_manifest_checksum="frozen-data",
        feature_names=("x", "y"),
        config=regressor_qrc,
    )

    assert classifier_qrc == regressor_qrc
    assert classifier.raw["qrc"]["feature_cache"] == regressor.raw["qrc"]["feature_cache"]
    assert first.checksum == repeated.checksum


def _fake_completed_run(
    root: Path,
    *,
    task: str = "regime_classification",
    complete: bool = True,
) -> Path:
    directory = root / "run"
    (directory / "model").mkdir(parents=True)
    manifest = {
        "status": "success",
        "data_source_type": "public_market",
        "is_synthetic": False,
        "data_snapshot_id": "snapshot",
        "backend": "numpy_density_matrix_exact",
        "exact_noiseless": True,
        "model_selection_data": "validation only",
        "test_evaluated_after_readout_freeze": True,
        "reservoir_seed": 2026,
        "task": task,
        "qrc_configuration": {
            "n_qubits": 2,
            "virtual_nodes": 2,
            "state_policy": "reset_each_input",
        },
    }
    _write_json(directory / "manifest.json", manifest)
    (directory / "config.yaml").write_text(
        yaml.safe_dump({"final_architecture": {"id": STUDY_ID}}), encoding="utf-8"
    )
    if complete:
        for split in ("validation", "test"):
            _write_json(directory / f"{split}_metrics.json", {})
            (directory / f"{split}_predictions.csv").write_text(
                "date,value\n2025-01-01,1\n", encoding="utf-8"
            )
        np.savez_compressed(directory / "model/readout.npz", readout=np.ones((2, 1)))
    return directory


def test_partial_resumption_discovers_only_complete_runs(tmp_path: Path) -> None:
    complete_root = tmp_path / "complete"
    partial_root = tmp_path / "partial"
    _fake_completed_run(complete_root)
    _fake_completed_run(partial_root, complete=False)

    assert discover_completed_final_runs(complete_root, snapshot_id="snapshot") == {
        (2026, "regime_classification"): complete_root / "run"
    }
    assert discover_completed_final_runs(partial_root, snapshot_id="snapshot") == {}


def test_partial_resumption_rejects_stale_feature_cache(tmp_path: Path) -> None:
    directory = _fake_completed_run(tmp_path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["qrc_feature_cache_key_checksum"] = "current-cache"
    _write_json(manifest_path, manifest)

    assert _completed_run_uses_cache(directory, "current-cache")
    assert not _completed_run_uses_cache(directory, "stale-cache")


def test_reset_reference_reproduction_and_finiteness_are_recorded(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    (directory / "model").mkdir(parents=True)
    manifest = {
        "reservoir_seed": 2027,
        "task": "regime_classification",
        "selected_hyperparameters": {"ridge_alpha": 1.0e-5},
        "qrc_raw_feature_dimension": 6,
        "readout_shape": [7, 3],
        "trainable_readout_parameters": 21,
        "qrc_feature_cache_key_checksum": "cache",
        "qrc_feature_cache_hit": True,
        "data_snapshot_id": "snapshot",
        "backend": "numpy_density_matrix_exact",
        "exact_noiseless": True,
        "model_selection_data": "validation only",
        "test_evaluated_after_readout_freeze": True,
    }
    metrics = {
        "macro_f1": 0.4,
        "balanced_accuracy": 0.5,
        "transition_pr_auc": 0.6,
        "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    _write_json(directory / "manifest.json", manifest)
    _write_json(
        directory / "timing.json",
        {"state_generation_seconds": 0.2, "readout_fitting_seconds": 0.01},
    )
    _write_json(directory / "test_metrics.json", metrics)
    pd.DataFrame(
        {
            "true_regime": [0, 1],
            "predicted_regime": [0, 1],
            "probability_low": [0.8, 0.1],
            "probability_medium": [0.1, 0.8],
            "probability_high": [0.1, 0.1],
        }
    ).to_csv(directory / "test_predictions.csv", index=False)
    np.savez_compressed(directory / "model/readout.npz", readout=np.full((7, 3), 0.25))
    diagnostics = {
        "effective_rank": 2.5,
        "numerical_rank": 6,
        "condition_number": 5.77e12,
        "largest_singular_value": 10.0,
        "smallest_retained_singular_value": 10.0 / 5.77e12,
    }
    reference = {(2027, "regime_classification", "test"): metrics}

    row = _run_row(
        directory,
        split="test",
        diagnostics=diagnostics,
        reference_rows=reference,
        repository_root=tmp_path,
    )

    assert row["reset_ablation_reproduced_within_1e_12"] is True
    assert row["condition_number"] == 5.77e12
    assert row["finite_coefficients"] is True
    assert row["finite_predictions"] is True


def test_condition_recording_and_aggregation_are_deterministic() -> None:
    x = np.linspace(0.0, 1.0, 40)
    features = np.column_stack((x, x + 1.0e-12 * np.sin(np.arange(len(x))), x**2))
    diagnostics = _feature_condition_diagnostics(features)
    assert diagnostics["condition_number"] > 1.0e12

    rows = [
        {
            "task": "rv_regression",
            "split": "test",
            "reservoir_seed": seed,
            "qlike": -2.0 - index / 10,
            "rmse": 0.1,
            "mae": 0.05,
            "correlation": 0.5,
            "condition_number": float(seed),
            "effective_rank": 2.0,
            "readout_coefficient_l2_norm": 1.0,
        }
        for index, seed in enumerate(SEEDS)
    ]
    assert aggregate_exact_rows(rows) == aggregate_exact_rows(list(reversed(rows)))


def test_final_comparison_selects_latest_complete_public_baseline() -> None:
    source = (_root() / "src/qtyche_qrc/experiments/final_financial_qrc.py").read_text(
        encoding="utf-8"
    )

    assert "latest[model_type] = directory" in source
    assert 'manifest.get("status") != "success"' in source
    assert 'manifest = json.loads((directory / "manifest.json")' in source
