import json
from pathlib import Path
from typing import Any

import pandas as pd

from qtyche_qrc.experiments.public_compare import compare_public_baselines


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _experiment(root: Path, model: str, task: str, metrics: dict[str, Any]) -> None:
    experiment_id = f"20260101T000000.000000Z_{model}_{task}_seed7"
    directory = root / experiment_id
    directory.mkdir(parents=True)
    manifest = {
        "experiment_id": experiment_id,
        "status": "success",
        "model_type": model,
        "task": task,
        "seed": 7,
        "selected_hyperparameters": {},
        "data_snapshot_id": "snapshot_v1",
        "data_manifest_checksum": "abc123",
        "data_source_type": "public_market",
        "is_synthetic": False,
        "git": {"commit": "deadbeef", "dirty": False},
    }
    _write_json(directory / "manifest.json", manifest)
    _write_json(directory / "validation_metrics.json", metrics)
    _write_json(directory / "test_metrics.json", metrics)


def test_public_comparison_writes_separate_tables_with_provenance(tmp_path: Path) -> None:
    classification = {
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "macro_f1": 0.4,
        "weighted_f1": 0.45,
        "log_loss": 1.0,
        "multiclass_brier_score": 0.6,
        "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "per_class_precision": {"low": 0.4, "medium": 0.5, "high": 0.6},
        "per_class_recall": {"low": 0.4, "medium": 0.5, "high": 0.6},
        "per_class_f1": {"low": 0.4, "medium": 0.5, "high": 0.6},
        "transition_rate": 0.3,
        "transition_accuracy": 0.6,
        "transition_balanced_accuracy": 0.5,
        "transition_f1": 0.4,
        "transition_roc_auc": 0.7,
        "transition_pr_auc": 0.6,
        "transition_brier_score": 0.2,
        "transition_subgroups": {
            "low_origin": {"count": 10, "positive_count": 3, "unstable": True}
        },
    }
    regression = {
        "rmse": 0.1,
        "mae": 0.08,
        "qlike": -2.0,
        "r_squared": 0.2,
        "prediction_mean": 0.01,
        "prediction_median": 0.009,
        "prediction_minimum": 0.001,
        "prediction_maximum": 0.05,
        "non_finite_prediction_count": 0,
        "floored_prediction_count": 0,
        "prediction_floor": 1e-12,
    }
    results = tmp_path / "results"
    _experiment(results, "majority_classifier", "regime_classification", classification)
    _experiment(results, "rv_persistence", "rv_regression", regression)

    outputs = compare_public_baselines(results, tmp_path / "tables")

    assert outputs["validation"].name == "public_market_validation_comparison.csv"
    assert outputs["test"].name == "public_market_test_comparison.csv"
    assert outputs["validation"] != outputs["test"]
    for path in outputs.values():
        assert path.is_file()
        table = pd.read_csv(path)
        assert {
            "model",
            "task",
            "seed",
            "selected_configuration",
            "data_snapshot_id",
            "data_manifest_checksum",
            "data_source_type",
            "is_synthetic",
            "git_commit",
            "dirty",
            "split",
        }.issubset(table.columns)
