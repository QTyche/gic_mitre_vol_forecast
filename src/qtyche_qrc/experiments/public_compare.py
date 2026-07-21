"""Public-market benchmark tables with separate validation and test designations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _common(manifest: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "model": manifest["model_type"],
        "task": manifest["task"],
        "seed": manifest["seed"],
        "selected_configuration": json.dumps(
            manifest.get("selected_hyperparameters"), sort_keys=True, separators=(",", ":")
        ),
        "data_snapshot_id": manifest.get("data_snapshot_id"),
        "data_manifest_checksum": manifest.get("data_manifest_checksum"),
        "data_source_type": manifest["data_source_type"],
        "is_synthetic": manifest["is_synthetic"],
        "git_commit": manifest.get("git", {}).get("commit"),
        "dirty": manifest.get("git", {}).get("dirty"),
        "split": split,
        "experiment_id": manifest["experiment_id"],
    }


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _latest_public_experiments(results_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    experiments: list[tuple[Path, dict[str, Any]]] = []
    for path in results_dir.rglob("manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "success":
            continue
        if manifest.get("data_source_type") != "public_market" or manifest.get("is_synthetic"):
            continue
        experiments.append((path.parent, manifest))
    if not experiments:
        raise ValueError(f"no completed public-market experiments found below {results_dir}")
    experiments.sort(key=lambda item: str(item[1]["experiment_id"]))
    latest: dict[tuple[str, str, int], tuple[Path, dict[str, Any]]] = {}
    for experiment in experiments:
        manifest = experiment[1]
        identity = (manifest["task"], manifest["model_type"], int(manifest["seed"]))
        latest[identity] = experiment
    return list(latest.values())


def compare_public_baselines(results_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Create the five requested public-market comparison and diagnostic tables."""

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    for experiment_dir, manifest in _latest_public_experiments(results_dir):
        for split in ("validation", "test"):
            metrics = json.loads(
                (experiment_dir / f"{split}_metrics.json").read_text(encoding="utf-8")
            )
            common = _common(manifest, split)
            comparison = {**common, **_scalar_metrics(metrics)}
            if split == "validation":
                validation_rows.append(comparison)
            else:
                test_rows.append(comparison)
            if manifest["task"] == "regime_classification":
                classification = dict(common)
                for family in ("per_class_precision", "per_class_recall", "per_class_f1"):
                    for class_name, value in metrics[family].items():
                        classification[f"{family}_{class_name}"] = value
                for name in (
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "weighted_f1",
                    "log_loss",
                    "multiclass_brier_score",
                ):
                    classification[name] = metrics[name]
                classification["confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
                classification_rows.append(classification)

                overall = {
                    **common,
                    "subgroup": "overall",
                    **{
                        name: metrics[name]
                        for name in (
                            "transition_rate",
                            "transition_accuracy",
                            "transition_balanced_accuracy",
                            "transition_f1",
                            "transition_roc_auc",
                            "transition_pr_auc",
                            "transition_brier_score",
                        )
                    },
                    "unstable": False,
                }
                transition_rows.append(overall)
                for subgroup, values in metrics.get("transition_subgroups", {}).items():
                    transition_rows.append({**common, "subgroup": subgroup, **values})
            else:
                regression_rows.append({**common, **_scalar_metrics(metrics)})

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "validation": output_dir / "public_market_validation_comparison.csv",
        "test": output_dir / "public_market_test_comparison.csv",
        "classification": output_dir / "public_market_classification_diagnostics.csv",
        "transition": output_dir / "public_market_transition_diagnostics.csv",
        "regression": output_dir / "public_market_regression_diagnostics.csv",
    }
    pd.DataFrame(validation_rows).to_csv(outputs["validation"], index=False)
    pd.DataFrame(test_rows).to_csv(outputs["test"], index=False)
    pd.DataFrame(classification_rows).to_csv(outputs["classification"], index=False)
    pd.DataFrame(transition_rows).to_csv(outputs["transition"], index=False)
    pd.DataFrame(regression_rows).to_csv(outputs["regression"], index=False)
    return outputs
