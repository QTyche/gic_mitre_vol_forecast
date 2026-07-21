"""Separate validation and test baseline comparison tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from qtyche_qrc.evaluation.plots import plot_baseline_comparison


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def compare_baselines(
    results_dir: Path,
    output_dir: Path,
    *,
    latest_per_model: bool = False,
) -> tuple[Path, Path]:
    """Write independent validation and test tables; never create a combined ranking."""

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for manifest_path in sorted(results_dir.rglob("manifest.json")):
        experiment_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "success":
            continue
        selected = json.dumps(
            manifest.get("selected_hyperparameters"), sort_keys=True, separators=(",", ":")
        )
        common = {
            "data_source_type": manifest["data_source_type"],
            "is_synthetic": manifest["is_synthetic"],
            "data_warning": manifest.get("data_warning"),
            "task": manifest["task"],
            "seed": manifest["seed"],
            "model": manifest["model_type"],
            "selected_configuration": selected,
            "experiment_id": manifest["experiment_id"],
            "selection_metric": manifest["model_selection_metric"],
        }
        validation = json.loads(
            (experiment_dir / "validation_metrics.json").read_text(encoding="utf-8")
        )
        test = json.loads((experiment_dir / "test_metrics.json").read_text(encoding="utf-8"))
        validation_rows.append({**common, **_scalar_metrics(validation)})
        test_rows.append({**common, **_scalar_metrics(test)})
    if not validation_rows:
        raise ValueError(f"no completed experiments found below {results_dir}")
    validation_table = pd.DataFrame(validation_rows)
    test_table = pd.DataFrame(test_rows)
    if latest_per_model:
        identity = ["task", "seed", "model"]
        # ``rglob`` ordering groups runs by directory path, not execution time.
        # Sort on the timestamp-prefixed experiment ID so archived/nested runs
        # cannot displace a newer top-level run.
        validation_table = validation_table.sort_values("experiment_id")
        test_table = test_table.sort_values("experiment_id")
        validation_table = validation_table.drop_duplicates(identity, keep="last")
        test_table = test_table.drop_duplicates(identity, keep="last")
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "baseline_validation_comparison.csv"
    test_path = output_dir / "baseline_test_comparison.csv"
    validation_table.to_csv(validation_path, index=False)
    test_table.to_csv(test_path, index=False)
    validation_table["primary_metric_value"] = validation_table.apply(
        lambda row: row.get(str(row["selection_metric"])), axis=1
    )
    test_table["primary_metric_value"] = test_table.apply(
        lambda row: row.get(str(row["selection_metric"])), axis=1
    )
    plot_baseline_comparison(
        validation_table,
        "primary_metric_value",
        output_dir / "baseline_validation_comparison.png",
    )
    plot_baseline_comparison(
        test_table,
        "primary_metric_value",
        output_dir / "baseline_test_comparison.png",
    )
    return validation_path, test_path
