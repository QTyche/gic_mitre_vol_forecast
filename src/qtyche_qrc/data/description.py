"""Descriptive public-market reporting before any model fitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qtyche_qrc.data.download import sha256_file
from qtyche_qrc.data.validation import DataValidationError


def _repository_root(processed_dir: Path) -> Path:
    for candidate in (processed_dir, *processed_dir.parents):
        if (candidate / ".git").exists():
            return candidate
    raise DataValidationError("could not locate repository root for descriptive report")


def _resolve_source(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _summary(values: pd.Series[float]) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "standard_deviation": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "q01": float(values.quantile(0.01)),
        "q05": float(values.quantile(0.05)),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "q95": float(values.quantile(0.95)),
        "q99": float(values.quantile(0.99)),
    }


def _boundaries(axis: Any, manifest: dict[str, Any]) -> None:
    for split_name in ("validation", "test"):
        date = pd.Timestamp(manifest["split_boundaries"][split_name]["start"])
        axis.axvline(date, color="black", linestyle="--", linewidth=0.8)


def describe_public_data(processed_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Write split statistics and seven non-predictive descriptive figures."""

    processed_dir = processed_dir.resolve()
    manifest_path = processed_dir / "data_manifest.json"
    if not manifest_path.is_file():
        raise DataValidationError(f"missing data manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("data_source_type") != "public_market" or manifest.get("is_synthetic"):
        raise DataValidationError("describe-data requires a non-synthetic public-market dataset")
    root = _repository_root(processed_dir)
    destination = output_dir or root / "results/data_audit"
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    unscaled = pd.read_csv(processed_dir / "features_unscaled.csv", parse_dates=["date"])
    rows: list[dict[str, Any]] = []
    split_documents: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        split = unscaled.loc[unscaled["split"].eq(split_name)].copy()
        regimes = split["target_regime_5d"].value_counts()
        document: dict[str, Any] = {
            "observation_count": len(split),
            "start_date": split["date"].min().date().isoformat(),
            "end_date": split["date"].max().date().isoformat(),
            "regime": {
                str(label): {
                    "count": int(regimes.get(label, 0)),
                    "percentage": float(regimes.get(label, 0) / len(split)),
                }
                for label in (0, 1, 2)
            },
            "transition_count": int(split["target_transition"].sum()),
            "transition_rate": float(split["target_transition"].mean()),
            "upward_transition_count": int(split["target_upward_transition"].sum()),
            "upward_transition_rate": float(split["target_upward_transition"].mean()),
            "downward_transition_count": int(split["target_downward_transition"].sum()),
            "downward_transition_rate": float(split["target_downward_transition"].mean()),
            "target_rv_5d": _summary(split["target_rv_5d"]),
            "vix": _summary(np.exp(split["vix_log_level"])),
            "spy_log_return_1d": _summary(split["spy_log_return_1d"]),
        }
        split_documents[split_name] = document
        row: dict[str, Any] = {
            "split": split_name,
            "observation_count": len(split),
            "start_date": document["start_date"],
            "end_date": document["end_date"],
            "transition_count": document["transition_count"],
            "transition_rate": document["transition_rate"],
            "upward_transition_count": document["upward_transition_count"],
            "upward_transition_rate": document["upward_transition_rate"],
            "downward_transition_count": document["downward_transition_count"],
            "downward_transition_rate": document["downward_transition_rate"],
        }
        for label, name in ((0, "low"), (1, "medium"), (2, "high")):
            row[f"{name}_regime_count"] = document["regime"][str(label)]["count"]
            row[f"{name}_regime_percentage"] = document["regime"][str(label)]["percentage"]
        for family in ("target_rv_5d", "vix", "spy_log_return_1d"):
            for statistic, value in document[family].items():
                row[f"{family}_{statistic}"] = value
        rows.append(row)

    summary = {
        "schema_version": 1,
        "data_source_type": "public_market",
        "is_synthetic": False,
        "snapshot_id": manifest.get("source_snapshot_id"),
        "data_manifest_sha256": sha256_file(manifest_path),
        "requested_date_range": manifest.get("requested_date_range"),
        "splits": split_documents,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "public_market_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(destination / "public_market_summary.csv", index=False)

    source_files = manifest["source_files"]
    spy = pd.read_csv(_resolve_source(root, source_files["spy"]["path"]), parse_dates=["date"])
    vix = pd.read_csv(_resolve_source(root, source_files["vix"]["path"]), parse_dates=["date"])
    series_specs = (
        (spy, "close", "SPY closing price", "spy_closing_price.png"),
        (vix, "close", "VIX closing level", "vix.png"),
        (unscaled, "target_rv_5d", "Five-day forward realized variance", "target_rv_5d.png"),
    )
    for frame, column, title, filename in series_specs:
        figure, axis = plt.subplots(figsize=(11, 3.5))
        axis.plot(frame["date"], frame[column], linewidth=0.8)
        _boundaries(axis, manifest)
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(figures / filename, dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 3.5))
    axis.step(unscaled["date"], unscaled["target_regime_5d"], where="post")
    _boundaries(axis, manifest)
    axis.set_yticks([0, 1, 2], ["Low", "Medium", "High"])
    axis.set_title("Forward volatility regimes")
    figure.tight_layout()
    figure.savefig(figures / "volatility_regimes.png", dpi=150)
    plt.close(figure)

    transitions = unscaled.loc[unscaled["target_transition"].eq(1)]
    figure, axis = plt.subplots(figsize=(11, 3.5))
    axis.scatter(transitions["date"], transitions["target_regime_5d"], s=8)
    _boundaries(axis, manifest)
    axis.set_yticks([0, 1, 2], ["Low", "Medium", "High"])
    axis.set_title("Regime-transition events")
    figure.tight_layout()
    figure.savefig(figures / "regime_transitions.png", dpi=150)
    plt.close(figure)

    class_table = pd.crosstab(unscaled["split"], unscaled["target_regime_5d"]).reindex(
        index=["train", "validation", "test"], columns=[0, 1, 2], fill_value=0
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    class_table.plot.bar(ax=axis)
    axis.set_title("Class distribution by split")
    axis.legend(["Low", "Medium", "High"])
    figure.tight_layout()
    figure.savefig(figures / "class_distribution.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    for split_name in ("train", "validation", "test"):
        values = unscaled.loc[unscaled["split"].eq(split_name), "target_rv_5d"]
        axis.hist(values, bins=50, alpha=0.4, density=True, label=split_name)
    axis.set_title("Target realized-variance distribution by split")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "target_rv_distribution.png", dpi=150)
    plt.close(figure)
    return summary
