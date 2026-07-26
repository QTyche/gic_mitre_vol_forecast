#!/usr/bin/env python3
"""Prepare the frozen and reproducibly transformed Phase 3 paper assets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
PAPER_BASE_COMMIT = "49c1fde09b56df9a5d9c1bef04dfa039275926b0"
SCIENTIFIC_FREEZE_COMMIT = "3b562e72c655e6e4fb38b45ec22cfb4a1f96b530"
RESAMPLING_REPETITIONS = 10_000
TWO_SIDED_REPORTING_RESOLUTION = 2.0 / RESAMPLING_REPETITIONS

MODEL_COLOURS = {
    "Majority classifier": "#999999",
    "Regime persistence": "#CC79A7",
    "Logistic regression": "#0072B2",
    "ESN": "#E69F00",
    "RV persistence": "#999999",
    "GARCH(1,1)": "#009E73",
    "QRC mean": "#D55E00",
    "Flattened logistic": "#0072B2",
    "Size-controlled ESN": "#E69F00",
    "Exact QRC mean": "#D55E00",
}


@dataclass(frozen=True)
class FrozenInput:
    path: str
    sha256: str


FROZEN_INPUTS = {
    "financial_figure": FrozenInput(
        "paper_assets/figures/publication_figure_2_financial_comparison.pdf",
        "4df86f6a34dbe2bc6e91daf6e8385ae2aa491009482101ae58da5142b25444a5",
    ),
    "mnist_figure": FrozenInput(
        "paper_assets/figures/publication_figure_4_mnist_benchmark.pdf",
        "96eed1a4175cd1928895887da5be8dfae9c916c0719e140c63c4e3169cd77bde",
    ),
    "financial_table": FrozenInput(
        "paper_assets/tables/publication_table_2_financial_benchmark.json",
        "fe82b17b076bf7188d470b3142acda186338cccdbc42cbea44007ee7a68a70f2",
    ),
    "mnist_table": FrozenInput(
        "paper_assets/tables/publication_table_3_mnist_benchmark.json",
        "5a6a8ace43d107cf4c09b6153f0b4cb359e53f0e6f3765396cece1e621e6d202",
    ),
    "final_results_manifest": FrozenInput(
        "paper_assets/final_results_manifest.json",
        "a6cc26b63c6931e70e07b4c513fded101786c4dc52c0bc115bdc5294cfbe32d8",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_source(item: FrozenInput) -> Path:
    source = REPOSITORY / item.path
    if not source.is_file():
        raise FileNotFoundError(f"required frozen asset is missing: {item.path}")
    actual = sha256(source)
    if actual != item.sha256:
        raise ValueError(
            f"frozen asset checksum mismatch for {item.path}: "
            f"expected {item.sha256}, found {actual}"
        )
    return source


def load_json(key: str) -> dict[str, Any]:
    source = checked_source(FROZEN_INPUTS[key])
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {source}")
    return cast(dict[str, Any], payload)


def latex_value(row: dict[str, Any], field: str, *, bold: bool = False) -> str:
    value = str(row[f"{field}_display"]).replace("±", r"$\pm$")
    if value == "—":
        value = r"\textemdash"
    return rf"\textbf{{{value}}}" if bold else value


def _save_figure(figure: Figure, destination: Path) -> None:
    """Write a deterministic vector-only paper figure."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        bbox_inches="tight",
        metadata={
            "Creator": "Team QTyche Phase 3 paper asset compiler",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def _direct_metric_panel(
    axis: Axes,
    rows: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    direction: str,
) -> None:
    usable = [row for row in rows if row.get(metric) is not None]
    y = np.arange(len(usable))
    values = np.asarray([float(row[metric]) for row in usable])
    errors = np.asarray([float(row.get(f"{metric}_sd") or 0.0) for row in usable])
    colours = [MODEL_COLOURS[str(row["model"])] for row in usable]
    axis.errorbar(
        values,
        y,
        xerr=errors,
        fmt="none",
        ecolor="#555555",
        capsize=3.5,
        linewidth=1.8,
    )
    axis.scatter(values, y, c=colours, s=60, zorder=3)
    axis.set_yticks(y, [str(row["model"]) for row in usable], fontsize=9.5)
    axis.tick_params(axis="x", labelsize=9)
    axis.invert_yaxis()
    axis.set_title(title, fontsize=13)
    axis.set_xlabel(direction, fontsize=10.5)
    axis.grid(axis="x", alpha=0.2)


def _fact_values() -> dict[str, float]:
    payload = load_json("final_results_manifest")
    values: dict[str, float] = {}
    for fact in payload["facts"]:
        value = fact["exact_value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values[str(fact["fact_id"])] = float(value)
    return values


def _inference_rows(
    facts: dict[str, float],
    specifications: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prefix, label in specifications:
        rows.append(
            {
                "comparison": label,
                "difference": facts[f"{prefix}.difference"],
                "ci_lower": facts[f"{prefix}.ci_lower"],
                "ci_upper": facts[f"{prefix}.ci_upper"],
                "holm_p": facts[f"{prefix}.holm_p"],
            }
        )
    return rows


def _display_holm_p(value: float) -> str:
    if value == 0.0:
        return r"Holm $p<2\times10^{-4}$"
    return f"Holm p={value:.3g}"


def _difference_panel(
    axis: Axes,
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    y = np.arange(len(rows))
    centres = np.asarray([float(row["difference"]) for row in rows])
    lower = np.asarray([float(row["ci_lower"]) for row in rows])
    upper = np.asarray([float(row["ci_upper"]) for row in rows])
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.1)
    for index, row in enumerate(rows):
        adjusted = float(row["holm_p"])
        significant = adjusted < 0.05
        axis.plot(
            [lower[index], upper[index]],
            [index, index],
            color="#444444",
            linewidth=2.0,
        )
        axis.scatter(
            [centres[index]],
            [index],
            marker="o",
            s=52,
            facecolor="#D55E00" if significant else "white",
            edgecolor="#D55E00",
            linewidth=1.5,
            zorder=3,
        )
        axis.text(
            upper[index],
            index - 0.19,
            _display_holm_p(adjusted),
            fontsize=8.7,
            ha="right",
        )
    axis.set_yticks(y, [str(row["comparison"]) for row in rows], fontsize=9)
    axis.tick_params(axis="x", labelsize=9)
    axis.invert_yaxis()
    axis.set_xlabel(label, fontsize=9.5)
    axis.grid(axis="x", alpha=0.2)


def financial_figure(destination: Path) -> None:
    """Regenerate Figure 2 from frozen facts with presentation-only changes."""

    benchmark_rows = cast(list[dict[str, Any]], load_json("financial_table")["rows"])
    facts = _fact_values()
    inference = {
        "macro_f1": _inference_rows(
            facts,
            (
                (
                    "financial.inference.macro_f1.qrc_vs_logistic_regression",
                    "vs logistic",
                ),
                ("financial.inference.macro_f1.qrc_vs_esn_classifier", "vs ESN"),
            ),
        ),
        "transition_pr_auc": _inference_rows(
            facts,
            (
                (
                    "financial.inference.transition_pr_auc.qrc_vs_regime_persistence",
                    "vs persistence",
                ),
                (
                    "financial.inference.transition_pr_auc.qrc_vs_esn_classifier",
                    "vs ESN",
                ),
            ),
        ),
        "qlike": _inference_rows(
            facts,
            (
                ("financial.inference.qlike.qrc_vs_garch_1_1", "vs GARCH"),
                ("financial.inference.qlike.qrc_vs_esn_regressor", "vs ESN"),
            ),
        ),
        "rmse": _inference_rows(
            facts,
            (
                (
                    "financial.inference.squared_error.qrc_vs_garch_1_1",
                    "vs GARCH",
                ),
                (
                    "financial.inference.squared_error.qrc_vs_esn_regressor",
                    "vs ESN",
                ),
            ),
        ),
    }
    figure = plt.figure(figsize=(13.2, 7.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=(2.1, 1.15))
    metrics = [
        ("macro_f1", "Macro-F1", "higher is better"),
        ("transition_pr_auc", "Transition PR-AUC", "higher is better"),
        ("qlike", "QLIKE", "lower is better"),
        ("rmse", "RMSE", "lower is better"),
    ]
    for column, (metric, title, direction) in enumerate(metrics):
        direct_axis = figure.add_subplot(grid[0, column])
        difference_axis = figure.add_subplot(grid[1, column])
        _direct_metric_panel(
            direct_axis,
            benchmark_rows,
            metric=metric,
            title=title,
            direction=direction,
        )
        difference_label = (
            "QRC - baseline\nsquared-loss difference"
            if metric == "rmse"
            else "QRC - baseline difference"
        )
        _difference_panel(
            difference_axis,
            inference[metric],
            label=difference_label,
        )
    figure.suptitle(
        "Frozen financial test comparison and architecture-level uncertainty",
        fontsize=16,
    )
    _save_figure(figure, destination)


def mnist_figure(destination: Path) -> None:
    """Regenerate Figure 3 from frozen table rows with larger typography."""

    rows = cast(list[dict[str, Any]], load_json("mnist_table")["rows"])
    comparison_rows = [
        {
            "model": row["model_or_condition"],
            "accuracy": row["accuracy"],
            "accuracy_sd": row["accuracy_sd"],
            "macro_f1": row["macro_f1"],
            "macro_f1_sd": row["macro_f1_sd"],
        }
        for row in rows
        if row["section"] == "Model comparison"
    ]
    robustness_labels = {
        "Analytic (seed 2026)": "analytic",
        "2,048 shots (seed 2026)": "2,048 shots",
        "2,048 shots + depolarising 0.01": "depol. 0.01",
        "2,048 shots + measurement flip 0.02": "bit flip 0.02",
    }
    robustness_rows = [
        {
            "condition": robustness_labels[str(row["model_or_condition"])],
            "accuracy": row["accuracy"],
            "macro_f1": row["macro_f1"],
        }
        for row in rows
        if row["section"] == "QRC robustness"
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.9), constrained_layout=True)
    metrics = ("accuracy", "macro_f1")
    offsets = (-0.13, 0.13)
    models = [str(row["model"]) for row in comparison_rows]
    x = np.arange(len(models))
    for metric, offset, marker in zip(metrics, offsets, ("o", "s")):
        axes[0].errorbar(
            x + offset,
            [float(row[metric]) for row in comparison_rows],
            yerr=[float(row.get(f"{metric}_sd") or 0.0) for row in comparison_rows],
            fmt=marker,
            markersize=7,
            linewidth=1.8,
            color="#0072B2" if metric == "accuracy" else "#D55E00",
            capsize=4,
            label="Accuracy" if metric == "accuracy" else "Macro-F1",
        )
    axes[0].set_xticks(x, models, rotation=18, ha="right", fontsize=11)
    axes[0].tick_params(axis="y", labelsize=10)
    axes[0].set_ylabel("test score (higher is better)", fontsize=11.5)
    axes[0].set_title("(a) Frozen model comparison", fontsize=13)
    axes[0].legend(frameon=False, fontsize=11)
    axes[0].grid(axis="y", alpha=0.2)
    conditions = [str(row["condition"]) for row in robustness_rows]
    robustness_x = np.arange(len(conditions))
    for metric, offset, marker in zip(metrics, offsets, ("o", "s")):
        axes[1].plot(
            robustness_x + offset,
            [float(row[metric]) for row in robustness_rows],
            marker=marker,
            markersize=7,
            linewidth=2.0,
            color="#0072B2" if metric == "accuracy" else "#D55E00",
            label="Accuracy" if metric == "accuracy" else "Macro-F1",
        )
    axes[1].set_xticks(
        robustness_x,
        conditions,
        rotation=20,
        ha="right",
        fontsize=11,
    )
    axes[1].tick_params(axis="y", labelsize=10)
    axes[1].set_ylabel("test score (higher is better)", fontsize=11.5)
    axes[1].set_title("(b) QRC seed 2026 robustness", fontsize=13)
    axes[1].legend(frameon=False, fontsize=11)
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle(
        "MNIST benchmark and controlled finite-shot/noise behaviour",
        fontsize=16,
    )
    _save_figure(figure, destination)


def financial_table() -> str:
    payload = load_json("financial_table")
    rows = {row["model"]: row for row in payload["rows"]}
    classification = [
        ("Majority classifier", False),
        ("Regime persistence", False),
        ("Logistic regression", True),
        ("ESN", False),
        ("GARCH(1,1)", False),
        ("QRC mean", False),
    ]
    regression = [
        ("RV persistence", False, False, False),
        ("ESN", False, True, True),
        ("GARCH(1,1)", True, False, False),
        ("QRC mean", False, False, False),
    ]
    lines = [
        r"% Generated by prepare_assets.py from the frozen Stage 2C JSON.",
        r"% SOURCE: paper_assets/tables/publication_table_2_financial_benchmark.json :: rows",
        r"\begin{center}",
        r"\begin{minipage}{\linewidth}",
        r"\centering",
        r"{\small\textbf{Table 1:} Frozen 2024--2025 financial test results. "
        r"Higher is better except for "
        r"QLIKE, RMSE and MAE. QRC is mean across three reservoir seeds; the full-precision "
        r"population standard deviations are retained in the source manifest. Bold marks "
        r"the leading listed point estimate per metric. GARCH has no transition score.\par}",
        r"\smallskip",
        r"\begin{minipage}[t]{0.515\textwidth}",
        r"\centering\small",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Classifier & Macro-F1 & Bal.\ acc. & Trans.\ PR-AUC \\",
        r"\midrule",
    ]
    for model, is_best in classification:
        row = rows[model]
        lines.append(
            f"{model.replace('GARCH(1,1)', r'GARCH(1,1)')} & "
            f"{latex_value(row, 'macro_f1', bold=is_best)} & "
            f"{latex_value(row, 'balanced_accuracy', bold=is_best)} & "
            f"{latex_value(row, 'transition_pr_auc', bold=is_best)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{minipage}\hfill",
            r"\begin{minipage}[t]{0.455\textwidth}",
            r"\centering\small",
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            r"Forecaster & QLIKE & RMSE & MAE & Corr. \\",
            r"\midrule",
        ]
    )
    for model, qlike_best, rmse_best, mae_best in regression:
        row = rows[model]
        lines.append(
            f"{model} & {latex_value(row, 'qlike', bold=qlike_best)} & "
            f"{latex_value(row, 'rmse', bold=rmse_best)} & "
            f"{latex_value(row, 'mae', bold=mae_best)} & "
            f"{latex_value(row, 'correlation', bold=model == 'QRC mean')} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{minipage}",
            r"\end{minipage}",
            r"\end{center}",
            "",
        ]
    )
    return "\n".join(lines)


def mnist_table() -> str:
    payload = load_json("mnist_table")
    rows = {
        row["model_or_condition"]: row
        for row in payload["rows"]
        if row["section"] == "Model comparison"
    }
    order = ["Flattened logistic", "Size-controlled ESN", "Exact QRC mean"]
    lines = [
        r"% Generated by prepare_assets.py from the frozen Stage 2C JSON.",
        r"% SOURCE: paper_assets/tables/publication_table_3_mnist_benchmark.json :: "
        r"rows[section=Model comparison]",
        r"\begin{center}",
        r"\begin{minipage}{\linewidth}",
        r"\centering\small",
        r"\textbf{Table 2:} Genuine-MNIST test results. QRC values are mean $\pm$ population "
        r"standard deviation across the three frozen reservoir seeds; baselines are "
        r"single deterministic fits.\par",
        r"\smallskip",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Model & Accuracy & Macro-F1 & Bal.\ acc. & Macro AUC \\",
        r"\midrule",
    ]
    for model in order:
        row = rows[model]
        bold = model == "Size-controlled ESN"
        lines.append(
            f"{model} & {latex_value(row, 'accuracy', bold=bold)} & "
            f"{latex_value(row, 'macro_f1', bold=bold)} & "
            f"{latex_value(row, 'balanced_accuracy', bold=bold)} & "
            f"{latex_value(row, 'macro_roc_auc', bold=bold)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{minipage}",
            r"\end{center}",
            "",
        ]
    )
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def asset_entry(
    destination: Path,
    source: FrozenInput,
    purpose: str,
    number: str,
    experiment: str,
    *,
    directly_included: bool,
    transformed: bool,
    transformation: str | None,
    transformation_sources: list[FrozenInput] | None = None,
) -> dict[str, Any]:
    return {
        "destination_filename": destination.relative_to(HERE).as_posix(),
        "original_repository_path": source.path,
        "original_sha256": source.sha256,
        "copied_file_sha256": sha256(destination),
        "purpose": purpose,
        "figure_or_table_number": number,
        "source_experiment": experiment,
        "directly_included": directly_included,
        "transformed": transformed,
        "transformation": transformation,
        "transformation_sources": [
            {"path": item.path, "sha256": item.sha256} for item in (transformation_sources or [])
        ],
    }


def main() -> None:
    figures = HERE / "figures"
    tables = HERE / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    financial_figure_path = figures / "financial_comparison.pdf"
    mnist_figure_path = figures / "mnist_benchmark.pdf"
    checked_source(FROZEN_INPUTS["financial_figure"])
    checked_source(FROZEN_INPUTS["mnist_figure"])
    financial_figure(financial_figure_path)
    mnist_figure(mnist_figure_path)

    financial_tex = tables / "financial_results.tex"
    mnist_tex = tables / "mnist_results.tex"
    write_text(financial_tex, financial_table())
    write_text(mnist_tex, mnist_table())

    assets = [
        asset_entry(
            financial_figure_path,
            FROZEN_INPUTS["financial_figure"],
            "Financial test comparison and Stage 2A uncertainty",
            "Figure 2",
            "Frozen financial benchmark and statistical validation",
            directly_included=True,
            transformed=True,
            transformation=(
                "Deterministic paper-local vector regeneration from frozen table and fact "
                "values. Plotted estimates and intervals are unchanged; typography is "
                "enlarged and the stored empirical zero is displayed at the finite "
                "two-sided reporting resolution p<2e-4."
            ),
            transformation_sources=[
                FROZEN_INPUTS["financial_table"],
                FROZEN_INPUTS["final_results_manifest"],
            ],
        ),
        asset_entry(
            mnist_figure_path,
            FROZEN_INPUTS["mnist_figure"],
            "Genuine-MNIST model comparison and controlled robustness",
            "Figure 3",
            "Frozen genuine-MNIST benchmark",
            directly_included=True,
            transformed=True,
            transformation=(
                "Deterministic paper-local vector regeneration from frozen table values; "
                "only figure dimensions, typography, marker size and line weight change."
            ),
            transformation_sources=[FROZEN_INPUTS["mnist_table"]],
        ),
        asset_entry(
            financial_tex,
            FROZEN_INPUTS["financial_table"],
            "Compact two-panel financial benchmark table",
            "Table 1",
            "Frozen financial benchmark",
            directly_included=True,
            transformed=True,
            transformation=(
                "Deterministic field selection and LaTeX formatting by "
                "submission/phase3/prepare_assets.py; no numeric recomputation"
            ),
        ),
        asset_entry(
            mnist_tex,
            FROZEN_INPUTS["mnist_table"],
            "Compact genuine-MNIST model-comparison table",
            "Table 2",
            "Frozen genuine-MNIST benchmark",
            directly_included=True,
            transformed=True,
            transformation=(
                "Deterministic selection of model-comparison rows and LaTeX formatting "
                "by submission/phase3/prepare_assets.py; no numeric recomputation"
            ),
        ),
    ]
    manifest = {
        "schema_version": 1,
        "paper_base_commit": PAPER_BASE_COMMIT,
        "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
        "generator": "submission/phase3/prepare_assets.py",
        "generation_command": "python3 submission/phase3/prepare_assets.py",
        "policy": (
            "Frozen source checksums are verified before transformation. Original frozen "
            "assets are never edited; every paper-local destination is hash-committed."
        ),
        "statistical_display_contract": {
            "affected_fact_id": (
                "financial.inference.transition_pr_auc.qrc_vs_regime_persistence.holm_p"
            ),
            "frozen_exact_value": 0.0,
            "implementation": "src/qtyche_qrc/statistics/bootstrap.py::bootstrap_interval",
            "rule": (
                "The frozen strict empirical two-sided value is zero because no "
                "opposing-sign draw occurred and the implementation adds no pseudocount. "
                "The paper never prints p=0; it reports p<2e-4, the smallest non-zero "
                "two-sided value on a 10,000-draw grid. No inferential result changes."
            ),
            "resampling_repetitions": RESAMPLING_REPETITIONS,
            "two_sided_reporting_resolution": TWO_SIDED_REPORTING_RESOLUTION,
        },
        "assets": assets,
    }
    write_text(
        HERE / "asset_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
