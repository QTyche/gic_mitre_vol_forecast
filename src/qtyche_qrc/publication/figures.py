"""Publication-resolution figures built only from already-frozen table rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy.typing as npt

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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


def _save(figure: Figure, destination: Path, *, dpi: int) -> dict[str, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    png = destination.with_suffix(".png")
    pdf = destination.with_suffix(".pdf")
    figure.savefig(
        png,
        dpi=dpi,
        bbox_inches="tight",
        metadata={"Software": "qtyche-qrc publication asset compiler"},
    )
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "qtyche-qrc publication asset compiler",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return {"png": png, "pdf": pdf}


def _metric_lines(
    axis: Axes,
    x: npt.NDArray[np.int_],
    values: dict[str, npt.NDArray[np.float64]],
    *,
    selected_x: int,
    right_metric: str,
) -> None:
    left = axis
    right = axis.twinx()
    styles = {
        "Macro-F1": ("#0072B2", "o"),
        "Transition PR-AUC": ("#E69F00", "s"),
        "QLIKE": ("#009E73", "^"),
    }
    for label, y in values.items():
        colour, marker = styles[label]
        target = right if label == right_metric else left
        target.plot(x, y, color=colour, marker=marker, linewidth=1.8, label=label)
        chosen = int(np.where(x == selected_x)[0][0])
        target.scatter(
            [x[chosen]],
            [y[chosen]],
            marker="*",
            s=120,
            color=colour,
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )
    left.set_ylabel("classification score (higher is better)")
    right.set_ylabel("QLIKE (lower is better)", color=styles[right_metric][0])
    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    left.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        fontsize=7,
        loc="best",
        frameon=False,
    )
    left.grid(axis="y", alpha=0.2)


def architecture_selection_figure(
    *,
    qubit_rows: list[dict[str, Any]],
    virtual_rows: list[dict[str, Any]],
    destination: Path,
    dpi: int,
) -> dict[str, Path]:
    """Plot validation-only performance/cost evidence for the frozen architecture."""

    figure = plt.figure(figsize=(11.0, 6.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(2.2, 1.0))
    q_metric_axis = figure.add_subplot(grid[0, 0])
    q_cost_axis = figure.add_subplot(grid[1, 0])
    v_metric_axis = figure.add_subplot(grid[0, 1])
    v_cost_axis = figure.add_subplot(grid[1, 1])

    qubit_rows = sorted(qubit_rows, key=lambda row: int(row["n_qubits"]))
    q = np.asarray([int(row["n_qubits"]) for row in qubit_rows])
    _metric_lines(
        q_metric_axis,
        q,
        {
            "Macro-F1": np.asarray([float(row["macro_f1"]) for row in qubit_rows]),
            "Transition PR-AUC": np.asarray(
                [float(row["transition_pr_auc"]) for row in qubit_rows]
            ),
            "QLIKE": np.asarray([float(row["qlike"]) for row in qubit_rows]),
        },
        selected_x=2,
        right_metric="QLIKE",
    )
    q_metric_axis.set_title("(a) Qubit scaling: validation evidence")
    q_metric_axis.set_xticks(q)
    q_metric_axis.axvline(2, color="#666666", linestyle=":", linewidth=1.0)
    q_metric_axis.text(
        2.05,
        q_metric_axis.get_ylim()[1],
        "selected",
        fontsize=8,
        va="top",
    )
    q_cost_axis.plot(
        q,
        [float(row["state_generation_seconds"]) for row in qubit_rows],
        color="#56B4E9",
        marker="o",
        linewidth=1.8,
    )
    q_cost_axis.set_yscale("log")
    q_cost_axis.set_ylabel("state generation (s, log)")
    q_cost_axis.set_xlabel("reservoir qubits")
    q_cost_axis.set_xticks(q)
    q_cost_axis.grid(axis="y", alpha=0.2)
    for row in qubit_rows:
        q_cost_axis.annotate(
            f"d={int(row['feature_dimension'])}",
            (int(row["n_qubits"]), float(row["state_generation_seconds"])),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

    virtual_rows = sorted(virtual_rows, key=lambda row: int(row["virtual_nodes"]))
    v = np.asarray([int(row["virtual_nodes"]) for row in virtual_rows])
    _metric_lines(
        v_metric_axis,
        v,
        {
            "Macro-F1": np.asarray([float(row["macro_f1"]) for row in virtual_rows]),
            "Transition PR-AUC": np.asarray(
                [float(row["transition_pr_auc"]) for row in virtual_rows]
            ),
            "QLIKE": np.asarray([float(row["qlike"]) for row in virtual_rows]),
        },
        selected_x=2,
        right_metric="QLIKE",
    )
    v_metric_axis.set_title("(b) Temporal multiplexing: validation evidence")
    v_metric_axis.set_xticks(v)
    v_metric_axis.axvline(2, color="#666666", linestyle=":", linewidth=1.0)
    v_metric_axis.text(
        2.1,
        v_metric_axis.get_ylim()[1],
        "selected",
        fontsize=8,
        va="top",
    )
    v_cost_axis.plot(
        v,
        [float(row["condition_number"]) for row in virtual_rows],
        color="#CC79A7",
        marker="s",
        linewidth=1.8,
    )
    v_cost_axis.set_yscale("log")
    v_cost_axis.set_ylabel("condition number (log)")
    v_cost_axis.set_xlabel("virtual nodes")
    v_cost_axis.set_xticks(v)
    v_cost_axis.grid(axis="y", alpha=0.2)
    for row in virtual_rows:
        v_cost_axis.annotate(
            f"d={int(row['feature_dimension'])}",
            (int(row["virtual_nodes"]), float(row["condition_number"])),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

    figure.suptitle(
        "Validation-only architecture selection: performance and computational cost",
        fontsize=13,
    )
    return _save(figure, destination, dpi=dpi)


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
        capsize=2.5,
        linewidth=1.3,
    )
    axis.scatter(values, y, c=colours, s=38, zorder=3)
    axis.set_yticks(y, [str(row["model"]) for row in usable], fontsize=7)
    axis.invert_yaxis()
    axis.set_title(title, fontsize=10)
    axis.set_xlabel(direction, fontsize=8)
    axis.grid(axis="x", alpha=0.2)


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
    significant = np.asarray([bool(row["holm_significant"]) for row in rows])
    axis.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
    for index, row in enumerate(rows):
        axis.plot([lower[index], upper[index]], [index, index], color="#444444", linewidth=1.5)
        axis.scatter(
            [centres[index]],
            [index],
            marker="o",
            s=32,
            facecolor="#D55E00" if significant[index] else "white",
            edgecolor="#D55E00",
            linewidth=1.2,
            zorder=3,
        )
        axis.text(
            upper[index],
            index - 0.18,
            f"Holm p={float(row['holm_p']):.3g}",
            fontsize=6.5,
            ha="right",
        )
    axis.set_yticks(y, [str(row["comparison"]) for row in rows], fontsize=6.5)
    axis.invert_yaxis()
    axis.set_xlabel(label, fontsize=7)
    axis.grid(axis="x", alpha=0.2)


def financial_comparison_figure(
    *,
    benchmark_rows: list[dict[str, Any]],
    inference: dict[str, list[dict[str, Any]]],
    destination: Path,
    dpi: int,
) -> dict[str, Path]:
    """Show direct test metrics above selected Stage 2A paired intervals."""

    figure = plt.figure(figsize=(12.0, 6.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=(2.1, 1.1))
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
        label = (
            "QRC - baseline squared-loss difference"
            if metric == "rmse"
            else "QRC - baseline difference"
        )
        _difference_panel(difference_axis, inference[metric], label=label)
    figure.suptitle(
        "Frozen financial test comparison and architecture-level uncertainty",
        fontsize=13,
    )
    return _save(figure, destination, dpi=dpi)


def financial_robustness_figure(
    *,
    shot_rows: list[dict[str, Any]],
    noise_rows: list[dict[str, Any]],
    destination: Path,
    dpi: int,
) -> dict[str, Path]:
    """Plot controlled finite-shot and noise simulations for three headline metrics."""

    figure, axes = plt.subplots(2, 3, figsize=(11.5, 6.5), constrained_layout=True)
    metrics = [
        ("macro_f1", "Macro-F1", "higher is better"),
        ("transition_pr_auc", "Transition PR-AUC", "higher is better"),
        ("qlike", "QLIKE", "lower is better"),
    ]
    shot_labels = ["analytic", "128", "512", "2,048", "8,192"]
    for column, (metric, title, direction) in enumerate(metrics):
        top = axes[0, column]
        selected = [row for row in shot_rows if row["metric"] == metric]
        selected.sort(key=lambda row: int(row["order"]))
        x = np.arange(len(selected))
        top.errorbar(
            x,
            [float(row["value"]) for row in selected],
            yerr=[float(row["sd"]) for row in selected],
            color="#D55E00",
            marker="o",
            capsize=3,
            linewidth=1.7,
        )
        top.set_xticks(x, shot_labels, rotation=25, ha="right", fontsize=7)
        top.set_title(title)
        top.set_ylabel(direction, fontsize=8)
        top.grid(axis="y", alpha=0.2)

        bottom = axes[1, column]
        conditions = [row for row in noise_rows if row["metric"] == metric]
        conditions.sort(key=lambda row: int(row["order"]))
        nx = np.arange(len(conditions))
        bottom.errorbar(
            nx,
            [float(row["value"]) for row in conditions],
            yerr=[float(row["sd"]) for row in conditions],
            color="#0072B2",
            marker="s",
            capsize=3,
            linewidth=1.7,
        )
        bottom.set_xticks(
            nx,
            ["2,048 shots", "+ depol. 0.01", "+ bit flip 0.02"],
            rotation=25,
            ha="right",
            fontsize=7,
        )
        bottom.set_ylabel(direction, fontsize=8)
        bottom.grid(axis="y", alpha=0.2)
    axes[0, 0].text(
        -0.22,
        1.10,
        "(a) Shot-count study",
        transform=axes[0, 0].transAxes,
        fontsize=10,
        fontweight="bold",
    )
    axes[1, 0].text(
        -0.22,
        1.10,
        "(b) Controlled noise at 2,048 shots",
        transform=axes[1, 0].transAxes,
        fontsize=10,
        fontweight="bold",
    )
    figure.suptitle("Final financial QRC robustness (classical simulation)", fontsize=13)
    return _save(figure, destination, dpi=dpi)


def mnist_benchmark_figure(
    *,
    comparison_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
    destination: Path,
    dpi: int,
) -> dict[str, Path]:
    """Show the common MNIST benchmark and seed-2026 finite-shot sensitivity."""

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
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
            color="#0072B2" if metric == "accuracy" else "#D55E00",
            capsize=3,
            label="Accuracy" if metric == "accuracy" else "Macro-F1",
        )
    axes[0].set_xticks(x, models, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("test score (higher is better)")
    axes[0].set_title("(a) Frozen model comparison")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].annotate(
        "ESN highest",
        (1, max(float(comparison_rows[1][metric]) for metric in metrics)),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )

    conditions = [str(row["condition"]) for row in robustness_rows]
    rx = np.arange(len(conditions))
    for metric, offset, marker in zip(metrics, offsets, ("o", "s")):
        axes[1].plot(
            rx + offset,
            [float(row[metric]) for row in robustness_rows],
            marker=marker,
            linewidth=1.5,
            color="#0072B2" if metric == "accuracy" else "#D55E00",
            label="Accuracy" if metric == "accuracy" else "Macro-F1",
        )
    axes[1].set_xticks(rx, conditions, rotation=22, ha="right", fontsize=8)
    axes[1].set_ylabel("test score (higher is better)")
    axes[1].set_title("(b) QRC seed 2026 robustness")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle("MNIST benchmark and controlled finite-shot/noise behaviour", fontsize=13)
    return _save(figure, destination, dpi=dpi)
