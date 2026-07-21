"""Matplotlib-only experiment and baseline visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qtyche_qrc.evaluation.calibration import calibration_bins

SYNTHETIC_WARNING = "SYNTHETIC FIXTURE DATA — NOT A FINANCIAL PERFORMANCE RESULT"


def _title(title: str, synthetic: bool) -> str:
    return f"{title}\nSYNTHETIC FIXTURE DATA" if synthetic else title


def plot_confusion_matrix(matrix: list[list[int]], path: Path, synthetic: bool) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(np.asarray(matrix), cmap="Blues")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
    axis.set_xticks(range(3), ["Low", "Medium", "High"])
    axis.set_yticks(range(3), ["Low", "Medium", "High"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(_title("Regime confusion matrix", synthetic))
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_transition_series(predictions: pd.DataFrame, path: Path, synthetic: bool) -> None:
    figure, axis = plt.subplots(figsize=(10, 3.5))
    dates = pd.to_datetime(predictions["date"])
    axis.plot(dates, predictions["predicted_transition_probability"], label="P(transition)")
    axis.scatter(
        dates[predictions["true_transition"].eq(1)],
        np.ones(predictions["true_transition"].eq(1).sum()),
        marker="|",
        color="black",
        label="Observed transition",
    )
    axis.set_ylim(-0.05, 1.05)
    axis.set_title(_title("Transition probability through time", synthetic))
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_transition_calibration(
    truth: NDArray[np.int_],
    probabilities: NDArray[np.float64],
    path: Path,
    synthetic: bool,
) -> None:
    predicted, observed, _ = calibration_bins(truth, probabilities)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Ideal")
    axis.plot(predicted, observed, marker="o", label="Observed")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted", ylabel="Observed")
    axis.set_title(_title("Transition calibration", synthetic))
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_rv_series(predictions: pd.DataFrame, path: Path, synthetic: bool) -> None:
    figure, axis = plt.subplots(figsize=(10, 3.5))
    dates = pd.to_datetime(predictions["date"])
    axis.plot(dates, predictions["true_rv_5d"], label="True RV")
    axis.plot(dates, predictions["predicted_rv_5d"], label="Predicted RV", alpha=0.8)
    axis.set_title(_title("Five-day realized variance", synthetic))
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_baseline_comparison(
    table: pd.DataFrame,
    metric: str,
    path: Path,
) -> None:
    usable = table.loc[table[metric].notna()].copy()
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(usable["model"], usable[metric])
    axis.tick_params(axis="x", rotation=30)
    synthetic = bool(usable["is_synthetic"].any()) if len(usable) else False
    axis.set_title(_title(f"Baseline {metric} comparison", synthetic))
    axis.set_ylabel(metric)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
