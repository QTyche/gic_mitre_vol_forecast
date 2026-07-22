"""Deterministic label-free memory, nonlinearity, rank, and fading-memory analysis."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from qtyche_qrc.experiments.run import _git_metadata, _write_json
from qtyche_qrc.models.qrc.backends import trace_distance
from qtyche_qrc.models.qrc.encoding import array_checksum
from qtyche_qrc.models.qrc.readout import ridge_solution
from qtyche_qrc.models.qrc.reservoir import QRCConfig, QuantumReservoir
from qtyche_qrc.runtime import runtime_metadata


@dataclass(frozen=True)
class CapacityConfig:
    """Synthetic sequence and linear-readout settings for characterization."""

    analysis_id: str
    project_root: Path
    output_root: Path
    sequence_seed: int
    sequence_length: int
    washout: int
    train_fraction: float
    max_delay: int
    ridge_alpha: float
    autocorrelation_max_lag: int
    contractivity_steps: int
    cross_delay_pairs: tuple[tuple[int, int], ...]
    qrc: QRCConfig
    input_scalings: tuple[float, ...]
    j_strengths: tuple[float, ...]


def squared_correlation_capacity(
    target: NDArray[np.float64], prediction: NDArray[np.float64]
) -> float:
    """Return Corr(target,prediction)^2 with controlled constant handling."""

    truth = np.asarray(target, dtype=float).reshape(-1)
    estimate = np.asarray(prediction, dtype=float).reshape(-1)
    if len(truth) != len(estimate) or len(truth) < 2:
        raise ValueError("capacity vectors must align and contain at least two values")
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError("capacity vectors must be finite")
    truth_centered = truth - truth.mean()
    estimate_centered = estimate - estimate.mean()
    denominator = float(np.linalg.norm(truth_centered) * np.linalg.norm(estimate_centered))
    if denominator <= 1e-15:
        return 1.0 if np.allclose(truth, estimate, atol=1e-12) else 0.0
    correlation = float(np.dot(truth_centered, estimate_centered) / denominator)
    return float(np.clip(correlation * correlation, 0.0, 1.0))


def effective_rank_from_singular_values(
    singular_values: NDArray[np.float64], tolerance: float = 1e-12
) -> dict[str, float | int | None]:
    """Compute entropy effective rank and retained-spectrum conditioning."""

    values = np.asarray(singular_values, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("singular values must be a non-empty finite non-negative vector")
    retained = values[values > tolerance]
    total = float(values.sum())
    if total <= 0 or not len(retained):
        return {
            "effective_rank": 0.0,
            "numerical_rank": 0,
            "largest_singular_value": float(values.max(initial=0.0)),
            "smallest_retained_singular_value": None,
            "condition_number": None,
            "rank_tolerance": tolerance,
        }
    probabilities = values[values > 0] / total
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return {
        "effective_rank": effective_rank,
        "numerical_rank": len(retained),
        "largest_singular_value": float(values[0]),
        "smallest_retained_singular_value": float(retained[-1]),
        "condition_number": float(values[0] / retained[-1]),
        "rank_tolerance": tolerance,
    }


def effective_feature_rank(
    features: NDArray[np.float64], tolerance: float = 1e-12
) -> tuple[dict[str, float | int | None], NDArray[np.float64]]:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("feature-rank input must be a finite matrix")
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.asarray(np.linalg.svd(centered, compute_uv=False), dtype=float)
    return effective_rank_from_singular_values(singular_values, tolerance), singular_values


def _capacity_for_targets(
    features: NDArray[np.float64],
    targets: dict[str, NDArray[np.float64]],
    *,
    train_fraction: float,
    ridge_alpha: float,
) -> dict[str, float]:
    split = int(len(features) * train_fraction)
    if split <= 1 or split >= len(features) - 1:
        raise ValueError("capacity train_fraction leaves an empty chronological partition")
    results: dict[str, float] = {}
    for name, target in targets.items():
        weights = ridge_solution(features[:split], target[:split, None], ridge_alpha).reshape(-1)
        design = np.column_stack((np.ones(len(features) - split, dtype=float), features[split:]))
        prediction = np.asarray(np.einsum("ij,j->i", design, weights), dtype=float)
        results[name] = squared_correlation_capacity(target[split:], prediction)
    return results


def delayed_capacities(
    features: NDArray[np.float64],
    inputs: NDArray[np.float64],
    *,
    washout: int,
    max_delay: int,
    train_fraction: float,
    ridge_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate linear delayed inputs and delayed second Legendre targets."""

    start = washout + max_delay
    if start >= len(inputs) - 2:
        raise ValueError("washout and maximum delay remove the capacity sequence")
    indices = np.arange(start, len(inputs))
    usable_features = features[indices]
    linear_targets = {
        str(delay): np.asarray(inputs[indices - delay], dtype=float)
        for delay in range(1, max_delay + 1)
    }
    quadratic_targets = {
        str(delay): np.asarray(0.5 * (3.0 * inputs[indices - delay] ** 2 - 1.0), dtype=float)
        for delay in range(1, max_delay + 1)
    }
    linear = _capacity_for_targets(
        usable_features,
        linear_targets,
        train_fraction=train_fraction,
        ridge_alpha=ridge_alpha,
    )
    quadratic = _capacity_for_targets(
        usable_features,
        quadratic_targets,
        train_fraction=train_fraction,
        ridge_alpha=ridge_alpha,
    )
    return (
        pd.DataFrame(
            {"delay": [int(key) for key in linear], "linear_memory_capacity": list(linear.values())}
        ),
        pd.DataFrame(
            {
                "delay": [int(key) for key in quadratic],
                "quadratic_capacity": list(quadratic.values()),
            }
        ),
    )


def cross_delay_capacities(
    features: NDArray[np.float64],
    inputs: NDArray[np.float64],
    *,
    washout: int,
    max_delay: int,
    pairs: tuple[tuple[int, int], ...],
    train_fraction: float,
    ridge_alpha: float,
) -> pd.DataFrame:
    """Reconstruct configured cross-delay products u(t-k)u(t-l)."""

    start = washout + max_delay
    indices = np.arange(start, len(inputs))
    targets: dict[str, NDArray[np.float64]] = {}
    for first, second in pairs:
        if first == second or min(first, second) < 1 or max(first, second) > max_delay:
            raise ValueError(f"invalid cross-delay pair: {(first, second)}")
        targets[f"{first}:{second}"] = np.asarray(
            inputs[indices - first] * inputs[indices - second], dtype=float
        )
    values = _capacity_for_targets(
        features[indices], targets, train_fraction=train_fraction, ridge_alpha=ridge_alpha
    )
    return pd.DataFrame(
        [
            {
                "first_delay": int(name.split(":")[0]),
                "second_delay": int(name.split(":")[1]),
                "cross_delay_capacity": capacity,
            }
            for name, capacity in values.items()
        ]
    )


def feature_autocorrelation(features: NDArray[np.float64], max_lag: int) -> pd.DataFrame:
    """Return mean absolute per-feature autocorrelation at each lag."""

    values = np.asarray(features, dtype=float)
    rows: list[dict[str, float | int]] = []
    for lag in range(1, max_lag + 1):
        correlations: list[float] = []
        for column in range(values.shape[1]):
            first = values[:-lag, column]
            second = values[lag:, column]
            if np.std(first) <= 1e-15 or np.std(second) <= 1e-15:
                continue
            correlations.append(abs(float(np.corrcoef(first, second)[0, 1])))
        rows.append(
            {
                "lag": lag,
                "mean_absolute_autocorrelation": float(np.mean(correlations))
                if correlations
                else 0.0,
                "nonconstant_feature_count": len(correlations),
            }
        )
    return pd.DataFrame(rows)


def empirical_contractivity(
    config: QRCConfig, inputs: NDArray[np.float64]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drive two valid initial states identically and measure trace distance."""

    first = QuantumReservoir(1, config)
    second = QuantumReservoir(1, config)
    dimension = 2**config.n_qubits
    alternate = np.zeros((dimension, dimension), dtype=complex)
    alternate[-1, -1] = 1.0
    second.set_state(np.asarray(alternate, dtype=complex))
    distances = [trace_distance(first.get_state(), second.get_state())]
    for value in inputs:
        row = np.asarray([value], dtype=float)
        first.step(row)
        second.step(row)
        distances.append(trace_distance(first.get_state(), second.get_state()))
    curve = pd.DataFrame({"step": np.arange(len(distances)), "trace_distance": distances})
    positive = np.asarray(distances, dtype=float) > 1e-14
    decay_rate: float | None = None
    if int(positive.sum()) >= 3:
        indices = np.arange(len(distances), dtype=float)[positive]
        decay_rate = float(np.polyfit(indices, np.log(np.asarray(distances)[positive]), 1)[0])
    changes = np.diff(np.asarray(distances, dtype=float))
    non_monotonic = np.where(changes > 1e-10)[0] + 1
    initial = float(distances[0])
    final = float(distances[-1])
    summary = {
        "initial_trace_distance": initial,
        "final_trace_distance": final,
        "final_to_initial_distance_ratio": final / initial if initial > 0 else None,
        "fitted_log_distance_decay_rate_per_step": decay_rate,
        "distance_decreases_overall": final < initial,
        "non_monotonic_interval_count": len(non_monotonic),
        "non_monotonic_interval_end_steps": [int(value) for value in non_monotonic],
        "interpretation_scope": "empirical fading-memory evidence for this tested configuration",
        "formal_global_contraction_claim": False,
    }
    return curve, summary


def _load_capacity_config(path: Path) -> tuple[CapacityConfig, dict[str, Any]]:
    source = path.resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("QRC capacity configuration schema_version must be 1")
    analysis = raw.get("analysis")
    qrc_raw = raw.get("qrc")
    ablation = raw.get("ablation")
    if (
        not isinstance(analysis, dict)
        or not isinstance(qrc_raw, dict)
        or not isinstance(ablation, dict)
    ):
        raise ValueError("capacity configuration requires analysis, qrc, and ablation mappings")
    project_root = (source.parent / str(analysis["project_root"])).resolve()
    qrc_values = dict(qrc_raw)
    if "chords" in qrc_values:
        qrc_values["chords"] = tuple(tuple(edge) for edge in qrc_values["chords"])
    qrc_config = QRCConfig(**qrc_values)
    qrc_config.validate()
    config = CapacityConfig(
        analysis_id=str(analysis["id"]),
        project_root=project_root,
        output_root=(project_root / str(analysis["output_root"])).resolve(),
        sequence_seed=int(analysis.get("sequence_seed", 2026)),
        sequence_length=int(analysis.get("sequence_length", 800)),
        washout=int(analysis.get("washout", 100)),
        train_fraction=float(analysis.get("train_fraction", 0.6)),
        max_delay=int(analysis.get("max_delay", 10)),
        ridge_alpha=float(analysis.get("ridge_alpha", 1e-6)),
        autocorrelation_max_lag=int(analysis.get("autocorrelation_max_lag", 10)),
        contractivity_steps=int(analysis.get("contractivity_steps", 150)),
        cross_delay_pairs=tuple(
            (int(pair[0]), int(pair[1])) for pair in analysis["cross_delay_pairs"]
        ),
        qrc=qrc_config,
        input_scalings=tuple(float(value) for value in ablation["input_scaling"]),
        j_strengths=tuple(float(value) for value in ablation["j_strength"]),
    )
    if config.sequence_length <= config.washout + config.max_delay + 10:
        raise ValueError("capacity sequence is too short")
    if not 0 < config.train_fraction < 1 or config.ridge_alpha <= 0:
        raise ValueError("capacity readout settings are invalid")
    return config, raw


def _analyze_one(
    qrc_config: QRCConfig, inputs: NDArray[np.float64], config: CapacityConfig
) -> dict[str, Any]:
    reservoir = QuantumReservoir(1, qrc_config)
    features = reservoir.transform(inputs[:, None], reset=True)
    linear, quadratic = delayed_capacities(
        features,
        inputs,
        washout=config.washout,
        max_delay=config.max_delay,
        train_fraction=config.train_fraction,
        ridge_alpha=config.ridge_alpha,
    )
    cross = cross_delay_capacities(
        features,
        inputs,
        washout=config.washout,
        max_delay=config.max_delay,
        pairs=config.cross_delay_pairs,
        train_fraction=config.train_fraction,
        ridge_alpha=config.ridge_alpha,
    )
    rank, singular_values = effective_feature_rank(features[config.washout :])
    autocorrelation = feature_autocorrelation(
        features[config.washout :], config.autocorrelation_max_lag
    )
    contractivity, contractivity_summary = empirical_contractivity(
        qrc_config, inputs[: config.contractivity_steps]
    )
    return {
        "features": features,
        "linear": linear,
        "quadratic": quadratic,
        "cross": cross,
        "rank": rank,
        "singular_values": singular_values,
        "autocorrelation": autocorrelation,
        "contractivity": contractivity,
        "contractivity_summary": contractivity_summary,
        "reservoir": reservoir,
    }


def _plot_capacity_outputs(
    output_dir: Path, primary: dict[str, Any], ablation: pd.DataFrame
) -> None:
    figures = output_dir / "figures"
    figures.mkdir()
    figure, axis = plt.subplots(figsize=(5.5, 3.5))
    linear = primary["linear"]
    axis.bar(linear["delay"], linear["linear_memory_capacity"])
    axis.set(xlabel="delay", ylabel="linear memory capacity")
    figure.tight_layout()
    figure.savefig(figures / "linear_memory_by_delay.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 3.5))
    quadratic = primary["quadratic"]
    axis.bar(quadratic["delay"], quadratic["quadratic_capacity"])
    axis.set(xlabel="delay", ylabel="quadratic capacity")
    figure.tight_layout()
    figure.savefig(figures / "quadratic_capacity_by_delay.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 3.8))
    scatter = axis.scatter(
        ablation["total_linear_memory"],
        ablation["total_quadratic_capacity"],
        c=ablation["input_scaling"],
        s=45 + 20 * ablation["j_strength"],
    )
    axis.set(xlabel="total linear memory", ylabel="total quadratic capacity")
    figure.colorbar(scatter, ax=axis, label="input scaling")
    figure.tight_layout()
    figure.savefig(figures / "memory_nonlinearity_ablation.png", dpi=160)
    plt.close(figure)

    labels = [f"a={row.input_scaling:g}, J={row.j_strength:g}" for row in ablation.itertuples()]
    figure, axis = plt.subplots(figsize=(8, 3.8))
    axis.bar(np.arange(len(ablation)), ablation["effective_rank"])
    axis.set_xticks(np.arange(len(ablation)), labels, rotation=45, ha="right")
    axis.set_ylabel("effective rank")
    figure.tight_layout()
    figure.savefig(figures / "effective_rank_by_configuration.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 3.5))
    curve = primary["contractivity"]
    axis.plot(curve["step"], curve["trace_distance"])
    axis.set(xlabel="input step", ylabel="trace distance")
    figure.tight_layout()
    figure.savefig(figures / "empirical_trace_distance.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 3.5))
    singular = primary["singular_values"]
    axis.semilogy(np.arange(1, len(singular) + 1), singular, marker="o")
    axis.set(xlabel="singular-value index", ylabel="singular value")
    figure.tight_layout()
    figure.savefig(figures / "qrc_feature_singular_values.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 3.8))
    trajectories = primary["features"][:150, : min(4, primary["features"].shape[1])]
    axis.plot(trajectories)
    axis.set(xlabel="input step", ylabel="reservoir feature")
    figure.tight_layout()
    figure.savefig(figures / "selected_feature_trajectories.png", dpi=160)
    plt.close(figure)


def characterize_qrc(config_path: Path) -> Path:
    """Run the primary capacity analysis and the fixed 3x3 analytical ablation."""

    config, raw = _load_capacity_config(config_path)
    output_dir = config.output_root / config.analysis_id
    if output_dir.exists():
        raise FileExistsError(f"QRC capacity analysis already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    shutil.copyfile(config_path, output_dir / "config.yaml")
    rng = np.random.default_rng(config.sequence_seed)
    inputs = np.asarray(rng.uniform(-1.0, 1.0, config.sequence_length), dtype=float)
    primary = _analyze_one(config.qrc, inputs, config)
    primary["linear"].to_csv(output_dir / "linear_memory_by_delay.csv", index=False)
    primary["quadratic"].to_csv(output_dir / "quadratic_capacity_by_delay.csv", index=False)
    primary["cross"].to_csv(output_dir / "cross_delay_capacity.csv", index=False)
    pd.DataFrame(
        {
            "index": np.arange(1, len(primary["singular_values"]) + 1),
            "singular_value": primary["singular_values"],
        }
    ).to_csv(output_dir / "singular_values.csv", index=False)
    _write_json(output_dir / "feature_rank.json", primary["rank"])
    primary["contractivity"].to_csv(output_dir / "contractivity.csv", index=False)
    _write_json(output_dir / "contractivity_summary.json", primary["contractivity_summary"])
    primary["autocorrelation"].to_csv(output_dir / "feature_autocorrelation.csv", index=False)

    ablation_rows: list[dict[str, Any]] = []
    ablation_base = replace(
        config.qrc,
        n_qubits=4,
        virtual_nodes=2,
        h_strength=1.0,
        tau=1.0,
    )
    for input_scaling in config.input_scalings:
        for j_strength in config.j_strengths:
            candidate = replace(ablation_base, input_scaling=input_scaling, j_strength=j_strength)
            result = _analyze_one(candidate, inputs, config)
            ablation_rows.append(
                {
                    "input_scaling": input_scaling,
                    "j_strength": j_strength,
                    "reservoir_seed": candidate.reservoir_seed,
                    "total_linear_memory": float(result["linear"]["linear_memory_capacity"].sum()),
                    "total_quadratic_capacity": float(
                        result["quadratic"]["quadratic_capacity"].sum()
                    ),
                    "total_cross_delay_capacity": float(
                        result["cross"]["cross_delay_capacity"].sum()
                    ),
                    "effective_rank": result["rank"]["effective_rank"],
                    "condition_number": result["rank"]["condition_number"],
                    "final_trace_distance_ratio": result["contractivity_summary"][
                        "final_to_initial_distance_ratio"
                    ],
                }
            )
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(output_dir / "input_scaling_interaction_ablation.csv", index=False)
    _plot_capacity_outputs(output_dir, primary, ablation)
    reservoir: QuantumReservoir = primary["reservoir"]
    manifest = {
        "schema_version": 1,
        **runtime_metadata(),
        "analysis_id": config.analysis_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(config.project_root),
        "configuration_checksum": hashlib.sha256(
            (output_dir / "config.yaml").read_bytes()
        ).hexdigest(),
        "configuration": raw,
        "synthetic_input": True,
        "financial_target_labels_consumed": False,
        "sequence_seed": config.sequence_seed,
        "sequence_checksum": array_checksum(inputs),
        "qrc_configuration": asdict(config.qrc),
        "qrc_configuration_checksum": config.qrc.checksum,
        "backend_metadata": reservoir.resource_metadata(),
        "numerical_diagnostics": reservoir.numerical_diagnostics(),
        "total_linear_memory": float(primary["linear"]["linear_memory_capacity"].sum()),
        "total_quadratic_capacity": float(primary["quadratic"]["quadratic_capacity"].sum()),
        "total_cross_delay_capacity": float(primary["cross"]["cross_delay_capacity"].sum()),
        "effective_feature_rank": primary["rank"]["effective_rank"],
        "feature_condition_number": primary["rank"]["condition_number"],
        "contractivity_summary": primary["contractivity_summary"],
        "ablation_configuration_count": len(ablation),
        "ablation_seed_count": 1,
        "ablation_used_for_financial_model_selection": False,
        "status": "success",
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir
