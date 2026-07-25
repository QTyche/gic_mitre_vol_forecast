import hashlib
from pathlib import Path

import numpy as np
import pytest

from qtyche_qrc.diagnostics.calibration import (
    calibration_bin_assignments,
    calibration_errors,
    multiclass_calibration_summary,
    reliability_table,
)
from qtyche_qrc.diagnostics.regimes import (
    assess_lead_time_identifiability,
    fixed_regime_labels,
    transition_type_diagnostics,
    variance_regime_diagnostics,
)
from qtyche_qrc.diagnostics.temporal import (
    cumulative_loss_difference,
    rolling_variance_diagnostics,
)
from qtyche_qrc.experiments.benchmark_diagnostics import (
    QRC_SEEDS,
    _generated_output_manifest,
    _load_financial_splits,
    _load_mnist,
    _mnist_tables,
    load_benchmark_diagnostics_config,
    verify_frozen_diagnostic_sources,
)
from qtyche_qrc.statistics.bootstrap import (
    circular_block_bootstrap_indices,
    stratified_bootstrap_indices,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_path() -> Path:
    return _root() / "configs/benchmark_diagnostics.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_exact_frozen_identity_and_checksum_validation() -> None:
    config = load_benchmark_diagnostics_config(_config_path())
    records = verify_frozen_diagnostic_sources(config)

    assert records
    assert all(_sha256(_root() / row["path"]) == row["sha256"] for row in records)
    assert config.data_snapshot_id == "yahoo_chart_20100101_20251231_v1"


def test_calibration_bin_assignment_includes_boundaries() -> None:
    probabilities = np.asarray([0.0, 0.099, 0.1, 0.5, 0.999, 1.0])
    assert np.array_equal(
        calibration_bin_assignments(probabilities, 10),
        np.asarray([0, 0, 1, 5, 9, 9]),
    )


def test_reliability_table_records_empty_bins_explicitly() -> None:
    rows = reliability_table(
        np.asarray([0, 1]),
        np.asarray([0.1, 0.9]),
        5,
    )

    assert len(rows) == 5
    assert sum(row["observation_count"] for row in rows) == 2
    assert sum(row["empty_bin"] for row in rows) == 3
    assert all(
        row["mean_probability"] is None and row["event_rate"] is None
        for row in rows
        if row["empty_bin"]
    )


def test_multiclass_brier_score_is_sum_of_classwise_squared_errors() -> None:
    summary = multiclass_calibration_summary(
        np.asarray([0, 1]),
        np.asarray([[0.8, 0.2], [0.4, 0.6]]),
        bin_count=5,
    )
    assert summary["multiclass_brier_score"] == pytest.approx(0.2)
    assert summary["multiclass_log_loss"] == pytest.approx(-0.5 * (np.log(0.8) + np.log(0.6)))


def test_expected_and_maximum_calibration_errors_use_observation_weights() -> None:
    rows = reliability_table(
        np.asarray([1, 1]),
        np.asarray([0.6, 0.8]),
        10,
    )
    ece, mce = calibration_errors(rows)
    assert ece == pytest.approx(0.3)
    assert mce == pytest.approx(0.4)


def test_regime_labels_use_only_fixed_training_thresholds() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 0.4])
    labels = fixed_regime_labels(values, low_medium=0.2, medium_high=0.3)
    assert np.array_equal(labels, np.asarray([0, 0, 1, 2]))


def test_training_tail_thresholds_are_not_derived_from_test() -> None:
    config = load_benchmark_diagnostics_config(_config_path())
    splits, thresholds, payload = _load_financial_splits(config)
    training = np.genfromtxt(
        config.financial_training.path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    assert payload["fit_split"] == "train"
    assert thresholds["training_p90"] == pytest.approx(np.quantile(training["target_rv_5d"], 0.90))
    assert thresholds["training_p95"] == pytest.approx(np.quantile(training["target_rv_5d"], 0.95))
    assert thresholds["training_p95"] != pytest.approx(
        np.quantile(splits["test"].realised_variance, 0.95)
    )


def test_regime_conditioned_metrics_preserve_masks() -> None:
    truth = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    forecasts = np.asarray([1.0, 3.0, 2.0, 5.0, 4.0, 7.0])
    regimes = np.asarray([0, 0, 1, 1, 2, 2])
    rows = variance_regime_diagnostics(truth, forecasts, regimes)

    assert [row["sample_count"] for row in rows] == [2, 2, 2]
    assert rows[0]["bias"] == pytest.approx(0.5)
    assert rows[1]["bias"] == pytest.approx(0.0)
    assert rows[2]["bias"] == pytest.approx(0.0)


def test_transition_types_use_supplied_scores_without_derivation() -> None:
    rows = transition_type_diagnostics(
        np.asarray([0, 0, 1, 1, 2]),
        np.asarray([1, 2, 0, 1, 0]),
        np.asarray([1, 0, 0, 1, 2]),
        np.asarray([0.2, 0.4, 0.6, 0.1, 0.8]),
    )

    assert {row["transition_type"] for row in rows} == {
        "low_to_medium",
        "low_to_high",
        "medium_to_low",
        "high_to_low",
    }
    low_high = next(row for row in rows if row["transition_type"] == "low_to_high")
    assert low_high["mean_supplied_transition_score"] == pytest.approx(0.4)
    assert low_high["derived_transition_score"] is False


def test_lead_time_analysis_refuses_ambiguous_aggregate_target() -> None:
    result = assess_lead_time_identifiability(
        prediction_columns={"date", "true_regime", "predicted_regime"},
        target_horizon=5,
        target_definition="realised-variance regime target",
    )

    assert result["identifiable"] is False
    assert result["performed"] is False
    assert result["lead_times_inferred"] is False
    assert "transition_date" in result["missing_columns"]


def test_rolling_window_alignment_uses_window_end_date() -> None:
    dates = np.asarray(["d0", "d1", "d2", "d3"])
    truth = np.asarray([1.0, 2.0, 3.0, 4.0])
    forecast = np.asarray([1.0, 2.0, 4.0, 2.0])
    rows = rolling_variance_diagnostics(dates, truth, forecast, window=3)

    assert [row["date"] for row in rows] == ["d2", "d3"]
    assert rows[0]["window_start_date"] == "d0"
    assert rows[0]["window_end_date"] == "d2"
    assert rows[0]["rolling_bias"] == pytest.approx(1.0 / 3.0)


def test_cumulative_loss_difference_direction_is_model_minus_reference() -> None:
    result = cumulative_loss_difference(
        np.asarray([1.0, 1.0, 1.0]),
        np.asarray([2.0, 0.0, 2.0]),
    )
    assert np.array_equal(result, np.asarray([-1.0, 0.0, -1.0]))


def test_all_three_qrc_seeds_are_preserved() -> None:
    config = load_benchmark_diagnostics_config(_config_path())
    assert tuple(sorted(config.qrc_numerical)) == QRC_SEEDS
    assert {
        model.reservoir_seed
        for model in config.regression_models.values()
        if model.reservoir_seed is not None
    } == set(QRC_SEEDS)


def test_mnist_paired_predictions_have_exact_official_identity_alignment() -> None:
    data = _load_mnist(load_benchmark_diagnostics_config(_config_path()))

    assert len(data.identities) == 1000
    assert {partition for partition, _ in data.identities} == {"official_test"}
    assert all(len(model.predictions) == 1000 for model in data.models.values())
    assert all(len(model.predictions) == 1000 for model in data.robustness.values())


def test_mnist_gain_and_loss_digits_use_paired_per_digit_accuracy() -> None:
    config = load_benchmark_diagnostics_config(_config_path())
    data = _load_mnist(config)
    truth = data.truth
    qrc_correct = data.models["qrc_2026"].predictions == truth
    baseline_correct = data.models["flattened_logistic"].predictions == truth
    differences = {
        digit: float(qrc_correct[truth == digit].mean() - baseline_correct[truth == digit].mean())
        for digit in range(10)
    }

    largest_gain_digit = max(differences, key=lambda digit: (differences[digit], -digit))
    largest_loss_digit = min(differences, key=lambda digit: (differences[digit], digit))
    rows = _mnist_tables(config, data, repetitions=20)["mnist_pairwise_error_overlap"]
    row = next(
        row
        for row in rows
        if row["qrc_model"] == "qrc_2026" and row["baseline"] == "flattened_logistic"
    )

    assert row["per_digit_accuracy_difference"] == {
        str(digit): difference for digit, difference in differences.items()
    }
    assert row["largest_qrc_accuracy_gain_digit"] == largest_gain_digit
    assert row["largest_qrc_accuracy_loss_digit"] == largest_loss_digit
    assert row["largest_qrc_accuracy_gain"] == max(differences.values())
    assert row["largest_qrc_accuracy_loss"] == min(differences.values())


def test_diagnostic_bootstraps_are_deterministic_and_paired() -> None:
    first_blocks = circular_block_bootstrap_indices(40, 20, 10, 2026)
    second_blocks = circular_block_bootstrap_indices(40, 20, 10, 2026)
    labels = np.repeat(np.arange(10), 4)
    first_stratified = stratified_bootstrap_indices(labels, 20, 2026)
    second_stratified = stratified_bootstrap_indices(labels, 20, 2026)

    assert np.array_equal(first_blocks, second_blocks)
    assert np.array_equal(first_stratified, second_stratified)
    for row in first_stratified:
        assert np.array_equal(np.bincount(labels[row], minlength=10), np.full(10, 4))


def test_verification_does_not_mutate_frozen_sources() -> None:
    config = load_benchmark_diagnostics_config(_config_path())
    before = {row["path"]: row["sha256"] for row in verify_frozen_diagnostic_sources(config)}
    after = {row["path"]: row["sha256"] for row in verify_frozen_diagnostic_sources(config)}
    assert after == before


def test_generated_output_manifest_records_exact_checksums(tmp_path: Path) -> None:
    output = tmp_path / "diagnostic.csv"
    output.write_bytes(b"frozen diagnostic output\n")

    records = _generated_output_manifest(tmp_path, {"diagnostic_csv": output.name})

    assert records == [
        {
            "output_id": "diagnostic_csv",
            "path": "diagnostic.csv",
            "sha256": _sha256(output),
            "bytes": output.stat().st_size,
        }
    ]


def test_checksum_preservation_of_every_prior_result_tree() -> None:
    result_root = _root() / "results"
    excluded = result_root / "benchmark_diagnostics"

    def checksums() -> dict[str, str]:
        return {
            path.relative_to(result_root).as_posix(): _sha256(path)
            for path in sorted(result_root.rglob("*"))
            if path.is_file() and excluded not in path.parents
        }

    before = checksums()
    verify_frozen_diagnostic_sources(load_benchmark_diagnostics_config(_config_path()))
    assert checksums() == before


def test_experiment_contains_no_fit_recalibration_or_ensemble_path() -> None:
    source = (_root() / "src/qtyche_qrc/experiments/benchmark_diagnostics.py").read_text(
        encoding="utf-8"
    )

    assert ".fit(" not in source
    assert "probabilities_recalibrated" in source
    assert "ensembles_created" in source
    assert '"models_fitted": False' in source
