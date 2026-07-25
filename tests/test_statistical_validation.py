import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qtyche_qrc.experiments.statistical_validation import (
    FINANCIAL_SEEDS,
    _direction,
    architecture_level_differences,
    load_statistical_validation_config,
    require_exact_alignment,
    verify_frozen_sources,
)
from qtyche_qrc.statistics.bootstrap import (
    circular_block_bootstrap_indices,
    indices_to_counts,
    stratified_bootstrap_indices,
)
from qtyche_qrc.statistics.hac import hac_mean_test, newey_west_long_run_variance
from qtyche_qrc.statistics.pairwise import (
    classification_metric_distribution,
    holm_adjust,
    mcnemar_exact,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_path() -> Path:
    return _root() / "configs/statistical_validation.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_prediction_alignment_accepts_identical_ordered_dates() -> None:
    reference = pd.DataFrame({"date": ["a", "b"], "truth": [1.0, 2.0]})
    candidate = reference.copy()
    require_exact_alignment(
        reference,
        candidate,
        identity_columns=("date",),
        truth_columns=("truth",),
        candidate_name="candidate",
    )


@pytest.mark.parametrize(
    "candidate",
    [
        pd.DataFrame({"date": ["a"], "truth": [1.0]}),
        pd.DataFrame({"date": ["b", "a"], "truth": [2.0, 1.0]}),
        pd.DataFrame({"date": ["a", "b"], "truth": [1.0, 3.0]}),
    ],
)
def test_prediction_alignment_rejects_incomplete_or_changed_inputs(
    candidate: pd.DataFrame,
) -> None:
    reference = pd.DataFrame({"date": ["a", "b"], "truth": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"alignment|identity|truth"):
        require_exact_alignment(
            reference,
            candidate,
            identity_columns=("date",),
            truth_columns=("truth",),
            candidate_name="candidate",
        )


def test_hac_is_deterministic_and_lag_zero_matches_population_variance() -> None:
    values = np.asarray([1.0, 2.0, 4.0, 3.0, 8.0])
    first = hac_mean_test(values, lag=0)
    second = hac_mean_test(values, lag=0)
    expected_lrv = float(np.sum((values - values.mean()) ** 2) / len(values))

    assert first == second
    assert first["long_run_variance"] == pytest.approx(expected_lrv)
    assert first["hac_standard_error"] == pytest.approx(np.sqrt(expected_lrv / len(values)))


def test_newey_west_uses_bartlett_lag_weights() -> None:
    values = np.asarray([2.0, -1.0, 3.0, 0.0, 4.0])
    centered = values - values.mean()
    gamma_zero = np.dot(centered, centered) / len(values)
    gamma_one = np.dot(centered[1:], centered[:-1]) / len(values)
    gamma_two = np.dot(centered[2:], centered[:-2]) / len(values)
    expected = gamma_zero + 2.0 * ((2.0 / 3.0) * gamma_one + (1.0 / 3.0) * gamma_two)

    assert newey_west_long_run_variance(values, 2) == pytest.approx(expected)


def test_circular_block_bootstrap_is_deterministic_and_keeps_blocks() -> None:
    first = circular_block_bootstrap_indices(11, 8, 4, 2026)
    second = circular_block_bootstrap_indices(11, 8, 4, 2026)

    assert np.array_equal(first, second)
    assert first.shape == (8, 11)
    for row in first:
        for start in range(0, 8, 4):
            assert np.array_equal(
                row[start : start + 4],
                (row[start] + np.arange(4)) % 11,
            )


def test_counts_preserve_paired_observation_multiplicities() -> None:
    indices = circular_block_bootstrap_indices(13, 10, 5, 2026)
    counts = indices_to_counts(indices, 13)
    values = np.arange(13)

    assert np.all(counts.sum(axis=1) == 13)
    assert np.array_equal(
        np.einsum("ij,j->i", counts, values),
        values[indices].sum(axis=1),
    )


def test_architecture_level_averages_differences_not_forecasts() -> None:
    differences = {
        2026: np.asarray([1.0, 2.0]),
        2027: np.asarray([3.0, 4.0]),
        2028: np.asarray([5.0, 6.0]),
    }
    result = architecture_level_differences(differences)

    assert np.array_equal(result, np.asarray([3.0, 4.0]))
    assert tuple(sorted(differences)) == FINANCIAL_SEEDS


def test_architecture_level_rejects_misaligned_missing_metrics() -> None:
    differences = {
        2026: np.asarray([1.0, np.nan]),
        2027: np.asarray([2.0, 3.0]),
        2028: np.asarray([4.0, np.nan]),
    }
    with pytest.raises(ValueError, match="do not align"):
        architecture_level_differences(differences)


def test_holm_adjustment_matches_step_down_definition() -> None:
    adjusted = holm_adjust(np.asarray([0.01, 0.04, 0.03, 0.20]))
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.20])


def test_direction_conventions_are_explicit() -> None:
    assert _direction(-0.2, positive_favours_qrc=False) == "QRC"
    assert _direction(0.2, positive_favours_qrc=False) == "classical baseline"
    assert _direction(0.2, positive_favours_qrc=True) == "QRC"
    assert _direction(-0.2, positive_favours_qrc=True) == "classical baseline"


def test_mcnemar_reports_paired_discordant_counts() -> None:
    truth = np.asarray([0, 0, 1, 1, 0, 1])
    first = np.asarray([0, 0, 1, 0, 1, 1])
    second = np.asarray([1, 0, 0, 1, 0, 1])
    result = mcnemar_exact(truth, first, second)

    assert result["first_correct_second_wrong"] == 2
    assert result["first_wrong_second_correct"] == 2
    assert result["discordant_count"] == 4
    assert result["test_statistic"] == 2


def test_mnist_stratified_bootstrap_preserves_every_class_and_pairing() -> None:
    labels = np.repeat(np.arange(10), 3)
    indices = stratified_bootstrap_indices(labels, 20, 2026)

    assert np.array_equal(indices, stratified_bootstrap_indices(labels, 20, 2026))
    for row in indices:
        assert np.array_equal(np.bincount(labels[row], minlength=10), np.full(10, 3))


def test_invalid_transition_pr_auc_replicates_are_retained_as_nan() -> None:
    truth = np.asarray([0, 0, 1, 1])
    predictions = np.asarray([0, 1, 1, 0])
    scores = np.asarray([0.1, 0.4, 0.8, 0.2])
    counts = np.asarray(
        [
            [2, 2, 0, 0],
            [0, 0, 2, 2],
            [1, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    draws = classification_metric_distribution(
        truth,
        predictions,
        counts,
        "transition_pr_auc",
        class_labels=(0, 1),
        scores=scores,
    )

    assert np.isnan(draws[:2]).all()
    assert np.isfinite(draws[2])


def test_frozen_prediction_checksums_are_preserved_and_outputs_are_isolated() -> None:
    config = load_statistical_validation_config(_config_path())
    before = {source["path"]: source["sha256"] for source in verify_frozen_sources(config)}
    after = {source["path"]: source["sha256"] for source in verify_frozen_sources(config)}

    assert before == after
    assert all(_sha256(_root() / path) == checksum for path, checksum in before.items())
    assert config.output_root == _root() / "results/statistical_validation"
    assert all(
        config.output_root != (_root() / path)
        and config.output_root not in (_root() / path).parents
        for path in before
    )


def test_checksum_preservation_of_every_prior_result_tree() -> None:
    result_root = _root() / "results"
    excluded = result_root / "statistical_validation"

    def checksums() -> dict[str, str]:
        return {
            path.relative_to(result_root).as_posix(): _sha256(path)
            for path in sorted(result_root.rglob("*"))
            if path.is_file() and excluded not in path.parents
        }

    before = checksums()
    verify_frozen_sources(load_statistical_validation_config(_config_path()))
    assert checksums() == before


def test_validation_module_contains_no_model_fit_or_ensemble_path() -> None:
    source = (_root() / "src/qtyche_qrc/experiments/statistical_validation.py").read_text(
        encoding="utf-8"
    )

    assert ".fit(" not in source
    assert "forecast_averaging_used" in source
    assert "forecast_or_probability_averaging_used" in source
