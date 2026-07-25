import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from qtyche_qrc.data.download import sha256_file
from qtyche_qrc.evaluation.metrics import regression_metrics
from qtyche_qrc.experiments.garch_baseline import (
    STUDY_ID,
    _execute_garch_run,
    discover_completed_garch_runs,
    load_garch_return_inputs,
    load_garch_study_config,
    pending_garch_modes,
    verify_garch_public_data,
)
from qtyche_qrc.experiments.run import SyntheticResultsError
from qtyche_qrc.models.baselines.garch import GaussianGARCH11


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fitted_models() -> tuple[GaussianGARCH11, GaussianGARCH11, np.ndarray]:
    rng = np.random.default_rng(918)
    innovations = rng.normal(size=500)
    returns = np.empty_like(innovations)
    variance = 0.0001
    for index, innovation in enumerate(innovations):
        returns[index] = 0.0003 + np.sqrt(variance) * innovation
        variance = 0.000003 + 0.08 * (returns[index] - 0.0003) ** 2 + 0.88 * variance
    first = GaussianGARCH11()
    second = GaussianGARCH11()
    first.fit(returns, maximum_starts=3)
    second.fit(returns, maximum_starts=3)
    return first, second, returns


@pytest.fixture(scope="module")
def public_smoke_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Any, Any, Any]:
    config = load_garch_study_config(_root() / "configs/garch_baseline.yaml")
    dataset, provenance = verify_garch_public_data(config)
    inputs = load_garch_return_inputs(config, dataset)
    isolated = replace(
        config,
        output_root=tmp_path_factory.mktemp("garch_results"),
    )
    experiment_dir = _execute_garch_run(
        isolated,
        dataset,
        inputs,
        smoke=True,
        data_provenance=provenance,
    )
    return experiment_dir, isolated, dataset, inputs


def test_parameter_constraints_and_lowest_converged_start(
    fitted_models: tuple[GaussianGARCH11, GaussianGARCH11, np.ndarray],
) -> None:
    model, _, _ = fitted_models
    fit = model.fit_result
    assert fit is not None
    parameters = fit.parameters

    assert parameters.omega > 0
    assert parameters.alpha >= 0
    assert parameters.beta >= 0
    assert parameters.alpha + parameters.beta < 1
    successful = [
        float(attempt.negative_log_likelihood)
        for attempt in fit.attempts
        if attempt.success and attempt.negative_log_likelihood is not None
    ]
    assert fit.training_log_likelihood == pytest.approx(-min(successful))


def test_estimation_is_deterministic(
    fitted_models: tuple[GaussianGARCH11, GaussianGARCH11, np.ndarray],
) -> None:
    first, second, _ = fitted_models

    assert first.fit_result == second.fit_result


def test_all_unconverged_starts_fail_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    returns = np.linspace(-0.01, 0.01, 100)

    def failed_minimize(_function: Any, initial: np.ndarray, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            x=initial,
            fun=1.0,
            success=False,
            status=2,
            message="forced non-convergence",
            nit=3,
            nfev=8,
        )

    monkeypatch.setattr(
        "qtyche_qrc.models.baselines.garch.minimize",
        failed_minimize,
    )

    with pytest.raises(RuntimeError, match="all deterministic GARCH fits failed"):
        GaussianGARCH11().fit(returns, maximum_starts=2)


def test_recursive_one_step_update_is_exact(
    fitted_models: tuple[GaussianGARCH11, GaussianGARCH11, np.ndarray],
) -> None:
    model, _, _ = fitted_models
    parameters = model.parameters
    current_variance = 0.00017
    observed_return = -0.012
    expected = (
        parameters.omega
        + parameters.alpha * (observed_return - parameters.mu) ** 2
        + parameters.beta * current_variance
    )

    assert model.one_step_variance(current_variance, observed_return) == pytest.approx(
        expected,
        rel=0,
        abs=1e-18,
    )


def test_five_step_cumulative_forecast_and_target_units_are_exact(
    fitted_models: tuple[GaussianGARCH11, GaussianGARCH11, np.ndarray],
) -> None:
    model, _, _ = fitted_models
    parameters = model.parameters
    one_day = 0.0002
    manual = [one_day]
    for _ in range(4):
        manual.append(parameters.omega + parameters.persistence * manual[-1])
    cumulative = model.cumulative_variance_forecast(one_day)

    assert cumulative == pytest.approx(sum(manual), rel=0, abs=1e-18)
    assert (model.annualization / model.horizon) * cumulative == pytest.approx(
        (252.0 / 5.0) * sum(manual),
        rel=0,
        abs=1e-18,
    )


def test_forecasts_do_not_depend_on_future_returns(
    fitted_models: tuple[GaussianGARCH11, GaussianGARCH11, np.ndarray],
) -> None:
    model, _, returns = fitted_models
    post_returns = returns[:30].copy()
    changed_future = post_returns.copy()
    changed_future[12:] += 4.0
    initial_variance = model.parameters.unconditional_variance

    reference = model.forecast_sequence(
        post_returns,
        initial_variance=initial_variance,
    )
    perturbed = model.forecast_sequence(
        changed_future,
        initial_variance=initial_variance,
    )

    assert np.array_equal(
        reference.filtered_variance_at_origin[:12], perturbed.filtered_variance_at_origin[:12]
    )
    assert np.array_equal(reference.one_day_variance[:12], perturbed.one_day_variance[:12])
    assert np.array_equal(
        reference.five_day_cumulative_variance[:12],
        perturbed.five_day_cumulative_variance[:12],
    )


def test_numerical_floor_accounting_is_explicit() -> None:
    evaluation = regression_metrics(
        np.asarray([0.01, 0.02, 0.03]),
        np.asarray([0.02, 0.0, np.nan]),
        epsilon=1e-12,
    )

    assert evaluation.metrics["non_finite_prediction_count"] == 1
    assert evaluation.metrics["floored_prediction_count"] == 2
    assert evaluation.metrics["prediction_floor"] == 1e-12
    assert np.array_equal(evaluation.floored, [False, True, True])


def test_public_prediction_alignment_and_target_unit_consistency(
    public_smoke_run: tuple[Path, Any, Any, Any],
) -> None:
    experiment_dir, _, dataset, _ = public_smoke_run
    for split_name, split in (
        ("validation", dataset.validation),
        ("test", dataset.test),
    ):
        predictions = pd.read_csv(
            experiment_dir / f"{split_name}_predictions.csv",
            parse_dates=["date"],
        )
        expected_dates = pd.to_datetime(split.dates)

        assert np.array_equal(predictions["date"].to_numpy(), expected_dates.to_numpy())
        assert np.allclose(predictions["true_rv_5d"], split.y_rv, rtol=0, atol=0)
        assert np.allclose(
            predictions["raw_predicted_rv_5d"],
            (252.0 / 5.0) * predictions["five_day_cumulative_variance"],
            rtol=0,
            atol=1e-12,
        )


def test_parameter_fit_is_training_only_and_frozen(
    public_smoke_run: tuple[Path, Any, Any, Any],
) -> None:
    experiment_dir, _, _, inputs = public_smoke_run
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    fitted = json.loads((experiment_dir / "fitted_parameters.json").read_text(encoding="utf-8"))

    assert manifest["training_only_parameter_fit"] is True
    assert manifest["parameters_frozen_after_training"] is True
    assert manifest["validation_or_test_parameter_refit"] is False
    assert manifest["future_return_lookahead"] is False
    assert manifest["post_training_filter_consumes_targets"] is False
    assert fitted["fit_split"] == "train"
    assert pd.Timestamp(fitted["fit_date_end"]) <= pd.Timestamp("2020-12-31")
    assert inputs.training_dates.max() < inputs.post_training_dates.min()


def test_public_data_guard_rejects_synthetic_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_garch_study_config(_root() / "configs/garch_baseline.yaml")
    fake_dataset = SimpleNamespace(is_synthetic=True, data_source_type="fixture")
    monkeypatch.setattr(
        "qtyche_qrc.experiments.garch_baseline.verify_public_snapshot",
        lambda _config: {
            "snapshot_id": "yahoo_chart_20100101_20251231_v1",
            "files": {},
        },
    )
    monkeypatch.setattr(
        "qtyche_qrc.experiments.garch_baseline.load_model_dataset",
        lambda _processed_dir: fake_dataset,
    )

    with pytest.raises(SyntheticResultsError, match="non-synthetic"):
        verify_garch_public_data(config)


def _fake_completed_run(root: Path, name: str, *, complete: bool) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "experiment_id": name,
        "status": "success",
        "mode": "smoke",
        "data_source_type": "public_market",
        "is_synthetic": False,
        "data_snapshot_id": "yahoo_chart_20100101_20251231_v1",
        "training_only_parameter_fit": True,
        "future_return_lookahead": False,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    required = (
        "fitted_parameters.json",
        "validation_predictions.csv",
        "test_predictions.csv",
        "validation_metrics.json",
        "test_metrics.json",
        "conditional_variance_path.csv",
    )
    for filename in required:
        if complete or filename != "test_metrics.json":
            (directory / filename).write_text("{}\n", encoding="utf-8")
    return directory


def test_partial_resumption_uses_only_complete_matching_run(tmp_path: Path) -> None:
    complete = _fake_completed_run(
        tmp_path,
        "20260101T000000.000000Z_gaussian_garch_1_1_smoke",
        complete=True,
    )
    _fake_completed_run(
        tmp_path,
        "20260102T000000.000000Z_gaussian_garch_1_1_smoke",
        complete=False,
    )

    completed = discover_completed_garch_runs(
        tmp_path,
        study_id=STUDY_ID,
        snapshot_id="yahoo_chart_20100101_20251231_v1",
    )

    assert completed == {"smoke": complete}
    assert pending_garch_modes("smoke", completed) == ()
    assert pending_garch_modes("full", completed) == ("full",)


def test_manifest_records_required_provenance(
    public_smoke_run: tuple[Path, Any, Any, Any],
) -> None:
    experiment_dir, _, _, _ = public_smoke_run
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))

    required = {
        "git",
        "python_version",
        "package_versions",
        "operating_system",
        "execution_platform",
        "configuration",
        "data_snapshot_id",
        "data_manifest_checksum",
        "processed_data_checksums",
        "raw_return_checksum",
        "training_return_checksum",
        "post_training_return_checksum",
        "evaluation_date_checksum",
        "target_definition",
        "fit",
        "timing",
    }
    assert required.issubset(manifest)
    assert manifest["data_source_type"] == "public_market"
    assert manifest["is_synthetic"] is False
    assert manifest["target_definition"]["formula"] == ("(252 / 5) * sum(r_(t+1)^2,...,r_(t+5)^2)")


def test_output_namespace_preserves_existing_result_contracts() -> None:
    root = _root()
    config = load_garch_study_config(root / "configs/garch_baseline.yaml")

    assert config.output_root == root / "results/garch_baseline"
    assert config.output_root not in {
        root / "results/public_market",
        root / "results/qrc_public_pilot",
        root / "results/qrc_qubit_scaling",
        root / "results/qrc_noise_robustness",
        root / "results/qrc_encoding_density",
        root / "results/qrc_state_memory_ablation",
    }
    assert sha256_file(root / "configs/models/qrc_classifier_pilot.yaml") == (
        "65a8b098aa466b2ab0b336cb8f5f6cabe412c43a2e8b27673d427304c24e0002"
    )
    assert sha256_file(root / "configs/models/qrc_regressor_pilot.yaml") == (
        "3a3c1d75521c6e0ac027686fade31d319c16108a699efbde1eeb147bc4158bb3"
    )
    assert sha256_file(config.qrc_selection_summary) == (
        "21e70d85f586af6bce9eb615ab2d9b36001654887c18c278bfe2d7b0375a6fc8"
    )
