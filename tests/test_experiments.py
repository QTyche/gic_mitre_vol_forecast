import json
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from qtyche_qrc.data.fixtures import create_or_verify_fixture_snapshots
from qtyche_qrc.data.pipeline import prepare_data
from qtyche_qrc.experiments.compare import compare_baselines
from qtyche_qrc.experiments.run import SyntheticResultsError, run_baseline_experiment
from tests.data_helpers import write_test_data_config, write_test_model_config


def _prepared_experiment_root(tmp_path: Path) -> Path:
    data_config = write_test_data_config(tmp_path)
    create_or_verify_fixture_snapshots(data_config)
    prepare_data(data_config)
    return write_test_model_config(tmp_path)


def test_headline_command_rejects_fixture_without_override(tmp_path: Path) -> None:
    model_config = _prepared_experiment_root(tmp_path)

    with pytest.raises(SyntheticResultsError, match="fixture data"):
        run_baseline_experiment(model_config)


def test_completed_fixture_experiment_has_required_marked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_config = _prepared_experiment_root(tmp_path)

    def forbidden_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("baseline smoke experiments must remain offline")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)
    experiment_dir = run_baseline_experiment(model_config, allow_synthetic_results=True)
    required_files = {
        "config.yaml",
        "manifest.json",
        "model_metadata.json",
        "selection_results.csv",
        "validation_metrics.json",
        "test_metrics.json",
        "validation_predictions.csv",
        "test_predictions.csv",
        "timing.json",
    }

    assert required_files.issubset({path.name for path in experiment_dir.iterdir()})
    assert (experiment_dir / "model").is_dir()
    assert (experiment_dir / "figures").is_dir()
    assert (experiment_dir / "logs").is_dir()
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_source_type"] == "fixture"
    assert manifest["is_synthetic"] is True
    assert "SYNTHETIC FIXTURE DATA" in manifest["data_warning"]


def test_validation_and_test_comparison_tables_are_separate(tmp_path: Path) -> None:
    model_config = _prepared_experiment_root(tmp_path)
    run_baseline_experiment(model_config, allow_synthetic_results=True)
    output_dir = tmp_path / "tables"

    validation_path, test_path = compare_baselines(tmp_path / "results", output_dir)

    assert validation_path.name == "baseline_validation_comparison.csv"
    assert test_path.name == "baseline_test_comparison.csv"
    assert validation_path != test_path
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    for table in (validation, test):
        assert {
            "data_source_type",
            "is_synthetic",
            "task",
            "seed",
            "model",
            "selected_configuration",
        }.issubset(table.columns)


def test_latest_comparison_uses_experiment_time_not_directory_path(
    tmp_path: Path,
) -> None:
    model_config = _prepared_experiment_root(tmp_path)
    older_dir = run_baseline_experiment(model_config, allow_synthetic_results=True)
    archive_dir = older_dir.parent / "z_archive"
    archive_dir.mkdir()
    older_dir.rename(archive_dir / older_dir.name)
    newer_dir = run_baseline_experiment(model_config, allow_synthetic_results=True)

    validation_path, test_path = compare_baselines(
        tmp_path / "results", tmp_path / "tables", latest_per_model=True
    )

    for path in (validation_path, test_path):
        table = pd.read_csv(path)
        assert table["experiment_id"].tolist() == [newer_dir.name]
