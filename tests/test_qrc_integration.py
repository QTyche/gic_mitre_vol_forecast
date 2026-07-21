import json
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import yaml

from qtyche_qrc.data.fixtures import create_or_verify_fixture_snapshots
from qtyche_qrc.data.pipeline import prepare_data
from qtyche_qrc.experiments.model_config import load_model_config
from qtyche_qrc.experiments.qrc_run import (
    generate_qrc_features,
    run_qrc_experiment,
    select_qrc_readout,
)
from qtyche_qrc.experiments.run import SyntheticResultsError
from qtyche_qrc.models.dataset import load_model_dataset
from tests.data_helpers import write_test_data_config, write_test_qrc_model_config


def _prepared_qrc(tmp_path: Path, *, search: bool = False) -> Path:
    data_config = write_test_data_config(tmp_path)
    create_or_verify_fixture_snapshots(data_config)
    prepare_data(data_config)
    return write_test_qrc_model_config(tmp_path, search=search)


def test_fixture_qrc_smoke_experiment_runs_fully_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _prepared_qrc(tmp_path)

    def forbidden_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("QRC fixture smoke must not access the network")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)
    experiment_dir = run_qrc_experiment(
        config_path, allow_synthetic_results=True, reservoir_seed=23
    )
    required = {
        "model/qrc_hamiltonian.npz",
        "model/input_projection.npy",
        "model/observables.json",
        "model/readout.npz",
        "qrc_backend_metadata.json",
        "qrc_numerical_diagnostics.json",
        "qrc_feature_metadata.json",
    }
    assert all((experiment_dir / value).is_file() for value in required)
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["is_synthetic"] is True
    effective_config = yaml.safe_load((experiment_dir / "config.yaml").read_text(encoding="utf-8"))
    assert manifest["reservoir_seed"] == 23
    assert effective_config["experiment"]["seed"] == 23
    assert effective_config["model"]["parameters"]["reservoir_seed"] == 23
    assert manifest["qrc_features_generated_without_labels"] is True
    assert manifest["test_evaluated_after_readout_freeze"] is True


def test_public_pilot_command_rejects_synthetic_data_without_override(tmp_path: Path) -> None:
    config_path = _prepared_qrc(tmp_path)
    with pytest.raises(SyntheticResultsError, match="rejects synthetic data"):
        generate_qrc_features(config_path)


def test_hyperparameter_selection_has_no_test_data_or_metrics(tmp_path: Path) -> None:
    config_path = _prepared_qrc(tmp_path, search=True)
    config = load_model_config(config_path)
    data = load_model_dataset(config.processed_dir)
    selection = data.for_selection()
    train = np.zeros((len(selection.train.X), 6), dtype=float)
    validation = np.zeros((len(selection.validation.X), 6), dtype=float)

    selected, results = select_qrc_readout(
        config=config,
        data=selection,
        train_features=train,
        validation_features=validation,
    )

    assert selected in {0.01, 0.1}
    assert all(result.validation_score is not None for result in results)
    with pytest.raises(AttributeError, match="test data are unavailable"):
        _ = selection.test_metrics


def test_validation_and_test_label_changes_do_not_change_cached_features(tmp_path: Path) -> None:
    config_path = _prepared_qrc(tmp_path)
    before = generate_qrc_features(config_path, allow_synthetic_results=True)
    data = load_model_dataset(tmp_path / "data/processed")
    data.validation.y_regime[:] = data.validation.y_regime[::-1]
    data.test.y_regime[:] = data.test.y_regime[::-1]
    after = generate_qrc_features(config_path, allow_synthetic_results=True)
    assert after.cache_hit is True
    assert np.array_equal(before.validation, after.validation)
    assert np.array_equal(before.test, after.test)
