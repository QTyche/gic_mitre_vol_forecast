import json
from pathlib import Path

import pytest

from qtyche_qrc.data.config import load_data_config
from qtyche_qrc.data.fixtures import create_or_verify_fixture_snapshots
from qtyche_qrc.data.pipeline import prepare_data
from qtyche_qrc.models.dataset import DatasetIntegrityError, load_model_dataset
from tests.data_helpers import prepared_model_dataset, write_test_data_config


def test_frozen_fixture_row_counts_remain_exact() -> None:
    config = load_data_config(Path("configs/data.yaml"))
    create_or_verify_fixture_snapshots(config)
    result = prepare_data(config)
    manifest = json.loads(result.output_paths["data_manifest"].read_text(encoding="utf-8"))

    assert manifest["row_counts"] == {
        "canonical_source": 4225,
        "features_unscaled": 4159,
        "train": 2865,
        "validation": 776,
        "test": 518,
    }


def test_model_loader_marks_fixture_data_synthetic(tmp_path: Path) -> None:
    dataset = prepared_model_dataset(tmp_path)

    assert dataset.data_source_type == "fixture"
    assert dataset.is_synthetic is True
    assert dataset.X_train.shape[1] == len(dataset.feature_names)


def test_processed_checksum_mismatch_fails_clearly(tmp_path: Path) -> None:
    config = write_test_data_config(tmp_path)
    create_or_verify_fixture_snapshots(config)
    prepare_data(config)
    manifest_path = config.processed_path / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["processed_checksums"] = {"train.csv": "definitely-not-the-real-checksum"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetIntegrityError, match="checksum mismatch"):
        load_model_dataset(config.processed_path)


def test_selection_dataset_exposes_no_test_metrics(tmp_path: Path) -> None:
    selection = prepared_model_dataset(tmp_path).for_selection()

    with pytest.raises(AttributeError, match="test data are unavailable"):
        _ = selection.X_test
