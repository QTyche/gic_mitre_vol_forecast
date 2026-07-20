import json
import urllib.request
from pathlib import Path

import pytest

from qtyche_qrc.data.fixtures import create_or_verify_fixture_snapshots
from qtyche_qrc.data.pipeline import prepare_data
from qtyche_qrc.data.validation import audit_processed_data
from tests.data_helpers import write_test_data_config


def test_fixture_pipeline_is_network_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_test_data_config(tmp_path)

    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden in fixture tests")

    monkeypatch.setattr(urllib.request, "urlopen", fail_network)
    create_or_verify_fixture_snapshots(config)
    result = prepare_data(config)

    assert audit_processed_data(result.processed_dir)["status"] == "passed"
    assert all(path.is_file() for path in result.output_paths.values())
    assert result.quality_report["output_missing_value_counts"] == {
        "train": 0,
        "validation": 0,
        "test": 0,
    }


def test_pipeline_is_deterministic_except_generation_timestamp(tmp_path: Path) -> None:
    config = write_test_data_config(tmp_path)
    create_or_verify_fixture_snapshots(config)
    first = prepare_data(config)
    first_features = first.output_paths["features_unscaled"].read_bytes()
    first_manifest = json.loads(first.output_paths["data_manifest"].read_text(encoding="utf-8"))

    second = prepare_data(config)
    second_features = second.output_paths["features_unscaled"].read_bytes()
    second_manifest = json.loads(second.output_paths["data_manifest"].read_text(encoding="utf-8"))
    first_manifest.pop("generation_timestamp")
    second_manifest.pop("generation_timestamp")

    assert first_features == second_features
    assert first_manifest == second_manifest


def test_pipeline_applies_five_row_purge_to_every_split(tmp_path: Path) -> None:
    config = write_test_data_config(tmp_path)
    create_or_verify_fixture_snapshots(config)

    result = prepare_data(config)

    assert result.quality_report["splitting"]["purged_forward_window_rows"] == {
        "train": 5,
        "validation": 5,
        "test": 5,
    }
