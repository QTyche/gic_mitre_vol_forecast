from pathlib import Path

import pytest

from qtyche_qrc.config import ConfigError, load_config


def test_yaml_configuration_loads() -> None:
    config = load_config(Path("configs/qrc_smoke.yaml"))

    assert config.schema_version == 1
    assert config.experiment.name == "qrc_smoke"
    assert config.experiment.seed == 2026


def test_invalid_configuration_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("schema_version: 1\nexperiment:\n  name: missing-fields\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"experiment\.seed"):
        load_config(path)
