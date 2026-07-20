import json
from pathlib import Path

from qtyche_qrc.config import load_config
from qtyche_qrc.experiments.manifest import create_manifest


def test_smoke_experiment_creates_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "schema_version: 1\n"
        "experiment:\n"
        "  name: test_smoke\n"
        "  seed: 11\n"
        f"  output_dir: {tmp_path.as_posix()}/results\n",
        encoding="utf-8",
    )

    path = create_manifest(load_config(config_path), repository=tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert path.is_file()
    assert manifest["experiment"] == "test_smoke"
    assert manifest["random_seed"] == 11
    assert manifest["configuration"]["content"]["experiment"]["seed"] == 11
    assert manifest["outputs"]["manifest"] == str(path)
