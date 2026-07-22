"""Machine-readable experiment provenance manifests."""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qtyche_qrc.config import ProjectConfig
from qtyche_qrc.runtime import runtime_metadata


def _git_metadata(repository: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def create_manifest(config: ProjectConfig, repository: Path | None = None) -> Path:
    """Write a provenance manifest for a validated, manifest-only smoke run."""

    repository = (repository or Path.cwd()).resolve()
    created_at = datetime.now(timezone.utc)
    output_root = config.experiment.output_dir
    if not output_root.is_absolute():
        output_root = repository / output_root
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
    manifest_path = manifest_dir / f"smoke_{timestamp}.json"

    config_bytes = config.source.read_bytes()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        **runtime_metadata(),
        "experiment": config.experiment.name,
        "created_at_utc": created_at.isoformat(),
        "git": _git_metadata(repository),
        "environment": {
            "python_version": sys.version,
            "hostname": socket.gethostname(),
        },
        "random_seed": config.experiment.seed,
        "configuration": {
            "path": str(config.source),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "content": config.raw,
        },
        "outputs": {"manifest": str(manifest_path)},
        "run_kind": "smoke_manifest_only",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path
