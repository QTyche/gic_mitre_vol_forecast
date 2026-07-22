"""Portable runtime metadata shared by every experiment manifest."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from typing import Any

ALLOWED_EXECUTION_PLATFORMS = frozenset({"local", "qbraid_lab", "qbraid_simulator", "qbraid_qpu"})

TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "PyYAML",
    "pytest",
    "ruff",
    "mypy",
    "qtyche-qrc",
)


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Return installed versions without making optional tools mandatory at runtime."""

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _first_environment_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def execution_platform() -> str:
    """Return the explicit supported platform, defaulting conservatively to local."""

    value = os.environ.get("QTYCHE_EXECUTION_PLATFORM")
    if value is None:
        value = "qbraid_lab" if os.environ.get("QBRAID_ENVS_PATH") else "local"
    if value not in ALLOWED_EXECUTION_PLATFORMS:
        allowed = ", ".join(sorted(ALLOWED_EXECUTION_PLATFORMS))
        raise ValueError(f"unsupported execution platform {value!r}; expected one of: {allowed}")
    return value


def _qbraid_sdk_version() -> str | None:
    for distribution in ("qbraid", "qbraid-core", "qbraid-cli"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def runtime_metadata() -> dict[str, Any]:
    """Return portable local/qBraid provenance for experiment manifests."""

    return {
        "execution_platform": execution_platform(),
        "qbraid_environment_name": _first_environment_value(
            "QTYCHE_QBRAID_ENVIRONMENT_NAME", "QBRAID_ENV_NAME", "QBRAID_ENV_SLUG"
        ),
        "qbraid_environment_id": _first_environment_value(
            "QTYCHE_QBRAID_ENVIRONMENT_ID", "QBRAID_ENV_ID"
        ),
        "qbraid_lab_image": _first_environment_value(
            "QTYCHE_QBRAID_LAB_IMAGE", "QBRAID_LAB_IMAGE", "QBRAID_IMAGE"
        ),
        "qbraid_sdk_version": _qbraid_sdk_version(),
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "package_versions": package_versions(),
    }
