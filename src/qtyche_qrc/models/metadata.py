"""Shared model provenance helpers."""

from __future__ import annotations

import importlib.metadata
from datetime import datetime, timezone


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def package_versions(*names: str) -> dict[str, str]:
    """Return installed versions for explicitly relevant packages."""

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions
