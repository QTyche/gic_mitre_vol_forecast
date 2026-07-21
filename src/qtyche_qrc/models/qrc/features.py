"""Checksum-guarded reusable QRC reservoir-feature cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qtyche_qrc.models.qrc.backends import BACKEND_VERSION
from qtyche_qrc.models.qrc.encoding import array_checksum
from qtyche_qrc.models.qrc.reservoir import QRCConfig, QuantumReservoir, split_qrc_features


class FeatureCacheIntegrityError(ValueError):
    """Raised when cached QRC features or their key metadata have changed."""


@dataclass(frozen=True)
class FeatureCacheKey:
    """Every field that can change the deterministic QRC representations."""

    processed_data_manifest_checksum: str
    feature_column_checksum: str
    qrc_configuration_checksum: str
    reservoir_seed: int
    state_policy: str
    backend_version: str

    @property
    def checksum(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QRCFeatureBundle:
    """Chronological train/validation/test features plus generation evidence."""

    train: NDArray[np.float64]
    validation: NDArray[np.float64]
    test: NDArray[np.float64]
    metadata: dict[str, Any]
    cache_dir: Path
    cache_hit: bool


def feature_column_checksum(feature_names: tuple[str, ...]) -> str:
    payload = json.dumps(list(feature_names), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_feature_cache_key(
    *,
    processed_data_manifest_checksum: str,
    feature_names: tuple[str, ...],
    config: QRCConfig,
) -> FeatureCacheKey:
    return FeatureCacheKey(
        processed_data_manifest_checksum,
        feature_column_checksum(feature_names),
        config.checksum,
        config.reservoir_seed,
        config.state_policy,
        BACKEND_VERSION,
    )


def _cache_paths(cache_root: Path, key: FeatureCacheKey) -> tuple[Path, Path, Path]:
    directory = cache_root / key.checksum
    return directory, directory / "qrc_features.npz", directory / "metadata.json"


def load_feature_cache(cache_root: Path, key: FeatureCacheKey) -> QRCFeatureBundle | None:
    """Load and checksum-validate an exact-key cache, or return None if absent."""

    directory, arrays_path, metadata_path = _cache_paths(cache_root, key)
    if not directory.exists():
        return None
    if not arrays_path.is_file() or not metadata_path.is_file():
        raise FeatureCacheIntegrityError(f"incomplete QRC feature cache: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("cache_key") != asdict(key)
        or metadata.get("cache_key_checksum") != key.checksum
    ):
        raise FeatureCacheIntegrityError("QRC feature cache key checksum mismatch")
    with np.load(arrays_path) as values:
        train = np.asarray(values["train"], dtype=float)
        validation = np.asarray(values["validation"], dtype=float)
        test = np.asarray(values["test"], dtype=float)
    arrays = {"train": train, "validation": validation, "test": test}
    expected = metadata.get("array_checksums")
    if not isinstance(expected, dict):
        raise FeatureCacheIntegrityError("QRC feature cache omits array checksums")
    for name, value in arrays.items():
        actual = array_checksum(value)
        if actual != expected.get(name):
            raise FeatureCacheIntegrityError(
                f"QRC feature cache checksum mismatch for {name}: {actual} != {expected.get(name)}"
            )
    return QRCFeatureBundle(train, validation, test, metadata, directory, True)


def generate_or_load_features(
    *,
    cache_root: Path,
    key: FeatureCacheKey,
    feature_names: tuple[str, ...],
    config: QRCConfig,
    X_train: NDArray[np.float64],
    X_validation: NDArray[np.float64],
    X_test: NDArray[np.float64],
) -> QRCFeatureBundle:
    """Generate label-free split features once and persist all numerical evidence."""

    cached = load_feature_cache(cache_root, key)
    if cached is not None:
        return cached
    reservoir = QuantumReservoir(len(feature_names), config)
    train, validation, optional_test = split_qrc_features(
        reservoir, X_train, X_validation, X_test, config.state_policy
    )
    if optional_test is None:
        raise RuntimeError("QRC feature generation unexpectedly omitted test inputs")
    test = optional_test
    directory, arrays_path, metadata_path = _cache_paths(cache_root, key)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(arrays_path, train=train, validation=validation, test=test)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "cache_key": asdict(key),
        "cache_key_checksum": key.checksum,
        "label_free_generation": True,
        "target_columns_consumed": [],
        "split_shapes": {
            "train": list(train.shape),
            "validation": list(validation.shape),
            "test": list(test.shape),
        },
        "array_checksums": {
            "train": array_checksum(train),
            "validation": array_checksum(validation),
            "test": array_checksum(test),
        },
        "observable_metadata": reservoir.observables.metadata(),
        "observable_checksum": reservoir.observables.checksum,
        "resource_metadata": reservoir.resource_metadata(),
        "numerical_diagnostics": reservoir.numerical_diagnostics(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return QRCFeatureBundle(train, validation, test, metadata, directory, False)
