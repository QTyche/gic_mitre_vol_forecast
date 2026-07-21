"""Read-only model dataset loader for frozen processed artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qtyche_qrc.data.targets import TARGET_NAMES


class DatasetIntegrityError(ValueError):
    """Raised when processed files disagree with their data manifest."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelSplit:
    """Chronological model arrays and observation-level context for one split."""

    X: NDArray[np.float64]
    y_regime: NDArray[np.int_]
    y_transition: NDArray[np.int_]
    y_rv: NDArray[np.float64]
    dates: NDArray[np.datetime64]
    current_regime: NDArray[np.int_]
    current_rv_unscaled: NDArray[np.float64]


@dataclass(frozen=True)
class SelectionDataset:
    """Search-only view that deliberately has no test-set attributes."""

    train: ModelSplit
    validation: ModelSplit
    feature_names: tuple[str, ...]

    def __getattr__(self, name: str) -> Any:
        if "test" in name.lower():
            raise AttributeError("test data are unavailable during hyperparameter selection")
        raise AttributeError(name)


@dataclass(frozen=True)
class ModelDataset:
    """Frozen train/validation/test arrays plus manifest provenance."""

    train: ModelSplit
    validation: ModelSplit
    test: ModelSplit
    feature_names: tuple[str, ...]
    manifest: dict[str, Any]
    processed_checksums: dict[str, str]
    data_source_type: str
    is_synthetic: bool

    @property
    def X_train(self) -> NDArray[np.float64]:
        return self.train.X

    @property
    def X_validation(self) -> NDArray[np.float64]:
        return self.validation.X

    @property
    def X_test(self) -> NDArray[np.float64]:
        return self.test.X

    def for_selection(self) -> SelectionDataset:
        """Return a structural guard that exposes no test data."""

        return SelectionDataset(self.train, self.validation, self.feature_names)


def _source_metadata(manifest: dict[str, Any]) -> tuple[str, bool]:
    explicit_type = manifest.get("data_source_type")
    explicit_synthetic = manifest.get("is_synthetic")
    if explicit_type in {"fixture", "public_market"} and isinstance(explicit_synthetic, bool):
        if explicit_synthetic != (explicit_type == "fixture"):
            raise DatasetIntegrityError("data manifest source flags are inconsistent")
        return str(explicit_type), explicit_synthetic
    source_files = manifest.get("source_files", {})
    paths = [str(value.get("path", "")) for value in source_files.values()]
    fixture = bool(paths) and all(Path(path).name.startswith("fixture_") for path in paths)
    return ("fixture", True) if fixture else ("public_market", False)


def _load_split(
    processed_dir: Path,
    split_name: str,
    feature_names: tuple[str, ...],
    expected_rows: int,
    unscaled_rv: pd.Series[float],
) -> ModelSplit:
    path = processed_dir / f"{split_name}.csv"
    if not path.is_file():
        raise DatasetIntegrityError(f"missing processed split: {path}")
    frame = pd.read_csv(path, parse_dates=["date"])
    expected_columns = ["date", "split", *feature_names, *TARGET_NAMES]
    if list(frame.columns) != expected_columns:
        raise DatasetIntegrityError(f"{split_name} columns disagree with data manifest")
    if len(frame) != expected_rows:
        raise DatasetIntegrityError(
            f"{split_name} row count disagrees with manifest: {len(frame)} != {expected_rows}"
        )
    if not frame["split"].eq(split_name).all():
        raise DatasetIntegrityError(f"{split_name} file contains a different split label")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise DatasetIntegrityError(f"{split_name} dates must be unique and increasing")
    if frame.isna().any().any():
        raise DatasetIntegrityError(f"{split_name} contains missing values")
    features = frame.loc[:, list(feature_names)].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise DatasetIntegrityError(f"{split_name} features contain non-finite values")
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    lookup_dates = pd.DatetimeIndex(dates)
    try:
        current_rv = unscaled_rv.loc[lookup_dates].to_numpy(dtype=float)
    except KeyError as exc:
        raise DatasetIntegrityError(
            f"{split_name} dates are missing from features_unscaled.csv"
        ) from exc
    return ModelSplit(
        X=features,
        y_regime=frame["target_regime_5d"].to_numpy(dtype=int),
        y_transition=frame["target_transition"].to_numpy(dtype=int),
        y_rv=frame["target_rv_5d"].to_numpy(dtype=float),
        dates=dates,
        current_regime=frame["current_regime"].to_numpy(dtype=int),
        current_rv_unscaled=current_rv,
    )


def load_model_dataset(processed_dir: Path) -> ModelDataset:
    """Load processed data without fitting, scaling, or shuffling anything."""

    processed_dir = processed_dir.resolve()
    manifest_path = processed_dir / "data_manifest.json"
    if not manifest_path.is_file():
        raise DatasetIntegrityError(f"missing data manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_names = tuple(str(value) for value in manifest.get("feature_names", []))
    if not feature_names:
        raise DatasetIntegrityError("data manifest contains no feature names")
    if tuple(manifest.get("target_names", [])) != TARGET_NAMES:
        raise DatasetIntegrityError("data manifest target names disagree with frozen targets")

    required_files = [
        "features_unscaled.csv",
        "train.csv",
        "validation.csv",
        "test.csv",
        "preprocessing.json",
        "regime_thresholds.json",
        "data_manifest.json",
        "data_quality_report.json",
    ]
    checksums = {name: _sha256(processed_dir / name) for name in required_files}
    saved_checksums = manifest.get("processed_checksums")
    if isinstance(saved_checksums, dict):
        for name, expected in saved_checksums.items():
            if name in checksums and checksums[name] != expected:
                raise DatasetIntegrityError(
                    f"processed-data checksum mismatch for {name}: {checksums[name]} != {expected}"
                )

    unscaled_path = processed_dir / "features_unscaled.csv"
    unscaled = pd.read_csv(unscaled_path, parse_dates=["date"])
    if unscaled["date"].duplicated().any() or not unscaled["date"].is_monotonic_increasing:
        raise DatasetIntegrityError("features_unscaled dates must be unique and increasing")
    if "spy_rv_5d" not in unscaled:
        raise DatasetIntegrityError("features_unscaled.csv omits unscaled spy_rv_5d")
    unscaled_rv = unscaled.set_index("date")["spy_rv_5d"]
    row_counts = manifest.get("row_counts", {})
    train = _load_split(
        processed_dir, "train", feature_names, int(row_counts.get("train", -1)), unscaled_rv
    )
    validation = _load_split(
        processed_dir,
        "validation",
        feature_names,
        int(row_counts.get("validation", -1)),
        unscaled_rv,
    )
    test = _load_split(
        processed_dir, "test", feature_names, int(row_counts.get("test", -1)), unscaled_rv
    )
    source_type, synthetic = _source_metadata(manifest)
    return ModelDataset(
        train=train,
        validation=validation,
        test=test,
        feature_names=feature_names,
        manifest=manifest,
        processed_checksums=checksums,
        data_source_type=source_type,
        is_synthetic=synthetic,
    )
