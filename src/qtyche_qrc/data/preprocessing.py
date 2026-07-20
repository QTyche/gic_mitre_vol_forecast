"""Training-only standardization with persisted, inspectable parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainStandardizer:
    """Feature scaler whose parameters are fit exactly once on training rows."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    zero_variance_features: tuple[str, ...]
    fitted_rows: int

    @classmethod
    def fit(cls, training_features: pd.DataFrame) -> TrainStandardizer:
        """Fit population means and standard deviations on training features."""

        if training_features.empty:
            raise ValueError("cannot fit preprocessing on an empty training set")
        if training_features.columns.duplicated().any():
            raise ValueError("training feature columns must be unique")
        if training_features.isna().any().any():
            raise ValueError("cannot fit preprocessing with missing training features")
        means = training_features.mean(axis=0)
        standard_deviations = training_features.std(axis=0, ddof=0)
        if not np.isfinite(means.to_numpy(dtype=float)).all():
            raise ValueError("training feature means must be finite")
        if not np.isfinite(standard_deviations.to_numpy(dtype=float)).all():
            raise ValueError("training feature standard deviations must be finite")
        zero_variance = tuple(standard_deviations.index[standard_deviations.eq(0)].tolist())
        effective_standard_deviations = standard_deviations.mask(standard_deviations.eq(0), 1.0)
        return cls(
            feature_names=tuple(str(name) for name in training_features.columns),
            means=tuple(float(value) for value in means),
            standard_deviations=tuple(float(value) for value in effective_standard_deviations),
            zero_variance_features=zero_variance,
            fitted_rows=len(training_features),
        )

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen parameters, rejecting missing, extra, or reordered columns."""

        actual_names = tuple(str(name) for name in features.columns)
        if actual_names != self.feature_names:
            raise ValueError(
                f"feature columns differ from fitted preprocessing: expected "
                f"{self.feature_names}, received {actual_names}"
            )
        if features.isna().any().any():
            raise ValueError("cannot transform missing feature values")
        means = pd.Series(self.means, index=self.feature_names)
        scales = pd.Series(self.standard_deviations, index=self.feature_names)
        transformed = (features - means) / scales
        if not np.isfinite(transformed.to_numpy(dtype=float)).all():
            raise ValueError("preprocessing produced non-finite feature values")
        return transformed

    def as_dict(self) -> dict[str, Any]:
        """Return stable JSON-serializable preprocessing parameters."""

        return {
            "schema_version": 1,
            "fit_split": "train",
            "standard_deviation_ddof": 0,
            "feature_names": list(self.feature_names),
            "means": dict(zip(self.feature_names, self.means)),
            "standard_deviations": dict(zip(self.feature_names, self.standard_deviations)),
            "zero_variance_features": list(self.zero_variance_features),
            "zero_variance_policy": "scale_by_one_after_centering",
            "fitted_rows": self.fitted_rows,
        }

    def save(self, path: Path) -> None:
        """Persist parameters as strict, sorted JSON."""

        path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
