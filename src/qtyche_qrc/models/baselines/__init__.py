"""Inspectably implemented classical benchmark models."""

from qtyche_qrc.models.baselines.esn import (
    ESNClassifier,
    ESNConfig,
    ESNRegressor,
    ESNReservoir,
)
from qtyche_qrc.models.baselines.garch import (
    GARCHFitResult,
    GARCHForecastPath,
    GARCHOptimizationAttempt,
    GARCHParameters,
    GaussianGARCH11,
)
from qtyche_qrc.models.baselines.logistic import MultinomialLogisticClassifier
from qtyche_qrc.models.baselines.persistence import (
    CurrentRegimePersistenceClassifier,
    MajorityClassClassifier,
    RealizedVariancePersistenceRegressor,
)

__all__ = [
    "CurrentRegimePersistenceClassifier",
    "ESNClassifier",
    "ESNConfig",
    "ESNRegressor",
    "ESNReservoir",
    "GARCHFitResult",
    "GARCHForecastPath",
    "GARCHOptimizationAttempt",
    "GARCHParameters",
    "GaussianGARCH11",
    "MajorityClassClassifier",
    "MultinomialLogisticClassifier",
    "RealizedVariancePersistenceRegressor",
]
