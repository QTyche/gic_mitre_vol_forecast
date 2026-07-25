"""Descriptive diagnostics for immutable benchmark predictions."""

from qtyche_qrc.diagnostics.calibration import (
    calibration_bin_assignments,
    multiclass_calibration_summary,
    reliability_table,
    variance_bootstrap_intervals,
    variance_calibration_deciles,
    variance_point_metrics,
)
from qtyche_qrc.diagnostics.regimes import (
    assess_lead_time_identifiability,
    classification_diagnostics,
    transition_type_diagnostics,
)
from qtyche_qrc.diagnostics.temporal import (
    cumulative_loss_difference,
    rolling_variance_diagnostics,
)

__all__ = [
    "assess_lead_time_identifiability",
    "calibration_bin_assignments",
    "classification_diagnostics",
    "cumulative_loss_difference",
    "multiclass_calibration_summary",
    "reliability_table",
    "rolling_variance_diagnostics",
    "transition_type_diagnostics",
    "variance_bootstrap_intervals",
    "variance_calibration_deciles",
    "variance_point_metrics",
]
