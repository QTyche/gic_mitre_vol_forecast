"""Strict chronological splitting with complete forward-window containment."""

from __future__ import annotations

from typing import Any

import pandas as pd

from qtyche_qrc.data.config import SplitBoundary


def assign_chronological_splits(
    frame: pd.DataFrame,
    boundaries: tuple[SplitBoundary, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign rows to splits and purge labels whose window crosses an end date."""

    if not frame["date"].is_monotonic_increasing or frame["date"].duplicated().any():
        raise ValueError("dates must be strictly increasing before splitting")
    if "target_window_end" not in frame:
        raise ValueError("splitting requires target_window_end")

    selected: list[pd.DataFrame] = []
    purged: dict[str, int] = {}
    purged_dates: dict[str, list[str]] = {}
    rows_in_boundaries = 0
    for boundary in boundaries:
        date_mask = frame["date"].between(
            pd.Timestamp(boundary.start), pd.Timestamp(boundary.end), inclusive="both"
        )
        candidates = frame.loc[date_mask].copy()
        rows_in_boundaries += len(candidates)
        safe = candidates["target_window_end"].le(pd.Timestamp(boundary.end))
        purged[boundary.name] = int((~safe).sum())
        purged_dates[boundary.name] = candidates.loc[~safe, "date"].dt.strftime("%Y-%m-%d").tolist()
        candidates = candidates.loc[safe].copy()
        candidates["split"] = boundary.name
        selected.append(candidates)

    result = pd.concat(selected, ignore_index=True) if selected else frame.iloc[0:0].copy()
    result = result.sort_values("date", kind="stable").reset_index(drop=True)
    report: dict[str, Any] = {
        "rows_in_split_boundaries": rows_in_boundaries,
        "rows_outside_split_boundaries": int(len(frame) - rows_in_boundaries),
        "purged_forward_window_rows": purged,
        "purged_forward_window_dates": purged_dates,
        "rows_after_split_and_purge": len(result),
    }
    return result, report


def validate_forward_window_containment(
    frame: pd.DataFrame,
    boundaries: tuple[SplitBoundary, ...],
) -> None:
    """Raise if any retained target window crosses its configured split end."""

    by_name = {boundary.name: boundary for boundary in boundaries}
    for split_name, rows in frame.groupby("split", sort=False):
        boundary = by_name[str(split_name)]
        if rows["target_window_end"].gt(pd.Timestamp(boundary.end)).any():
            raise ValueError(f"{split_name} contains a target window crossing its end boundary")
