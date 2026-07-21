import pytest

from qtyche_qrc.experiments.sweep import CandidateResult, select_candidate


def test_selection_uses_only_configured_validation_metric() -> None:
    results = [
        CandidateResult(1, {"value": 1}, "macro_f1", 0.4, "success"),
        CandidateResult(2, {"value": 2}, "macro_f1", 0.7, "success"),
    ]

    assert select_candidate(results, "macro_f1", minimize=False) == {"value": 2}
    with pytest.raises(ValueError, match="unexpected selection metric"):
        select_candidate(results, "qlike", minimize=True)


def test_failed_trials_are_retained_but_not_selected() -> None:
    results = [
        CandidateResult(1, {"value": 1}, "qlike", None, "failure", "failed"),
        CandidateResult(2, {"value": 2}, "qlike", 3.0, "success"),
    ]

    assert len(results) == 2
    assert select_candidate(results, "qlike", minimize=True) == {"value": 2}
