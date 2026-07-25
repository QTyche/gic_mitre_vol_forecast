import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

import pytest

from qtyche_qrc.publication import (
    freeze_publication_assets,
    load_publication_config,
    verify_publication_sources,
)

TABLE_NAMES = (
    "publication_table_1_experiment_design",
    "publication_table_2_financial_benchmark",
    "publication_table_3_mnist_benchmark",
)
FIGURE_NAMES = (
    "publication_figure_1_architecture_selection",
    "publication_figure_2_financial_comparison",
    "publication_figure_3_financial_robustness",
    "publication_figure_4_mnist_benchmark",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_path() -> Path:
    return _root() / "configs/publication_assets.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _paper_checksums() -> dict[str, str]:
    root = _root() / "paper_assets"
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _prior_result_checksums() -> dict[str, str]:
    result_root = _root() / "results"
    publication_root = result_root / "publication_assets"
    return {
        path.relative_to(result_root).as_posix(): _sha256(path)
        for path in sorted(result_root.rglob("*"))
        if path.is_file() and publication_root not in path.parents
    }


@pytest.fixture(scope="module")
def regeneration_audit() -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    prior_before = _prior_result_checksums()
    freeze_publication_assets(_config_path())
    first = _paper_checksums()
    freeze_publication_assets(_config_path())
    second = _paper_checksums()
    prior_after = _prior_result_checksums()
    return first, second, prior_before, prior_after


def test_every_table_value_traces_to_a_checksum_pinned_fact() -> None:
    results = _load_json(_root() / "paper_assets/final_results_manifest.json")
    facts = {fact["fact_id"]: fact for fact in results["facts"]}
    configured_sources = {(source["path"], source["sha256"]) for source in results["sources"]}

    for name in TABLE_NAMES:
        table = _load_json(_root() / f"paper_assets/tables/{name}.json")
        for row in table["rows"]:
            assert row["trace_fact_ids"]
            for fact_id in row["trace_fact_ids"]:
                fact = facts[fact_id]
                assert (
                    fact["source_artifact_path"],
                    fact["source_artifact_sha256"],
                ) in configured_sources
                assert fact["source_column_or_key"]
                assert fact["display_rounded_value"] != ""


def test_every_figure_has_registered_frozen_facts() -> None:
    results = _load_json(_root() / "paper_assets/final_results_manifest.json")
    usages = {
        usage
        for fact in results["facts"]
        for usage in fact["usages"]
        if usage.startswith("publication_figure_")
    }
    assert usages == {f"publication_figure_{index}" for index in range(1, 5)}


def test_all_frozen_source_checksums_match() -> None:
    config = load_publication_config(_config_path())
    records = verify_publication_sources(config)
    assert records
    assert all(_sha256(_root() / row["path"]) == row["sha256"] for row in records)
    assert len(config.appendix_sources) == 7


def test_validation_and_test_evidence_are_not_mixed() -> None:
    results = _load_json(_root() / "paper_assets/final_results_manifest.json")
    facts = {fact["fact_id"]: fact for fact in results["facts"]}
    figure_one_facts = [fact for fact in facts.values() if "publication_figure_1" in fact["usages"]]
    assert figure_one_facts
    assert {fact["split"] for fact in figure_one_facts} == {"validation"}

    for table_name in TABLE_NAMES[1:]:
        table = _load_json(_root() / f"paper_assets/tables/{table_name}.json")
        traced = {fact_id for row in table["rows"] for fact_id in row["trace_fact_ids"]}
        splits = {facts[fact_id]["split"] for fact_id in traced}
        assert "test" in splits
        assert splits <= {"test", "not_applicable"}


def test_architecture_means_and_per_seed_robustness_are_explicit() -> None:
    results = _load_json(_root() / "paper_assets/final_results_manifest.json")
    facts = {fact["fact_id"]: fact for fact in results["facts"]}

    assert facts["financial.test.qrc_mean.macro_f1"]["value_scope"] == ("architecture_level")
    assert facts["mnist.test.exact_qrc_mean.accuracy"]["value_scope"] == ("architecture_level")
    assert facts["mnist.test.2_048_shots_seed_2026.accuracy"]["value_scope"] == ("per_seed")


def test_significance_annotations_require_holm_adjusted_support() -> None:
    results = _load_json(_root() / "paper_assets/final_results_manifest.json")
    facts = {fact["fact_id"]: fact for fact in results["facts"]}
    significant_rows: list[dict[str, Any]] = []
    for table_name in TABLE_NAMES[1:]:
        table = _load_json(_root() / f"paper_assets/tables/{table_name}.json")
        significant_rows.extend(
            row
            for row in table["rows"]
            if "Holm-adjusted" in row["holm_annotation"]
            and not row["holm_annotation"].startswith("No ")
        )

    assert significant_rows
    for row in significant_rows:
        holm_facts = [
            facts[fact_id] for fact_id in row["trace_fact_ids"] if fact_id.endswith(".holm_p")
        ]
        assert holm_facts
        assert all(fact["significance_adjusted"] is True for fact in holm_facts)
        assert any(float(fact["exact_value"]) < 0.05 for fact in holm_facts)


def test_claim_classification_and_architecture_level_correlation() -> None:
    results = _load_json(_root() / "paper_assets/final_results_manifest.json")
    claims = {int(claim["claim_id"]): claim for claim in results["claims"]}
    facts = {fact["fact_id"]: fact for fact in results["facts"]}

    assert {claims[index]["status"] for index in (1, 2, 11, 12)} == {"prohibited"}
    assert claims[13]["status"] == "unsupported"
    assert claims[8]["status"] == "supported"
    qrc = float(facts["financial.test.qrc_mean.correlation"]["exact_value"])
    baselines = [
        float(facts[f"financial.test.{model}.correlation"]["exact_value"])
        for model in ("esn", "garch_1_1", "rv_persistence")
    ]
    assert qrc > max(baselines)


def test_tracked_text_assets_contain_no_absolute_local_paths() -> None:
    markers = ("/" + "Users/", "\\" + "Users\\", str(_root().parent))
    for path in sorted((_root() / "paper_assets").rglob("*")):
        if path.is_file() and path.suffix in {".csv", ".json", ".md", ".tex"}:
            text = path.read_text(encoding="utf-8")
            assert not any(marker in text for marker in markers)


def test_selected_figures_have_png_pdf_and_publication_dimensions() -> None:
    figure_root = _root() / "paper_assets/figures"
    for name in FIGURE_NAMES:
        png = figure_root / f"{name}.png"
        pdf = figure_root / f"{name}.pdf"
        assert png.is_file() and pdf.is_file()
        header = png.read_bytes()[:24]
        assert header[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", header[16:24])
        assert width >= 2400
        assert height >= 1200
        assert pdf.read_bytes().startswith(b"%PDF")


def test_selected_tables_have_all_four_formats() -> None:
    table_root = _root() / "paper_assets/tables"
    for name in TABLE_NAMES:
        for suffix in (".csv", ".json", ".tex", ".md"):
            path = table_root / f"{name}{suffix}"
            assert path.is_file()
            assert path.stat().st_size > 0


def test_publication_manifest_checksums_every_selected_asset() -> None:
    manifest = _load_json(_root() / "paper_assets/publication_assets_manifest.json")
    assert manifest["publication_assets_frozen"] is True
    assert manifest["no_new_model_execution"] is True
    assert manifest["no_test_based_selection"] is True
    assert manifest["quantum_advantage_claim"] is False
    assert manifest["selected_asset_count"] == len(manifest["selected_assets"])
    for asset in manifest["selected_assets"]:
        path = _root() / asset["path"]
        assert path.is_file()
        assert _sha256(path) == asset["sha256"]


def test_asset_regeneration_is_byte_identical(
    regeneration_audit: tuple[
        dict[str, str],
        dict[str, str],
        dict[str, str],
        dict[str, str],
    ],
) -> None:
    first, second, _, _ = regeneration_audit
    assert first == second


def test_every_prior_result_tree_is_preserved(
    regeneration_audit: tuple[
        dict[str, str],
        dict[str, str],
        dict[str, str],
        dict[str, str],
    ],
) -> None:
    _, _, prior_before, prior_after = regeneration_audit
    assert prior_before
    assert prior_before == prior_after


def test_publication_generation_has_no_model_execution_path() -> None:
    package_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((_root() / "src/qtyche_qrc/publication").glob("*.py"))
    )
    runner_source = (_root() / "scripts/freeze_publication_assets.py").read_text(encoding="utf-8")
    source = package_source + runner_source
    forbidden = (
        ".fit(",
        "subprocess",
        "run_final_financial",
        "run_qrc_mnist",
        "train-qrc",
        "generate-qrc-features",
    )
    assert not any(re.search(re.escape(token), source) for token in forbidden)
