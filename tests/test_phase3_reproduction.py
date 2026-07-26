from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from qtyche_qrc.qbraid import scan_prohibited_paths
from qtyche_qrc.reproducibility import orchestrator
from qtyche_qrc.reproducibility.orchestrator import (
    ReproductionTaskError,
    Task,
    construct_tasks,
    load_reproduction_config,
    validate_artifact_reuse_changed_paths,
    validate_recorded_artifacts,
)
from qtyche_qrc.reproducibility.verification import (
    CLONE_URL,
    EXPECTED_ARCHITECTURE_SHA256,
    EXPECTED_PUBLICATION_TREE_DIGEST,
    ReproductionVerificationError,
    _verify_frozen_facts,
    _verify_publication_assets,
    compare_numeric,
    find_repository_root,
    publication_tree_digest,
    sha256_path,
    verify_frozen_repository,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config() -> dict[str, Any]:
    return load_reproduction_config(_root() / "configs/phase3_reproduction.yaml")


def test_clean_repository_root_detection(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    (root / ".git").mkdir(parents=True)
    (root / "src/qtyche_qrc").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    nested = root / "nested/path"
    nested.mkdir(parents=True)

    assert find_repository_root(nested) == root


def test_command_construction_is_shell_free_and_repository_relative() -> None:
    root = _root()
    evidence = root / "qbraid_evidence/test_command_construction"
    tasks = construct_tasks(
        _config(),
        tier="headline",
        root=root,
        evidence_dir=evidence,
        python_executable=sys.executable,
    )

    assert tasks[0].command[:3] == (
        sys.executable,
        "-m",
        "qtyche_qrc.reproducibility.verification",
    )
    assert all(task.command[0] == sys.executable for task in tasks)
    assert all(not Path(argument).is_absolute() for task in tasks for argument in task.command[1:])
    assert not evidence.exists()


def test_subprocess_failure_propagates(tmp_path: Path) -> None:
    task = Task(
        "intentional_failure",
        (sys.executable, "-c", "raise SystemExit(7)"),
        (),
    )
    transcript = io.StringIO()

    with pytest.raises(ReproductionTaskError, match="exit status 7"):
        orchestrator._run_task(
            task,
            root=_root(),
            transcript=transcript,
            fingerprint="0" * 64,
        )

    assert "exit=7" in transcript.getvalue()


def test_tier_task_selection() -> None:
    root = _root()
    evidence = root / "qbraid_evidence/test_tiers"
    selected = {
        tier: [
            task.task_id
            for task in construct_tasks(
                _config(),
                tier=tier,
                root=root,
                evidence_dir=evidence,
                python_executable=sys.executable,
            )
        ]
        for tier in ("verify", "headline", "full")
    }

    assert selected["verify"] == ["frozen_verification", "focused_tests"]
    assert "final_financial_exact" in selected["headline"]
    assert "mnist_smoke" in selected["headline"]
    assert "processed_semantic_verification" in selected["headline"]
    assert "final_financial_full" in selected["full"]
    assert "mnist_full" in selected["full"]
    assert "processed_semantic_verification" in selected["full"]
    assert "publication_regeneration" in selected["full"]
    assert "final_financial_full" not in selected["headline"]


def test_no_synthetic_data_fallback() -> None:
    config = _config()

    assert config["scientific_contract"]["synthetic_fallback_permitted"] is False
    commands = [argument for task in config["tasks"].values() for argument in task["command"]]
    assert "--allow-synthetic-results" not in commands
    assert "create-fixture-data" not in commands


def test_numeric_tolerance_is_explicit() -> None:
    inside = compare_numeric(
        1.0 + 5e-11,
        1.0,
        absolute_tolerance=1e-10,
        relative_tolerance=0.0,
    )
    outside = compare_numeric(
        1.0 + 2e-8,
        1.0,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
    )

    assert inside["passed"] is True
    assert outside["passed"] is False
    assert outside["absolute_difference"] > outside["permitted_difference"]
    config = _config()
    assert config["tolerances"]["regenerated_metrics"]["absolute"] == 1e-10
    assert config["tolerances"]["regenerated_metrics"]["relative"] == 1e-9
    garch = config["tolerances"]["garch_regression_portability"]
    assert garch["scope"] == ["qlike", "rmse", "mae"]
    assert sha256_path(_root() / garch["reference"]) == garch["reference_sha256"]
    mnist = config["tolerances"]["mnist_exact_portability"]
    assert mnist["scope"] == [
        "exact_qrc_mnist_accuracy",
        "exact_qrc_mnist_macro_f1",
        "exact_qrc_mnist_balanced_accuracy",
        "exact_qrc_mnist_macro_roc_auc",
    ]
    assert sha256_path(_root() / mnist["reference"]) == mnist["reference_sha256"]


def test_exact_publication_assets_and_tree_digest() -> None:
    report = _verify_publication_assets(_root())

    assert report["passed"] is True
    assert report["asset_count"] == 42
    assert report["publication_tree_digest"] == EXPECTED_PUBLICATION_TREE_DIGEST
    assert publication_tree_digest(_root()) == EXPECTED_PUBLICATION_TREE_DIGEST


def test_frozen_architecture_source_registry_and_prohibited_claims() -> None:
    report = _verify_frozen_facts(_root())

    assert report["passed"] is True
    assert report["architecture"]["manifest_sha256"] == EXPECTED_ARCHITECTURE_SHA256
    assert report["fact_sources_registered"] is True
    assert report["prohibited_claims_preserved"] is True


def test_dirty_git_state_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    passing = {"passed": True}
    monkeypatch.setattr(
        "qtyche_qrc.reproducibility.verification._verify_expected_files",
        lambda _root: passing,
    )
    monkeypatch.setattr(
        "qtyche_qrc.reproducibility.verification._verify_publication_assets",
        lambda _root: passing,
    )
    monkeypatch.setattr(
        "qtyche_qrc.reproducibility.verification._verify_frozen_facts",
        lambda _root: passing,
    )
    monkeypatch.setattr(
        "qtyche_qrc.reproducibility.verification._verify_data_declarations",
        lambda _root: passing,
    )
    monkeypatch.setattr(
        "qtyche_qrc.reproducibility.verification._environment_report",
        lambda: passing,
    )
    monkeypatch.setattr(
        "qtyche_qrc.reproducibility.verification.git_report",
        lambda _root: {
            "clean": False,
            "compatible_submission_descendant": True,
        },
    )
    output = tmp_path / "dirty.json"

    with pytest.raises(ReproductionVerificationError, match="git"):
        verify_frozen_repository(_root(), report_path=output)

    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "fail"


def test_partial_resumption_requires_matching_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    root = tmp_path
    record = {
        "status": "success",
        "fingerprint": "a" * 64,
        "artifacts": [
            {
                "path": "artifact.json",
                "bytes": artifact.stat().st_size,
                "sha256": sha256_path(artifact),
            }
        ],
    }

    assert orchestrator._resume_valid(record, fingerprint="a" * 64, root=root)
    artifact.write_text('{"changed": true}\n', encoding="utf-8")
    assert not orchestrator._resume_valid(record, fingerprint="a" * 64, root=root)


def test_artifact_reuse_rehashes_every_unique_recorded_output(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"status": "success"}\n', encoding="utf-8")
    identity = {
        "path": "artifact.json",
        "bytes": artifact.stat().st_size,
        "sha256": sha256_path(artifact),
    }
    records = [
        {"task_id": "first", "artifacts": [identity]},
        {"task_id": "second", "artifacts": [dict(identity)]},
    ]

    report = validate_recorded_artifacts(records, root=tmp_path)

    assert report["recorded_artifact_entries"] == 2
    assert report["unique_artifacts"] == 1
    assert report["all_recorded_artifacts_exact"] is True

    artifact.write_text('{"status": "changed"}\n', encoding="utf-8")
    with pytest.raises(ReproductionVerificationError, match="changed or is missing"):
        validate_recorded_artifacts(records, root=tmp_path)


def test_artifact_reuse_accepts_only_the_final_identity_after_a_task_overwrite(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "shared-summary.json"
    artifact.write_text('{"task": "garch_fit"}\n', encoding="utf-8")
    first_identity = {
        "path": artifact.name,
        "bytes": artifact.stat().st_size,
        "sha256": sha256_path(artifact),
    }
    artifact.write_text('{"task": "garch_comparison"}\n', encoding="utf-8")
    final_identity = {
        "path": artifact.name,
        "bytes": artifact.stat().st_size,
        "sha256": sha256_path(artifact),
    }

    report = validate_recorded_artifacts(
        [
            {"task_id": "garch_fit", "artifacts": [first_identity]},
            {"task_id": "garch_comparison", "artifacts": [final_identity]},
        ],
        root=tmp_path,
    )

    assert report["recorded_artifact_entries"] == 2
    assert report["unique_artifacts"] == 1
    assert report["superseded_artifact_identity_count"] == 1
    assert report["superseded_artifact_identities"] == [
        {
            "path": artifact.name,
            "prior_task_id": "garch_fit",
            "prior_bytes": first_identity["bytes"],
            "prior_sha256": first_identity["sha256"],
            "superseding_task_id": "garch_comparison",
            "final_bytes": final_identity["bytes"],
            "final_sha256": final_identity["sha256"],
        }
    ]

    artifact.write_text('{"task": "unexpected"}\n', encoding="utf-8")
    with pytest.raises(ReproductionVerificationError, match="changed or is missing"):
        validate_recorded_artifacts(
            [
                {"task_id": "garch_fit", "artifacts": [first_identity]},
                {"task_id": "garch_comparison", "artifacts": [final_identity]},
            ],
            root=tmp_path,
        )


def test_artifact_reuse_allows_only_validation_implementation_changes() -> None:
    changed = validate_artifact_reuse_changed_paths(
        [
            "docs/qbraid_reproduction.md",
            "src/qtyche_qrc/reproducibility/orchestrator.py",
            "tests/test_mnist_portability.py",
        ]
    )

    assert changed == [
        "docs/qbraid_reproduction.md",
        "src/qtyche_qrc/reproducibility/orchestrator.py",
        "tests/test_mnist_portability.py",
    ]
    with pytest.raises(ReproductionVerificationError, match="non-validation"):
        validate_artifact_reuse_changed_paths(["src/qtyche_qrc/experiments/qrc_mnist.py"])


def test_pinned_failed_comparison_source_is_compatible_with_validation_commit() -> None:
    root = _root()
    config = _config()
    reference = json.loads(
        (root / "configs/reproduction/mnist_exact_portability_reference.json").read_text(
            encoding="utf-8"
        )
    )
    source_commit = reference["linux_profile"]["artifact_commit"]
    config_path = root / "configs/phase3_reproduction.yaml"
    source_config = subprocess.run(
        ["git", "show", f"{source_commit}:configs/phase3_reproduction.yaml"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = {
        "schema_version": 1,
        "status": "failure",
        "tier": "full",
        "git": {"commit": source_commit},
        "configuration": {
            "path": "configs/phase3_reproduction.yaml",
            "sha256": hashlib.sha256(source_config).hexdigest(),
        },
        "error": {"message": "task full_comparison failed with exit status 1"},
        "tasks": [
            {
                "task_id": task_id,
                "status": "success",
                "exit_status": 0,
                "artifacts": [],
            }
            for task_id in config["tiers"]["full"][:-1]
        ],
    }

    records, report = orchestrator._validate_source_execution(
        source,
        config=config,
        config_source=config_path,
        root=root,
        current_git={"commit": current_commit},
    )

    assert len(records) == len(config["tiers"]["full"]) - 1
    assert report["source_commit"] == source_commit
    assert report["validation_commit"] == current_commit
    assert report["all_changes_validation_only"] is True


def test_reproduction_config_preserves_frozen_architecture() -> None:
    original = yaml.safe_load(
        (_root() / "configs/final_financial_qrc.yaml").read_text(encoding="utf-8")
    )
    reproduction = yaml.safe_load(
        (_root() / "configs/reproduction/final_financial_qrc.yaml").read_text(encoding="utf-8")
    )

    assert reproduction["architecture"] == original["architecture"]
    assert reproduction["readout"] == original["readout"]
    assert reproduction["freeze"] == original["freeze"]
    assert reproduction["study"]["reservoir_seeds"] == original["study"]["reservoir_seeds"]
    assert reproduction["study"]["classifier_config"] == original["study"]["classifier_config"]
    assert reproduction["study"]["regressor_config"] == original["study"]["regressor_config"]
    processed = reproduction["study"]["processed_file_sha256"]
    assert set(processed) == {
        "features_unscaled.csv",
        "preprocessing.json",
        "regime_thresholds.json",
        "test.csv",
        "train.csv",
        "validation.csv",
    }
    assert all(len(checksum) == 64 for checksum in processed.values())
    semantic_reference = reproduction["study"]["processed_semantic_reference"]
    semantic_checksum = reproduction["study"]["processed_semantic_reference_sha256"]
    assert semantic_reference == ("configs/reproduction/processed_data_semantic_reference.json")
    assert len(semantic_checksum) == 64
    assert sha256_path(_root() / semantic_reference) == semantic_checksum


def test_prior_result_and_publication_trees_are_not_deleted_or_overwritten() -> None:
    config = _config()
    command_text = json.dumps(config["tasks"], sort_keys=True)

    assert "rm " not in command_text
    assert "unlink" not in command_text
    publication_command = config["tasks"]["prepare_publication_config"]["command"]
    assert "{evidence}/regenerated_publication_assets" in publication_command
    assert "paper_assets" not in publication_command


def test_no_absolute_local_paths_in_tracked_text() -> None:
    findings = scan_prohibited_paths(_root())

    assert findings == []


def test_readme_commands_and_launch_url_match_entry_points() -> None:
    readme = (_root() / "README.md").read_text(encoding="utf-8")

    assert f"gitHubUrl={CLONE_URL}" in readme
    for command in (
        "python scripts/reproduce_phase3.py --verify",
        "python scripts/reproduce_phase3.py --headline",
        "python scripts/reproduce_phase3.py --full",
        "python scripts/package_qbraid_evidence.py",
    ):
        assert command in readme
    process = subprocess.run(
        [sys.executable, "scripts/reproduce_phase3.py", "--help"],
        cwd=_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0
    assert all(
        option in process.stdout
        for option in (
            "--verify",
            "--headline",
            "--full",
            "--finalize-artifact-reuse",
        )
    )


def test_qbraid_skill_has_required_fields_and_constraints() -> None:
    skill = _root() / ".agents/skills/qbraid-phase3-reproduction/SKILL.md"
    text = skill.read_text(encoding="utf-8")
    _empty, frontmatter, body = text.split("---", maxsplit=2)
    metadata = yaml.safe_load(frontmatter)

    assert metadata["name"] == "qbraid-phase3-reproduction"
    assert "qBraid" in metadata["description"]
    for phrase in (
        "--verify",
        "--headline",
        "--full",
        "no physical\nQPU",
        "no quantum-advantage",
        "Diagnose failures",
    ):
        assert phrase in body
    interface = yaml.safe_load(
        (_root() / ".agents/skills/qbraid-phase3-reproduction/agents/openai.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert "$qbraid-phase3-reproduction" in interface["interface"]["default_prompt"]


def test_evidence_report_contract_is_complete() -> None:
    source = (_root() / "src/qtyche_qrc/reproducibility/orchestrator.py").read_text(
        encoding="utf-8"
    )
    for field in (
        "clone_url",
        "started_at_utc",
        "completed_at_utc",
        "runtime_seconds",
        "environment",
        "git",
        "resumed_task_ids",
        "recomputed_task_ids",
        "exit_status",
        "peak_child_rss",
        "maximum_peak_child_rss_bytes",
    ):
        assert f'"{field}"' in source


def test_evidence_package_includes_dataset_checksum_report() -> None:
    source = (_root() / "scripts/package_qbraid_evidence.py").read_text(encoding="utf-8")

    assert '"dataset_checksum_report.json"' in source
    assert '"processed_model_inputs_byte_exact"' in source
    assert '"processed_model_inputs_semantically_exact"' in source
    assert '"processed_semantic_verification"' in source
    assert '"provider_revision_detected"' in source
    assert '"garch_portability_report.json"' in source
    assert "passing GARCH portability checks" in source
    assert '"mnist_exact_portability_report.json"' in source
    assert "passing MNIST portability checks" in source
    assert "verify_artifact_reuse_execution" in source
    assert "execution_chain_verification.json" in source
    assert "ARTIFACT_REUSE_EXECUTION_MODE" in source
    assert "ARTIFACT_REUSE_SOURCE_REPORT" in source
    assert "ARTIFACT_REUSE_VALIDATION_REPORT" in source


def test_fast_verification_pins_the_frozen_scientific_stack() -> None:
    requirements = (_root() / "requirements-qbraid.txt").read_text(encoding="utf-8")
    verifier = (_root() / "src/qtyche_qrc/reproducibility/verification.py").read_text(
        encoding="utf-8"
    )

    for requirement in ("numpy==2.4.6", "scipy==1.17.1", "scikit-learn==1.9.0"):
        assert requirement in requirements
    assert "EXPECTED_SCIENTIFIC_VERSIONS" in verifier
    assert '"scientific_version_mismatches"' in verifier


def test_dynamic_diagnostics_pins_only_after_processed_input_verification() -> None:
    source = (_root() / "src/qtyche_qrc/reproducibility/artifacts.py").read_text(encoding="utf-8")

    assert "_verify_frozen_processed_inputs(root)" in source
    assert '("manifest", "data_manifest.json")' in source
