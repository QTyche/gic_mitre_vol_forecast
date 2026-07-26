---
name: qbraid-phase3-reproduction
description: Validate or reproduce the frozen Team QTyche Phase 3 financial and MNIST submission in a clean qBraid Lab clone. Use for environment checks, frozen-architecture inspection, genuine-data acquisition, fast verification, headline or full reproduction, publication regeneration, evidence packaging, output discovery, and qBraid setup diagnosis.
---

# qBraid Phase 3 Reproduction

Use the repository entry points as the source of truth. Do not change model
architecture, seeds, temporal splits, thresholds, selection grids, or paper
facts while diagnosing reproduction.

## Choose the tier

1. Run `python scripts/reproduce_phase3.py --verify` for a fast tracked-contract
   and focused-test check.
2. Run `python scripts/reproduce_phase3.py --headline` for exact financial QRC,
   GARCH, genuine-MNIST smoke, and frozen-fact comparison.
3. Run `python scripts/reproduce_phase3.py --full` only after reviewing the
   displayed runtime, disk, and memory planning range.
4. Preserve the default resume behavior. Add `--no-resume` only when the user
   explicitly wants completed, checksum-verified tasks recomputed.

Read [references/contracts.md](references/contracts.md) before running headline
or full reproduction.

## Establish a clean qBraid checkout

- Require a fresh clone with no downloaded data, results, caches, bytecode, or
  editable install from another checkout.
- Check `git status --short` and `git rev-parse HEAD`.
- Accept submission commit
  `cd9fa988f854009e408af1774a97ed663b0e8b86` or an explicitly compatible
  descendant only.
- In an already active empty qBraid environment, run
  `./scripts/setup_qbraid.sh`.
- Do not guess a qBraid environment-creation command. Inspect
  `qbraid envs create --help` in the current qBraid image and use only syntax
  confirmed there. Record the exact successful command, environment name, and
  environment ID in the evidence.
- Set `QTYCHE_QBRAID_ENVIRONMENT_NAME` and
  `QTYCHE_QBRAID_ENVIRONMENT_ID` when qBraid does not expose them automatically.

Earlier qBraid images did not reliably handle
`qbraid envs create -n NAME -f environment-qbraid.yaml -y`. Prefer an empty
environment followed by the repository setup script. The requirements file is
for pip; do not feed its compound ranges to a qBraid-specific parser.

## Inspect the frozen architecture

Open `configs/reproduction/final_financial_qrc.yaml` and the referenced model
configs. Confirm:

- two qubits and two virtual nodes;
- `reset_each_input`;
- reservoir seeds 2026, 2027, and 2028;
- exact `numpy_density_matrix_exact` backend;
- training-only preprocessing and validation-only ridge selection;
- test evaluation only after readout freeze.

Use `paper_assets/final_results_manifest.json` for frozen numeric facts and
claims. Treat its prohibited claims as constraints, not prompts to test new
hypotheses.

## Locate and package evidence

Use `qbraid_evidence/final_clean_room/` for reports and transcripts. Generated
scientific data remain in ignored `data/` and `results/` trees. Package after a
successful run:

```bash
python scripts/package_qbraid_evidence.py
```

Report the archive path and SHA-256 sidecar. Never commit downloaded datasets,
feature caches, model outputs, or the archive.

## Diagnose failures

- Dirty checkout: start another clone; do not bypass the check.
- Existing data/results on a first headline/full run: start another clone.
- Public-data checksum failure: remove only the named failed snapshot file and
  rerun the download task; do not substitute fixture data.
- Historical raw SPY hash warning with all six processed checks passing:
  preserve the provider-revision record. The changed `adjusted_close` digits
  are unused; do not rewrite data or relax the processed-file checks.
- MNIST checksum failure: remove only the named corrupt IDX gzip and rerun with
  `--download`; never generate digits.
- Missing fonts/display: retain `MPLBACKEND=Agg`; no system LaTeX is required.
- Interrupted task: rerun the same tier and let artifact checksums authorize
  resumption.
- Numeric comparison failure: inspect the recorded actual, expected, absolute
  difference, and declared tolerance. Do not widen tolerances silently.

This project uses classical exact density-matrix simulation and controlled
finite-shot/noise simulation of a quantum reservoir. It performs no physical
QPU execution and makes no quantum-advantage claim.
