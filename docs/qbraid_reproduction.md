# Final Phase 3 qBraid clean-room reproduction

This guide is the judge and agent path for Stage 3A. It preserves the frozen
financial architecture, classical comparators, MNIST benchmark, temporal
splits, thresholds, validation-only selection, and Stage 2C paper facts.

The QRC is simulated classically. The exact financial reference uses NumPy
density matrices; the robustness grids use labelled sampling and controlled
noise models. No physical QPU is executed and no quantum advantage is claimed.

## Clean-room boundary

Start from a new clone of:

```text
https://github.com/QTyche/gic_mitre_vol_forecast.git
```

The frozen Stage 2C submission ancestor is
`cd9fa988f854009e408af1774a97ed663b0e8b86`. Stage 3A accepts that commit or a
compatible descendant. The fast verifier requires a clean Git state.

Do not carry any of these into the clone:

- `.venv`, bytecode, tool caches, or an editable install from another checkout;
- `data/raw/`, `data/processed/`, or a prior MNIST cache;
- generated `results/` or QRC feature caches;
- a prior evidence directory.

A first headline or full run checks for those generated trees and refuses to
continue if it finds them. A later invocation can resume the same evidence
session.

## qBraid environment

Python 3.12 is preferred and Python 3.11 is the minimum. In qBraid Lab, create
an empty persistent Python environment in the Environment Manager and activate
it before installing.

The qBraid CLI has changed between images. Earlier testing found that this
command was unreliable and it is deliberately not part of the final path:

```text
qbraid envs create -n NAME -f environment-qbraid.yaml -y
```

Likewise, do not pass `requirements-qbraid.txt` to a qBraid-specific
requirements parser; its compound ranges are valid pip syntax. If an agent must
create the environment from a terminal, first run `qbraid envs create --help`
in that Lab image and record the exact syntax that succeeds. Do not infer it
from this document.

After activating the empty environment:

```bash
git clone https://github.com/QTyche/gic_mitre_vol_forecast.git
cd gic_mitre_vol_forecast
./scripts/setup_qbraid.sh
```

The setup script confirms Python, exports headless plotting defaults, upgrades
pip, installs the recorded scientific package versions with pip, installs this
package from the current clone, and runs the repository qBraid environment
check. The fast verifier rejects scientific-package drift. Setup does not need
a system LaTeX installation.

When qBraid does not expose its identity to child processes, record it:

```bash
export QTYCHE_QBRAID_ENVIRONMENT_NAME="THE-ACTUAL-NAME"
export QTYCHE_QBRAID_ENVIRONMENT_ID="THE-ACTUAL-ID"
```

Replace those values with qBraid’s displayed values. Never commit account
credentials or API keys.

## Tier 1: fast verification

```bash
python scripts/reproduce_phase3.py --verify
```

This checks:

- supported Python and required packages;
- clean Git state and compatible submission ancestry;
- exact hashes of the frozen financial, MNIST, publication, reproduction, and
  processed-data semantic-reference configs;
- the frozen public-data declaration and four official MNIST download hashes;
- final two-qubit architecture identity and architecture-manifest checksum;
- the 42 files selected by the publication asset manifest;
- a canonical publication-tree digest;
- source hashes registered by every paper fact;
- preservation of all prohibited claims;
- focused Stage 3A tests.

It reads tracked files only and executes no model. Expected evidence includes:

```text
qbraid_evidence/final_clean_room/
├── command_log.json
├── environment_report.json
├── execution_report.json
├── fast_verification_report.json
├── git_report.json
└── terminal_transcript.log
```

## Tier 2: headline reproduction

Run this only in the same fresh clone after Tier 1:

```bash
python scripts/reproduce_phase3.py --headline
```

The orchestrator:

1. repeats the frozen checks;
2. downloads or verifies the named Yahoo snapshot and records provider hashes;
3. causally prepares and audits the frozen temporal dataset;
4. requires exact raw-file SHA-256 equality against the generated immutable
   snapshot manifest, then verifies all six generated processed files against
   the tracked semantic commitments;
5. fits the required public classical readouts;
6. fits leakage-safe Gaussian GARCH(1,1);
7. runs the frozen two-qubit, two-virtual-node exact financial QRC for seeds
   2026, 2027, and 2028, sharing each seed’s feature cache across readouts;
8. downloads checksum-pinned official MNIST and runs the genuine smoke subset;
9. compares financial QRC and GARCH headline values with the Stage 2C facts.

No task can fall back to fixtures or generated digits. The global floating
comparison remains absolute tolerance `1e-10` plus relative tolerance `1e-9`
for QRC, GARCH classification, and publication comparisons. It does not relax
model outputs, tracked config, or publication checksums.

The qBraid Linux/x86 fit reproduced GARCH macro-F1 and balanced accuracy
exactly, while test QLIKE, RMSE, and MAE differed by
`8.792821493130987e-8`, `6.647243633306488e-8`, and
`4.137040631943534e-8`. The frozen eight-start fit has six converged starts
within `2e-8` negative log-likelihood of the selected optimum. Across those
equivalent starts, all validation/test regime assignments remain identical;
the largest forecast-path differences are `9.29e-6` absolute and `1.63e-5`
relative.

`configs/reproduction/garch_portability_reference.json` therefore defines a
GARCH-only contract. Before a `2.5e-7` absolute, zero-relative tolerance can be
used for test QLIKE, RMSE, and MAE, the evidence must prove:

- successful convergence from the frozen deterministic start grid, using the
  same selected start or an equivalent optimum;
- training log-likelihood within `1e-7`, iteration-count difference at most
  15, and absolute parameter bounds of `2e-10` for omega, `5e-6` for alpha,
  `2e-6` for beta, `1e-8` for mu, `5e-6` for alpha plus beta, and `2e-8` for
  unconditional variance;
- candidate prediction files match forecasts independently reconstructed from
  the candidate parameters within `1e-12`;
- exact row counts, dates, labels, regime assignments and threshold crossings,
  with zero new non-finite or floored predictions;
- per-split forecast maximum/mean/median absolute differences no greater than
  `2e-5`/`1e-6`/`5e-7`, and maximum relative difference no greater than
  `5e-5`;
- unchanged displayed paper values and model rankings.

The frozen comparison path is reconstructed from the frozen parameters on the
same checksum- and semantic-verified return stream used by the candidate. This
isolates optimizer differences from the separately gated processed-data
serialization differences.

The limits are rounded upward from the equivalent-start parameter and
prediction envelope with documented safety margins. A materially different fit
cannot gain access to the GARCH metric tolerance. The complete parameter,
optimizer, forecast, classification, threshold, floor, display and ranking
comparison is written to
`qbraid_evidence/final_clean_room/garch_portability_report.json`.

The historical raw snapshot declaration is retained exactly. Yahoo currently
returns revised final digits for SPY’s `adjusted_close` field, which is not used
by any feature, target, split, threshold, or GARCH return. The evidence package
therefore records both the historical and current raw hashes, flags the
provider revision, and requires each downloaded file to remain byte-exact to
the manifest created with that immutable download.

The first qBraid x86 validation showed different byte hashes for every
float-bearing processed CSV and `preprocessing.json`, while
`regime_thresholds.json` remained byte-identical. The writers already force LF
line endings and deterministic JSON formatting. The source is final-bit
variation from platform `log` implementations and CPU-specific rolling,
mean, and standard-deviation reductions: CSV preserves 12 significant digits,
and preprocessing JSON preserves full binary64 round trips.

Generated processed artifacts therefore retain both historical and current
byte SHA-256 values but are gated by
`configs/reproduction/processed_data_semantic_reference.json`. The canonical
rule parses finite numbers, normalizes signed zero, and commits values at 10
significant decimal digits, whose maximum relative quantization width is at most
`1e-9`. Row counts, column names and order, date sequences, row order,
date/split membership, integral labels, missing-value positions, threshold and
preprocessing JSON structure, and all non-float values remain exact. A mismatch
in any of those fields, or in any canonical numeric digest, is fatal. The full
report is
`qbraid_evidence/final_clean_room/processed_data_semantic_verification.json`.

The main additional report is:

```text
qbraid_evidence/final_clean_room/headline_reproduction_report.json
qbraid_evidence/final_clean_room/garch_portability_report.json
qbraid_evidence/final_clean_room/processed_data_semantic_verification.json
```

Normal generated files remain in ignored `data/` and `results/` trees.

## Tier 3: full reproduction

Review the planning range printed in the execution report, then run:

```bash
python scripts/reproduce_phase3.py --full
```

This extends the final pipeline with:

- complete financial finite-shot and controlled-noise robustness;
- the aligned GARCH/classical comparison;
- full three-seed genuine-MNIST exact benchmark and limited robustness;
- dynamic checksum pinning of regenerated predictions to unchanged Stage 2A
  statistical controls;
- Stage 2B calibration, regime, temporal, numerical, and per-digit diagnostics;
- reconstruction of the compact validation-selection plotting sources from
  frozen facts (not exploratory model reruns);
- publication generation into
  `qbraid_evidence/final_clean_room/regenerated_publication_assets/`;
- source-to-display and frozen-fact comparison.

The historical `paper_assets/` tree is read and verified but never overwritten.
The full run does not rerun superseded qubit, encoding-density, or state-memory
searches.

## Resumption and failure semantics

Resume is the default:

```bash
python scripts/reproduce_phase3.py --headline
python scripts/reproduce_phase3.py --full
```

For each task, the report stores a fingerprint of the command, Git commit, and
orchestration config plus checksums of watched outputs. A task is labelled
`resumed_verified` only when every recorded artifact still exists with the same
size and SHA-256. Missing or changed output forces recomputation. A failed
subprocess stops the tier, records the nonzero exit status and output tail, and
never receives successful status.

Use this only when recomputation is intentional:

```bash
python scripts/reproduce_phase3.py --full --no-resume
```

## Evidence package

After a successful tier:

```bash
python scripts/package_qbraid_evidence.py
```

This adds the package freeze, dataset/checksum report, failures/resolutions
record, raw byte checks, processed historical/current byte checks, the semantic
verification report, the required passing GARCH portability report, and
checksum inventory, then creates:

```text
qbraid_evidence/phase3_final_clean_room.tar.gz
qbraid_evidence/phase3_final_clean_room.tar.gz.sha256
```

The archive uses normalized tar metadata and a zero gzip timestamp. The archive,
datasets, model outputs, and feature caches are ignored by Git.

## Expected runtime and resources

The config currently exposes conservative preflight planning ranges:

| Component | Planning range |
| --- | ---: |
| Environment installation | 3–12 minutes |
| Fast verification | 1–5 minutes |
| Headline reproduction | 8–35 minutes |
| Full financial workflow | 20–90 minutes |
| Full MNIST workflow | 10–60 minutes |
| Statistics and publication | 3–20 minutes |
| Total full pipeline | 35–170 minutes |
| Disk | 1–4 GiB |
| Peak memory | 1–6 GiB |

These values are planning guidance, not measured qBraid performance. The final
Stage 3A evidence must replace or accompany them with actual qBraid installation,
verification, financial, MNIST, statistics/publication, total wall time, disk
use, and peak-memory records. Until that actual run exists, qBraid validation is
not complete.

## Troubleshooting

- **Dirty repository:** use another clean clone; do not bypass the verifier.
- **Pre-existing generated trees:** use another clean clone for the first
  headline/full run.
- **Processed public-data checksum mismatch:** remove only the named corrupt
  download and rerun. Never replace the snapshot with fixture data.
- **Historical raw SPY hash differs but processed hashes pass:** retain the
  recorded provider-revision warning. Do not edit the download to force a raw
  hash; the exact per-download raw manifest and tracked processed semantic
  commitments remain the scientific gates.
- **GARCH-only regression mismatch:** inspect
  `garch_portability_report.json`. The special tolerance is unavailable unless
  every optimizer, likelihood, parameter, reconstructed-path, alignment,
  classification, threshold, floor, display, and ranking check passes.
- **MNIST checksum mismatch:** remove only the named IDX gzip and rerun with the
  download path. Never synthesize a replacement.
- **Package installed into qBraid base Python:** reactivate the persistent
  environment and confirm `python -m pip --version` points inside it.
- **Display/font failure:** retain `MPLBACKEND=Agg`; matplotlib creates its own
  cache on first use.
- **Numeric tolerance failure:** inspect the recorded delta and environment.
  Do not silently widen tolerances or change the model.
- **Interrupted run:** rerun the same tier. The evidence report distinguishes
  resumed and recomputed tasks.

## Agent Skill

The repository Skill is
`.agents/skills/qbraid-phase3-reproduction/SKILL.md`. It routes an AI coding
agent through architecture inspection, genuine-data acquisition, tier choice,
output discovery, failure handling, and evidence packaging without duplicating
this guide or authorizing scientific changes.
