# qBraid Lab reproduction

This guide reproduces the existing Phase 3 experiments from a terminal in
qBraid Lab. It does not change the frozen data contract, temporal splits, QRC
mathematics, or model-selection rules.

## Prerequisites

- A qBraid account and a running qBraid Lab instance.
- A public GitHub repository containing this project. Replace both placeholder
  fields below only after that repository exists.
- Git and a persistent qBraid Python environment with Python 3.11.
- For the public pilot only, the immutable raw snapshot and its processed files
  in the repository-relative locations described under **Cached public data**.

The smoke workflow is synthetic, CPU-only, and offline after package
installation. No qBraid API key or quantum-device allocation is required.

## Clone and create the environment

In a qBraid Lab terminal:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git qtyche-qrc
cd qtyche-qrc
qbraid envs create -n qtyche-qrc-phase3 -f environment-qbraid.yaml -y
qbraid envs activate qtyche-qrc-phase3
./scripts/setup_qbraid.sh
```

The YAML environment requests Python `>=3.11,<3.13`. The requirements file
contains bounded compatible ranges for the numerical, plotting, test, lint,
and type-check dependencies. The setup script installs the repository into the
active environment and finishes by running the qBraid verifier. It deliberately
does not refer to a workstation virtual environment or operating-system package
manager.

If the qBraid environment already exists, activate it and run:

```bash
qbraid envs activate qtyche-qrc-phase3
./scripts/setup_qbraid.sh
```

qBraid environments are persistent virtual environments. Confirm that the
terminal is using the environment's Python before installing packages:

```bash
python --version
python -m pip --version
```

## Verify the environment

```bash
./scripts/verify_qbraid_environment.sh
```

Equivalent direct command:

```bash
python -m qtyche_qrc.cli verify-qbraid
```

Success writes `results/qbraid/qbraid_environment_report.json`. The report
checks Python, imports, installed distributions, repository files and relative
configuration paths, data-manifest readiness, fixture configurations, output
writability, Git commit and dirty state, CLI commands, and availability of the
exact NumPy density-matrix backend. A dirty tree is recorded for provenance but
is not by itself a verification failure.

The fixture data manifest may initially be absent in a fresh clone. That state
is reported together with its bootstrap command; the smoke stage creates the
deterministic fixture before using the manifest. The public manifest is optional
for smoke and mandatory for the public pilot.

## Synthetic qBraid smoke

```bash
./scripts/reproduce_qbraid_smoke.sh
```

This invokes the verifier, prepares deterministic fixture data, runs all six
classical fixture baselines, generates three-qubit exact QRC features, fits the
QRC classification and log-variance regression heads, runs a reduced capacity
characterization, and verifies output checksums. It does not open a notebook or
make a network request.

The single orchestration summary is
`results/qbraid/qbraid_smoke_summary.json`. Model runs retain their normal
scientific artifacts under `results/` and `results/qrc_smoke/`. All fixture
outputs state that they are synthetic and are not financial-performance
evidence.

On the Phase 3 reference workstation, the post-install smoke took about 3.3
seconds. Allow roughly one to five minutes in a shared qBraid Lab CPU session;
font-cache initialization and available CPU capacity can dominate the first
run. The summary records the actual wall time and all command durations.

Reference deterministic SHA-256 values from the Python 3.11 smoke are:

| Artifact | SHA-256 |
| --- | --- |
| QRC classifier test predictions | `6d6c9811724ec4e6c1a5e9d7c607b5c18fd1ba91afbfa7fdf7befadb26d8d78a` |
| QRC regressor test predictions | `30421a377fca3abfa209391a4da3145583c9d87875b7196ba8d1de0949d59310` |
| Reduced linear-memory CSV | `252a1988b9922bac3b588868883070ed9dfb8a50d0d6854b658d1c9d9ecba80c` |
| Reduced quadratic-capacity CSV | `488b71466703662142c091dd88f51cb56b329155302f51a00d8c86b7ee5ac402` |
| Reduced cross-delay-capacity CSV | `a6b38a2aba482fd9cb52191ff6545fa97ba4a31a588d2caa4d89ce671a4a5293` |

Use `deterministic_output_checksums` in the generated summary as the complete
machine-readable reference. Manifests themselves intentionally change when the
timestamp, Git state, platform, or package environment changes.

## Cached public data and the six-qubit pilot

The public raw files are intentionally not committed. Before entering qBraid,
place the immutable snapshot at:

```text
data/raw/public_market/yahoo_chart_20100101_20251231_v1/
```

It must contain `spy.csv`, `vix.csv`, `qqq.csv`, and
`snapshot_manifest.json`. Also provide the already prepared files at:

```text
data/processed/public_market/
```

The pilot verifies the snapshot identity, every raw checksum, the processed
manifest, and every processed checksum before running. It never redownloads
market data. Run one fixed seed first:

```bash
./scripts/reproduce_qbraid_public_pilot.sh --seed 2026
```

After reviewing that result, run the frozen three-seed set:

```bash
./scripts/reproduce_qbraid_public_pilot.sh --all-seeds
```

The workflow uses the six-qubit exact backend and reuses checksum-keyed QRC
features when the data manifest, feature names, reservoir configuration, and
seed agree. It writes the orchestration summary to
`results/qbraid/qbraid_public_pilot_summary.json`, model artifacts under
`results/qrc_public_pilot/`, and per-seed comparison tables under
`results/tables/`.

Reference feature-state generation took about ten seconds per seed on the
development workstation. Allow approximately two to ten minutes for one
uncached seed and six to thirty minutes for all three in a shared Lab CPU
session. A cache hit should be substantially faster. These are planning ranges,
not performance claims; the summary and experiment manifests record actual
times and checksums.

## Agent-executable stages

The same workflows can be driven without shell wrappers or notebooks:

```bash
python scripts/reproduce_phase3.py --stage smoke
python scripts/reproduce_phase3.py --stage capacity
python scripts/reproduce_phase3.py --stage public-pilot --seed 2026
python scripts/reproduce_phase3.py --stage all --all-seeds
```

Each run validates its arguments, calls the repository CLI entry points, records
every command and output, writes structured JSON, and exits nonzero after any
failure.

## Runtime metadata

The qBraid wrappers set `execution_platform` to `qbraid_lab`. If the Lab image
does not expose environment identity automatically, these optional variables
can make the manifest more specific:

```bash
export QTYCHE_QBRAID_ENVIRONMENT_NAME="qtyche-qrc-phase3"
export QTYCHE_QBRAID_ENVIRONMENT_ID="ENVIRONMENT-ID"
export QTYCHE_QBRAID_LAB_IMAGE="LAB-IMAGE-NAME"
```

Missing optional identity fields are recorded as null. The Python, operating
system, package versions, Git commit, and dirty state are still captured.

## Troubleshooting

- **Python verification fails:** activate the environment and confirm that
  `python --version` is at least 3.11.
- **Imports or dependencies fail:** run `./scripts/setup_qbraid.sh` with the
  environment's Python, not the base Lab interpreter.
- **A prohibited path is reported:** replace the committed host-specific path
  with a path relative to the repository root. Runtime-generated manifests may
  describe their working directory, but no command may depend on it.
- **The public snapshot is missing:** transfer the frozen snapshot and processed
  directory to the exact cached locations above. The pilot does not download it.
- **A checksum mismatch is reported:** do not bypass the check. Restore the
  immutable file matching its manifest or rebuild the complete snapshot through
  the separately documented explicit acquisition workflow.
- **A QRC cache is not reused:** compare the data-manifest checksum, reservoir
  seed, QRC configuration checksum, and selected feature names recorded in the
  feature metadata.
- **The first plot is slow:** matplotlib may be building its font cache; later
  runs should avoid that one-time cost.

## Limitations and execution meaning

The density-matrix implementation is exact only up to floating-point numerical
tolerances and is deliberately capped at six qubits. It models no physical
noise, shot sampling, calibration drift, compilation constraints, or device
queueing. The public pilot is correctness and stability evidence, not evidence
of quantum advantage or superiority over the classical controls.

**The current qBraid run uses an exact NumPy density-matrix simulator inside
qBraid Lab. It is not a physical quantum-hardware result.**
