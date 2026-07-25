# Team QTyche Phase 3 benchmark

This repository is the reproducible experiment harness for **Quantum Reservoir
Computing for Forecasting Equity Volatility Regime Transitions**. The planned
primary benchmark asks whether a fixed transverse-field Ising quantum reservoir
provides better or more robust features than a resource- and search-budget-matched
Echo State Network for forecasting the volatility regime five trading days ahead.

The repository currently contains the versioned data contract, deterministic
fixture workflow, immutable public-market snapshot workflow, classical
baselines, Echo State Network controls, exact noiseless quantum reservoir,
analytical capacity diagnostics, evaluation framework, and reproducible
experiment manifests.

## Repository layout

- `configs/`: versioned experiment and data-contract configurations
- `src/qtyche_qrc/`: package code, interfaces, and experiment utilities
- `scripts/`: command-line reproduction and packaging entry points
- `tests/`: automated contract and smoke tests
- `data/`: provenance notes plus ignored raw and processed data areas
- `results/`: ignored generated manifests, metrics, figures, and tables
- `notebooks/`: optional audits and exploration, never the sole reproduction path
- `paper/`: manuscript source

## Local setup

Python 3.9 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
make install
```

This creates `.venv`, installs the package and development tools, and uses the
committed lock file once it has been generated. Run the checks with:

```bash
make test
make lint
```

## Smoke test

```bash
make smoke
```

The smoke command creates or verifies deterministic synthetic raw fixtures,
prepares causal features and five-day targets, writes purged and normalized
splits plus their data manifest, audits them, and runs six small classical
baseline experiments. It then evaluates them, creates separate validation and
test comparison tables, and runs the complete test/lint/type-check suite. It is
offline and CPU-oriented.

**All smoke forecasting outputs use synthetic fixture data. They are pipeline
tests, not financial performance evidence.** Every fixture artifact is visibly
marked `SYNTHETIC FIXTURE DATA — NOT A FINANCIAL PERFORMANCE RESULT`.

The CLI is also available directly:

```bash
uv run python -m qtyche_qrc.cli --help
uv run python -m qtyche_qrc.cli validate-config --config configs/qrc_smoke.yaml
uv run python -m qtyche_qrc.cli prepare-data --config configs/data.yaml
uv run python -m qtyche_qrc.cli audit-data --processed-dir data/processed
```

## Classical baseline experiments

Headline-capable commands reject fixtures unless the explicit integration-test
override is present:

```bash
uv run python -m qtyche_qrc.cli train-baseline \
  --config configs/models/logistic_regression.yaml \
  --allow-synthetic-results

uv run python -m qtyche_qrc.cli search-baseline \
  --config configs/models/esn_classifier_smoke.yaml \
  --allow-synthetic-results
```

The override retains synthetic warnings and does not authorize a market claim.

## Public-market benchmark

Download or verify the versioned Yahoo chart snapshot explicitly, then process
it from the checksum-verified local cache:

```bash
./scripts/reproduce_public_data.sh
```

The canonical snapshot is `yahoo_chart_20100101_20251231_v1`. Raw provider
files are ignored by Git; configuration, acquisition code, checksums, and
provenance metadata are committed. Provider terms must be reviewed before raw
files are redistributed.

After processing, run the six real-data baselines and diagnostic package:

```bash
./scripts/reproduce_core_baselines.sh
```

This script refuses missing, synthetic, or incorrectly marked processed data.
It runs majority and regime persistence, validation-selected logistic and ESN
classification, realized-variance persistence, and the validation-selected
log-variance ESN regression workflow. Outputs appear in
`results/public_market/`, `results/diagnostics/esn_regression/`,
`results/data_audit/`, and `results/tables/`.

Each run creates `results/<experiment_id>/` with its exact configuration,
manifest, model, all candidate validation results, separate validation/test
metrics and predictions, timing, figures, and logs. Inspect or compare with:

```bash
uv run python -m qtyche_qrc.cli inspect-experiment \
  --experiment-dir results/<experiment_id>

uv run python -m qtyche_qrc.cli compare-baselines \
  --results-dir results \
  --output-dir results/tables
```

Comparison always writes separate validation and test tables. See
`docs/evaluation_protocol.md`, `docs/model_interface.md`, `docs/esn_design.md`,
and `docs/result_schema.md` for the scientific contracts.

For the public package, `compare-public-baselines` writes separate
validation/test tables plus classification, transition, and regression
diagnostics. See `docs/public_data_provenance.md`,
`docs/public_data_audit.md`, `docs/esn_regression_diagnostics.md`, and
`docs/real_baseline_protocol.md`.

## Exact QRC workflows

Run the deterministic three-qubit fixture integration entirely offline:

```bash
./scripts/reproduce_qrc_smoke.sh
```

These outputs are marked synthetic and are not financial results. Characterize
linear and nonlinear memory, feature rank, autocorrelation, and empirical
contractivity on deterministic synthetic inputs with:

```bash
./scripts/characterize_qrc.sh
```

After the verified public processed data exist, run the fixed six-qubit,
two-virtual-node pilot for reservoir seeds 2026, 2027, and 2028:

```bash
./scripts/run_qrc_public_pilot.sh
```

Equivalent individual commands include:

```bash
uv run python -m qtyche_qrc.cli generate-qrc-features \
  --config configs/models/qrc_classifier_pilot.yaml
uv run python -m qtyche_qrc.cli train-qrc \
  --config configs/models/qrc_classifier_pilot.yaml
uv run python -m qtyche_qrc.cli inspect-qrc \
  --experiment-dir results/qrc_public_pilot/<experiment_id>
uv run python -m qtyche_qrc.cli compare-qrc-seeds \
  --results-dir results/qrc_public_pilot \
  --output-dir results/tables
```

The backend is exact and noiseless, has a six-qubit safety limit, and includes
no shots, hardware, physical noise, or feedback. See
`docs/qrc_mathematical_definition.md`, `docs/qrc_backend.md`,
`docs/qrc_capacity_analysis.md`, and `docs/qrc_esn_fairness.md`. The pilot is
correctness and stability evidence only; it makes no ESN-superiority or quantum
advantage claim.

### Exact QRC qubit-scaling study

The controlled scaling study changes only the reservoir qubit count while
holding the public snapshot, temporal splits, two virtual nodes, reservoir
dynamics, observables, readout selection, targets, and seed grid fixed. Run its
small resume-safe smoke grid or the complete 2–6-qubit, three-seed grid with:

```bash
python scripts/run_qrc_qubit_scaling.py \
  --qubits 2 4 6 \
  --seeds 2026 \
  --smoke

python scripts/run_qrc_qubit_scaling.py \
  --qubits 2 3 4 5 6 \
  --seeds 2026 2027 2028
```

Outputs are written under `results/qrc_qubit_scaling/`. They include per-run
and seed-aggregated JSON/CSV tables, analytical resource estimates, PNG/PDF
figures, run manifests, and a top-level resume summary. The runner verifies the
frozen public-data checksums and rejects synthetic fixture data before any model
is run. This is exact classical density-matrix simulation of a quantum
reservoir, not physical-QPU execution or evidence of quantum advantage. See
`docs/qrc_qubit_scaling.md` for the output contract and interpretation guidance.

### Finite-shot and simulated-noise robustness

The next controlled study uses the two-qubit scaling winner by default and
varies one measurement factor at a time: finite shots, local depolarising
probability, or independent measurement-bit-flip probability. Run the compact
public-data smoke or the complete configured grid with:

```bash
python scripts/run_qrc_noise_robustness.py --smoke
python scripts/run_qrc_noise_robustness.py
```

`--n-qubits N` overrides the default selected architecture without changing the
other controls. The runner verifies the frozen raw and processed data,
validation-selects each readout, shares label-free features between classifier
and regressor, and resumes complete task runs. Results, seed-level
aggregations, prediction-stability diagnostics, resource estimates, and PNG/PDF
figures are isolated below `results/qrc_noise_robustness/`.

The channels are explicit controlled simulations, not hardware-calibrated
device models. State evolution and channel application use classical
density-matrix computation; no physical QPU is used and no quantum-advantage
claim is supported. See `docs/qrc_noise_robustness.md`.

### QRC temporal-multiplexing and encoding-density study

For the fixed two-qubit candidate, this controlled study varies only the
implemented within-input virtual-node density, `V=1,2,4,8`, over reservoir
seeds 2026, 2027, and 2028. The total evolution interval remains `tau=1.0` for
every market input; each condition uses `V` equal substeps of duration
`tau/V`. Run the compact smoke grid or complete study with:

```bash
python scripts/run_qrc_encoding_density.py \
  --virtual-nodes 1 2 8 \
  --seeds 2026 \
  --smoke

python scripts/run_qrc_encoding_density.py \
  --virtual-nodes 1 2 4 8 \
  --seeds 2026 2027 2028
```

Outputs under `results/qrc_encoding_density/` include split-level and
seed-aggregated JSON/CSV tables, a validation-only candidate table, resource
and training-feature conditioning diagnostics, nine PNG/PDF figures, and an
array-level comparison of `V=2` with the existing exact two-qubit reference
when its cache is available. Runs are resume-safe, use an isolated cache,
verify frozen public-data checksums, reject synthetic data, and share each
feature cache between both readouts.

The candidate table does not freeze an architecture because the state-memory
ablation remains outstanding. This is exact classical density-matrix
simulation of a quantum reservoir—not physical-QPU execution or evidence of
quantum advantage. See `docs/qrc_encoding_density.md`.

### QRC state-memory ablation

The state-memory ablation fixes the validation/performance-cost candidate at
two qubits and `V=2`, then changes only whether the full reservoir state is
carried between market observations or reset before every observation. Both
conditions preserve the established partial input-qubit reinjection and
within-observation virtual-node evolution.

```bash
python scripts/run_qrc_state_memory_ablation.py \
  --state-policies carry_inputs reset_each_input \
  --seeds 2026 \
  --smoke

python scripts/run_qrc_state_memory_ablation.py \
  --state-policies carry_inputs reset_each_input \
  --seeds 2026 2027 2028
```

Outputs under `results/qrc_state_memory_ablation/` include split-level and
seed-aggregated tables, paired carry-minus-reset differences, a
validation-only policy decision, five PNG/PDF figure pairs, feature
autocorrelation and lagged-input correlation tables, effective-rank and
conditioning diagnostics, and initial-state perturbation-decay evidence. Runs
are deterministic, resume-safe, checksum-verified, synthetic-data rejecting,
and isolated from every previous experiment.

QLIKE and RMSE are lower-is-better metrics; their paired tables make the sign
interpretation explicit. Test results do not influence the state-policy
choice. See `docs/qrc_state_memory_ablation.md`.

## Reproducibility principles

Temporal splits are never shuffled. Preprocessing must be fitted on training
data only. Every reported run must have a saved configuration and manifest,
and headline claims require multiple seeds and uncertainty estimates. Frozen
processed data will include provenance and checksums so judging does not depend
on a live market-data endpoint. Hardware claims will identify the backend,
transpiled circuit statistics, shot count, and measured runtime.

## Launch on qBraid

<!-- Replace both uppercase placeholders only after the public repository exists. -->
[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150" alt="Launch on qBraid">](https://account.qbraid.com?gitHubUrl=https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git)

The button uses qBraid's official public-repository launch format, but its URL is
intentionally a placeholder until this project has a final GitHub location.
After launching qBraid Lab, clone the repository if the button has not already
done so, then create and activate the pinned Python 3.11 environment:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git qtyche-qrc
cd qtyche-qrc
qbraid envs create -n qtyche-qrc-phase3 -f environment-qbraid.yaml -y
qbraid envs activate qtyche-qrc-phase3
./scripts/setup_qbraid.sh
```

Verify the environment and run the fully offline synthetic smoke with:

```bash
./scripts/verify_qbraid_environment.sh
./scripts/reproduce_qbraid_smoke.sh
```

The smoke writes one orchestration report at
`results/qbraid/qbraid_smoke_summary.json`, in addition to the normal model
artifacts. It took about 3.3 seconds after installation on the reference
workstation; allow one to five minutes in a shared Lab CPU session. Reference
SHA-256 values include `6d6c9811...d8d78a` for classifier test predictions,
`30421a37...59310` for regressor test predictions, and
`252a1988...80c` for the reduced linear-memory table. The full checksums are in
the summary and in [the qBraid reproduction guide](docs/qbraid_reproduction.md).

When the immutable raw and processed public-market snapshot is already present,
run one fixed six-qubit seed before the complete seed set:

```bash
./scripts/reproduce_qbraid_public_pilot.sh --seed 2026
./scripts/reproduce_qbraid_public_pilot.sh --all-seeds
```

The public wrapper verifies raw and processed checksums and does not redownload
data. Plan for roughly two to ten minutes per uncached seed on a shared Lab CPU;
cache hits should be faster. Missing dependencies, host-specific paths, absent
snapshots, and checksum mismatches fail clearly. Troubleshooting, cached-data
layout, exact output expectations, and the agent-executable stage commands are
documented in `docs/qbraid_reproduction.md`.

The main reproduction workflow will remain backend-agnostic. A targeted hardware
validation can be added once the challenge organisers confirm the required device
and mandatory Track A metrics.

The current qBraid run uses an exact NumPy density-matrix simulator inside
qBraid Lab. It is not a physical quantum-hardware result.

## Responsible use of generative AI

Generative AI may assist with coding, testing, and writing. Team QTyche remains
responsible for every technical decision, dataset choice, experiment, result,
interpretation, citation, and submission statement. AI-generated claims are not
treated as evidence.
