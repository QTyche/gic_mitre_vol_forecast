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

## For Judges

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150" alt="Launch on qBraid">](https://account.qbraid.com?gitHubUrl=https://github.com/QTyche/gic_mitre_vol_forecast.git)

The button targets `https://github.com/QTyche/gic_mitre_vol_forecast.git` on
`main`. In qBraid Lab, create and activate an empty persistent Python 3.12
environment using the Environment Manager, open a terminal, and run:

```bash
git clone https://github.com/QTyche/gic_mitre_vol_forecast.git
cd gic_mitre_vol_forecast
./scripts/setup_qbraid.sh
python scripts/reproduce_phase3.py --verify
```

The fast check verifies the clean Git state, compatible submission ancestry,
dependencies, frozen configs, the tracked processed-data semantic reference,
data declarations, final architecture checksum, all 42 manifest-selected paper
assets, publication-tree digest, source-to-fact records, prohibited claims, and
focused deterministic tests. It does not execute a model.

Continue with one command per tier:

```bash
python scripts/reproduce_phase3.py --headline
python scripts/reproduce_phase3.py --full
python scripts/package_qbraid_evidence.py
```

Headline reproduction downloads the named public-market snapshot, verifies the
raw files byte-for-byte against their download manifest, then verifies the six
generated processed model-input files against a tracked semantic commitment.
That commitment requires exact rows, column order, dates, labels, split
membership, missingness, and JSON structure plus numeric equality after
canonicalization to 10 significant decimal digits. It then runs the required
classical readouts, GARCH, the three-seed exact financial QRC, and genuine-MNIST
smoke before comparing regenerated financial values with the frozen facts. Full
reproduction adds financial robustness, full genuine MNIST, statistical
validation, diagnostics, and publication regeneration into the isolated
evidence tree. Both tiers resume only tasks whose recorded artifacts still
exist with matching checksums.

Outputs are under ignored `data/`, `results/`, and
`qbraid_evidence/final_clean_room/`. The evidence archive and SHA-256 sidecar
remain outside Git. Before a full run, the command displays conservative
planning ranges of 35–170 minutes, 1–4 GiB disk, and 1–6 GiB peak memory; these
are not qBraid measurements. Actual environment-install, verify, financial,
MNIST, statistics/publication, total runtime, disk, and peak-memory values must
come from the final qBraid evidence.

The frozen submission ancestor is
`cd9fa988f854009e408af1774a97ed663b0e8b86`; the verifier also accepts a
descendant containing only compatible reproduction work. A first headline/full
run refuses pre-existing generated data or result trees, synthetic financial
data, substituted MNIST, a dirty checkout, or an unrelated commit.

Known limitations: Yahoo and the checksum-pinned Google MNIST mirror must be
reachable on the first run; exact checks can pass offline only after those
downloads exist. Yahoo has revised final digits in SPY’s unused
`adjusted_close` field since the historical snapshot was recorded. The
evidence reports that historical difference explicitly while requiring each
downloaded raw file to match its own immutable snapshot manifest exactly.
Apple ARM and qBraid x86 implementations of `log` and rolling/reduction
operations can differ in final binary digits; 12-digit CSV output and
full-round-trip preprocessing JSON turn those harmless differences into
different byte hashes. All historical and current byte hashes remain recorded,
but generated processed data pass only if their tracked semantic digests match.
Ten-significant-digit canonicalization has a relative quantization width at most
`1e-9`; non-finite values are rejected, and any material value, threshold,
date, label, split, missingness, row, or schema change is fatal. Regenerated
metrics separately use a declared `1e-10` absolute plus `1e-9` relative
tolerance. That global tolerance remains unchanged for QRC, GARCH
classification, and publication comparisons. GARCH test QLIKE, RMSE, and MAE
have a separate `2.5e-7` absolute-only portability bound, activated only after
`garch_portability_report.json` proves converged equivalent likelihood,
tightly bounded parameters and forecast paths, identical dates and regime
threshold crossings, zero new floors/non-finite values, and unchanged displayed
values and rankings. The bound is 2.84 times the largest observed qBraid GARCH
metric delta and is constrained by a separately documented equivalent-optimum
parameter/forecast envelope.

The full qBraid Linux/x86 MNIST run also exposed a platform-sensitive
multinomial L-BFGS readout path. Checksum-identical inputs produced exact
reservoir features differing by at most `6.44e-15`; applying either already
fitted model to the other platform's features changed no decisions. Fitting
the frozen `tol=1e-4` readout on designs with condition numbers from `1.57e5`
to `6.41e5`, however, stopped at platform-dependent iterates. This changed 8
validation and 14 test decisions across 6,000 seed-example evaluations and
changed the mean test accuracy from the frozen `0.874` to `0.875`.
`mnist_exact_portability_report.json` accepts only the frozen path or the
checksum-pinned qBraid e0l4 Linux/x86 path after exact source, index, label,
configuration, feature commitment, selected readout, score, probability,
changed-index, confusion-matrix, displayed-value, and ranking checks. It does
not widen the global metric tolerance. The paper retains the frozen `0.874`
and the evidence reports `0.875` explicitly.

This is classical exact density-matrix and controlled finite-shot/noise
simulation of a quantum reservoir. No physical QPU is executed, and no quantum
advantage is claimed.

See [the qBraid clean-room guide](docs/qbraid_reproduction.md) for the evidence
schema, recovery rules, and full output inventory.

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

### Frozen final financial QRC

Stage 1D freezes the validation-selected financial architecture at two qubits,
`V=2`, and `reset_each_input`, while preserving partial input-qubit
reinjection, `tau=1.0`, the ring Hamiltonian, analytic `Z_i` and unique
`Z_i Z_j` observables, the frozen public data/splits, and validation-only ridge
selection. No further architecture tuning using test results is permitted.

```bash
python scripts/run_final_financial_qrc.py --smoke
python scripts/run_final_financial_qrc.py --smoke
python scripts/run_final_financial_qrc.py
```

The first command runs one exact reservoir seed and the compact robustness
grid; the repeated smoke verifies deterministic table checksums and resumable
execution. The full command runs all three reservoir seeds and the complete
finite-shot, depolarising-noise, and measurement-noise grids.

Outputs under `results/final_financial_qrc/` include the architecture manifest,
validation-selection evidence, exact seed-level and aggregate tables, an
isolated reset-policy robustness study, the final classical benchmark table,
and six publication-ready PNG/PDF figure pairs. The previous carry-input
robustness and every earlier experiment tree remain unchanged.

The seed-2027 reset feature matrix is near singular (condition number about
`5.77e12`), so the final report records coefficient and prediction finiteness
without changing the frozen architecture or ridge grid. GARCH transition
PR-AUC remains not applicable, and no formal significance testing is performed
in Stage 1D. This is exact classical simulation of a quantum reservoir, not
physical-QPU execution or a quantum-advantage claim. See
`docs/final_financial_qrc.md`.

## Gaussian GARCH(1,1) volatility baseline

The classical volatility workstream fits a stationary Gaussian GARCH(1,1) to
training-period SPY returns only, freezes its parameters, and causally filters
validation/test returns observed at each forecast origin. Its five-day expected
cumulative variance is converted to the exact annualized units and dates of the
frozen `target_rv_5d` benchmark.

```bash
python scripts/run_garch_baseline.py --smoke
python scripts/run_garch_baseline.py
```

Use `--no-resume` to force a new deterministic fit. Results under
`results/garch_baseline/` include fitted parameters and all optimizer attempts,
aligned validation/test predictions and metrics, a continuous conditional
variance path, comparison CSV/JSON files for persistence, ESN, GARCH, and the
validation-selected final QRC, provenance manifests, and five publication-ready
PNG/PDF figure pairs.

The runner verifies frozen raw and processed checksums, rejects synthetic data,
fits no parameter outside the training period, and records the numerical floor
used only for QLIKE evaluation. Deterministic regime labels use the frozen
training thresholds; transition PR-AUC is marked not applicable because GARCH
does not produce calibrated class probabilities here. This is a classical
econometric baseline, and the QRC comparator is exact classical simulation—not
physical-QPU execution or evidence of quantum advantage. See
`docs/garch_baseline.md`.

## Common MNIST QRC benchmark

Stage 1E is an isolated challenge-compliance workflow for genuine MNIST digit
classification. It does not reopen the frozen financial architecture. Download
and checksum-verify the four official IDX files, then run the deterministic
one-seed smoke or the complete three-seed benchmark:

```bash
python scripts/run_qrc_mnist.py --download-only
python scripts/run_qrc_mnist.py --smoke
python scripts/run_qrc_mnist.py --smoke
python scripts/run_qrc_mnist.py
```

The full split contains 6,000 training, 1,000 validation, and 1,000 test images,
balanced across digits 0–9. Selection seed 2026 draws disjoint training and
validation subsets only from the official training partition and draws test
examples only from the official test partition. The runner never substitutes
synthetic data.

Each image becomes a 28-step sequence of five fixed column-band means. The
five-qubit, two-virtual-node QRC resets between images and carries state only
between rows of the same image. Its final, mean, population-standard-deviation,
and four window-mean summaries produce 140 features. Validation-only
regularisation selection is used for the multinomial readouts. Directly
comparable flattened logistic and 32-state ESN baselines use the same images;
one fixed QRC seed also covers analytic, 2,048-shot, depolarising, and
measurement-bit-flip simulations.

Outputs are isolated under `results/qrc_mnist/` and include manifests, selected
indices, deterministic feature caches, validation/test predictions, per-run and
aggregated CSV/JSON tables, models, runtime/resource diagnostics, and seven
PNG/PDF figure pairs. The compressed source download is about 11.0 MiB and is
cached under `data/raw/mnist/`. See `docs/qrc_mnist_benchmark.md` for the exact
data, feature, output, runtime, resumption, and qBraid contracts.

This workflow is exact classical density-matrix simulation of a quantum
reservoir plus explicitly labelled finite-shot/noise simulations. It does not
execute on a physical QPU and provides no evidence or claim of quantum
advantage.

## Formal benchmark statistical validation

Stage 2A performs paired inference on the frozen financial and MNIST
predictions without refitting, retuning, test-set selection, forecast
ensembling, or changes to any earlier result tree:

```bash
python scripts/run_statistical_validation.py --smoke
python scripts/run_statistical_validation.py --smoke
python scripts/run_statistical_validation.py
```

The financial analysis reports per-seed and architecture-level QLIKE,
squared-error, absolute-error, macro-F1, balanced-accuracy, and transition
PR-AUC differences. It uses lag-4 HAC/DM-style inference with lag 0/10/20
sensitivity, deterministic circular-block intervals with block lengths
5/10/20, Holm correction, and Mincer–Zarnowitz diagnostics. The MNIST analysis
uses exact McNemar tests and paired class-stratified bootstrap intervals for
accuracy, macro-F1, balanced accuracy, and macro ROC-AUC.

Outputs under `results/statistical_validation/` include eight CSV/JSON table
pairs, seven publication-ready PNG/PDF figure pairs, a summary, and a
checksum-complete environment/provenance manifest. Every frozen input is
SHA-256 verified before and after inference. See
`docs/statistical_validation.md` for the comparison families, sign
conventions, output contract, and interpretation limits.

This is classical statistical analysis of predictions produced by exact
classical density-matrix simulation of a quantum reservoir. It is not physical
QPU execution and makes no quantum-advantage claim.

## Frozen benchmark diagnostics

Stage 2B adds post-freeze calibration, regime, transition, temporal,
conditioning, and MNIST digit-level diagnostics. It consumes checksum-pinned
prediction files and never fits or recalibrates a model, changes thresholds or
architectures, creates an ensemble, or uses diagnostics for selection.

```bash
python scripts/run_benchmark_diagnostics.py --smoke
python scripts/run_benchmark_diagnostics.py --smoke
python scripts/run_benchmark_diagnostics.py
```

Outputs under `results/benchmark_diagnostics/` include 15 CSV/JSON table
pairs, 13 publication-ready PNG/PDF figure pairs, a run summary, and complete
provenance. Financial uncertainty uses deterministic moving-block bootstrap;
MNIST digit and paired-accuracy intervals use class-stratified paired
bootstrap.

Lead-time analysis is explicitly omitted because the frozen financial rows
provide a five-day aggregate target but no unique target or transition date.
No lead is inferred. See `docs/benchmark_diagnostics.md` for the fixed
thresholds, sign conventions, output contract, and limitations.

The workflow is classical analysis of frozen exact-simulation and labelled
finite-shot/noise predictions. It is not physical-QPU execution and makes no
quantum-advantage claim.

## Frozen publication assets

Stage 2C compiles the final five-page-paper tables, figures, claims manifest,
and factual prose support from checksum-pinned frozen results. It does not fit
or rerun models, change architectures or thresholds, recalibrate predictions,
select on test results, create ensembles, or conduct new hypothesis tests.

```bash
uv run python scripts/freeze_publication_assets.py
uv run python scripts/freeze_publication_assets.py
```

Tracked final files are under `paper_assets/`; detailed untracked compiler
records are under `results/publication_assets/`. The package contains three
tables in CSV/JSON/LaTeX/Markdown, four 300-DPI PNG/PDF main figures, seven
preserved appendix figure pairs, a checksum-complete asset manifest, a
fact-by-fact results and claims manifest, figure captions, an estimated page
footprint, and results, limitations and reproducibility factsheets.

Every reported value identifies its frozen source, locator, checksum, split,
scope, adjustment status and metric direction. Regeneration is deterministic,
and tracked assets contain only repository-relative paths. See
`docs/publication_assets.md` for the exact selection, output contract,
interpretation limits, and validation commands.

These assets report classical exact density-matrix and explicitly labelled
finite-shot/noise simulations of a quantum reservoir. No physical QPU was
executed and no quantum-advantage claim is made.

## Reproducibility principles

Temporal splits are never shuffled. Preprocessing must be fitted on training
data only. Every reported run must have a saved configuration and manifest,
and headline claims require multiple seeds and uncertainty estimates. Frozen
processed data will include provenance and checksums so judging does not depend
on a live market-data endpoint. Hardware claims will identify the backend,
transpiled circuit statistics, shot count, and measured runtime.

## Responsible use of generative AI

Generative AI may assist with coding, testing, and writing. Team QTyche remains
responsible for every technical decision, dataset choice, experiment, result,
interpretation, citation, and submission statement. AI-generated claims are not
treated as evidence.
