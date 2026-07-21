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

## Reproducibility principles

Temporal splits are never shuffled. Preprocessing must be fitted on training
data only. Every reported run must have a saved configuration and manifest,
and headline claims require multiple seeds and uncertainty estimates. Frozen
processed data will include provenance and checksums so judging does not depend
on a live market-data endpoint. Hardware claims will identify the backend,
transpiled circuit statistics, shot count, and measured runtime.

## Launch on qBraid

> Placeholder: add the final qBraid Lab launch URL and badge after the hosted
> repository environment has been created and validated.

The main reproduction workflow will remain backend-agnostic. A targeted hardware
validation can be added once the challenge organisers confirm the required device
and mandatory Track A metrics.

## Responsible use of generative AI

Generative AI may assist with coding, testing, and writing. Team QTyche remains
responsible for every technical decision, dataset choice, experiment, result,
interpretation, citation, and submission statement. AI-generated claims are not
treated as evidence.
