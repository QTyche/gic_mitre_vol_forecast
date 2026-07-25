# Formal benchmark statistical validation

## Purpose and freeze boundary

Stage 2A adds paired statistical inference to the final financial and MNIST
benchmarks. It reads the existing frozen prediction files and never calls a
model fit, changes an architecture or hyperparameter, selects from test
results, or writes into an earlier result tree. Every input path and SHA-256
digest is pinned in `configs/statistical_validation.yaml` and verified before
and after inference.

The financial QRC remains the validation-selected two-qubit, two-virtual-node,
`reset_each_input` architecture with reservoir seeds 2026–2028. The MNIST QRC
remains the five-qubit architecture on the fixed genuine-MNIST 1,000-image
official-test subset. Architecture-level statistics average paired
per-seed loss or metric differences. They do not average forecasts or
probabilities into an ensemble.

## Commands

From the repository root, run the compact deterministic calculation and then
the full 10,000-repetition analysis:

```bash
python scripts/run_statistical_validation.py --smoke
python scripts/run_statistical_validation.py --smoke
python scripts/run_statistical_validation.py
```

The repeated smoke command should reproduce every scientific CSV and JSON table
byte for byte. Smoke mode retains all comparisons and sensitivity settings but
uses 200 bootstrap repetitions. The full run uses 10,000. The shell wrapper
accepts the same options:

```bash
./scripts/run_statistical_validation.sh --smoke
./scripts/run_statistical_validation.sh
```

Use `--config PATH` only with a complete checksum-pinned Stage 2A contract.
There is deliberately no refit or resume option: inference is inexpensive and
is regenerated deterministically from immutable predictions.

## Financial inference

All models must have exactly the same 497 ordered test dates and frozen target
values. A missing, reordered, duplicated, or truth-mismatched observation
stops the analysis.

For regression, each QRC seed is paired with GARCH(1,1), the headline ESN
regressor, realised-variance persistence, and the existing directly comparable
ESN log-variance diagnostic. The datewise differential is

```text
d_t = loss(QRC)_t - loss(baseline)_t
```

for QLIKE, squared error, and absolute error. Negative values favour QRC.
The primary Diebold–Mariano-style mean test uses a Bartlett-kernel
Newey–West/HAC variance with lag 4 and the Harvey–Leybourne–Newbold
five-step finite-sample correction. Lags 0, 10, and 20 are recorded as
sensitivity checks. Paired circular-block intervals use seed 2026, primary
block length 10, and sensitivity lengths 5 and 20.

Classification compares each QRC seed with logistic regression, the ESN
classifier, regime persistence, and the majority classifier. Every circular
block is applied to the shared dates, truth, predictions, and supplied frozen
transition scores. Differences are QRC minus baseline, so positive values
favour QRC. Macro-F1, balanced accuracy, and transition PR-AUC are recomputed
inside each replicate. A PR-AUC replicate without both transition classes is
retained as invalid and counted rather than silently replaced.

Holm correction is applied separately within every metric, analysis, and
inference-level family. The architecture-level regression series is the
datewise mean of three loss differentials. Classification uses the
replicate-wise mean of three paired metric differences.

Mincer–Zarnowitz diagnostics fit

```text
RV_t = alpha + beta * forecast_t + error_t
```

to frozen test forecasts only, with lag-4 HAC covariance. These diagnostic
regressions report separate tests of `alpha = 0` and `beta = 1`, their joint
Wald test, R-squared, and sample size. They do not alter any forecast.

## MNIST inference

Each exact QRC seed is compared with flattened logistic regression and the
size-controlled ESN on the same 1,000 official-test identities. Accuracy uses
an exact paired McNemar/binomial test and reports both discordant counts.
Accuracy, macro-F1, balanced accuracy, and one-vs-rest macro ROC-AUC use 10,000
paired class-stratified replicates. Sampling occurs independently within each
true digit, preserving all ten classes and both models' predictions and
probabilities for an image. Invalid AUC replicates are counted. Architecture
results average the three per-seed metric differences within each replicate;
probabilities are never averaged.

## Output contract

Generated files are isolated below `results/statistical_validation/`, which is
excluded from Git:

```text
results/statistical_validation/
├── tables/
│   ├── financial_regression_pairwise_per_seed.{csv,json}
│   ├── financial_regression_architecture_level.{csv,json}
│   ├── financial_classification_pairwise_per_seed.{csv,json}
│   ├── financial_classification_architecture_level.{csv,json}
│   ├── mincer_zarnowitz.{csv,json}
│   ├── mnist_pairwise_per_seed.{csv,json}
│   ├── mnist_architecture_level.{csv,json}
│   └── multiple_testing_adjustments.{csv,json}
├── figures/
│   ├── financial_qlike_loss_difference_forest.{png,pdf}
│   ├── financial_squared_error_difference_forest.{png,pdf}
│   ├── financial_macro_f1_difference_forest.{png,pdf}
│   ├── financial_transition_pr_auc_difference_forest.{png,pdf}
│   ├── mincer_zarnowitz_alpha_beta.{png,pdf}
│   ├── mnist_accuracy_difference_forest.{png,pdf}
│   └── mnist_macro_f1_difference_forest.{png,pdf}
├── environment_manifest.json
└── statistical_validation_summary.json
```

JSON tables retain nested HAC and block-length sensitivity results. CSV tables
encode those nested records as deterministic compact JSON. The environment
manifest records every frozen source checksum, the architecture and subset
checksums, Git state, platform and package versions, and explicit freeze
flags. Timestamps and runtime appear only in the summary; scientific tables
contain no time-dependent fields.

## Interpretation and limitations

An adjusted p-value or bootstrap interval measures evidence for a paired
difference under the stated dependence and resampling assumptions. It is not
an effect-size substitute. Report the observed difference, uncertainty,
adjusted result, direction, and sensitivity together. Failure to reject zero
does not establish equivalence. Sensitivity across HAC lags or block lengths
should be disclosed rather than resolved by selecting the most favourable
setting.

Only three QRC reservoir seeds are available. Financial targets overlap,
classification samples are temporally dependent, the MNIST evaluation uses a
fixed 1,000-image subset, and bootstrap tail probabilities have Monte Carlo
resolution. Mincer–Zarnowitz regressions are post-evaluation diagnostics, not
new forecasting models.

All QRC predictions were produced by exact classical density-matrix simulation
of a quantum reservoir. Stage 2A performs classical statistical inference over
those frozen results. It is not physical-QPU execution and makes no
quantum-advantage claim.
