# Frozen benchmark diagnostic analysis

## Purpose and freeze boundary

Stage 2B describes calibration, regime behaviour, temporal errors, numerical
conditioning, and digit-level MNIST behaviour after the financial and MNIST
models have been frozen. It consumes existing validation and test prediction
files only. It does not fit or recalibrate a model, alter a threshold,
hyperparameter or architecture, average predictions into an ensemble, or use a
diagnostic to select a model.

Every source path and SHA-256 digest is pinned in
`configs/benchmark_diagnostics.yaml`. The driver verifies all unique sources
before and after analysis. It explicitly requires the non-synthetic
`yahoo_chart_20100101_20251231_v1` public-market tree and the fixed genuine
MNIST official-test identities. Generated files are isolated below
`results/benchmark_diagnostics/`.

## Commands

From the repository root:

```bash
python scripts/run_benchmark_diagnostics.py --smoke
python scripts/run_benchmark_diagnostics.py --smoke
python scripts/run_benchmark_diagnostics.py
```

Smoke mode retains every model, split, diagnostic, sensitivity setting and
figure, but uses 200 bootstrap repetitions. The full run uses 5,000. Repeated
smoke runs should produce byte-identical scientific CSV and JSON tables. The
thin shell wrapper accepts the same arguments:

```bash
./scripts/run_benchmark_diagnostics.sh --smoke
./scripts/run_benchmark_diagnostics.sh
```

There is deliberately no fitting, recalibration or resume option. The
diagnostics are regenerated deterministically from immutable inputs.

## Financial variance diagnostics

RV persistence, ESN, GARCH(1,1), and QRC seeds 2026–2028 are evaluated on all
748 validation and 497 test dates. The overall table contains forecast and
realised means and standard deviations, mean and median error, mean ratio,
over/underprediction rates, QLIKE, RMSE, MAE, and correlation.

Each model and split also receives ten post-hoc forecast-quantile bins. These
are descriptive split-specific bins, not reusable thresholds or
model-selection rules. Low, medium and high conditioning uses only the stored
training-derived regime boundaries:

```text
low / medium = 0.006637091876978361
medium / high = 0.019546076157892393
```

The high-volatility tails use the 90th and 95th empirical quantiles of the
frozen training `target_rv_5d` column. Validation and test observations never
contribute to either threshold. Overall, regime and tail summaries include
descriptive 95% circular moving-block intervals with 5,000 repetitions, block
length 10, and seed 2026.

## Probability, regime and transition diagnostics

Frozen logistic-regression, ESN and three QRC probability vectors are assessed
without temperature scaling, isotonic regression, Platt scaling or any other
transformation. Primary ten-bin results report multiclass Brier score, log
loss, top-label ECE/MCE, confidence, accuracy, confidence gap and entropy.
Sensitivity tables use 5, 15 and 20 bins. Top-label and classwise reliability
records retain every bin, including empty bins.

Label, regime and transition diagnostics also include majority and regime
persistence. The output records fixed-label confusion matrices, per-class
precision/recall/F1/support, predicted/true distributions, each observed
origin-to-destination transition type, and transition versus non-transition
subsets. Transition diagnostics use the supplied
`predicted_transition_probability` column unchanged; no replacement score is
constructed.

### Lead-time decision

Lead-time analysis is omitted. The frozen financial prediction rows contain
one `date`, a current regime, and a five-trading-day aggregate target regime.
They do not contain distinct forecast-origin, target, or within-window
transition dates. Because an aggregate future realised-variance regime has no
unique transition date, a legitimate transition lead cannot be identified
without inventing semantics. The machine-readable summary records the missing
columns and refusal reason.

## Temporal and QRC-seed diagnostics

The temporal table contains trailing 60-observation QLIKE, RMSE and bias;
cumulative model-minus-GARCH and model-minus-ESN QLIKE; and cumulative
classification-error difference relative to logistic regression. Negative
cumulative differences favour the named model. These trajectories are
descriptive and are not searched for favourable subperiods.

The same table identifies the five largest test realised-variance
observations and includes every frozen variance forecast on those dates.

The QRC numerical table places all three seeds side by side. It records feature
condition number/rank, selected classifier and regressor ridge values,
coefficient norms and maxima, forecast dispersion, confidence, entropy,
calibration, high-regime and P95-tail behaviour, state-validation counts, and
all available finiteness checks. Seed 2027 remains present despite its known
near-singular training matrix. Comparisons are factual and do not attribute
its predictive behaviour to conditioning.

## MNIST diagnostic appendix

The fixed 1,000-image official-test subset supplies per-digit
precision/recall/F1, confusion rows, confidence and entropy for three exact QRC
seeds, flattened logistic regression, and ESN. Paired overlap tables count
both-correct, QRC-only, baseline-only and both-wrong images.
Each paired row also records every per-digit accuracy difference and names the
largest QRC gain and loss digit, with deterministic lowest-digit tie-breaking.

For seed 2026, analytic predictions are compared by digit with the existing
2,048-shot, depolarising-0.01 and measurement-flip-0.02 predictions. Tables
report accuracy, F1, signed degradation, absolute change, changed predictions,
and destination-specific confusion-count changes. No reservoir or measurement
sampling is rerun. Per-digit and paired-accuracy intervals use 5,000
class-stratified paired bootstrap repetitions with seed 2026.

## Output contract

```text
results/benchmark_diagnostics/
├── tables/
│   ├── variance_overall_calibration.{csv,json}
│   ├── variance_decile_calibration.{csv,json}
│   ├── variance_regime_diagnostics.{csv,json}
│   ├── variance_tail_diagnostics.{csv,json}
│   ├── probability_calibration.{csv,json}
│   ├── classwise_probability_calibration.{csv,json}
│   ├── regime_classification_diagnostics.{csv,json}
│   ├── transition_type_diagnostics.{csv,json}
│   ├── transition_vs_nontransition.{csv,json}
│   ├── temporal_error_diagnostics.{csv,json}
│   ├── qrc_seed_numerical_diagnostics.{csv,json}
│   ├── mnist_per_digit_diagnostics.{csv,json}
│   ├── mnist_pairwise_error_overlap.{csv,json}
│   ├── mnist_robustness_by_digit.{csv,json}
│   └── benchmark_diagnostics_summary.{csv,json}
├── figures/                       # 13 PNG/PDF pairs
├── benchmark_diagnostics_summary.json
└── provenance_manifest.json
```

Nested confusion matrices and reliability bins remain structured in JSON and
are encoded as deterministic compact JSON inside CSV cells. Scientific tables
contain no timestamps. The root summary contains runtime and completion time;
the provenance manifest records every source checksum, Git and package state,
fixed thresholds, lead-time refusal, freeze flags, and SHA-256 plus byte size
for every generated table, figure and the root summary. The provenance
manifest excludes its own self-referential checksum explicitly.

## Interpretation and limitations

Calibration and error bins are post-hoc descriptions, not formal Stage 2A
hypothesis tests. Bootstrap intervals describe sampling variation under the
chosen resampling scheme and should not be used to rank or select models.
Only three QRC reservoir seeds, 497 financial test observations and 1,000
MNIST images are available. Tail subsets are small, decile boundaries are
split-specific, and overlapping five-day targets remain serially dependent.

All QRC results originate from exact classical density-matrix or explicitly
labelled finite-shot/noise simulation of a quantum reservoir. Stage 2B is
classical analysis of frozen predictions, not physical-QPU execution and not a
quantum-advantage claim.
