# Gaussian GARCH(1,1) volatility baseline

## Purpose

This workstream adds a recognised classical econometric volatility benchmark to
the frozen public-market evaluation. It measures how a stationary Gaussian
GARCH(1,1) forecast compares with realized-variance persistence, the ESN
regressor, and the validation-selected final QRC on exactly the same SPY
five-trading-day realized-variance rows.

The model is

\[
\epsilon_t=r_t-\mu,\qquad
h_{t+1}=\omega+\alpha\epsilon_t^2+\beta h_t ,
\]

with \(\omega>0\), \(\alpha\geq0\), \(\beta\geq0\), and
\(\alpha+\beta<1\). A softplus and scaled-softmax transformation enforces these
conditions by construction. Parameters are estimated by deterministic Gaussian
quasi-maximum likelihood using several fixed starting points; the lowest
training negative log-likelihood among converged fits is selected. The run
fails if no start converges.

## Frozen data and leakage controls

The runner verifies the raw snapshot and processed-data checksums before doing
any fitting and rejects synthetic fixture data. It reads the SPY close-to-close
log-return stream identified by the processed manifest. It then independently
reconstructs

\[
\mathrm{target\_rv\_5d}_t =
\frac{252}{5}\sum_{k=1}^{5}r_{t+k}^{2}
\]

only as an alignment audit and requires agreement with every frozen validation
and test target. Targets never enter estimation or filtering.

The full fit uses all finite SPY returns whose dates lie inside the frozen
training period, ending 2020-12-31. The smoke fit uses the last 750 of those
training-period returns to reduce execution time. Validation and test returns
are never used for optimization, initialization selection, clipping selection,
or model selection. After fitting, \(\omega,\alpha,\beta,\mu\) remain fixed.

For each post-training origin \(t\), the filter first has \(h_t\), consumes only
the return \(r_t\) observed at that origin, and produces \(h_{t+1}\). Expected
future variances use

\[
h_{t+k}=\omega+(\alpha+\beta)h_{t+k-1},\quad k=2,\ldots,5.
\]

The saved cumulative forecast is
\(\sum_{k=1}^{5}h_{t+k}\); the regression prediction converts it to the frozen
target units with \((252/5)\sum_{k=1}^{5}h_{t+k}\). Filtering is continuous
across every available post-training trading date, including dates excluded
from target evaluation by purging, while predictions are selected only for the
exact frozen validation and test origins.

## Reproduction

Run the checksum-verified reduced smoke:

```bash
python scripts/run_garch_baseline.py --smoke
```

Run it again without resuming an earlier completed smoke:

```bash
python scripts/run_garch_baseline.py --smoke --no-resume
```

Run the complete public-data fit using all training-period returns and all eight
fixed starts:

```bash
python scripts/run_garch_baseline.py
```

The shell wrapper accepts the same arguments:

```bash
./scripts/run_garch_baseline.sh --smoke
./scripts/run_garch_baseline.sh
```

By default, a complete matching smoke or full run is resumed. An incomplete run
is ignored and a new complete run is produced. `--no-resume` always creates a
new run.

## Outputs

Outputs are isolated under `results/garch_baseline/`. The top-level
`garch_run_summary.json` points to the active run directory. Each run contains:

- `fitted_parameters.json`: fitted parameters, persistence, unconditional
  variance, all optimizer attempts, selected training likelihood, convergence
  details, and fit-date/checksum evidence;
- `validation_predictions.csv` and `test_predictions.csv`: aligned returns,
  filtered state, one-day variance, five-day cumulative variance, target-unit
  prediction, floor indicator, truth, and deterministic regime labels;
- `validation_metrics.json` and `test_metrics.json`: QLIKE, RMSE, MAE,
  correlation, floor diagnostics, macro-F1, balanced accuracy, and confusion
  matrix;
- `conditional_variance_path.csv`: the continuous post-training causal filter
  and forecast path;
- `aggregate_comparison.csv` and `.json`, plus `comparison_per_run.csv` and
  `.json`: date-revalidated persistence, ESN, GARCH, and final-QRC results;
- `manifest.json` and `timing.json`: data, temporal, Git, Python, package,
  platform, fitting, and runtime provenance;
- `figures/`: PNG and PDF versions of the test time series, QLIKE comparison,
  RMSE comparison, realized-regime forecast distribution, and conditional
  variance/five-day path.

QLIKE applies the configured \(10^{-12}\) positive floor at evaluation only.
The metrics and predictions record every affected value. No fitted state or raw
forecast is clipped. Regime classes are assigned with the existing
training-derived thresholds. Because deterministic thresholding does not
produce calibrated class probabilities, transition PR-AUC is explicitly
reported as not applicable.

## Interpretation

QLIKE, RMSE, and MAE are lower-is-better; correlation, macro-F1, and balanced
accuracy are higher-is-better. The comparison aggregates the three final-QRC
reservoir seeds but has one deterministic run for each classical model. These
descriptive results do not constitute a formal significance test; correlated
forecast-loss testing remains a later workstream.

GARCH(1,1) assumes a stationary, symmetric Gaussian conditional-variance
dynamics and does not model leverage, heavy-tailed innovations, or
time-varying parameters. Its five-day forecast is an expected variance path,
whereas the target is one realized future path, so pointwise differences are
expected.

This is a classical econometric baseline. The referenced QRC results are exact
classical simulations of a quantum reservoir, not physical-QPU execution, and
the comparison does not support a quantum-advantage claim.
