# Real classical-baseline protocol

The public benchmark uses snapshot `yahoo_chart_20100101_20251231_v1` and the
same frozen targets, features, splits, training-only scaler, and training-only
regime thresholds as the fixture workflow. Public and fixture raw, processed,
experiment, and comparison paths are separate.

Training labels fit model parameters. Validation metrics select logistic and
ESN configurations. Search functions receive a structural view with no test
attributes. After selection, the chosen configuration is frozen, the model is
refit on training labels only, and validation and test predictions are saved.
Test metrics are evaluated once and never enter configuration selection.

Logistic regression searches three configured regularization strengths. The
ESN classifier samples 20 configurations deterministically and records macro
F1, transition PR-AUC, timing, reservoir dimension, measured spectral radius,
warnings, and status for every trial. The ESN regressor first compares direct
and log target heads on validation data, then runs a deterministic 20-trial
search with the validated log head.

Reproduce acquisition and processing:

```bash
./scripts/reproduce_public_data.sh
```

Once the snapshot is present, processing uses cached mode and is offline. Run
the six public baselines and generate the benchmark package with:

```bash
./scripts/reproduce_core_baselines.sh
```

Outputs are written below `results/public_market/`,
`results/diagnostics/esn_regression/`, `results/data_audit/`, and
`results/tables/`. Validation and test comparisons remain separate. A larger
metric does not imply statistical superiority; Mincer-Zarnowitz and
Diebold-Mariano analysis are intentionally deferred.
