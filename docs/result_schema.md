# Experiment result schema

Every successful experiment directory contains:

| Path | Contents |
|---|---|
| `config.yaml` | Exact input model configuration |
| `manifest.json` | Git state, source/synthetic status, checksums, rows, features, target, seed, selected configuration, timing, environment, selection policy, and warnings |
| `model_metadata.json` | Versioned fitted-model metadata and dimensions |
| `selection_results.csv` | Every attempted candidate, validation-only score, status, and error |
| `validation_metrics.json` | Validation metrics only |
| `test_metrics.json` | Frozen test metrics only |
| `validation_predictions.csv` | Validation observation predictions |
| `test_predictions.csv` | Test observation predictions |
| `timing.json` | Training, validation prediction, and test prediction seconds |
| `model/` | Reloadable model parameters |
| `figures/` | Matplotlib outputs with fixture warnings where applicable |
| `logs/` | Failure or diagnostic records |

Classification prediction files contain `date`, `current_regime`, `true_regime`,
`predicted_regime`, low/medium/high probabilities, `true_transition`, derived
transition probability, and predicted transition. Regression prediction files
contain `date`, true and predicted five-day RV, and a flooring indicator.

Comparison creates independent `baseline_validation_comparison.csv` and
`baseline_test_comparison.csv` files. Both begin with source type, synthetic
flag/warning, task, seed, model, selected configuration, and experiment ID.
They are deliberately never merged into a single ranking.

Public-market runs additionally record `data_snapshot_id`,
`data_manifest_checksum`, source snapshot-manifest checksum, Git commit, and
dirty status. Regression prediction files retain both
`raw_predicted_rv_5d` and the evaluation-adjusted `predicted_rv_5d`, plus
negative and floor indicators.

`compare-public-baselines` writes five tables under `results/tables/`:

- `public_market_validation_comparison.csv`
- `public_market_test_comparison.csv`
- `public_market_classification_diagnostics.csv`
- `public_market_transition_diagnostics.csv`
- `public_market_regression_diagnostics.csv`

Every row contains model, task, seed, selected configuration, snapshot ID,
processed-manifest checksum, source/synthetic flags, Git provenance, split
designation, and experiment ID. Nested class and transition metrics are
flattened only in their task-specific diagnostic tables.

QRC experiment directories preserve the same core schema and additionally
contain `model/qrc_hamiltonian.npz`, `model/qrc_hamiltonian.json`,
`model/input_projection.npy`, `model/observables.json`, `model/readout.npz`,
`qrc_backend_metadata.json`, `qrc_numerical_diagnostics.json`, and
`qrc_feature_metadata.json`. The feature metadata includes the complete cache
key, per-split checksums, label-free-generation declaration, timing, and
observable ordering.

Capacity outputs live below `results/qrc_capacity/<analysis_id>/` and include
the exact config/manifest, delayed linear and quadratic tables, cross-delay
table, singular values, feature-rank JSON, contractivity curve/summary,
feature autocorrelation, the fixed analytical ablation table, and Matplotlib
figures.

Public pilot aggregation writes `qrc_pilot_validation_by_seed.csv`,
`qrc_pilot_test_by_seed.csv`, and `qrc_pilot_seed_summary.csv`. Test rows are
never used during ridge selection.
