# Forecast model interface

Classical models extend `ForecastClassifier` or `ForecastRegressor` in
`src/qtyche_qrc/models/base.py`. Classifiers implement `fit`, `predict`,
`predict_proba`, `save`, `get_params`, and `get_model_metadata`; regressors use
the same contract except that they do not expose class probabilities. Calling a
prediction method before fitting is an error.

Metadata records model name/version, task, ordered feature names, fitted state,
hyperparameters, seed, training timestamp, and relevant package versions.
Serialization must include every learned array. Loading a saved model must
preserve predictions.

The model dataset loader consumes the existing scaled split CSVs but never fits
or changes preprocessing. It separately joins unscaled `spy_rv_5d` by date for
the realized-variance persistence baseline. `current_regime` is passed explicitly
to regime persistence and is not treated as a target-derived feature.

Sequential reservoirs additionally expose state reset, sequence transformation,
and state get/set operations. State updates consume feature inputs only, never
validation or test labels.
