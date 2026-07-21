# Evaluation and model-selection protocol

Training observations fit model parameters and ESN readouts. Validation labels
select hyperparameters using macro F1 for regime classification or QLIKE for
realized-variance regression. The search API receives a restricted dataset view
with no test attributes. Test predictions are constructed only after a candidate
has been selected and frozen, and test metrics never affect selection.

Dates remain chronological and are never shuffled. Under the default
`carry_inputs` state policy, a reservoir processes training, validation, and test
features continuously: validation state may depend on earlier training inputs,
and test state may depend on earlier training and validation inputs. Labels are
not state inputs. Under `reset`, reservoir state is reset at each split boundary.
The future QRC benchmark must use the same declared state policy.

The primary metric is regime macro F1. Classification reporting also includes
accuracy, balanced accuracy, weighted F1, per-class precision/recall/F1, log
loss, multiclass Brier score, and confusion matrix. Observation-level transition
probability is `1 - P(predicted regime = current_regime)` and defaults to a 0.5
decision threshold. Transition ROC-AUC, PR-AUC, Brier score, F1, accuracy, and
balanced accuracy are secondary metrics.

Regression reports RMSE, MAE, QLIKE, and R-squared in original annualized
variance units. QLIKE is `mean(log(y_hat) + y/y_hat)`. True variance must be
positive. Non-finite or sub-floor predictions are replaced at evaluation only
by the configured positive floor (default `1e-12`); every replacement and every
non-finite prediction is counted. Mincer-Zarnowitz and Diebold-Mariano analysis
are reserved for a later statistical task.

Synthetic fixtures are permitted only for tests, smoke runs, and pipeline
validation. Headline-capable commands reject them unless
`--allow-synthetic-results` is explicit. An override never converts fixtures
into financial evidence: manifests, tables, metrics, and figures remain marked
`SYNTHETIC FIXTURE DATA — NOT A FINANCIAL PERFORMANCE RESULT`.

Public-market acquisition, processing, and results use separate directories and
explicit `public_market`/`false` source flags. ESN regression target
transformations are selected on validation QLIKE only. The direct head remains
in diagnostics; the log head is not chosen using test metrics. Public
validation and test tables are generated separately and are not interpreted as
tests of statistical superiority.
