# ESN regression diagnostics

The direct ESN variance readout is retained as a diagnostic control. The
recommended head fits ridge regression to
`log(target_rv_5d + epsilon)` and converts readout scores back with
`exp(score) - epsilon`, where `epsilon = 1e-12`. The reservoir, chronological
`carry_inputs` policy, and training-only readout discipline are otherwise
unchanged.

The fixed 50-state diagnostic reservoir has measured spectral radius
0.899999999999999 and a ridge design condition number of 50.29. Its state
values remain bounded approximately within [-1, 1]. This does not indicate
severe design ill-conditioning.

| Validation head | Ridge alpha | Negative/floored | RMSE | MAE | QLIKE |
|---|---:|---:|---:|---:|---:|
| Direct variance | 0.1 | 149 | 0.040996 | 0.028116 | 3,899,010,237.169994 |
| Log variance | 0.1 | 0 | 0.027087 | 0.016413 | -2.604905 |

The direct head produced predictions as low as -0.083712 and failed primarily
because an unconstrained linear readout can forecast negative variance. The
log head was selected using validation QLIKE only. Test data were accessed
after freezing that choice; the fixed diagnostic configuration produced test
QLIKE -2.844808, RMSE 0.069641, MAE 0.019863, and zero negative, non-finite, or
floored predictions.

The full deterministic ESN regression search therefore keeps `log_variance`
fixed while searching reservoir and ridge hyperparameters. Transform name,
epsilon, inverse rule, matrices, and readout are saved with the model. Unit
tests cover round-trip accuracy, positive realistic inverse forecasts,
serialization, label-free states, and test-set unavailability during head
selection. These numerical comparisons are benchmark diagnostics, not tests of
statistical superiority.
