# Echo State Network design

For input `u_t` and state `x_t`, the fixed reservoir uses:

```text
x_t = (1 - leaking_rate) x_(t-1)
      + leaking_rate tanh(W_in [1; u_t] + W_res x_(t-1))
```

`W_in` is sampled once per seed from a uniform distribution bounded by
`input_scaling`. `W_res` is an inspectable sparse uniform random matrix, where
`sparsity` is the non-zero connection probability. It is rescaled by its
measured largest absolute eigenvalue to the requested spectral radius. Tests
cover seed determinism, dimensions, and the post-scaling radius.

`leaking_rate` controls how quickly the new nonlinear candidate replaces the
previous state. `washout` removes initial training states from readout fitting.
The classification readout is ridge regression against three-column one-hot
labels; its raw scores become probabilities through a stable softmax. The
regression readout is ridge regression directly in annualized realized-variance
units. Ridge intercepts are not penalized.

`carry_inputs` processes all chronological feature inputs continuously, while
`reset` restarts at each split. Neither policy consumes labels during state
updates. Saved models include `W_in`, `W_res`, the fitted readout, current state,
configuration, ordered feature names, dimensions, and measured spectral radius.

This ESN is the principal classical reservoir control for the future QRC. Fair
comparison requires identical processed inputs, split chronology, state policy,
readout discipline, seeds/search budgets, and evaluation metrics.

For realized-variance regression, the reservoir is unchanged but the readout
target is configurable. `direct_variance` fits physical variance directly.
`log_variance` fits `log(target + epsilon)` and inverts with
`exp(score) - epsilon`. The selected transformation and epsilon are serialized;
invalid forecasts remain visible to the evaluation floor accounting.
