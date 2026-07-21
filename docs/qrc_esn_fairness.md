# Initial QRC–ESN fairness report

The public QRC pilot is an integration, correctness, and stability exercise.
It is not yet a fully size-matched comparison and supports no superiority or
quantum-advantage claim.

The fixed pilot uses six qubits, two virtual nodes, and a ring, producing 24
raw QRC features. Its classifier readout therefore has shape `25 x 3`, or 75
trainable classical coefficients including the intercept. The existing ESN
reference highlighted for later matching uses reservoir size 25 and a
`26 x 3` classifier readout, or 78 coefficients. The dimensions are close but
not identical, so comparisons are descriptive.

Every QRC experiment manifest records raw feature dimension, readout shape and
parameter count, training observations, four-candidate ridge budget,
state-generation/readout-fitting time, peak estimated density-matrix memory,
reservoir seed, backend, and exact/noiseless status. Three reservoir seeds are
reported independently and aggregated by mean, population standard deviation,
minimum, and maximum.

A later headline comparison must match or explicitly budget:

- readout feature dimension;
- trainable readout family and intercept treatment;
- number of reservoir seeds;
- hyperparameter trial budget;
- training observations;
- validation criterion;
- one-time frozen test protocol.

Runtime and memory should remain separate resource axes. The QRC pilot uses
exact density matrices, so its compute cost cannot be interpreted as physical
quantum hardware cost.
