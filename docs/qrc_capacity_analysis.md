# QRC capacity analysis

The analytical workflow uses a deterministic synthetic sequence sampled
uniformly on `[-1, 1]`. It does not load financial targets and is separate from
the market pilot.

For each delay `k`, linear memory capacity is the squared test correlation
between `u_(t-k)` and its ridge reconstruction. The reported total is the sum
over delays 1 through 10. This is an empirical finite-sample/readout quantity;
it is not an architecture-independent memory bound.

Quadratic capacity applies the same protocol to the second Legendre target
`0.5 (3 u_(t-k)^2 - 1)`. Cross-delay capacity reconstructs configured products
`u_(t-k) u_(t-l)`. These probes demonstrate accessible nonlinear functions but
do not establish usefulness for a particular financial target.

For centered feature matrix `Z` and singular values `sigma_i`, the workflow
uses `p_i = sigma_i / sum_j sigma_j` and reports
`exp(-sum_i p_i log p_i)`, numerical rank, retained spectral extremes, and
condition number. Effective rank depends on scaling, sample length, and the
documented numerical tolerance.

Empirical contractivity starts two valid density matrices from different pure
states, drives both with identical subsequent inputs, and records
`0.5 ||rho_t - sigma_t||_1`. The fitted log-distance rate, overall decrease,
non-monotonic intervals, and final/initial ratio are evidence only for fading
memory under that tested configuration. They are not a formal global
contraction theorem.

Mean absolute per-feature autocorrelation supplies an additional practical
memory diagnostic. It can reflect redundant or slowly varying features and is
not equivalent to reconstructive memory capacity.

The fixed 3x3 ablation varies input scaling `[0.1, 0.5, 1.0]` and interaction
strength `[0.1, 1.0, 2.0]` at four qubits, two virtual nodes, and one reservoir
seed. It probes the expected memory/nonlinearity trade-off and is not used
alone to select the financial pilot or support a headline statement.
