# QRC mathematical definition

## Hamiltonian and graph

For interaction graph `E`,

```text
H = sum_(i,j in E) J_ij Z_i Z_j + sum_i h_i X_i.
```

For `N >= 3`, the primary graph is the ordered ring
`(0,1), (1,2), ..., (N-2,N-1), (N-1,0)`. At `N = 2`, the ring has the single
unique undirected edge `(0,1)` rather than duplicating the same interaction in
opposite directions. The interface also accepts an explicit
`ring_plus_chords` graph. Once per reservoir seed,

```text
J_ij ~ J_strength Uniform(-1, 1)
h_i  ~ h_strength Uniform(h_min_factor, h_max_factor),
```

with default field factors 0.5 and 1.5. These values are frozen. For slice
duration `delta_tau = tau / V`, the exact backend caches
`U_delta = exp(-i H delta_tau)`.

## Input projection and reset channel

For normalized financial feature vector `x_t` and virtual slice `v`,

```text
theta_(t,v) = input_scaling b_v^T x_t.
```

Rows of the seed-fixed Gaussian projection `B` are normalized to unit L2 norm.
Angles are neither clipped nor wrapped. With `q_in = 0`,

```text
|psi(theta)> = Ry(theta)|0>
rho_reset = |psi(theta)><psi(theta)|_q0 tensor Tr_q0(rho_previous)
rho_new = U_delta rho_reset U_delta^dagger.
```

The code orders q0 as the most-significant Kronecker factor. Density-matrix row
axes are followed by column axes when taking a partial trace. This convention
is exercised by tensor-ordering tests.

## Observables and temporal multiplexing

After every slice, the reservoir measures exact expectations of every `Z_i`
and every graph-edge `Z_i Z_j`, in that order. Slice order is outermost. Thus

```text
z_t = concat_v [<Z_0>, ..., <Z_(N-1)>, <Z_i Z_j>_(i,j in E)]
raw dimension = V (N + |E|).
```

For rings with `N >= 3`, this is `2 V N`. For the two-qubit ring it is
`3 V`, because there are two single-qubit and one unique edge observable per
slice. Connected correlations
`<Z_i Z_j> - <Z_i><Z_j>` are numerical diagnostics only and are excluded from
the fitted readout.

The total interval for one input is fixed at `tau`, not `V tau`. Slice
endpoints are `tau/V, 2 tau/V, ..., tau`. Under the implemented encoding
semantics, every slice applies its own fixed-seed projection row and partial
input-qubit reset/reinjection before evolving for `tau/V`; the final state is
then carried to the next input under `carry_inputs`. Thus changing `V` changes
within-interval encoding and readout density while preserving the total
evolution duration.

## Readouts

The classification head fits intercept-unpenalized ridge regression against
three-column one-hot targets, then applies a stable softmax. Ridge alpha is
selected by validation macro F1 only.

The regression head fits
`log(target_rv_5d + epsilon)` with the same ridge family and returns
`exp(prediction) - epsilon`. Ridge alpha is selected by validation QLIKE only.
Non-finite or predictions below the positive evaluation floor are counted and
floored only by the existing evaluation policy.

## Temporal state policies

- `reset`: reset the complete density matrix before each split.
- `carry_inputs`: reset before train only, then carry state through validation
  and test inputs chronologically.

Validation state may depend on training inputs; test state may depend on train
and validation inputs. No label enters either state transition.

## Finite-shot observable extension

For a finite budget `S`, one computational-basis batch
`b^(1), ..., b^(S)` is sampled from `diag(rho)` after each virtual-node
evolution. With `z_i^(s) = 1 - 2 b_i^(s)`,

```text
<Z_i>_S     = (1/S) sum_s z_i^(s)
<Z_i Z_j>_S = (1/S) sum_s z_i^(s) z_j^(s).
```

The same sampled bitstrings determine every Z and ZZ value; observables are not
sampled independently. The analytic path above remains the infinite-shot
reference. Optional controlled channels and their placement are defined in
`docs/qrc_noise_robustness.md`.
