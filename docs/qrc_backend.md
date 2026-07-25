# Exact density-matrix backend

`numpy_density_matrix_exact` constructs the full complex Hamiltonian with
NumPy and caches the SciPy matrix exponential for each reservoir instance. A
density matrix has shape `2^N x 2^N`; dense evolution and eigenspectrum checks
therefore grow exponentially. The supported controlled-scaling range is
`2 <= N <= 6`; the implementation rejects larger reservoirs before allocation.

The backend is exact, noiseless, and analytic at measurement time. It uses no
shots, sampling, physical noise model, device calibration, transpilation, or
hardware. These omissions are deliberate: the first stage isolates reservoir
dynamics and software correctness before introducing hardware confounders.

The default numerical tolerances are:

| Check | Absolute tolerance |
|---|---:|
| trace one | `1e-10` |
| Hermiticity | `1e-10` |
| positive semidefiniteness | minimum eigenvalue `>= -1e-10` |
| cached-unitary identity error | `1e-10` |

Every reset and evolved state must be finite, have the expected shape, and
pass these checks. A trace drift no larger than the trace tolerance is divided
out and counted. Larger trace errors, excess anti-Hermitian components, or
negative eigenvalues beyond tolerance raise immediately; the backend does not
hide a badly invalid state through normalization. Diagnostics persist the
number of validations and trace corrections, maximum pre-correction trace and
Hermiticity errors, minimum observed eigenvalue, and unitary error.

The peak-memory estimate records three dense complex matrices. It is a
transparent analytical estimate, not a process-level profiler measurement.

| Qubits | Density shape | Three-matrix estimate |
|---:|---:|---:|
| 2 | `4 x 4` | 768 B |
| 3 | `8 x 8` | 3,072 B |
| 4 | `16 x 16` | 12,288 B |
| 5 | `32 x 32` | 49,152 B |
| 6 | `64 x 64` | 196,608 B |

Actual peak resident memory is higher because SciPy operations, eigensolver
workspace, Python objects, cached features, readouts, and artifacts are not
included.
