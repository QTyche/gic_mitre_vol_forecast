# Exact QRC architecture

The first quantum reservoir is a fixed, exact, noiseless density-matrix model.
Its purpose is to establish a correct and inspectable quantum feature pipeline,
not to demonstrate hardware performance or quantum advantage.

The reservoir uses three to six qubits, a sparse disordered transverse-field
Ising Hamiltonian, q0 reset-and-reinjection, and exact Z and ZZ expectations.
Hamiltonian couplings, transverse fields, and the input projection are sampled
once from the reservoir seed and never trained. Only an intercept-plus-ridge
classical readout is fitted.

For each chronological input row, every virtual-node slice performs:

1. Compute the unbounded angle from the frozen random projection.
2. Trace out input qubit q0.
3. Tensor the encoded pure q0 state with the retained subsystem state.
4. Evolve by the cached exact unitary for `tau / virtual_nodes`.
5. Validate the density matrix and measure all configured observables.

`carry_inputs` processes train, validation, and test inputs continuously. The
alternative `reset` policy starts each split from the all-zero pure state.
Neither policy consumes target labels. Classifier and regressor heads share the
same checksum-keyed reservoir feature cache when their dynamics and data match.

The exact backend is intentionally limited to six qubits. This stage contains
no finite-shot estimation, physical noise, hardware execution, measurement
feedback, or trainable quantum parameter.
