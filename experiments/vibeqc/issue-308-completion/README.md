# CUDA DF host-eigensolve campaign

Baseline: live `origin/master` at `15d6936390723edf9e9eb0c91fecde4390490573`.
The only open implementation PRs at the start were #306/#307, both for DFT.
This increment implements #308 instrumentation and prepares the subsequent
#309/#310/#311 ablations using #206's existing measurement/acceptance workflow.
It does not replace #5 or Direct-vs-Direct
gates. No performance gain has been measured in this initial instrumentation
increment.

## Initial-density caller audit

The shared helpers currently return a core-Hamiltonian orbital frame even
when a warm density is supplied. Any lazy-guess change must account for the
following callers, rather than assuming all output orbital objects are unused.

| Caller | Use of initial orbitals | Required preservation |
| --- | --- | --- |
| `scf/solver/mean_field_driver.cpp`, RHF/UHF CPU plans | Iterative Fock solve replaces the frame before forming a proposed density; strict proposal callbacks run after that solve | Cold densities, UHF frontier mixing, nonconverged and hook/export contracts |
| `scf/rhf.cpp`, CUDA DF single/bucket paths | Device loop consumes density and X, not initial orbitals; host fallback overwrites the frame on its first iteration | Explicit fallback and per-item failure isolation; finalization never consumes stale core orbitals |
| `dft/rks.cpp`, RKS occupied-factor path | A cold-only factor is packed before iteration; warm input intentionally has no such factor until a new orbital solve | Preserve the cold guard and generation-linked factor identity |
| `dft/rks.cpp`, UKS | First physical iteration replaces alpha/beta orbitals | Independent spin populations, empty beta and cold frontier perturbation |

The CPU helper normalizes warm RHF density by symmetry and electron trace,
and UHF channels independently, clearing a zero-occupation channel. Both
validate finite values and shape after their current unconditional hcore
solve. Moving that solve must preserve these density checks and explicitly
represent any unavailable initial orbitals.

## Existing provider audit

`scf/cuda/eigensolver.cpp::launch_solver` already dispatches native small,
batched Jacobi, `XsyevBatched`, and ordinary `Xsyevd` families. The last one
serializes a batch on its owning stream with shared workspace, and does not
require graph capture. Existing eligibility/probe decisions belong to the
prepared owner. #310 should consume this boundary and workspace ownership,
not add another Jacobi implementation or repeat #49's graph qualification.

DF also has a retained `DeviceSolver` in `scf/cuda/df_scf_state.hpp` and
`df_scf_library.cpp`: <=32 uses `DsyevjBatched`; larger dimensions use
`XsyevBatched`. Its persistent SCF state already owns density, proposal,
physical Fock, eigenvalue and occupied-factor buffers. These are candidate
inputs to #311, but the generations are not automatically a consistent final
physical state. The current finalizer rebuilds J/K, diagonalizes F on the CPU,
projects density and rebuilds again before returning energy/forces.

## Instrumentation increment

The new optional host collector observes actual oracle calls by reason,
dimension and item, with nested host phases and distinct thread CPU/wall
durations. Its observer interface preserves reference-module dependency
direction. The existing #206 force probe retains both host and CUDA ledgers;
the existing #206 matrix CLI adds native cold/replay/rebuild protocol controls.
See [the timing contract](../../../docs/df_component_trace.md).

This increment does not yet remove a solve. Subsequent measured ablations are
lazy core guesses, retained X, qualified required solves, verified final-state
reuse and the combined endpoint. Every final correctness and resource gate
in #308 and its children remains required.

The [first retained baseline](../../../benchmarks/results/issue308-host-baseline/README.md)
covers 96-AO RHF batch 1, clean energy/force cold and warm endpoints, changed
geometry, actual host solves and a separate CUDA graph-node/transfer capture.
