# Strict DF final-state contract

The internal `scf/solver/final_state.hpp` contract implements slice A of issue
#311. It is independent of an eigensolver backend. CUDA snapshot production,
SCF finalizer integration, resource accounting and complete-force endpoint
qualification remain subsequent slices. This contract alone changes no public
C/Python result layout or production finalization behavior.

A candidate binds its prepared basis/geometry/representation/device owner,
source item, solve epoch, orbital and density generations, resolved J/K model
and occupied counts. The prepared owner must supply immutable unique identities;
dimensions cannot establish ownership. A new solve must advance its epoch even
when local iteration counters restart. A producer retains both the candidate's
output-density generation and the generation at which its origin Fock was
evaluated. Zero identities and effective/DIIS/shifted origins fail validation.

Selection always evaluates a physical Fock at the supplied density through a
borrowed synchronous callback. The provider must evaluate that density for its
bound model and return the exact current identity; it cannot relabel a cached
Fock. Origin Fock generations may precede the candidate density only when the
frame passes against this current evaluation. This numerical test does not
require artificial equality of adjacent nonlinear SCF iterates.

Validation checks finite symmetric matrices, ordered eigenvalues, physical
eigen residual and S-orthogonality, reconstruction of D from occupied C, the
commutator `F D S - S D F`, electron counts and `D S D = occupation_weight * D`.
RHF uses weight two; UHF uses one for each spin, including an empty beta block.
The existing absolute eigenframe gates are unchanged. Density RMS and maximum
commutator are capped at the smaller of the requested density tolerance and
1e-8; trace and metric idempotency errors are capped at 1e-8. Energy is recomputed
as the nuclear term plus `1/2 sum_spin Tr[D (H + F)]` for the resolved quadratic
J/K model. The contract does not implement a KS/XC energy functional.

A rejected or absent candidate enters bounded strict correction. Each correction
diagonalizes the current physical Fock through the borrowed operation, validates
its output before projection, advances both determinant generations, and
evaluates the physical Fock again at the new density. Success requires all state
checks and the requested energy-change tolerance. A single rebuild is therefore
insufficient when its energy change or current-F residual fails. The default
budget is four corrections; zero allows validation only, and values above 64
are invalid. Generation overflow, exhaustion and provider failures return no
successful state. An independent later call can recover without retained state.

The explicit force-rebuild control executes that same solve/project/evaluate
path even for a valid candidate. Counts report actual physical evaluations,
per-spin eigensolves, joint density updates and rejected candidates. The host
component ledger separately records validation, physical Fock construction,
strict correction and weighted-density construction. It uses the existing
`final_fock` reason for actual provider solves.

Energy-only selection leaves W empty. Requests needing W compute
`C diag(occupation_weight * epsilon) C^T` only after the state passes; there is no
cached-W input. A downstream occupied factor must match the verified determinant
identity and density witness. Downstream integration must additionally enforce
the solve epoch and full model identity, budget snapshots and scratch, preserve
all DF derivative terms, and validate physical-reference export.

`tests/native/test_final_state.cpp` supplies analytic nonidentity-metric RHF/UHF
frames, degenerate rotations, independent energy/W expectations, identity and
numerical faults, actual rebuild counts, old-factor rejection and an oscillating
nonlinear Fock. Neither production nor reference diagonalization constructs the
expected answers. Molecular forces, device snapshot lifecycle and performance
claims require the later integrated paths and their independent qualification.
