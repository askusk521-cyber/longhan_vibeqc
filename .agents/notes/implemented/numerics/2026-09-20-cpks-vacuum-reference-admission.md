# Decision: share SCF and CPKS vacuum reference admission

Status: implemented
Date: 2026-09-20

## Problem

The production-grid integration admits a subnormal reference gradient when AO
contractions round its spin density to zero. The SCF evaluator already accepts
that reference, but the RKS/UKS directional point evaluators still required an
exactly zero reference gradient. Consequently the same valid SCF reference
could reject an exactly zero response direction. This was reproduced in the
native point contracts for both LDA and PBE after #690 merged.

## Decision

Use one reference-gradient admission predicate in `xc_point.hpp` for SCF and
both shared CPU/device response functions. It validates finite components and
preserves the existing SCF `abs(gradient) < DBL_MIN` allowance only at exactly
zero spin density. It does not modify any input or point formula.

Empty-spin tangent admission is separate and unchanged: density and gradient
directions must be exactly zero. Positive-density algebra, independently scaled
roundoff gates and explicit nonrepresentable-output rejection remain unchanged.
No full empty-spin Hessian or extra functional domain is promoted.

## Rejected alternatives

- Bypassing reference validation for a zero direction would also accept invalid
  normal vacuum gradients or nonfinite inputs.
- Broadening tangent admission would change the response's derivative domain;
  this defect concerns an already accepted reference.
- Adding a positive-density floor would modify qualified low-density physics.
- Duplicating the SCF point evaluation for validation would add work where a
  common admission predicate suffices.

## Evidence

New private-ABI RKS and UKS regressions reject the pre-fix library at the
SCF-admitted reference with a zero direction. They cover both signs, every
Cartesian component/spin, minimum and maximum subnormal gradients, normal
boundary rejection and unchanged nonzero tangent rejection. The device test
checks the same zero-response invariant directly and preserves all 30 RKS plus
48 UKS independent 450-digit reference directions and their numerical gates.

A separate baseline probe on the actual default 55,296-point native grids found
ordinary H2 RKS and LiH+ UKS LDA/PBE zero/nonzero actions already passed; those
specific molecular states contained no affected PBE vacuum residues. The
point-contract failure is not being relabeled as an observed molecular failure.

## Consequences and revisit conditions

SCF and response cannot drift independently on reference-gradient admission.
Any future change to empty-spin tangent semantics requires separate derivative
qualification; it must not be inferred from a reference-domain change.

## References

- Issues #179 and #690.
- `src/dft/xc_point.hpp`, `src/dft/xc_point_response.hpp`.
- `tests/native/test_rks_response.cpp`, `tests/native/test_uks_response.cpp`.
- `tests/native/test_xc_response_cuda.cu`, `docs/xc_scf_domain.md`.
