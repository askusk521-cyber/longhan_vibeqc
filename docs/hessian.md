# Analytic HF Hessians (issue #180, slice A)

This document is the second-derivative dependency graph requested by step 1 of
issue #180: it maps every term of the RHF energy to the Hessian contribution it
produces, and records which layer supplies that term. It is deliberately
written before any assembly code exists, so the term list and the sign
conventions are fixed in one place instead of being reverse-engineered from
three different providers later.

## Scope of this slice

Slice A targets a **tiny dense analytic RHF Hessian** with component checks, on
systems small enough for the existing reference exporter (`nbf <= 12`). It is
CPU-only.

Explicitly outside this slice, and left fail-closed rather than approximated:

- **DFT** (slice C) — needs the complete LDA/GGA nuclear gradients from #163 and
  #161's derivative kernels;
- **HVP and bounded full-Hessian execution** (slice B);
- **DF, ECP, range-separated and meta-GGA Hessians** — each needs its own
  complete second-derivative/response chain and is *not* inherited from energy
  or first-force support;
- **CUDA execution** and any performance claim.

## What the providers do and do not supply

The two upstream layers are needed here, and both stop short of a molecular
Hessian. Keeping that boundary explicit is the point of this section.

**#178 — second integral derivatives**
(`python/vibeqc_compiler/integral/second_derivatives.py`,
`docs/second_integral_derivatives.md`)

Supplies S/T/V and four-center Coulomb second derivative **integral primitives**
with a fixed external weight, in `raw_hessian`, `weighted_hessian` and
`weighted_hvp` forms. Its own scope note states that external weights,
directions and primitive parameters are fixed, that the provider is unscreened,
and that **electronic response and molecular Hessian assembly are excluded**.

The caller therefore owns, and must fold itself:

- the density and energy-weighted-density values that become the fixed weights;
- every orbit/multiplicity factor — a per-tile weight is not restricted to a
  triangular domain and the provider applies no HF density formula;
- all signs and prefactors (the output name never changes a sign);
- basis-representation conversion for spherical inputs.

**#179 — shared orbital response**
(`tools/vibeqc_response/`, `docs/response.md`)

Supplies the matrix-free CPHF operator (`RHFResponseOperator`), a true-residual
Krylov solver with multi-RHS strategies, and two independent oracles
(`explicit_rhf_response_matrix`, `finite_rotation_jvp`). It constructs **no
right-hand side**: `docs/response.md` assigns nuclear-perturbation RHS
construction to the caller. This work reuses that operator and does not add a
second response solver.

**#141 / #144 — first derivatives**, used to build the response RHS:

- one-electron raw tensors, `build_one_electron_derivative_ir(..., weighted=False)`
  yields a `RawBlock` with layout `(center, xyz, *tensor_indices)`;
- four-center ERI first derivatives exist only as a **weighted contraction**
  (`build_weighted_eri_ir`), not as a raw tensor.

## Term map

Writing the closed-shell RHF energy as

```text
E = E_nuc + Tr[P h] + ½ Tr[P G(P)]
G(P)_μν = Σ_λσ P_λσ [ (μν|λσ) - ½ (μλ|νσ) ]
P = 2 C_occ C_occᵀ
W = 2 Σ_i ε_i C_μi C_νi      (energy-weighted density)
```

the Hessian `H_{R R'} = ∂²E/∂R∂R'` splits into a **skeleton** part, taken with
the density and the orbital coefficients held fixed, and a **relaxation** part
carried by the orbital response.

| # | Term | Form | Supplier |
|---|---|---|---|
| 1 | Nuclear repulsion | `∂²E_nuc/∂R∂R'` | caller, closed form |
| 2 | One-electron skeleton | `Tr[P ∂²h/∂R∂R']` | #178 `build_one_electron_second_ir`, families `overlap`/`kinetic`/`nuclear_attraction`, `weighted_hessian`, W = P |
| 3 | Overlap (Pulay) skeleton | `-Tr[W ∂²S/∂R∂R']` | same provider, W = -W |
| 4 | Two-electron skeleton | `½ Σ_{μνλσ} W_μνλσ ∂²(μν|λσ)/∂R∂R'` with `W_μνλσ = ½ P_μν P_λσ - ¼ P_μλ P_νσ` | #178 `build_eri_second_ir`, `weighted_hessian` |
| 5 | Fock-derivative RHS | `b_ai = (∂F/∂R)_ai` at frozen P, occ-virt block, MO basis | caller: one-electron part from #141 raw `∂h/∂R`; two-electron part from #144 weighted ERI first derivative |
| 6 | Orbital response | solve `A u = -b` | #179 `RHFResponseOperator` + `solve_many` |
| 7 | Relaxation contribution | `u` combined with first derivatives of h, S, and the ERIs | caller |

Terms 2–4 are the "skeleton": they use the *second* derivatives of the
integrals with the density frozen, and they are exactly the shape the #178
provider emits. Terms 5–7 are the "relaxation": they exist because the
coefficients depend on the geometry, and they are what `docs/response.md` hands
to its callers.

**Component separation is a deliverable, not a debugging aid.** Terms 1–4 and
terms 5–7 are accumulated and reported separately, so a missing contribution
shows up as an isolated component error instead of a plausible-looking total.

## Conventions to pin down, and how

Three layers meet here with independently chosen conventions, which is the
highest-risk part of this work:

- **#178** carries `output_sign` on the consumer and `sign` / `prefactor` on the
  weight descriptor. The output name alone never changes a sign.
- **#179** implements `A x = -(∂g/∂t)` for the rotation generator, so a caller
  solving for an external perturbation λ supplies `b = -(∂g_ov/∂λ)` restricted
  to the nonredundant occ-virt block.
- **W** above is the energy-weighted density in the same doubled-occupation
  convention as `P = 2 C_occ C_occᵀ`.

Two further conventions are fixed by position, not by symbol, and must be
carried through assembly explicitly:

- **Hessian axes.** #178 emits `(center_row, xyz_row, center_column, xyz_column)`
  in requested mathematical-center order, *not* physical-atom order. The
  center→atom mapping (which also handles several mathematical centers sharing
  one atom) is applied through the caller's chain rule, and translation recovery
  is already performed inside the generated kernel.
- **Sign of a reported force.** Native forces are `-dE/dR`. The numerical oracle
  below compares against the *gradient*, so the conversion happens once, at a
  named boundary.

These are pinned by component-wise finite-difference checks rather than by
overall numerical agreement, because a wrong sign or factor in one term can
otherwise cancel against another.

## Numerical oracle

Step 2 supplies a finite-difference-of-analytic-gradient Hessian: central
differences of the *analytic energy gradient* over nuclear coordinates, at
several step sizes, with no best-step selection. It is clearly labelled a
numerical Hessian and is an oracle and early utility — it does not constitute
analytic Hessian support.

It is independent of the assembly in step 4 in the useful direction: it depends
only on the analytic first derivatives, so an error shared between the skeleton
and relaxation assembly cannot hide from it.

## Failure behaviour

Unsupported requests fail explicitly rather than substituting a lower-level
result. A full-Hessian request that cannot be expressed within the provider's
tile bounds is rejected; an unimplemented DF/ECP/range-separated/meta-GGA
Hessian is reported as unsupported rather than silently answered with an
HF or energy-only quantity.
