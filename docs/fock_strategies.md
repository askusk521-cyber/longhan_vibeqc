# Fock provider dispatch audit

Audit baseline: `3da58410bb02a903ea6341a7caac0afc9314355b` (after #239).
This records the existing implementation and the remaining #202 migration;
it does not advertise an executable mixed-provider or DFT SCF method.

## Implementation progress after the audit

The CPU provider integration now executes all exact/DF J/K combinations,
independently absent terms and finite coefficients. `CpuFockProviderView` binds
the existing immutable integral owners; `CpuFockPlanView` validates both sources
before execution and shares one provider call when both terms use the same
source. It introduces neither another integral cache nor another solver.

CPU RHF/UHF direct and fitted entry points now use the same iteration and
finalization code. The generic `run_cpu_fock_strategy` endpoint also accepts
explicit mixed semantics. Fock, energy and first derivatives use one binding;
the DF metric cutoff and complete spectrally truncated response are retained.
The source owner's DF buffers remain included in CPU resource observations.

Single-item method and fleet dispatch now call `run_fock_strategy`. Standard
CUDA HF still uses its established fused solver. CUDA DF raw services now
accept a typed output selection for resident, streamed, tiled, batch, item and
device-pointer execution. Unselected device outputs may be null and are never
accessed; host outputs are empty. Plan scratch capacity remains accounted and
available for subsequent selections. This raw service change does not yet
make arbitrary CUDA strategies executable through the full SCF/force layer.

Initial validation: 14 CPU native suites passed, including all independent
exact/DF pairings against a separate dense contraction, central differences
with a nonzero discarded metric eigenvalue, eight molecular spin/provider
endpoints, warm replay, changed geometry and a ragged fleet with a rejected
neighbor. CUDA validation is recorded in the working artifact directory.

Remaining: production independent CUDA direct bindings, matching CUDA
derivative dispatch, public independent choices and diagnostics, available XC
integration, full prepared identity checks and matched endpoint overhead
evidence. The sections below preserve the baseline audit; they describe the
coupling that existed before these implementation changes.

## Existing mathematical contract

`src/scf/fock_build.hpp` already defines `FockBuildSpec`, independent Coulomb
and exchange term specifications, spin conventions, operator/range parameters,
approximation identity and derivative order. `ResolvedFockBuild` keeps the
mathematical request separate from backend, schedule, screening and DF metric
threshold. Absent terms are canonicalized; malformed parameters and unsupported
range operators fail before execution.

Restricted density includes double occupation. Unrestricted J consumes the
total spin density and K consumes each matching-spin density. Raw J/K matrices
are unscaled; assembly applies each requested coefficient once. Fixed-density
two-electron energy/derivative assembly contains the additional one-half energy
factor. One-electron, Pulay and nuclear terms belong to complete method assembly.

The current independent consumer is `build_exact_direct_jk`: a CPU reference
over an explicitly supplied dense chemists'-order ERI tensor. The dense tensor
is an existing CPU reference representation, not a proposed production CUDA
memory model. Its tests cover absent terms, arbitrary coefficients, spin
conventions, fixed-density derivatives and preflight rejection.

## Dispatch and ownership map

| Boundary | Existing responsibility | Remaining coupling |
| --- | --- | --- |
| `methods/hf_method.cpp:resolve_hf_options` | Translate legacy method/DF options into one resolved request | Both terms inherit one approximation and one backend |
| `HfPreparedSingle::execute` | Select existing CPU/CUDA RHF/UHF solvers | Method code still branches on `legacy_density_fitting` |
| `HfPreparedBatch` / `scf::FleetPlan` | Own compatible buckets, geometry and warm state | Separate direct/DF booleans select the whole bucket |
| `scf/rhf.cpp` | CPU iterations and final energy/force assembly | Direct/DF entry points remain separate; exact entry preflight requires standard complete HF |
| `scf/cuda_rhf.cu` | Persistent direct-HF queues, generated/fallback kernels and solver state | Fused CUDA consumers require the standard coupled HF coefficients |
| `scf/cuda_density_fitting.cu` | Prepared metric/three-center data, bounded J/K, response and device SCF | Public execution helpers always request both J and K |
| `scf/fock_build.cpp` | Validate capabilities and mathematical/execution identity | Nonstandard CUDA terms and independently fitted terms are rejected |

The public method ABI does not yet expose independent J/K provider choices or
the resolved strategy diagnostics. Its existing combined DF enum is an explicit
approximation choice: AUTO does not authorize changing an exact Hamiltonian to
a fitted one.

## Reusable DF primitives

The CPU implementation already separates private `build_coulomb` and
`build_exchange` contractions. Likewise its response implementation separates
the Coulomb quadratic derivative from matching-spin exchange quadratic
derivatives. Reuse the existing metric pseudoinverse and its spectrally
truncated Frechet response; an independent J/K selection must not change the
retained metric subspace or omit metric response.

`cuda_density_fitting.hpp` exposes host-returning batched and item-level raw
J/K execution, plus device-pointer variants on the plan's stream. The latter
perform no mandatory D2H. Their implementation calls separate `build_coulomb`
and `build_exchange` services, but currently requires every output pointer and
executes all terms. Independent-term execution should reuse these services and
skip unrequested work explicitly. The plan's retained and scratch allocations
must remain visible to the existing resource accounting.

The direct CUDA implementation exposes a fused HF Fock path and a diagnostic
Fock-only iteration mode, not a general raw independent J/K provider interface.
The diagnostic mode is not an interchangeable production provider and cannot
be substituted for an independently validated raw consumer.

## Remaining acceptance work

1. Provide typed independent provider execution and coefficient-aware
   assembly, retaining the established fused path for standard HF requests.
2. Integrate exact and fitted combinations, their matching derivative
   contributions, complete prepared identities and failure isolation.
3. Expose requested/resolved semantics and backend/schedule diagnostics while
   preserving existing public defaults and approximation authorization.
4. Connect the available fixed-density XC consumer to the common J/K boundary.
   Complete RKS/UKS and hybrid methods remain owned by #162/#165, which are
   currently open; this refactor must not imply those methods are complete.
5. Validate identical-approximation raw matrices before exact-versus-fitted
   comparisons, then complete CPU/CUDA energy/force, replay, changed-geometry,
   ragged-batch and dispatch-overhead checks under matched final accuracy.

Unsupported range-separated operators remain explicit until #166 supplies
validated values and derivatives. Future providers must extend this common
boundary rather than introduce another method-specific selector or cache.
