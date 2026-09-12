# HF module decomposition (#240)

The independent CPU reference arithmetic, initial-state preparation, iteration
control and stationary force assembly now have explicit interfaces in
`src/scf/reference/`, `src/scf/initial_guess/`, `src/scf/solver/` and
`src/scf/gradient/`.
They retain the original loop order, occupation factors, Jacobi thresholds,
overlap rejection threshold, and UHF frontier-rotation policy. This extraction
does not complete #240: CUDA/DF legacy orchestration, provider planning and
CUDA gradient/finalization still require the remaining moves below.

## Responsibility inventory

The source baseline is `9f5b625`. The table describes ownership groups, not
mathematical equivalence or a new replacement implementation. The scientific
CUDA classifications and exact source anchors remain in `cuda_ownership.json`
and `cuda_ownership_current.json`, under #231.

| Source / group | Functions and state | Destination / current boundary |
| --- | --- | --- |
| `rhf.cpp`: reference linear algebra | `Matrix`, `EigenResult`, `index`, `identity`, `multiply`, `transpose`, `symmetric_eigen`, `symmetric_orthogonalizer`, `generalized_eigen`, `dot`, `solve_linear` | Extracted to `reference/linalg.*`; no HF, CUDA, or external numerical-library dependency. |
| `rhf.cpp`: mean-field reference contractions | `density_from_orbitals`, `energy_weighted_density`, restricted/unrestricted electronic energies, joined/split spin matrices, `commutator_residual`, density/residual RMS | Extracted to `reference/mean_field.*`; full dense AO conventions and spin factors remain explicit. |
| `rhf.cpp`: core/warm initial state | `spin_occupations`, `mix_open_shell_frontier_orbitals`, `normalize_spin_density`, `prepare_initial_density`, `prepare_initial_uhf_density` | Extracted to `initial_guess/density.*`; callers still own physical state/topology validation. |
| `rhf.cpp`: iteration control and safeguards | `Diis`, generations, proposal validation, `safeguarded_update`, finalization, `run_rhf_host_plan`, `run_uhf_host_plan` | Extracted to `solver/diis.*`, `solver/proposal_control.*` and `solver/mean_field_driver.*`. The #186 proposal contract, independent physical residual checks and convergence/finalization order are unchanged. Legacy CUDA/DF recovery orchestration still uses the shared DIIS owner. |
| `rhf.cpp`: provider semantics and accounting | `CpuScfIntegralDataView`, integral capacity sampling, `build_fock`, `build_uhf_focks`, prepared strategy entry points, CUDA DF plan preparation | The common CPU driver consumes #202's `PreparedFockPlan` with no direct/DF storage branches. `PreparedFockPlan::cpu_observation_capacity()` now owns the prior capacity accounting, counting shared AO data once. Legacy CUDA/DF provider orchestration remains in the compatibility driver. |
| `rhf.cpp`: stationary forces | `analytic_forces`, `analytic_uhf_forces`, DF/CUDA gradient adapters and final force assembly | Provider-based stationary assembly is extracted to `gradient/hf_gradient.*`, taking the provider's explicit positive derivative rather than owning a plan. Legacy CUDA/DF gradient adapters and finalization remain; their provider, overlap/Pulay, one-electron and nuclear terms must remain distinct. |
| `rhf.cpp`: compatibility entry points | Public RHF/UHF CPU/CUDA wrappers and CPU-build CUDA stubs | Keep existing method/ABI signatures, failure behavior and per-item ordering. |
| `cuda_rhf.cu`: scientific reference/fallback/specializations | `Dual`, `Dual3`, `MixedPrecisionFloat`, angular/Hermite/Coulomb workspaces, primitive/contracted one-/two-electron values and gradients, order-specific Fock/force kernels | Move by the existing #231 scientific ownership regions; do not copy formulas or turn the independent oracle into generated production arithmetic. |
| `cuda_rhf.cu`: queues, work descriptors and compaction | `ActiveShellQuartetTile`, `DirectTileValidationRecord`, `PsssResidentTask`, pair bounds, descriptor validation, bounded page ranges and queue kernels | Host partitioning and page ranges use `cuda/queue_plan.*`. Device validation, density bounds, compaction, generated/resident tasks, bounded pages, scans and diagnostics now have separate `cuda/direct_*` owners. Fused native bounded consumers and host launch sequencing remain pending. |
| `cuda_rhf.cu`: generic SCF kernels and libraries | Density/Fock/update/convergence kernels, inactive-eigensolver profiles and launch selection | Generic eigensolver execution lives in `cuda/eigensolver.cpp` / `eigensolver_kernels.cu`. Shared matrix, density, DIIS, convergence and state kernels now have separate `cuda/scf_*_kernels.*` owners; public/direct basis transforms use `basis_transform_kernels.*`. Scientific integral/Fock kernels and host bucket control remain pending. |
| `cuda_rhf.cu`: host planning and replay | `DeviceBatch`, `HostBatch`, `ArenaLayout`, `CudaResources`, `CudaRhfBucketPlan`, integral source implementation, graph construction and bucket dispatch | Packed/POD contracts and host topology/arena planning already have separate owners. DF source/export uses `cuda/df_source*` / `df_integral_export*`. Stream/graph/arena lifetime now lives in `cuda/resources.*`, with borrowed matrix-library execution in `matrix_library.*`. Graph construction and bucket dispatch remain pending. |
| Former `cuda_density_fitting.cu`: metric and storage planning | `SetupBuffers`, `CudaDensityFittingJkPlan`, checked sizes, cuSOLVER setup, plan creation/release and diagnostics | Extracted to `cuda/df_plan*`, `df_setup_internal.hpp`, `df_runtime.*` and `df_metric_kernels.*`. The public handle remains opaque; source transfer and retained metric factors keep their single owner. |
| Former `cuda_density_fitting.cu`: J/K execution | `build_coulomb`, `build_exchange`, tile gather/transpose/reduction kernels, RHF/UHF host/device/item entry points | Extracted to `cuda/df_coulomb.cpp`, `df_exchange.cpp`, `df_jk*`. Resident, host-backed and source-backed execution retain the same provider semantics and memory sub-budget. |
| Former `cuda_density_fitting.cu`: force integration | `execute_cuda_density_fitting_generated_force_response` | The adapter now lives in `cuda/df_force_response.cpp`. The #205 source-backed response borrows device factors through `df_response_weights.*` / `df_gradient_bridge.*`; the host-value compatibility adapter retains its CPU metric path. |
| Former `cuda_density_fitting.cu`: iterative replay | `DeviceSolver`, `DeviceIterationGraph`, `PersistentScfState`, eigensolve wrappers, RHF/UHF device SCF loops and graph-tail kernels | Extracted to `cuda/df_scf_state.hpp`, `df_scf_library.*`, `df_rhf_scf.cpp`, `df_uhf_scf.cpp` and `df_scf_kernels.*`. Host replay/graph control compiles in C++; only equations and the graph-tail kernel require CUDA compilation. |

## Dependency and size gates

`tools/check_scf_structure.py` runs in pre-commit and exposes `--json` for the
current shared-module include graph and sizes. Reference numerics may depend
only on reference numerics and standard headers. Initial-guess preparation may
add core/integral data interfaces. Neither layer may include the method driver,
public ABI implementation, CUDA providers, or later DFT/CC methods. The solver
may consume the shared provider/types/proposal interfaces and gradient assembly;
gradient assembly cannot acquire solver state. Tests cover
relative and angle-bracket include spellings and the reverse dependency edge.

New shared CPU modules should remain below 600 physical lines per source file;
a larger module requires a documented responsibility and build-cost argument.
This is a review target, not a claim that legacy source units already pass a
structural-size gate. At the baseline, `rhf.cpp` contains 3,413 lines / 165,000
bytes, `cuda_rhf.cu` 19,754 lines / 1,048,372 bytes, and
`cuda_density_fitting.cu` 3,098 lines / 160,928 bytes. The original DF unit is now
removed; its largest replacement is the 549-line setup transaction. The large
direct CUDA unit still exceeds the target and remains unfinished work under
#240. Its kernel/template and runtime coupling explains the staged extraction,
not an exemption from the issue's final acceptance criteria.

## Validation and remaining acceptance

The first extraction is checked with the existing RHF/UHF, direct/DF, spherical,
warm-batch, proposal, precision, and public native CPU tests. CUDA builds and
allocated-device endpoint tests are also required because `rhf.cpp` supplies
host control and finalization for native CUDA execution. Function movement must
preserve arithmetic order; new scientific behavior belongs in a separate fix.

A local compiler-cost sample used GCC 11.4.0, `-O3 -DNDEBUG`,
`VIBEQC_HAS_CUDA=0`, and `CCACHE_DISABLE=1`. Each translation unit was compiled
once with a warm filesystem cache, using its existing Ninja command and a
separate object/dependency-file destination. These are compiler-work samples,
not whole-build or molecular runtime measurements:

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| Baseline `rhf.cpp` | 2.781 | 141,960 |
| Extracted driver `rhf.cpp` | 2.223 | 116,520 |
| `reference/linalg.cpp` | 0.805 | 19,136 |
| `reference/mean_field.cpp` | 0.504 | 11,432 |
| `initial_guess/density.cpp` | 0.667 | 14,416 |

Separate implementation edits now compile their own small object. The combined
serial compiler work increases from 2.781 to 4.199 seconds in this sample, and
the summed object size increases from 141,960 to 161,504 bytes. This extraction
therefore establishes narrower rebuild ownership, not a cold-build speedup.
The representative CUDA rebuild after editing these CPU implementations
compiled only the affected C++ objects and build-identity object, then linked;
it did not compile a CUDA kernel object. A before/after full CUDA build study
remains part of the later decomposition acceptance.

Final #240 acceptance still needs the remaining inventory groups moved behind
stable interfaces, a complete before/after largest-file inventory, cold and
representative incremental build measurements with touched-object sets, and
unchanged direct/DF energy/force and batch/failure gates. A successful CPU
extraction alone is insufficient to close the issue.

## CPU solver and gradient extraction

The common host loops now compile once against the prepared-provider interface
instead of being private templates embedded in `rhf.cpp`. Their reference
arithmetic remains independent of generated backends. A comparison against
`64b1551` verifies unchanged arithmetic in 16 moved functions; the explicitly
recorded interface changes move derivative acquisition to the caller and
numerical-capacity observation to the provider owner. The DIIS augmented solve,
singular fallback, proposal generation, damping schedule, three-failure limit,
per-item history, physical convergence comparator and final unextrapolated Fock
rebuilds are preserved.

The source header `types.hpp` now directly includes the public enum definitions
it uses; its previous reliance on include order surfaced when proposal control
became an independent translation unit. This changes no struct layout or ABI.
The largest new implementation is `solver/mean_field_driver.cpp`, below 300
lines. The compatibility driver is approximately 2,530 lines versus 3,413 at the
initial inventory. This is a CPU ownership improvement; it does not satisfy the
remaining large-CUDA-file acceptance criteria by itself.

A second GCC 11.4 Release sample disabled ccache for each affected translation
unit. These are individual compiler-work samples with a warm filesystem cache,
not a whole-build measurement:

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| Before: `rhf.cpp` | 2.233 | 116,520 |
| Before: `fock_prepared.cpp` | 1.165 | 48,432 |
| After: `rhf.cpp` | 1.110 | 38,680 |
| After: `fock_prepared.cpp` | 1.270 | 49,488 |
| `solver/diis.cpp` | 0.580 | 10,632 |
| `solver/proposal_control.cpp` | 0.880 | 20,136 |
| `solver/mean_field_driver.cpp` | 1.506 | 59,128 |
| `gradient/hf_gradient.cpp` | 0.329 | 3,200 |

Summed compiler work for these owners rises from 3.398 to 5.676 seconds;
summed object bytes rise from 164,952 to 181,264. A captured implementation-only
mean-field-driver edit rebuilds that C++ object and `c_api_tuning.cpp` (source
identity), then relinks the library and dependent probes/tests. It recompiles
no CUDA kernel object. Full cold-build and device-link measurements remain
part of the later CUDA decomposition acceptance.

## CUDA topology, planning and eigensolver extraction

The next runtime move removes approximately 2,100 lines from `cuda_rhf.cu`.
Host topology packing, checked arena sizing, bounded queue partitioning and
queue diagnostics compile in ordinary C++ owners. Generic native/library
eigensolver dispatch borrows the existing cuSOLVER handles and workspaces;
only six narrow launch wrappers and the unchanged native Jacobi/instrumentation
kernels require CUDA compilation. The largest new implementation is 458 lines.
Shared headers contain POD layouts or narrow contracts; there is no umbrella
header containing the remaining scientific recurrence implementations.

A source audit against `bd657ec` verifies 26 function bodies, including the
remaining bucket executor, after two explicit interface changes: borrowed
library resources and host-callable launch wrappers. Each wrapper preserves
launch geometry, stream, shared bytes and argument ordering. Provider input
sanitization, inactive-state profiling, Jacobi rotations, stable eigenpair
sorting and last-error checks retain their existing order. Dependency gates
reject method-driver imports in these runtime owners and direct queue-policy
imports in the generic eigensolver owner.

Validation completed on the extracted implementation: 16 native CPU tests,
3 native CUDA runtime/provider/composition tests, and 106 Python GPU checks
with no skips. GPU execution used Slurm on the RTX 5090. The Python checks
cover RHF/UHF direct/DF Fock composition, checkpoint replay, public batch
behavior and the larger def2-TZVP water eigensolver endpoint.

Compiler samples below use GCC 11.4 Release and NVCC 12.9 sm_120 with the
explicit development fast-compile mode enabled, disabling ccache for each
invocation. These are single compiler-work samples with a warm filesystem
cache on a shared machine; production cold-build, device-link and performance
acceptance remain outstanding.

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| baseline_cuda_rhf | 43.269 | 9,891,176 |
| extracted_cuda_rhf | 41.937 | 9,815,088 |
| arena | 0.674 | 15,832 |
| topology | 1.120 | 26,440 |
| queue_plan | 0.683 | 10,568 |
| queue_profile | 0.681 | 10,864 |
| eigensolver | 0.781 | 5,520 |
| eigensolver_kernels | 2.826 | 267,552 |

Summed compiler work rises from 43.269 to 48.702
seconds; summed object bytes rise from 9,891,176 to
10,151,864. Actual implementation-only edits to
`topology.cpp` and `eigensolver.cpp` each rebuild their own C++ object plus
`c_api_tuning.cpp` source identity, then relink dependents. Neither edit
recompiles a CUDA kernel. The exact source was restored and rebuilt after
each probe.

`cuda_rhf.cu` remains 17,658 lines. Scientific direct Fock/force kernels,
device queue execution, graph/bucket control, source-backed DF integration,
and `cuda_density_fitting.cu` decomposition are still required for full #240
acceptance. The lower CUDA ownership line count from moving host code into
C++ is a structural move, not retirement of scientific arithmetic.

## DF source and tensor-export ownership

The source boundary now separates `df_source_setup.cpp` (validation, transforms,
metadata and current metric setup), `df_source.cpp` (bounded replay and public
source diagnostics), and explicit Cartesian tensor export in
`df_integral_export.cpp` / `df_integral_export_batch.cpp`. Their three kernel
launch wrappers and generated-policy basis/public-layout contractions live in
`df_source_kernels.cu`. Shared allocation registration is in
`metadata_upload.hpp`, because both direct J/K and DF already used that helper.
This avoids making a direct provider depend on private DF source state.

The mathematical DF policy, metric/raw/public layouts, primitive contraction,
coordinate mapping, partial tiles, export budget calculation, allocation
failure cleanup, and diagnostic strings remain unchanged. An audit against
`4b2083a` verifies 29 function bodies and all value/response specialization
arguments in three launch wrappers. The device record layout and public opaque
handle remain unchanged; only the owning translation units move. No generated
or handwritten scientific formula is copied into a second maintained file.

`cuda_rhf.cu` decreases from 17,658 to 16,018 lines for this move. The largest
new implementation is 437 lines. The CUDA scientific ownership ledger counts
existing bounded layout contractions in their new owner and distinguishes
host launch wrappers; reduced counted CUDA host lines do not constitute
scientific-code retirement.

The following development compiler-work sample uses ccache-disabled NVCC 12.9
sm_120 fast-compile mode and GCC 11.4 Release. The baseline is the exact
`4b2083a` source copied to a separate evidence path and compiled against
unchanged parent headers, so absolute compiler source paths differ. These
single-invocation samples are not production cold-build or runtime evidence.

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| baseline_cuda_rhf | 41.761 | 9,896,208 |
| extracted_cuda_rhf | 37.381 | 8,652,736 |
| df_source_setup | 1.672 | 71,448 |
| df_source | 0.943 | 29,584 |
| df_integral_export | 1.423 | 52,328 |
| df_integral_export_batch | 1.526 | 59,216 |
| df_source_kernels | 4.376 | 2,225,240 |

Aggregate compiler work rises from 41.761 to 47.320 seconds;
object bytes rise from 9,896,208 to 11,090,552.
Scientific direct kernels, GPU queue execution, graph/bucket control, and
`cuda_density_fitting.cu` planning/SCF ownership remain under the full #240
acceptance criteria.

This extraction passed 36 CPU structure/ownership/source checks, three native
GPU DF/provider/composition tests, and 34 Python GPU endpoint/resource checks
without skips. The allocated RTX 5090 runs cover values, complete forces,
Cartesian/spherical RHF/UHF, geometry replay, rank crossings, failed neighbors,
and positive device budgets. Actual implementation-only edits to source setup
and batched tensor export each rebuilt only their C++ owner and source identity,
then relinked; neither edit recompiled a CUDA kernel. The exact source was
restored and rebuilt after each probe.

After integration with the device force-response implementation and current
upstream, the combined source extraction passed 75 CPU structure/ownership
checks, three native GPU tests, and 38 opt-in Python GPU endpoint/resource
tests with no skips. The 29-function and three-wrapper audit still passes.

## DF plan, J/K and persistent solver ownership

The former 3,076-line `cuda_density_fitting.cu` is removed. Its existing
implementations now compile under these separate responsibilities:

| Owner | Responsibility |
| --- | --- |
| `df_plan.cpp`, `df_plan_setup.cpp`, `df_plan_lifetime.cpp` | Public opaque handle, setup transaction and teardown. |
| `df_runtime.*`, `df_setup_internal.hpp`, `df_plan_internal.hpp` | Checked sizes/status mapping, temporary setup allocations and sole plan storage layout. |
| `df_coulomb.cpp`, `df_exchange.cpp`, `df_jk.cpp` | Bounded mathematical J/K and public host/item/device adapters. |
| `df_force_response.cpp` | Borrow retained device factors for source response, or the existing explicit host-value compatibility path. |
| `df_scf_state.hpp`, `df_scf_library.*` | Persistent allocations, graph handles and cuBLAS/cuSOLVER workspace integration. |
| `df_rhf_scf.cpp`, `df_uhf_scf.cpp` | Host replay, graph capture/fallback, convergence and result publication. |
| `df_metric_kernels.*`, `df_jk_kernels.*`, `df_scf_kernels.*` | Unchanged device equations/reductions and 18 exact launch wrappers. |

The largest new implementation is the 549-line setup transaction. It remains
one transaction to preserve validation, source transfer, allocation failure
cleanup, metric factorization, diagnostics and publication order. No kernel
owner includes host plan state. The new dependency checks reject that edge as
well as imports of the RHF method driver into shared DF runtime owners.

A source audit against `bd9f3de` checks 54 function bodies, five storage layouts
and all 18 launch wrappers. Arithmetic order, streams, launch dimensions,
shared bytes, reduction order, public signatures and the persistent-state
lifetime are unchanged. Default arguments live only in the shared declaration.
The host cuBLAS J/K composition remains native scientific code; moving it from
CUDA to C++ does not retire its mathematics from #231's ownership model.

The compiler-work sample below uses ccache-disabled NVCC 12.9 sm_120 development
fast-compile mode and GCC 11.4 Release, with a warm filesystem cache on a shared
machine. The baseline is the exact parent source in its separate DF-source
worktree; absolute source paths differ. These are individual compiler samples,
not production cold-build, device-link or molecular runtime measurements.

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| baseline_cuda_density_fitting | 6.381 | 456,488 |
| df_coulomb | 0.833 | 10,760 |
| df_exchange | 0.863 | 14,008 |
| df_force_response | 0.885 | 14,928 |
| df_jk | 1.017 | 30,848 |
| df_jk_kernels | 2.411 | 86,984 |
| df_metric_kernels | 2.337 | 35,024 |
| df_plan | 0.744 | 6,224 |
| df_plan_lifetime | 1.096 | 19,608 |
| df_plan_setup | 1.426 | 54,080 |
| df_rhf_scf | 1.314 | 39,888 |
| df_runtime | 1.182 | 32,248 |
| df_scf_kernels | 2.475 | 164,376 |
| df_scf_library | 0.978 | 12,200 |
| df_uhf_scf | 1.361 | 43,368 |

Aggregate compiler work rises from 6.381 to 18.922
seconds; object bytes rise from 456,488 to 564,544.
This extraction narrows rebuild ownership; it does not establish a cold-build
speedup. The direct scientific/queue/graph groups in `cuda_rhf.cu` and the full
production build/runtime acceptance remain unfinished under #240. Earlier
sections record the state at each intermediate extraction checkpoint.

Validation of this DF runtime extraction passed 16 native CPU tests,
83 structure/ownership/source checks, three native GPU tests and 138 Python GPU
tests with no skips. GPU checks ran through Slurm on the RTX 5090 and cover the
complete derivative, resource, Fock-composition, checkpoint-replay and native
CUDA-runtime suites. Hooks and the function/layout/launch audit pass.

Actual implementation-only edits to `df_plan_setup.cpp`, `df_scf_library.cpp`
and `df_exchange.cpp` each rebuild only that C++ owner plus the source-identity
object, then relink dependents. They compile no CUDA kernel object. Each probe
restores and rebuilds the exact validated source before the next edit.

## Shared SCF device kernels

State initialization and solver-result routing, generic matrix operations,
density/warm-state preparation, DIIS, physical convergence, and public/direct
basis transforms now have six distinct kernel owners in `src/scf/cuda/`.
The 34 host-callable wrappers preserve launch geometry, shared bytes, streams,
arguments and the two retained-density template specializations. Shared launch
and convergence constants live in `scf_constants.hpp`; common kernel owners
cannot include direct queue policy or host plan state.

An audit against `cb12e0a` checks all 36 moved function bodies and the entire
remaining direct CUDA implementation after removing only the definitions and
replacing launch syntax. It also checks the 34 wrapper bodies. The warm-density
block reduction, metric trace normalization, open-shell frontier rotation,
DIIS reduction/solve, mixed-to-target refinement reset, final-Fock reuse and
warm-state restoration retain their exact arithmetic and per-item routing.
No scientific formula is duplicated into another maintained owner.

`cuda_rhf.cu` drops from 16,017 to 15,060 lines; the largest new implementation
is 370 lines. The ownership ledger retains scientific classification for
occupied/energy-weighted density and energy equations, while matrix algebra,
state, DIIS, convergence and launch routing remain runtime code. This still
leaves the direct scientific, device queue and host graph/bucket groups, plus
full production build/runtime acceptance, under the open #240 issue.

The following compiler-work sample uses ccache-disabled NVCC 12.9 sm_120
in development fast-compile mode, with a warm filesystem cache on a shared
machine. The exact parent and extracted source use separate worktrees, so
absolute paths differ. It is not production cold-build, device-link or molecular
runtime evidence.

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| baseline_cuda_rhf | 37.385 | 8,654,208 |
| extracted_cuda_rhf | 36.564 | 8,440,480 |
| scf_state_kernels | 1.008 | 44,184 |
| scf_matrix_kernels | 1.073 | 128,592 |
| scf_density_kernels | 1.104 | 172,088 |
| scf_diis_kernels | 1.004 | 92,808 |
| scf_convergence_kernels | 1.179 | 226,176 |
| basis_transform_kernels | 1.024 | 93,000 |

Aggregate compiler work changes from 37.385 to 42.957
seconds; aggregate object bytes change from 8,654,208 to
9,197,328. These figures do not establish a cold-build speedup.

The formatted extraction passes four native GPU tests and 123 Python GPU tests
with no skips, covering direct/DF provider composition, checkpoint replay,
RHF/UHF batches, precision provenance, final-Fock reuse, higher-angular-momentum
calculator endpoints and warm-state failure isolation. GPU execution used Slurm
on the RTX 5090. The combined code-generation/structure run passed 376 checks
with 48 opt-in compiler probes skipped; its one stale launch-syntax assertion
was corrected and both affected regression checks pass in the focused rerun.
Hooks and the complete remaining-source audit also pass.

Actual DIIS and convergence implementation edits each rebuild their own CUDA
object plus the source-identity object, then relink. Neither probe compiles
`cuda_rhf.cu` or an unrelated kernel owner. The exact source was restored and
rebuilt after both probes.

## CUDA resource lifetime and matrix-library execution

`resources.*` owns the prepared bucket's stream, graphs, library workspaces and
arena. Its destructor now compiles in ordinary C++, preserving owning-device
selection, graph destruction, library-handle release, stream-ordered frees,
stream drain and host-workspace release in their original order. The resource
field layout is unchanged. Eigensolver and matrix consumers borrow explicit
views; they do not acquire allocation or graph ownership.

`matrix_library.*` consumes only a stream and cuBLAS handle. It preserves the
resolved native/cuBLAS route, column-major strides, active masks and spin-aware
broadcasting. `runtime_support.*` keeps the existing status mappings and
nonempty host-upload behavior. Dependency gates reject imports of the bucket
resource owner into matrix-library execution and reject method-driver imports
into both owners.

The audit against `36b7209` checks eight moved bodies, the resource field and
special-member layout, the two-handle borrowed matrix view, and the entire
remaining direct CUDA body after only definition removal and explicit view
construction. This is an ownership extraction: no equation, fallback decision,
resource budget, public ABI or iteration order changes. Host graph construction
and bucket dispatch, device queues and direct scientific kernel decomposition
remain unfinished under #240.

The direct CUDA implementation decreases from 15,060 to 14,902
lines; the largest new C++ implementation has 78 lines.
The following compiler-work samples disable ccache and use NVCC 12.9 sm_120
development fast-compile mode and GCC 11.4 Release. The exact parent source and
candidate use separate worktrees with different absolute paths, on a shared
machine with a warm filesystem cache. These are not production cold-build,
device-link or molecular runtime measurements.

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| baseline_cuda_rhf | 36.442 | 8,440,448 |
| extracted_cuda_rhf | 36.396 | 8,428,544 |
| resources | 1.030 | 12,616 |
| matrix_library | 0.382 | 4,584 |
| runtime_support | 0.354 | 1,968 |

Aggregate compiler work changes from 36.442 to 38.163
seconds; aggregate object bytes change from 8,440,448 to
8,447,712. This establishes ownership boundaries, not a cold-build speedup.

Validation passed 88 structure/ownership checks, four native GPU tests,
123 Python GPU tests and 14 allocation-budget GPU tests, with no GPU skips.
GPU execution used Slurm on the RTX 5090. The body/layout audit and hooks pass.
The allocation checks cover prepared execution, failure cleanup and repeated
plan lifetime behavior with the extracted resource destructor.

Actual implementation-only edits to `resources.cpp` and `matrix_library.cpp`
each rebuild only the edited C++ object plus source-identity metadata, then
relink dependents. The observed rebuilds take 1.585 and 1.589 seconds,
respectively, and compile no CUDA kernel object. Both probes restore and rebuild
the exact validated source. Full production acceptance remains under #240.


## Direct device queues and screening

Nine CUDA owners now separate tile validation, density-bound reduction, tile
compaction, generated exact-class tasks, resident-bra tasks, bounded pages,
bounded generated tasks, scans/retries and diagnostic counters. Five shared
headers contain only indexing, task encoding, physical screening, page-tail
bounds and profiling helpers; the largest is 199 lines. Queue implementations
consume borrowed packed metadata and cannot import host bucket resources or
integral recurrence implementations.

The exact-parent audit against `fda0f69` preserves 53 function/type definitions,
26 launch wrappers, the unclassified-slot sentinel, and the entire remaining
CUDA implementation after explicit launch-interface substitutions. It checks
macro-spliced launch sites as well as ordinary calls. RHF/UHF and Fock/force
specializations retain their original bodies and launch geometry. The bounded
generated wrapper exposes only the materializing specializations already used
by the driver; per-class overflow and exact-page recovery remain unchanged.

The physical density/Schwarz gates, conservative page tails and mixed-precision
contribution cutoff remain classified as scientific code in the #231 ledger.
Queue bookkeeping and diagnostic counts remain runtime code. This move does
not retire a screening formula or duplicate an integral evaluator.

`cuda_rhf.cu` decreases from 14,902 to 13,172 lines. The largest new CUDA
implementation is 241 lines. Host bucket/graph construction, host generated
launch sequencing and the direct scientific/force kernels remain under #240;
fused native bounded consumers still enumerate and drain their own work.

The following ccache-disabled compiler samples use NVCC 12.9 sm_120 development
fast-compile mode, a warm filesystem cache and separate parent/candidate
worktrees with different absolute paths on a shared machine. They measure
individual compiler invocations, not production cold builds, device linking,
resource acceptance or molecular runtime.

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| baseline_cuda_rhf | 36.337 | 8,428,512 |
| extracted_cuda_rhf | 33.013 | 8,041,040 |
| direct_tile_validation | 2.683 | 151,032 |
| direct_density_bounds | 2.713 | 174,536 |
| direct_tile_compaction | 2.831 | 325,744 |
| direct_generated_tasks | 2.670 | 143,560 |
| direct_resident_tasks | 2.669 | 121,144 |
| direct_bounded_pages | 2.878 | 366,888 |
| direct_bounded_tasks | 2.872 | 390,904 |
| direct_queue_scan | 5.236 | 322,976 |
| direct_queue_diagnostics | 2.667 | 150,408 |

Aggregate compiler work changes from 36.337 to 60.233
seconds; aggregate object bytes change from 8,428,512 to
10,188,232. The reduced scope of an implementation edit
comes with additional aggregate compiler work in this development sample.

Validation passed four native GPU tests and 137 Python GPU tests with no skips,
including the 14 allocation-budget cases. Six exact-parent comparisons cover
fixed, resident and paged RHF/UHF execution with ragged batches of three,
STO-3G, descriptor validation, cold/warm state and explicit changed geometry.
All 96 energy/force comparisons pass their existing tolerances; the largest
absolute difference across the compared arrays is `3.55e-14`.
Both GPU gates ran through Slurm on the RTX 5090. These correctness comparisons
do not establish production runtime acceptance for the remaining #240 work.

The combined source/structure suite passed 350 checks with 48 opt-in compiler
probes skipped. Its one stale page-screening source-location assertion was
updated for the extracted owner and passes in a focused rerun. The additional
ownership/publication checks, refreshed source snapshot, hooks and complete
body/launch audit pass.

Implementation-only comment edits to `direct_queue_scan.cu` and
`direct_bounded_pages.cu` each trigger only their own CUDA object and the
source-identity C++ object, followed by relinks. The ordinary cache-enabled
Ninja rebuilds take 1.751 and 1.606 seconds. Neither rebuild invokes the
compiler for `cuda_rhf.cu` or an unrelated CUDA owner. Both probes restore and
rebuild the exact source; the dependency scope is distinct from the uncached
compiler-work samples above.
