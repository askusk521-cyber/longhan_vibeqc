# Issue #162 implementation and acceptance record

This is an intermediate record. **Issue #162 is not complete.** The original
RKS/UKS CPU/CUDA prepared-execution scope and both issue addenda remain the
acceptance contract. Gradients are #163; DF is not advertised by this work.

PR #306 publishes the validated resident XC/SCF and native ragged batch
commits `05951fb` and `afad4e4`. Its next stage adds the public #203 KS resource
request/CLI, allocation-owned CUDA shape queries and one persistent ledger
covering both prepare and execute. CPU budgets include all retained grids,
bases, providers and warm states plus serialized setup/SCF workspace. Device
budgets sum all concurrent item arenas; geometry rebuilds retire old owners
before allocation. Failures retain their preparation or execution evidence.

Resource-stage validation on 2026-09-14: 25/25 native CPU tests; 83 selected
Python resource/calculator/batch/HF tests passed, with 3 optional skips;
10 CUDA ownership tests passed. The existing C++ heap audit, extended through
the native KS method adapter, passed ten HF/KS cases including >16-AO water,
cold/replay/changed/restored geometry and complete release after destruction.
On the Slurm RTX 5090, four native tests and 24 KS/HF CUDA resource cases
passed. Six CUDA KS resource cases also passed Compute Sanitizer memcheck
in 124.79 s with zero errors and zero bytes leaked. These capacity checks
do not replace the full #138 numerical/workload evidence gate.

```bash
PYTHONPATH=python:. VIBEQC_LIBRARY=$PWD/build/cpu/libvibeqc.so \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  benchmarks/resource-planning/cpu_inventory.py --build build/cpu \
  --include-ks --output /tmp/ks-cpu-resources.json
```

PR #305 is merged as `15d6936390723edf9e9eb0c91fecde4390490573`.
Implementation `05951fb` is integrated onto that actual squash commit. It retains
all CPU corrections: stable extreme-spin/gradient point algebra, evaluated
UKS returned states, OH occupation stabilization, separate public density and
physical residual diagnostics, and permanent independent endpoint evidence.
The merge-linked issue closure was corrected because the full #162 acceptance
scope is still incomplete.

The prior integration on `c08927e` passed 24 native CPU tests, 97 RTX 5090 point
references and 32 public CPU/CUDA matched-grid endpoint/API cases. Those are
historical measurements; the integration onto the final merged API is being
revalidated. CPU occupation stabilization still needs matching CUDA SCF
implementation and coverage before full cross-backend parity is established.

Validation of `05951fb` on 2026-09-14 passed 25 native CPU tests, 166 selected
Python CPU tests (3 optional skips), 10 CUDA-ownership inventory tests, four
native CUDA tests, and all 32 public CPU/CUDA matched-grid SCF cases. CUDA
execution used an RTX 5090 allocated through Slurm, CUDA 12.9 and sm_120.

The next working-tree change adds native prepared ragged KS scheduling,
geometry rebuilds, resident/frozen/cleared/imported last-good seeds and an
additive per-item SCF diagnostic query. CPU validation passed 25 native tests
and 47 selected Python tests (3 optional skips). Two subsequent derivative-
capability regressions also pass: PBE energy requires first AO jets, LDA only
AO values, and neither energy-only method requires nuclear derivatives.
The matching CUDA rebuild passed. Its four native tests and all 45 combined
public batch/independent-SCF cases pass on the Slurm-allocated RTX 5090.
Compute Sanitizer memcheck of all five CUDA batch cases passed in 239.50 s,
with 0 errors and 0 bytes leaked. This covers ordinary replay, geometry
rebuilds, item failure/recovery, frozen and cleared seeds, atomic source-metric
seed import and iteration-limit handling for both functionals/spin conventions.

```
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 env PYTHONPATH=python VIBEQC_LIBRARY=$PWD/build/cuda/libvibeqc.so \
  VIBEQC_PROFILE=off OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck \
  --error-exitcode 99 --leak-check full \
  .venv/bin/python -m pytest tests/python/test_dft_batch.py -q -k cuda
```

## Implemented and checked so far

- Existing #285 CPU RKS and #214 identical-grid XC integrator/oracles retained.
- Shared CPU/CUDA LDA/PBE point evaluator with stable positive-density PBE
  algebra, no LDA tail substitution, and explicit C2 spin endpoint identity.
  See [the numerical domain](../../../docs/xc_scf_domain.md).
- Independent 97-point E/V fixture: Libxc 7 interior and 450-digit mpmath
  original-formula tail/spin derivatives. CPU and RTX 5090 point tests pass.
- Native ordinary-stream device-buffer XC now reuses the generated AO kernel,
  minimal D/gradient ingredients and shared point coefficients. It consumes a
  method-owned exact-size arena and uploads host quadrature once; iteration
  enqueue has no bulk staging, allocation or synchronization. See
  [the device contract](../../../docs/xc_native_cuda.md).
- RTX 5090 fixed-density CPU/CUDA RKS/UKS E/V, spin finite differences,
  Cartesian/spherical f shells, empty-spin/vacuum tails, device-produced
  density, stale generation/grid rejection, failure isolation and arena
  canaries pass. Compute Sanitizer memcheck reports 0 errors and 0 leaks.
- The #202 exact direct provider now exposes its ordinary stream and a
  validated device-buffer enqueue seam. It reuses common specification checks
  and the existing J/K kernel. Resident RKS/UKS J/K through f agrees with the
  independent CPU integrals; absent terms, alias/shape rejection and numerical
  failure recovery pass alongside the existing DF/response provider checks.
  Both new device test executables pass Compute Sanitizer memcheck with full
  leak checking: 0 errors and 0 bytes leaked for each executable.
- Native ordinary-stream GPU SCF now composes the #202 J provider, XC arena,
  existing device matrix/DIIS/eigensolver/density kernels, and current physical
  state. All iteration matrices remain resident; only scalar records are read.
  Exact state/XC arena allocations are charged to the existing #203 ledger.
- Registered CPU/CUDA LDA/PBE RKS/UKS single-system execution reports the
  actual backend and rejects forces. Energy-only GPU results do not download
  a final density. Compatible last-good warm density is retained on the device;
  failed or exhausted runs cannot replace it. CPU prepared replay also reuses
  its compatible last-good density with fresh DIIS.
- RTX 5090 KS tests pass for LDA/PBE H2 RKS and H/H2+/H3 UKS, including
  independent CPU reconstruction of returned E/D/residual, same-geometry
  resident replay, changed-geometry normalization, stale grid rejection,
  numerical failure recovery, iteration limits and exact arena accounting.
  The native SCF executable passed Compute Sanitizer memcheck with full leak
  checking: 0 errors and 0 bytes leaked.
- CPU LDA/PBE UKS through the method registry and the #202 Coulomb-only
  strategy, with independent spin occupations, Fock/residual evaluation,
  trace-normalized warm density and per-run DIIS state.
- KS convergence gates energy, density change and physical residual; the
  residual gate is at most 1e-9. RKS checks its final physical rebuild too.
- Opt-in normalized DIIS metric avoids the absolute-pivot failure at tight
  residuals. HF retains its original default DIIS behavior.
- Native wrong-factor, XC double-counting, spin swap, empty-spin, stale-grid,
  stale-AO, changed-geometry warm and failed-run tests pass.
- Stable independent matched-grid CPU/CUDA SCF tests pass for LDA/PBE H2,
  He, water (RKS), H, Li and H2+ (UKS), with two independent PySCF initial
  guesses. The public CUDA backend must actually execute for CUDA cases.
  The extended test module passed 28 CPU/CUDA endpoint/API cases; four additional
  water/def2-SVP cases also passed above the small eigensolver dimension.

Validation on 2026-09-13, using GCC 11.4, Python 3.13.9 and PySCF 2.14.0:

```
cmake --build build/cpu -j 8
ctest --test-dir build/cpu -j 6 --output-on-failure
# 23/23 passed

PYTHONPATH=python VIBEQC_LIBRARY=$PWD/build/cpu/libvibeqc.so OMP_NUM_THREADS=1 \
.venv/bin/python -m pytest -q tests/python/test_calculator.py \
  tests/python/test_xc_integration.py tests/python/test_xc_expressions.py \
  tests/python/test_dft_scf.py -k 'not cuda'
# 130 passed, 3 skipped, 25 deselected

cmake --build build/cuda --target vibeqc_xc_point_cuda_tests -j 2
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:02:00 build/cuda/vibeqc_xc_point_cuda_tests
# 72 independent LDA/PBE SCF-domain E/V points passed

cmake --build build/cuda --target vibeqc_dft_cuda_tests \
  vibeqc_cuda_fock_provider_tests -j 2
# Full native CUDA library and both test executables built successfully.
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc 'build/cuda/vibeqc_dft_cuda_tests && \
  build/cuda/vibeqc_cuda_fock_provider_tests'
# Native device-buffer LDA/PBE RKS/UKS E/V and state gates passed.
# CUDA independent J/K: DF layouts/selection and direct through-f values,
# s/p derivatives PASS.

PYTHONPATH=python VIBEQC_LIBRARY=$PWD/build/cpu/libvibeqc.so OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/python/test_xc_integration.py \
  -k identical_grid_independent
# 24 passed, 15 deselected. Existing #214 reference/hash checks retained.
```

Additional native SCF / public adapter validation on 2026-09-13:

```
cmake --build build/cuda --target vibeqc_ks_cuda_tests vibeqc_dft_api_tests -j 2
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:05:00 ctest --test-dir build/cuda --output-on-failure \
  -R '(vibeqc_ks_cuda_tests|vibeqc_dft_api_tests)'
# 2/2 passed, including actual CUDA public RKS/UKS execution.

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 env PYTHONPATH=python VIBEQC_LIBRARY=$PWD/build/cuda/libvibeqc.so \
  VIBEQC_PROFILE=off OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/python/test_dft_scf.py -q
# 28 passed; both backends and both independent PySCF guesses.

ctest --test-dir build/cpu --output-on-failure -R '(dft|uks|xc_point)'
# 5/5 passed after the public adapter change.
```

The CUDA build uses `/group/software/cuda-12.9.1/bin/nvcc` (12.9.86), sm_120,
Release, AOT shells disabled for these independent-provider development
checks. The point test transfers explicit point inputs/outputs; it is not
evidence for GPU XC integration, resident SCF or complete endpoint timing.
The separate `vibeqc_dft_cuda_tests` executable now supplies native GPU
fixed-density integration evidence. It has passed under finite Slurm jobs,
including `/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck
--error-exitcode 99 --leak-check full`; this does not establish full CUDA SCF.
Formatting, evidence-retention, compiler/SCF dependency and CUDA ownership
checks pass. The new point algebra is honestly classified as maintained
scientific code in the CUDA ownership ledger.

Earlier unconstrained exploratory OH calculations are not accepted independent endpoints: native
LDA/PBE converges, while PySCF's stricter convergence flag remains false for
some degenerate-grid traces, including Newton traces despite tiny energy
differences. Preserve this distinction when recording branch/guess evidence.
The later #305 CPU OH records use the explicit C2v reference subgroup and
pass the full unrestricted AO residual gate; see the source-bound correction
record. They do not establish the earlier unconstrained or CUDA traces.

## Remaining work against the full issue

1. Preserve the implemented method-level #203 resource contract as richer
   model options/diagnostics are added; bind new capacities and identities.
2. Extend larger-solver SCF coverage to independent open-shell and replay/failure
   cases; water/def2-SVP CPU/CUDA closed-shell endpoints now pass.
3. Include the now-tested native CPU/CUDA ragged batch paths in the full
   workload/evidence runner. Active/converged/failed isolation, stable order,
   force rejection, warm/frozen/imported replay and changed-geometry rebuilds
   pass; failed items preserve valid warm states.
4. Complete functional/grid/model options and numerical identity. Invalidate
   geometry, basis, spin/charge, grid and functional/regularization changes;
   keep density/orbital generations current and clear stale DIIS/final state.
5. Expose physical residual, density change, occupations, iteration history,
   components, actual backend and transfers. The merged additive SCF query
   already exposes distinct density-update and physical RMS; richer KS
   occupations/history/components/transport remain to be published.
6. Publish permanent source-bound resource evidence alongside the final #138
   workload records. Public KS budgets now include prepare/execute observation,
   all J/grid/XC/state/DIIS owners and host setup bounds. The standalone CPU
   heap audit includes recurrence/eigensolver temporaries; CUDA ledger tests
   verify exact persistent capacities, prepare failure, rebuild and release.
7. Validate CPU/CUDA fixed-density E/V and full RKS/UKS endpoints, failure
   isolation, numerical state, replay, changed geometry and distinct guesses.
   Retain all HF gates. Run every real-GPU test/sanitizer/benchmark through
   finite-time Slurm jobs; do not override assigned CUDA_VISIBLE_DEVICES.
8. Extend #138 evidence runner for cold setup, changed/fixed geometry,
   energy-only, batch 1 and ragged batches. Record grid size, actual backend,
   iteration history, component costs, H2D/D2H bytes and synchronization,
   including host quadrature and final outputs in complete timings.
   The shared entry point is `benchmarks/validation_gate.py`, with protocol
   helpers in `tools/vibeqc_validation/{schema,performance,fixtures}.py`.
9. Keep staged results in PR #306 with exact required AI attribution, complete
   review/CI and the requirement-by-requirement audit before claiming completion.

Workspace: `/home/jzzeng/codes/vibeqc-issue-162`, branch
`codex/issue-162-completion`, starting from upstream `e32c6d4`. The user's
original worktree `/home/jzzeng/codes/qc-mixed-precision` is untouched.
