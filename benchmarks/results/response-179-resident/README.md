# Exact-CUDA resident multi-RHS response qualification

This compact record qualifies bounded RHF response execution for #179 and its
#180 block HVP consumer. Both modes use conventional unscreened FP64 direct
CUDA J/K, identical converged fixture references, four identical RHS columns
(one dependent column), solver thresholds, and final solution publication with
`collect_basis=False`. No CPU-vs-DF ratio is used.

The measured source is `cda824e2389ae2a44b4822029d56e0f790ba708d` (clean). The loaded native
binary SHA-256 is `bdb5ebdc583371c4b7e7575630d03597921443ea9af16b88f9b26077f84d47a1`. `evidence.json` records the
compiled native source identity, actual selected generic CUDA profile, device,
driver/toolkit, thread settings, all individual samples and memory/counter scopes.
The build uses Release, CUDA 12.9.1, sm_120 and AOT shells disabled.

## Numerical acceptance

All 54 response samples (H2/LiH/water, three strategies, two vector engines,
three repeats) passed independently. The matrix uses committed Libcint AO
integrals transformed into the fixture's MO frame, separately from native J/K.
Maximum relative residual: `4.271820e-14`;
maximum absolute solution error: `9.769963e-14`.
Gates are 1e-9 relative residual and 3e-9 absolute solution error for every
sample. All six complete H2 HVP endpoints pass a 1e-9 maximum-error gate against
the independent dense Hessian. No timing sample is discarded by its accuracy.

## Measured endpoints and work

Times below are medians of repeats 1 and 2 (repeat 0 is also retained). Bytes
and action counts are the first sample; every repeat's raw counts are in JSON.
Host/resident columns compare vector execution, while J/K runs on CUDA in both.

| Case | Strategy | Host solve s | Resident solve s | Actions H/R | H2D bytes H/R | D2H bytes H/R |
| --- | --- | ---: | ---: | --- | --- | --- |
| h2 | sequential | 0.015050 | 0.014772 | 8 / 8 | 256 / 32 | 512 / 256 |
| h2 | blocked | 0.016508 | 0.016040 | 9 / 9 | 288 / 32 | 576 / 332 |
| h2 | recycled | 0.021003 | 0.020478 | 12 / 11 | 384 / 32 | 768 / 420 |
| lih | sequential | 1.135620 | 1.134343 | 48 / 48 | 13824 / 256 | 27648 / 2400 |
| lih | blocked | 0.568179 | 0.569663 | 24 / 24 | 6912 / 256 | 13824 / 1808 |
| lih | recycled | 1.210033 | 1.216819 | 52 / 51 | 14976 / 256 | 29952 / 5260 |
| water | sequential | 2.319602 | 2.316502 | 80 / 80 | 31360 / 320 | 62720 / 5152 |
| water | blocked | 0.870444 | 0.870021 | 30 / 30 | 11760 / 320 | 23520 / 2448 |
| water | recycled | 2.407799 | 2.416411 | 84 / 83 | 32928 / 320 | 65856 / 8172 |

Host payloads/synchronizations are derived from observed successful restricted
J/K calls: one FP64 AO density upload, two AO matrix downloads and one stream
fence per action, as implemented in `src/scf/cuda/direct_jk.cpp`. Resident
counts are raw native diagnostic deltas, including scalar reductions and
4-byte action status. They exclude provider setup/teardown; resident setup
counters are recorded separately. They are API-level accounting, not profiler
measurements of total PCIe traffic or hidden library synchronizations.

The host recycled path checks its explicit zero initial guess on the first RHS;
the resident path recognizes an empty bound space, so its recorded action count
can be one lower. This work difference is exposed, not assumed to be speedup.
Reduced long-vector movement does not establish lower complete endpoint time.
The samples do not support a blanket resident or recycling speedup claim.

The complete-solve timer includes RHS validation, internal rank preparation,
all iteration, final solution publication, automatic recycle teardown and
assembly of the returned solution matrix. The JSON separately retains plan
setup, final owner teardown, and their sum with the first solve. The cold field
is that declared plan-level sum, not a process-start benchmark. Oracle/fixture
loading and RHS creation are outside the timed response endpoint. Warm repeats
reuse the same owner but begin with an empty automatic recycle space; reuse
occurs within each multi-RHS solve. Persistent cross-call recycling is covered
by lifecycle/numerical tests, not inferred from these warm timings.

Operator time measures engine application only. Orthogonalization includes
basis/projection/range factorization; recycling covers retained projection and
replacement. These are disjoint partial components, not a full decomposition:
residual vector arithmetic, small least squares, validation and publication
remain in complete wall time. Small SVD/least-squares and convergence are host
controlled. Native retained J/K/response allocations and the separate
conservative logical solver reservation are recorded; CUDA context/library
memory and provider preparation transients are outside this numeric scope.

## Complete consumer

The native H2 state and independent Hessian oracle are prepared before timing.
Each endpoint builds its own CUDA response provider, prepares three directional
RHS, solves, reconstructs the response, evaluates the declared CPU first/second
integral consumers and publishes complete HVPs. These are mixed host/device
endpoints. A single endpoint sample per combination is numerical/composition
and cost evidence, not a robust performance ranking.

| Strategy | Host HVP s | Resident HVP s | Maximum absolute error |
| --- | ---: | ---: | ---: |
| sequential | 6.391809 | 4.971216 | 3.686e-16 |
| blocked | 4.962943 | 4.972025 | 3.686e-16 |
| recycled | 4.999662 | 4.994462 | 3.686e-16 |

All requested budgets and modeled phase peaks are retained in the consumer
records. Resident consumer counters include resident setup, unlike response
sample deltas. Independent tests additionally cover capacity rejection before
work, changed references, failed replacement and exceptional lease cleanup.

## Reproduction and limits

Build the native library in Release with CUDA enabled, sm_120 and
`VIBEQC_ENABLE_AOT_SHELLS=OFF`. Use one coherent CUDA runtime installation and
preserve the Slurm-assigned device visibility. From the repository root:

```bash
export PYTHONPATH=python:.
export VIBEQC_LIBRARY="$PWD/build-cuda/libvibeqc.so"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 python tools/response_resident_benchmark.py \
  --repeats 3 --consumer --output .artifacts/benchmarks/resident/evidence.json
```

Acceptance is limited to the bounded tools domain. No default/auto-selection
policy changes. Larger systems, changed-geometry performance and constrained
budget performance are required before promotion; they are not established by
this three-fixture campaign. The old `response-179` CPU/DF records remain
historical and are not a speedup baseline for this exact-Hamiltonian comparison.
