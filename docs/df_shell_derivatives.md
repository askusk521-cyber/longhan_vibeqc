# Generated shell-block DF response

`VIBEQC_DF_WEIGHTED_EXECUTION=shell` selects generated weighted three-center
derivatives across all 64 s/p/d/f shell classes. `shell-sp` retains the original
seven non-SSS s/p classes as a comparison subset. The default remains `generic`
until complete endpoint measurements qualify promotion. The initial s/p-only
prototype did not resolve the 192-to-384-AO force scaling problem in #308.

The scalar and shell consumers use the same generated primitive geometry and
Boys routine. The compiler builds each shell class's axis-moment cache from
`axis_polynomial`, the existing Gaussian-moment DAG. Component derivatives use
the same raised/lowered Gaussian identity as the generic generated evaluator.
The auxiliary derivative follows from translation of the two independent
orbital centers. Exponents, normalization and external response weights remain
fixed in this derivative.

The native runtime owns compact angular-class shell lists and AO offsets. It
projects each shell block's public weights through the existing normalized
sparse Cartesian expansions once. One generated subgroup traverses primitive shell
products, reuses geometry/Boys/axis moments across components, and immediately
contracts their derivatives into six independent gradient coordinates. A local
reduction and the existing atom-gradient sink finish the shell block. There is
no nuclear-coordinate derivative tensor or expanded shell-triple task array.

An auxiliary-major response panel may cut through a shell. The host narrows each
class list to intersecting shells; the device treats out-of-panel public weights
as zero. The next panel can revisit the same shell with disjoint weights. There
is no symmetry multiplier. Full shell execution consumes every three-center
weight; `shell-sp` partitions its subset from the generic thread consumer.
Metric derivatives, serial mapping, and the host-weight compatibility adapter
retain the generic generated route. Generic three-center work is excluded only
after the corresponding shell launches have been submitted successfully.

`VIBEQC_DF_SHELL_SCHEDULE` chooses among three compiler-owned variants:

| Selector | Component ownership | Shells per block |
| --- | --- | --- |
| `warp` | 32 lanes per shell | 1 |
| `packed` | 32 lanes per shell | Up to 4, bounded by shared storage |
| `compact` | 4, 8, 16 or 32 lanes according to component count | Up to 128 threads, bounded by shared storage |

Large component blocks cycle across the same lanes. Distinct shell groups use
disjoint shared arrays and subgroup masks, so ragged primitive counts and
clipped panels never require an unrelated group to reach a barrier. The
generator reserves 1 KiB for compiler/runtime shared state and keeps total
storage below 48 KiB, including the FFF weights and moment cache. Resource
qualification compares these bounds with every compiled kernel's actual use.

Metadata allocation grows with shell count and is charged before choosing the
response tile from the remaining budget. Static shared storage is bounded by
the generated angular class. The launcher borrows the response owner's stream
and adds no synchronization or allocation of its own. Selecting the sharded
gradient diagnostic together with shell execution is rejected because that
diagnostic specifically controls the original AO-element sink layout.

Set `VIBEQC_DF_SHELL_COUNTERS=1` and enable the existing component trace to obtain
device execution counters. These diagnostic atomics are disabled for clean
timing:

| Counter | Meaning |
| --- | --- |
| `shell_triples_visited` | Device shell tasks reaching the panel consumer |
| `shell_triples_nonzero` | Tasks with a nonzero projected Cartesian weight |
| `shell_public_weights_nonzero` | Nonzero public weights consumed before projection |
| `shell_primitive_products` | Actually executed primitive geometry/Boys evaluations |
| `shell_cartesian_component_products` | Nonzero Cartesian contributions served by those evaluations |

Panel-boundary revisits count as separate executions. Counts describe the shell
consumer only; the full `three_center_derivative_weights` and
`metric_derivative_weights` trace counters continue to describe the complete
response. Launch counts alone do not establish reuse or endpoint performance.

The weighted-execution selector is recorded in prepared replay metadata, so a
route change cannot silently reuse a result from a different execution policy.
CPU emission tests compare all 64 classes and coincident/asymmetric centers
against Libcint. GPU checks include transitions back to generic execution,
independent complete RHF/UHF Cartesian/spherical forces, existing auxiliary and
metric/subspace response cases, bounded budgets, and memory sanitization.

Two additional controls isolate the measured response bottlenecks:

| Control | Default | Alternative |
| --- | --- | --- |
| `VIBEQC_DF_RESPONSE_ALGEBRA` | `scalar` | `blas`: parallel charge GEMV and density GEMM |
| `VIBEQC_DF_RAW_STAGING` | `pageable` | `pinned-panels`: two bounded host panels |

BLAS execution borrows the existing host-scalar handle and its owning stream.
Transpose flags preserve the original row-major contractions. Terms with
exactly zero exchange coefficients require no exchange products. The spectral
metric reverse map, including retained/discarded subspace motion, is unchanged.

Pinned staging gathers up to 16 auxiliary columns into each of two host panels.
Padded column strides avoid cache-set aliasing during the transpose. Bounded
groups of 128 AO-pair rows reuse source cache lines while writing each output
column contiguously, without allocating another copy buffer. Each
panel's last queued H2D read records an event; the CPU waits for that event before
recycling the panel. The response arena drains before either panel is freed,
including exceptional exits. Actual allocated host bytes are charged to the
host budget independently of the existing device response tile. Source-backed
values keep their device generation route and allocate no host staging panels.
Pinned panels cannot be combined with the original drain/packed upload probes.

Host-gather, event-synchronization and contiguous-copy trace regions distinguish
these operations. The original inclusive upload interval must still not be
interpreted as pure packing or transfer time. The pinned-panel controls do not
reduce the mathematical response work or reconstruct raw values from a
truncated transformed tensor.

Benchmark with `benchmarks.issue308_response_timeline`, retaining exact source
and library identity and the independent reference arrays. Use separate traced
qualification and clean complete-force timings, and compare both 192/384 AO and
batch 1/4. Repeat `--candidate` to sweep named consumers on one fixed post-cold
density. Each candidate primes outside timing, and the runner rejects changed
SCF branches, missing warm starts, numerical parity failures, and missing
selected-consumer counters in traced qualification. Every GPU invocation must
run in a finite Slurm `main` allocation with
`--gres=gpu:5090:1`; preserve the assigned device visibility.
