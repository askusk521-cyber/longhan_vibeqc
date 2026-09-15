# Generated shell-block DF response prototype

`VIBEQC_DF_WEIGHTED_EXECUTION=shell-sp` selects an experimental weighted
three-center derivative consumer for the seven non-SSS s/p shell classes.
The default remains `generic`. This prototype demonstrates shared execution;
it does not resolve the complete 192-to-384-AO force scaling problem in #308.

The scalar and shell consumers use the same generated primitive geometry and
Boys routine. The compiler builds each shell class's axis-moment cache from
`axis_polynomial`, the existing Gaussian-moment DAG. Component derivatives use
the same raised/lowered Gaussian identity as the generic generated evaluator.
The auxiliary derivative follows from translation of the two independent
orbital centers. Exponents, normalization and external response weights remain
fixed in this derivative.

The native runtime owns compact angular-class shell lists and AO offsets. It
projects each shell block's public weights through the existing normalized
sparse Cartesian expansions once. One warp then traverses primitive shell
products, reuses geometry/Boys/axis moments across components, and immediately
contracts their derivatives into six independent gradient coordinates. A local
reduction and the existing atom-gradient sink finish the shell block. There is
no nuclear-coordinate derivative tensor or expanded shell-triple task array.

An auxiliary-major response panel may cut through a shell. The host narrows each
class list to intersecting shells; the device treats out-of-panel public weights
as zero. The next panel can revisit the same shell with disjoint weights. There
is no symmetry multiplier. SSS, classes involving d/f, metric derivatives,
serial mapping, and the host-weight compatibility adapter retain the generic
generated route. The generic thread kernel excludes s/p weights only after the
corresponding shell launches have been submitted successfully.

Metadata allocation grows with shell count and is charged before choosing the
response tile from the remaining budget. Static shared storage is bounded by
the generated angular class. The launcher borrows the response owner's stream
and adds no synchronization or allocation of its own. Selecting the sharded
gradient diagnostic together with the prototype is rejected because that
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
CPU emission tests compare all seven classes and coincident/asymmetric centers
against Libcint. GPU checks include transitions back to generic execution,
independent complete RHF/UHF Cartesian/spherical forces, existing auxiliary and
metric/subspace response cases, bounded budgets, and memory sanitization.

Benchmark with `benchmarks.issue308_response_timeline`, retaining exact source
and library identity and the independent reference arrays. Use separate traced
qualification and clean complete-force timings, and compare both 192/384 AO and
batch 1/4. Every GPU invocation must run in a finite Slurm `main` allocation with
`--gres=gpu:5090:1`; preserve the assigned device visibility.
