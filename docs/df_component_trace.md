# CUDA DF component evidence

`VIBEQC_DF_TRACE=/absolute/path.jsonl` enables diagnostic CUDA event intervals,
NVTX ranges (when toolkit NVTX headers are available), transfer/work counters,
and a logical three-center tile ledger. It covers RI-J, RI-K, generated raw and
transformed tiles, DF-HF response weights, exchange response matrix products,
Coulomb response, spectral metric response, weighted generated derivatives,
and the one-electron/overlap-Pulay response.

The disabled route performs no extra CUDA calls, allocations, synchronizations,
NVTX calls, or file writes. An enabled ordinary operation records events on its
existing stream and waits on the final event before writing JSONL. This changes
submission cost and overlap: use separate **unprofiled** endpoint runs for any
performance claim.

## Reproducible force probe

Build a Release library with production compiler settings, then run both probes
through a finite Slurm allocation, preserving the assigned device visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH="$PWD/python" /path/to/python benchmarks/issue206_df_force_probe.py \
  --library "$PWD/build/cuda/libvibeqc.so" --repeats 3 \
  --output /path/to/evidence/unprofiled.json

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH="$PWD/python" /path/to/python benchmarks/issue206_df_force_probe.py \
  --library "$PWD/build/cuda/libvibeqc.so" --repeats 1 \
  --component-trace-dir /path/to/evidence/new-traces \
  --output /path/to/evidence/components.json
```

The default cases are the #206 96- and 192-AO systems. The probe retains energy
parity and iteration-count gates, stores exact library/patch/untracked-file
hashes, preserves each solve's raw JSONL with a hash, and rejects missing or
invalid instrumentation. Output paths should be outside the source tree (or
ignored by Git) so writing evidence does not change the identified checkout.
Trace files must be fresh; the probe never overwrites earlier captures.

## Host eigensolve ledger for #308

`VIBEQC_DF_HOST_TRACE=/absolute/fresh.jsonl` records host scopes and actual
CPU-reference eigensolve invocations. The existing force probe's
`--component-trace-dir` collects this companion file together with the CUDA
ledger. The two clocks are reported separately. Setting either trace variable
implicitly on a clean force-probe invocation is rejected.

Each `reference_eigensolve` leaf records its dimension, reason (`overlap`,
`core_guess`, `final_fock`, `reference_export`, `fallback`, or `unspecified`),
item, host wall milliseconds, thread CPU milliseconds and exceptional exit.
Only the actual oracle entry emits this leaf; an intended solve, cache flag or
graph declaration cannot count as executed work. The Jacobi arithmetic and
overlap singularity threshold are unchanged. A small observer interface keeps
the independent reference and initial-guess modules free of runtime/CUDA
dependencies. The runtime owns timers, bounded records and file output.

`benchmarks/df_component_ledger.py` validates the host JSONL and subtracts
immediate children within each root to produce exclusive host phases. Root
and ancestor IDs disambiguate source indices local to different CUDA buckets.
Thread CPU time is `null` when the platform cannot supply it; process CPU time
is never substituted. Host wall time includes waiting and descheduling, so
wall minus CPU time is not automatically all GPU waiting. Concurrent CPU
worker roots overlap: their wall times must not be summed into endpoint time.
Host and CUDA-event times must not be added together either.

The existing #206 matrix entry point also supports native protocol controls:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:20:00 \
  env PYTHONPATH="$PWD/python" /path/to/python benchmarks/issue206_df_matrix.py \
  --run --host-workloads --case water-tetramer-def2-svp-spherical --batch 1 \
  --library "$PWD/build/cuda/libvibeqc.so" --memory-budget-bytes 1073741824 \
  --energy-only --repeats 5 --host-trace-dir /path/to/fresh-host-traces \
  --output-dir /path/to/host-components
```

It retains cold creation/solve/destruction, fixed-seed replay, energy-only or
energy-plus-force, and a changed-geometry item. Every changed sample restores
the original geometry before starting its timer. Omit `--host-trace-dir` for
separate clean timing; choose batch 4 or the existing 192/384-AO inputs to
extend the domain. Identical A/B configurations are an ABBA protocol control,
not a speedup or an external-engine parity result. Complete traffic/device
work still comes from the separate CUDA/Nsight ledger; the original matched
DF-versus-DF matrix remains the owner of parity acceptance.

The default `--memory-budget-bytes 0` preserves the original probe's host
resident compatibility route. A positive value, for example `268435456`,
selects the existing bounded generated-source execution. Record and compare
both routes explicitly: the original #206 probe does not exercise source-backed
tile regeneration. Compatibility one-electron/nuclear derivative exports have
their own roots, and host response weights remain labeled as host work.

For executed graph-node attribution, wrap a separate probe with Nsight Systems:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH="$PWD/python" /group/software/cuda-12.9.1/bin/nsys profile \
  --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --output=/path/to/evidence/df-nsys \
  /path/to/python benchmarks/issue206_df_force_probe.py \
  --library "$PWD/build/cuda/libvibeqc.so" --repeats 1 \
  --component-trace-dir /path/to/evidence/new-nsys-traces \
  --output /path/to/evidence/nsys-components.json
```

Keep the `.nsys-rep`, exported tables and original JSON alongside source and
binary identities. Nsight timing also has instrumentation overhead.

## Interpretation and limitations

- `execution=graph_capture` contains construction counts and host time only.
  It creates no CUDA events and adds no synchronization during capture. Those
  counters **do not count graph replay**, including device tail launches.
- Regions record inclusive host and GPU milliseconds with parent indices.
  `benchmarks/df_component_ledger.py` subtracts only immediate children to
  produce exclusive components. Host and GPU views overlap and must not be
  added together. CUDA events include stream idle gaps between submissions.
- Root exclusive time remains unclassified runtime work. The force attribution
  compares named host intervals and measured synchronization to the same
  profiled pair's force increment. Host nuclear assembly, packing before the
  operation, and work outside the roots remain explicit residuals. A ratio
  over one may reflect variation between the two SCF solves.
- A tile key includes absolute source system, AO-pair and auxiliary ranges,
  derivative coordinate (`-1` for values), and raw/transformed representation.
  `system_offset` locates a single force call within a batched source.
  Production multiplicity exposes repeated generation within each call.
- Generated value byte counts measure logical output work, not device traffic.
  Weighted derivative bytes measure consumed weights; the fused derivative
  kernel does not allocate or write a full nuclear derivative tensor. They
  must not be interpreted as materialized derivative bytes.
- Scratch counters report per-operation allocation sizes. Their sums are not
  live-memory peaks; the #203 resource ledger remains the complete memory gate.
- Each operation is limited to 65,536 regions and logical tile keys. Dropped
  entries, CUDA timing errors, malformed hierarchy, missing timings, truncated
  JSONL and inconsistent logical work totals invalidate evidence. Missing sink
  output is rejected by the probe rather than changing the scientific status.

This instrumentation alone does not complete #282, #283 or #284. Resident and
streamed reuse, force optimizations, occupied-factor exchange, numerical gates,
and the full batch-1/batch-4 #206 matrix require separate implementation and
evidence.
