# Generated s/p/d/f response scheduling (#308)

This draft retains the first generalized-shell comparison. All new execution
controls remain opt-in. Clean 192/384-AO batch-one/four measurements and final
promotion are pending; these intrusive event traces do not establish clean
scaling acceptance or close #308.

The compiler now supplies all 64 s/p/d/f shell classes and three bounded subgroup
schedules. The same Gaussian geometry, Boys values, moment polynomials and
raised/lowered derivatives feed immediate weighted contraction. Metric,
auxiliary and retained/discarded subspace response remain complete.

| v3 candidate | Complete traced force | Derivative event interval |
| --- | ---: | ---: |
| Generic / scalar / pageable | 12.146 s | 3.141 s |
| Shell warp / scalar / pageable | 10.207 s | 1.411 s |
| Shell packed / scalar / pageable | 10.529 s | 1.448 s |
| Shell compact / scalar / pageable | 9.886 s | 0.990 s |
| Generic / BLAS / pageable | 10.598 s | 3.134 s |
| Generic / BLAS / pinned | 6.264 s | 3.142 s |
| Shell warp / BLAS / pinned | 4.533 s | 1.409 s |
| Shell packed / BLAS / pinned | 4.568 s | 1.445 s |
| Shell compact / BLAS / pinned | 4.112 s | 0.993 s |

Each row is one complete 384-AO batch-one warm force with the same fixed
post-cold density, untimed candidate priming and iteration branch `[3]`. All
nine independent complete-force comparisons pass the unchanged 1e-9 Ha /
1e-8 Ha/Bohr gates; maximum force error is below 1.064e-10 Ha/Bohr. Every traced
warm host ledger records zero CPU-reference eigensolves. The measurement JSON
retains full force/energy arrays, branches, controls, resource/work counts,
reference hashes and source/library/runner identities.

Event intervals include stream gaps and instrumentation effects. They are not
CUPTI kernel-only measurements, and host/device intervals overlap. In pinned
execution, explicit host wall/CPU regions measure about 2.36 s of gather work;
that is distinct from panel-reuse waits and H2D execution. All rows still upload
4,015,521,792 raw-response bytes over eight weight panels.

Full shell execution actually visits 7,225,344 nonzero shell triples, consumes
56,623,104 nonzero public weights, evaluates 44,605,440 primitive/Boys products
and serves 243,749,376 Cartesian component contributions. Pinned staging owns
two bounded host panels, 16 columns each, totaling 37,750,784 bytes. Last-read
events protect their reuse and the arena drains before freeing them.

The v3 memcheck run passes 28 transition/property-budget cases with zero errors.
The current integrated candidate, v4, also changes the host gather to groups of
128 AO-pair rows with contiguous column stores. A host-only layout experiment
roughly doubles gather throughput without adding a buffer; v3 timings do not
measure this change. V4 passes 128 independent generated-arithmetic checks and
10 ownership checks. All 192 compiled kernel variants fit their generated
shared-memory bounds: maximum 48,624 bytes; register counts 94–255. Its GPU,
memcheck/racecheck, Nsight and clean endpoint runs are queued.

`summary.json` pins the retained files and qualifying logs. The v3 source patch
applies to the recorded v3 Git head; its exact runner is retained separately.
Detailed CUDA event traces and full sanitizer logs remain under the ignored
`.artifacts/issue308-shell-block/v3/` directory. The reproduction script records
the commands and historical local paths; use fresh output paths and a finite
Slurm `main` / `gpu:5090:1` allocation when reproducing.

## CUDA ownership disclosure

```text
generated capability: shared s/p/d/f shell derivatives with three bounded schedules
handwritten scientific CUDA LOC: +136 / -17
runtime CUDA LOC: +59 / -33
legacy production path removed: no
retained duplicate reason: oracle
```

`ownership-delta.json` compares this slice with PR B at `672ec47` using the
ownership reporter's physical nonblank/noncomment line accounting. The +119 net
scientific lines are conservative method/provider/layout and bounded host
staging glue. Gaussian derivative mathematics remains generated. No source
lines are reclassified to hide that growth. Generic execution remains the
default and comparison route pending clean promotion; the existing serial
metric-dot oracle retains its #206 numerical/resource/endpoint retirement gate.
