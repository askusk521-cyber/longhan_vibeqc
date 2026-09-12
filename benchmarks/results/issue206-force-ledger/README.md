# Issue 206 DF force-response ledger

This diagnostic complements the matched DF matrix in
`benchmarks/results/issue206-df-a/`. It runs fresh single-system VibeQC CUDA
DF calculations with `properties=("energy",)` and
`properties=("energy", "forces")` and records the wall-time increment. The two
runs are deliberately not a warm-solve comparison; the ledger isolates the
cost that must be optimized in the energy-plus-force endpoint.

The recorded run used Slurm job `1103` on node5 with one RTX 5090, CUDA 12.9,
and source commit `b69ec53b3f1a83fcf886efd52195991111214698`.

| workload | energy-only | energy + force | force increment |
| --- | ---: | ---: | ---: |
| water tetramer, 96 AO | 0.632 s | 5.278 s | 4.646 s |
| water octamer, 192 AO | 5.346 s | 25.807 s | 20.461 s |

Both paired runs converged with the same iteration count per workload (34 and
39). The ledger is diagnostic evidence, not a performance gate or a claim of
warm endpoint parity. Raw values and provenance are retained in `ledger.json`.
