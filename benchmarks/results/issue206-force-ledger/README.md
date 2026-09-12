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

The version-2 runner requires both solves to converge with identical iteration
counts and energies agreeing within `1e-10` Hartree. It checks the requested
force outputs and finite energies, forces and timings before publishing an
increment. Invalid pairs fail without writing a new ledger.

Select the exact native binary explicitly. Each new ledger records its resolved
path and SHA-256 alongside checkout revision, dirty state and patch hash, and
rejects source or binary changes during timing:

```sh
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH=python:. python benchmarks/issue206_df_force_probe.py \
  --library /absolute/path/to/libvibeqc.so --output .artifacts/force-ledger.json
```
