# Direct-native extraction validation

`validation.json` records complete Release build samples and matched prepared
endpoints for the exact parent and extracted source identities. It retains each
numerical comparison, geometry, timing repeat, iteration count and library hash.
The original endpoint worker is `tools/validate_weighted_eri_endpoints.py`.

Reproduce a fixed RHF endpoint for each library with:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VIBEQC_LIBRARY=/absolute/path/to/libvibeqc.so VIBEQC_PROFILE=off \
  VIBEQC_BOUNDED_DIRECT_STREAMING=none VIBEQC_PSSS_RESIDENT_BRA=0 \
  VIBEQC_DIRECT_TILE_VALIDATION=validate \
  python tools/validate_weighted_eri_endpoints.py \
  --worker '{"method":"rhf","basis":"sto-3g","batch":3,"repeats":5}'
```

Repeat with `method=uhf`. For resident traversal set
`VIBEQC_PSSS_RESIDENT_BRA=1`; for paged traversal set
`VIBEQC_BOUNDED_DIRECT_STREAMING=force` and `VIBEQC_PSSS_RESIDENT_BRA=0`.
The worker checks every warm result against its corresponding cold geometry.
Cross-library comparisons use that module's `require_equal`, with energy
`atol=2e-8, rtol=2e-10` and per-molecule force `atol=2e-7, rtol=2e-8`.
The recorded maximum difference is much smaller than these established gates.

The build records retain full CMake flags: fresh Release build trees, NVCC
12.9.1, architecture `120` (real plus virtual images), fast-compile mode off,
compiler cache off, two CUDA jobs and four total Ninja jobs. The builds ran
concurrently on a shared host with a warm filesystem cache. The extraction was
committed during its build without changing the measured source; configure-time
HEAD and source commit are recorded separately. Runtime uses the additional
C++ `<cstdlib>` portability fix. Binary size and timing claims apply only to
these measured configurations.

The response suite requires `VIBEQC_DF_DERIVATIVE_CUDA_TEST=1`. The corrected
Slurm run and the preceding skipped attempt are distinguished in the record.
The virtual-image probe changed only the direct subsystem's images; it is not
a complete PTX-only whole-library benchmark.
