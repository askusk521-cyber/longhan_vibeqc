# #149 B Resident-vs-Ordinary GPU Parity Evidence

Device: NVIDIA H200 (141GB), sm_90, CUDA 12.8, Driver 570.124.06
NVCC: Cuda compilation tools, release 12.8, V12.8.61
Python: 3.11.16, NumPy 2.2.6
Commit: f2134c7 (claude/issue-0149-b)

## Results

4 molecules × 2 budgets (normal 256MiB / minimum) = 8 runtime cases.
Every case compares the ordinary host-staged path against the resident
inlined-kernel path on identical feeds.

| Case | Budget | Parity (3 runs) | Stale Lease | Nonfinite | Missing Input | Repeat |
|------|--------|-----------------|-------------|-----------|---------------|--------|
| random (2,2) | normal | ✓ | ✓ | ✓ | ✓ | ✓ |
| random (2,2) | minimum | ✓ | ✓ | ✓ | ✓ | ✓ |
| h2 (1,1) | normal | ✓ | ✓ | ✓ | ✓ | ✓ |
| h2 (1,1) | minimum | ✓ | ✓ | ✓ | ✓ | ✓ |
| water (5,2) | normal | ✓ | ✓ | ✓ | ✓ | ✓ |
| water (5,2) | minimum | ✓ | ✓ | ✓ | ✓ | ✓ |
| lih (2,4) | normal | ✓ | ✓ | ✓ | ✓ | ✓ |
| lih (2,4) | minimum | ✓ | ✓ | ✓ | ✓ | ✓ |

All 8 cases passed (32/32 parity runs, 32/32 error/repeat tests).

## Full evidence

The per-case JSON records (including per-run metrics, artifact keys and
binary hashes) are retained on the qz shared filesystem at:
  /inspire/qb-ilm/project/chemicalreaction/czxs25220150/scratch/vibeqc/issue-0149-b/results/gpu/

Manifest SHA256: eae15410ef551a27bbab9dfde5b11d44b8baa3c826d20511b9102e8b2c1c02bc

## Not-run

- RTX 5090 (not available on this platform)
- GPU-job / multi-GPU batching
- Non-RCCSD programs
