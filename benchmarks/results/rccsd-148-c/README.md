# RCCSD C: CPU solver and complete small-system endpoints

This archive completes the internal CPU scope of issue #148, following A at
`5059736e3644f273cdd2347803f4a9adccaed7de` and B at
`3f4c6c4942efc0e6712e1b36b0c8938fcaae86e7`. It does not implement the public
GPU method/API, (T), Lambda or gradients; those have separate issues.

## Fixed implementation and environment

The 19 files in `source-snapshot.json` were validated on qz and independently
reviewed as Git tree `55762b33a028dc3dfba2e6d2aaa9d918d0709fd4` over B. The
coordinator checked every raw file hash before staging; both reviewers checked
the tree and hashes before/after review. Evidence files were added afterward,
without changing the reviewed scientific implementation, tests or references.
The records correctly retain the B revision and `dirty=true` for the tested
candidate; exact source hashes identify the candidate rather than that revision
alone. Old Windows checkout CRLF files are compared through Git blobs where
needed to match the Linux source bytes.

Remote worktree:
`/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-148-a`.
The existing general Notebook in CPU资源空间 / 原子级化学反应基座模型2.0 was
used with its own `.venv`, `build/cpu` and one BLAS/OpenMP thread. No GPU was
allocated or tested for #148. `verification.sh` is the executed final script;
all platform calls used `inspire --no-env-file --account qz`.

`versions.log` independently records the actual library SHA-256 and Python
3.11.16, NumPy 2.2.6, PySCF 2.14.0 and CMake 3.31.10. Native library hash:
`d576f7b604ea422dcbe1f5e9aaae6bd98f7280e1adaf983392f8c3c783cdff56`.
The existing native implementation is unchanged from A/B; this slice adds the
CPU equation/solver consumer and uses the validated <=12 AO exporter for the
selected small systems. It does not lift or conceal that exporter limit.

## Actual verification

- `pytest.log`: 231 related CC, TensorIR, post-HF and validation tests passed.
- `source-check.log`: every frozen candidate hash matched on the remote host.
- `ruff.log`: check and format passed for 17 Python files.
- `endpoints.log` and `endpoints/`: ten converged executions, one fixed-orbital
  and one newly converged native VibeQC HF path for each of H2, He with an
  explicit virtual s function, H2O, NH3 and CH4.
- All seven MO integral blocks are compared with independently saved AO
  integrals transformed using the actual reference orbitals. Final amplitudes
  are passed to the pinned independent PySCF equations to recompute the
  physical R1/R2, not merely a solver update norm.
- Total energies, aligned amplitudes, independent physical residuals and the
  two-electron FCI checks pass unchanged gates. The maximum external physical
  residual is below `1.12e-11`; maximum total-energy error is `7.03e-13 Eh`.
- `reference-stability.json`: the coordinator loaded both independently
  generated NPZ sets still present on qz, compared every actual array and
  the input/generator identities for all five cases; all arrays were identical
  and all total-energy differences were zero. `check-generations.py` reproduces
  this read-only comparison. It did not rerun quantum-chemistry generation.

The related suite includes shifted denominators, damping, disabled DIIS,
preflight rejection, maximum iterations, nonfinite failure, failure replay and
a false shared-residual negative test. A small update or stable energy cannot
bypass the final freshly executed expanded physical-residual checks.

This archive is not a claim that the complete Python suite or GPU tests were
run locally for C. Earlier A/B native tests remain applicable to the unchanged
native binary; current PR CI separately checks full integration.

## Reproduction

The same isolated source/environment and native library are required. Run
`verification.sh` after supplying that checkout's frozen snapshot at its
documented build path. The main numerical entry points from the repo root are:

```bash
export PYTHONPATH=.:python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VIBEQC_LIBRARY="$PWD/build/cpu/libvibeqc.so"
.venv/bin/python -m tools.validate_cc_solver --output build/cc-c-reproduction
.venv/bin/python -m tools.replay_ccsd --help
```

Each `*-state.json` contains reference/integral/equation identities, solver
options, replay inputs, complete iteration history and final amplitudes, with
record/input hashes. Convergence is distinct from preflight errors, numerical
failure and exhausted iterations. Independent convergence is demonstrated
for the stated cases, not guaranteed for arbitrary initial guesses or systems.

## Independent review

See `review.md`. Both new read-only reviewers found no P0/P1/P2 issues. The
coordinator checked their conclusions against the frozen tree and original
logs, verified the live native binary identity and both reference generations,
and found no required implementation changes. No review finding was hidden by
relaxing tolerances or dropping tests.
