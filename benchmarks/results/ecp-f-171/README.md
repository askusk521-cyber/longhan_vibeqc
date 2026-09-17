# Scalar ECP orbital-f qualification

This slice extends the scalar ECP orbital basis through f in Cartesian and
real-spherical layouts. Nonlocal projectors remain s/p/d, local labels through
f, radial powers 0..4 and complete methods direct RHF/UHF. It makes no broad
heavy-element, additional-method or speedup claim. Refs #171; depends on #425.

## Source and environment

Baseline: `d0f6b127a2e9e0049994c5f14e4748441d71fc43` (PR #425). The measured
candidate is this public Git archive plus the 12 exact source/test files in
`source-overlay.tar.gz`, not a claimed clean Git checkout. The retained full
source manifests verify every archive file, and CPU/CUDA candidate manifests
must match. Documentation and ownership records do not alter measured code.
Archive compression can differ across Git versions; per-file source hashes
and the archive's base-commit PAX record bind the actual scientific baseline.

RTX 4090, CUDA 12.9, sm_89, Release, AOT shells disabled, one OpenMP/BLAS thread.
Compute Sanitizer uses CUDA 12.8. `summary.json` records exact hardware,
toolchain, source/library hashes, numerical errors, all timing samples, work
counts and planned/observed resources. `raw-evidence.zip` and its manifest retain
build/test logs, generated headers, kernel resource usage, source identities and
reproduction scripts. The initial failed native test build is retained; it
used a nonexistent CPU enum, subsequently corrected to `CPU_REFERENCE`.

## Gates and scope

Independent host tests cover every Cartesian component through f, analytic
Gaussian derivatives, gamma-function normalization and invalid dispatch aliases.
C API tests cover f acceptance/g rejection on both ECP and all-electron atoms,
Cartesian/spherical layouts, rejection of f projectors and the grid wrapper's
shape/weight/append contract. Existing s/p/d numerical and failure tests remain.

Contracted f shells on Na and H exercise mixed centers and signed coefficients.
Libcint matrices are normalized with the overlap diagonal for Cartesian f.
All-center finite differences use steps 2e-4 and 7e-5 bohr, with arbitrary
nonsymmetric AO weights. A separate ECP center without an orbital basis tests
f orbitals against d projectors. Complete RHF/UHF energies and forces use PySCF
2.14.0 for both representations and backends; CUDA adds complete energy finite
differences. Planned-budget execution checks allocation bounds and prepared
geometry replay. Sanitizer checks both native error recovery and actual f raw
matrices/derivatives in both representations.

Absolute gates: raw matrices 2e-9 Eh, grid-refinement derivatives 2e-8, raw
finite-difference derivatives 3e-7 (relative 3e-6), complete energies 2e-8 Eh,
complete forces 2e-6 Eh/bohr. The 160/32/64 and 224/44/88 quadratures are
empirical convergence checks, not a universal error estimate.

Four matched f endpoint cases per backend cover Cartesian/spherical RHF/UHF
with f on Na; CUDA also measures a larger Cartesian RHF case with f on both
atoms. Each has a warmup, then one CPU or three CUDA complete synchronous
`singlepoint` timing samples, followed by an explicitly budgeted prepared
calculation. Larger CPU two-f-center behavior is covered by the spherical
prepared-replay regression rather than repeated Cartesian reference timings.
These small samples are diagnostics, not a CPU/GPU speed comparison. Records include actual shells,
primitives, AO/pair counts, iterations, energies, forces and resource diagnostics.
The former baseline cannot execute f, so it supplies only an existing-domain
regression comparison; its timings must not be reported as f speedups.

## Reproduction

1. Obtain the exact public baseline with `git archive`, verify the retained
   archive manifest, and extract `source-overlay.tar.gz` over a separate source
   tree. The overlay can be used as `f-changes.tar.gz` for preparation.
2. Adapt only the workspace prefix in `f-prepare.sh`, `f-run.sh`,
   `f-provenance.py` and `f-collect.py`. Keep the flags, grids and test commands.
3. Run preparation, then `f-run-v2.sh cpu` and `f-run-v2.sh baseline`. After baseline
   succeeds, run `f-run-v2.sh cuda`; GPU timing must remain sequential. The
   original script and its initial orchestration failures are retained too;
   never edit a script while a running shell may still read it.
4. Candidate CUDA execution explicitly regenerates the header, touches native
   inputs and reconfigures CMake so archive mtimes cannot hide changed code or
   newly registered tests. The endpoint script takes its source and exact
   `VIBEQC_LIBRARY` explicitly.
5. `f-collect.py` requires successful exit markers, matching complete candidate
   source manifests, all four CPU/five CUDA endpoint cases and exact library
   identities before archiving. Retain failures and diagnostic follow-ups.

## Ownership

This extension reuses the common scalar DAG and existing molecular spherical
expansion. Maintained scientific CUDA and runtime LOC each change by +0/-0;
no native production path is removed. The independent CPU oracle/fallback,
including its own quadrature and normalization, remains separate from generated
production arithmetic. Generated header growth is recorded independently and
does not offset handwritten code. Other generated-family measurements remain
preserved in the current ownership report.

See the [current contract](../../../docs/ecp.md) and
[decision](../../../.agents/notes/implemented/numerics/2026-09-17-ecp-orbital-f.md).
