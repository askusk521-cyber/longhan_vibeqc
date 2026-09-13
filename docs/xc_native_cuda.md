# Native semilocal XC device boundary

`dft/cuda_xc.hpp` supplies the ordinary-stream fixed-density LDA/PBE RKS/UKS
component for #162. It consumes current device density matrices and returns
device XC energy/electron totals, potential matrices and a numerical-error
flag. `dft/cuda_ks.hpp` composes it into native ordinary-stream SCF; the
registered single-system method adapter exposes CPU/CUDA energy execution.
Prepared ragged batching and full public resource planning remain separate
work.

## Ownership and data movement

`cuda_xc_layout` returns an exact explicit device-memory request. The method
ResourcePlan supplies that arena; XC performs no device allocation and accepts
no separate user memory budget. `CudaXcPlan` borrows the arena and stream, which
must outlive it. Its destructor drains the stream before the caller releases
either resource. Concurrent calculations use independent plans and arenas.

Host GridSpec-v1 quadrature remains explicit. Setup validates the complete
normalized AO packing against the grid's current system, uploads packed basis,
points and complete weights once, and synchronizes before releasing borrowed
host inputs. It retains the grid on device and uses bounded AO/intermediate
tiles. Host grid generation and setup are part of complete calculation cost;
this is not a claim that quadrature preparation runs on GPU.

Every `enqueue` consumes a strictly newer density generation, invalidates the
old result view, clears its outputs/error and rebuilds the full XC contribution.
It does not allocate, transfer host data or synchronize. Device density producers
must enqueue on the same stream, or establish an explicit event dependency.
`read_scalars` downloads only energy, both spin populations and the numerical
status. `download_potential` is an explicit user/reference output operation.
Transfer counters distinguish setup H2D from these requested D2H operations
and count their synchronization boundaries. The surrounding method must also
account for its density input, one-electron setup, J provider and final outputs.

The geometry, basis, grid and functional are immutable within a plan. Changes
require a new plan. The currently supported compositions are exactly
LDA_X+LDA_C_PW and PBE_X+PBE_C, with the
[versioned SCF point-domain policy](xc_scf_domain.md). No meta-GGA or force
capability is implied by these interfaces.

## Numerical path

The native generated grid translation unit reuses the existing through-f AO
kernel and generated D/C ingredient bilinears. LDA retains one AO/value feature
per spin; PBE retains value and three ordinary spatial derivatives. Neither
requests tau, D times derivative-AO panels, or higher AO jets.

The sequence is AO -> D times AO -> density/gradient -> shared point energy
and Cartesian potential coefficients -> weighted E/V reductions. The point
evaluator differentiates the same stable energy used by the CPU consumer; no
separate singular sigma chain rule or CPU XC call is inserted. Matrix assembly
applies weights once, retains both differentiated AO legs, and does not double
the scalar term. All symmetric matrix cross terms are retained.

RKS input is the total density. The point layer receives half in each spin and
the returned single potential uses the corresponding total-density chain rule.
UKS input is `[alpha,beta,AO,AO]` with independent spin densities and potentials.
E/V outputs remain in public normalized Cartesian or real spherical AO order.

The current dense reduction is a correctness baseline. It has no point-by-AO-
by-AO tensor, but it does not yet use the promoted local-dense or GEMM potential
contractions. The ownership ledger counts this numerical glue conservatively;
neither a scientific-code retirement nor a performance advantage is claimed.

## Direct J connection

`scf/cuda_direct_jk_device.hpp` exposes the #202 provider's ordinary stream and
an enqueue operation over caller-owned device density/J/K arrays. It reuses
the existing operator validation and contracted-ERI kernel, with device finite
checks and no success-path staging or synchronization. This lets a method use
one stream for matrix production, J and XC without inheriting the HF graph
layout. `CudaKsPlan` uses this seam directly, with an exact state arena charged
through the #203 device ledger. It reuses the existing matrix, DIIS, eigensolver
and density kernels, keeps current physical Fock and proposal state separate,
and reads one scalar diagnostic record per iteration. Convergence returns the
current evaluated density; only converged states replace the resident warm
cache. An energy-only public call omits final matrix export. Full #203 public
planning remains separate integration work.

## Validation

`vibeqc_dft_cuda_tests` checks the actual device-buffer pipeline against CPU
full-matrix integration, the independent #214 H2 fixture, spin-resolved finite
differences, empty spin/vacuum tails, partial tiles and Cartesian/spherical f
shells. It checks a density changed by a device kernel without re-upload,
generation rejection, same-shape stale grid rejection, independent-stream
failure isolation/recovery and exact arena bounds.

All real-GPU invocations use finite Slurm allocations. Fixed-density results,
point-domain checks, full SCF, replay, changed geometry and batching must remain
distinct evidence until the corresponding native consumers are verified.

`vibeqc_ks_cuda_tests` independently rebuilds returned SCF densities with the
CPU providers, exercises resident replay and changed-geometry normalization,
and rejects stale grids and failed warm-state replacement. The registered
C API tests cover both spins/functionals on the actual CUDA backend and reject
forces. `tests/python/test_dft_scf.py` compares CPU/CUDA stable small endpoints
against two independently converged PySCF guesses on the identical grid.
These verified single-system paths do not establish native prepared batching.
