#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace vibeqc::dft::cuda_ks_detail {
/** Small per-iteration host-control record. Density and Fock stay on device. */
struct Scalars {
  double one_electron{}, hartree{}, xc{}, residual{}, density_change{};
  /** Joined-spin public RMS values, distinct from the per-spin maximum gates. */
  double residual_rms{}, density_rms{};
  double electrons[2]{}, grid_electrons[2]{};
  int failure{};
};

void assemble_fock(cudaStream_t stream, std::size_t n, unsigned spins, const double* hcore,
                   const double* coulomb, const double* potential, double* fock);
/** Evaluate the CURRENT E/D/F generation and the proposed density change.
 * RMS is gated per spin; an empty spin cannot dilute an unconverged one. */
void diagnostics(cudaStream_t stream, std::size_t n, unsigned spins, const double* density,
                 const double* proposal, const double* residual, const double* hcore,
                 const double* overlap, const double* coulomb, const double* xc_totals,
                 const int* xc_error, const int* jk_error, const int* solver_info, Scalars* output);
}  // namespace vibeqc::dft::cuda_ks_detail
