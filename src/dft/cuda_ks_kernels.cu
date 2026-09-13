#include <algorithm>
#include <cmath>

#include "dft/cuda_ks_kernels.hpp"

namespace vibeqc::dft::cuda_ks_detail {
namespace {
__global__ void fock_kernel(std::size_t matrix, unsigned spins, const double* hcore,
                            const double* coulomb, const double* potential, double* fock) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < spins * matrix;
       i += std::size_t(blockDim.x) * gridDim.x)
    fock[i] = hcore[i % matrix] + coulomb[i % matrix] + potential[i];
}

// One deterministic reduction owner is sufficient for scalar control of the
// baseline. The expensive XC, J and matrix operations remain separate kernels.
__global__ void diagnostic_kernel(std::size_t matrix, unsigned spins, const double* density,
                                  const double* proposal, const double* residual,
                                  const double* hcore, const double* overlap, const double* coulomb,
                                  const double* xc_totals, const int* xc_error, const int* jk_error,
                                  const int* solver_info, Scalars* output) {
  Scalars result{};
  result.failure = (*xc_error != 0 ? 1 : 0) | (*jk_error != 0 ? 2 : 0);
  result.xc = xc_totals[0];
  result.grid_electrons[0] = xc_totals[1];
  result.grid_electrons[1] = xc_totals[2];
  for (unsigned spin = 0; spin < spins; ++spin) {
    const auto offset = spin * matrix;
    double error2 = 0.0, change2 = 0.0, electrons = 0.0;
    if (solver_info[spin] != 0) result.failure |= 4;
    for (std::size_t i = 0; i < matrix; ++i) {
      const double d = density[offset + i];
      const double change = proposal[offset + i] - d;
      result.one_electron += d * hcore[i];
      result.hartree += 0.5 * d * coulomb[i];
      electrons += d * overlap[i];
      error2 += residual[offset + i] * residual[offset + i];
      change2 += change * change;
    }
    if (!isfinite(error2) || !isfinite(change2) || !isfinite(electrons)) result.failure |= 8;
    result.residual = fmax(result.residual, sqrt(error2 / matrix));
    result.density_change = fmax(result.density_change, sqrt(change2 / matrix));
    result.residual_rms += error2 / matrix / spins;
    result.density_rms += change2 / matrix / spins;
    if (spins == 1)
      result.electrons[0] = result.electrons[1] = 0.5 * electrons;
    else
      result.electrons[spin] = electrons;
  }
  result.residual_rms = sqrt(result.residual_rms);
  result.density_rms = sqrt(result.density_rms);
  if (!isfinite(result.one_electron) || !isfinite(result.hartree) || !isfinite(result.xc))
    result.failure |= 8;
  *output = result;
}
}  // namespace

void assemble_fock(cudaStream_t stream, std::size_t n, unsigned spins, const double* hcore,
                   const double* coulomb, const double* potential, double* fock) {
  const auto blocks = std::min<std::size_t>((spins * n * n + 127) / 128, 65535);
  fock_kernel<<<static_cast<unsigned>(blocks), 128, 0, stream>>>(n * n, spins, hcore, coulomb,
                                                                 potential, fock);
}

void diagnostics(cudaStream_t stream, std::size_t n, unsigned spins, const double* density,
                 const double* proposal, const double* residual, const double* hcore,
                 const double* overlap, const double* coulomb, const double* xc_totals,
                 const int* xc_error, const int* jk_error, const int* solver_info,
                 Scalars* output) {
  diagnostic_kernel<<<1, 1, 0, stream>>>(n * n, spins, density, proposal, residual, hcore, overlap,
                                         coulomb, xc_totals, xc_error, jk_error, solver_info,
                                         output);
}
}  // namespace vibeqc::dft::cuda_ks_detail
