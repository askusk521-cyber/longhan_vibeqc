#pragma once

// Included only by the native generated_grid_policy.cu, after cuda_grid.cu.
// Reuse its AO traversal and the exact generated D/C ingredient bilinears.
#include "dft/cuda_xc.hpp"
#include "dft/xc_point.hpp"

namespace vibeqc::dft::cuda_xc_detail {
namespace {
using vibeqc_tensor::blocks;
using vibeqc_tensor::cuda_check;
using vibeqc_tensor::finite;
using vibeqc_tensor::I;

__global__ void validate_density(const double* density, I n, I spins, int* error) {
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < spins * n * n;
       i += I(blockDim.x) * gridDim.x) {
    const I transpose = (i / (n * n) * n + i % n) * n + i / n % n;
    const double a = density[i], b = density[transpose];
    if (!isfinite(a) || fabs(a - b) > 1e-12 + 1e-10 * fmax(fabs(a), fabs(b)))
      atomicCAS(error, 0, 5);
  }
}

// Only D*phi is required for rho and its first derivatives. LDA/GGA do not
// need the three D*grad(phi) panels used by the universal tau implementation.
__global__ void density_product(const double* density, const double* ao, I n, I count, I spins,
                                double* work, int* error) {
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < spins * count * n;
       i += I(blockDim.x) * gridDim.x) {
    const I spin = i / (count * n), point = i / n % count, mu = i % n;
    const double* d = density + spin * n * n;
    double value = 0.0;
    for (I nu = 0; nu < n; ++nu)
      value += (0.5 * d[mu * n + nu] + 0.5 * d[nu * n + mu]) * ao[point * n + nu];
    work[i] = finite(value, error, 1);
  }
}

__global__ void density_features(const double* ao, const double* work, I n, I count, I spins,
                                 I jets, double* features, int* error) {
  const I stride = count * n;
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < spins * count;
       i += I(blockDim.x) * gridDim.x) {
    const I spin = i / count, point = i % count;
    double accum[5]{};
    for (I mu = 0; mu < n; ++mu) {
      const I index = point * n + mu;
      double derivatives[3]{}, panel[4]{work[spin * stride + index], 0.0, 0.0, 0.0};
      if (jets == 4)
        for (unsigned k = 0; k < 3; ++k) derivatives[k] = ao[(k + 1) * stride + index];
      vibeqc_grid_policy::add_features(ao[index], derivatives, panel, accum, jets == 4 ? 3 : 1);
    }
    for (I k = 0; k < jets; ++k)
      features[(spin * jets + k) * count + point] = finite(accum[k], error, 1);
  }
}

__global__ void evaluate_points(const double* features, const double* weights, I count, I spins,
                                I jets, bool pbe, double* coefficients, double* point_totals,
                                int* error) {
  for (I p = I(blockIdx.x) * blockDim.x + threadIdx.x; p < count; p += I(blockDim.x) * gridDim.x) {
    double rho[2]{}, gradient[2][3]{};
    for (I s = 0; s < 2; ++s) {
      const I source = spins == 1 ? 0 : s;
      const double scale = spins == 1 ? 0.5 : 1.0;
      rho[s] = scale * features[source * jets * count + p];
      if (pbe)
        for (I k = 0; k < 3; ++k)
          gradient[s][k] = scale * features[(source * jets + k + 1) * count + p];
    }
    const auto xc = point::evaluate(pbe, rho, gradient);
    if (!xc.valid) atomicCAS(error, 0, 3);
    point_totals[p] = finite(weights[p] * xc.energy, error, 2);
    for (I s = 0; s < 2; ++s)
      point_totals[(s + 1) * count + p] = finite(weights[p] * rho[s], error, 2);
    // These are the current unweighted Cartesian potential coefficients,
    // obtained by differentiation of the same stable energy. This avoids a
    // separate sigma chain rule and its singular 0*infinity in grid tails.
    for (I s = 0; s < spins; ++s) {
      coefficients[s * jets * count + p] =
          spins == 1 ? 0.5 * xc.rho[0] + 0.5 * xc.rho[1] : xc.rho[s];
      if (pbe)
        for (I k = 0; k < 3; ++k)
          coefficients[(s * jets + k + 1) * count + p] =
              spins == 1 ? 0.5 * xc.gradient[0][k] + 0.5 * xc.gradient[1][k] : xc.gradient[s][k];
    }
  }
}

__global__ void assemble_potential(const double* ao, const double* coefficients,
                                   const double* weights, I n, I count, I spins, I jets,
                                   double* potential, int* error) {
  const I stride = count * n;
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < spins * n * n;
       i += I(blockDim.x) * gridDim.x) {
    const I spin = i / (n * n), mu = i / n % n, nu = i % n;
    if (mu > nu) continue;
    double value = 0.0;
    for (I p = 0; p < count; ++p) {
      const double a = ao[p * n + mu], b = ao[p * n + nu];
      double integrand = coefficients[spin * jets * count + p] * a * b;
      if (jets == 4)
        for (I k = 0; k < 3; ++k)
          integrand +=
              coefficients[(spin * jets + k + 1) * count + p] *
              (ao[(k + 1) * stride + p * n + mu] * b + a * ao[(k + 1) * stride + p * n + nu]);
      // One quadrature weight, both differentiated AO legs, no scalar factor 2.
      value += weights[p] * integrand;
    }
    value = finite(potential[i] + value, error, 3);
    potential[i] = value;
    potential[(spin * n + nu) * n + mu] = value;
  }
}

__global__ void accumulate_totals(const double* point_totals, I count, double* totals, int* error) {
  const I channel = threadIdx.x;
  if (channel >= 3) return;
  double value = 0.0;
  for (I p = 0; p < count; ++p) value += point_totals[channel * count + p];
  totals[channel] = finite(totals[channel] + value, error, 3);
}
}  // namespace

void enqueue(const CudaXcLayout& l, cudaStream_t stream, const double* basis, const double* points,
             const double* weights, const double* density, double* ao, double* work,
             double* features, double* coefficients, double* point_totals, double* potential,
             double* totals, int* error) {
  const I matrices = l.spins * l.nao * l.nao;
  cuda_check(cudaMemsetAsync(error, 0, sizeof(int), stream));
  cuda_check(cudaMemsetAsync(totals, 0, 3 * sizeof(double), stream));
  cuda_check(cudaMemsetAsync(potential, 0, matrices * sizeof(double), stream));
  validate_density<<<blocks(matrices, 128), 128, 0, stream>>>(density, l.nao, l.spins, error);
  cuda_check(cudaGetLastError());
  for (std::size_t begin = 0; begin < l.npoint; begin += l.tile_points) {
    const I count = std::min(l.tile_points, l.npoint - begin);
    // The existing through-f AO kernel is compiled earlier in this same TU.
    ao_kernel<<<blocks(l.jets * count * l.nao, 128), 128, 0, stream>>>(
        basis, l.natom, l.nprimitive, l.nao, points + 3 * begin, count, l.jets, ao, error, nullptr);
    cuda_check(cudaGetLastError());
    density_product<<<blocks(l.spins * count * l.nao, 128), 128, 0, stream>>>(
        density, ao, l.nao, count, l.spins, work, error);
    cuda_check(cudaGetLastError());
    density_features<<<blocks(l.spins * count, 128), 128, 0, stream>>>(
        ao, work, l.nao, count, l.spins, l.jets, features, error);
    cuda_check(cudaGetLastError());
    evaluate_points<<<blocks(count, 128), 128, 0, stream>>>(features, weights + begin, count,
                                                            l.spins, l.jets, l.pbe, coefficients,
                                                            point_totals, error);
    cuda_check(cudaGetLastError());
    assemble_potential<<<blocks(matrices, 128), 128, 0, stream>>>(
        ao, coefficients, weights + begin, l.nao, count, l.spins, l.jets, potential, error);
    cuda_check(cudaGetLastError());
    accumulate_totals<<<1, 32, 0, stream>>>(point_totals, count, totals, error);
    cuda_check(cudaGetLastError());
  }
}
}  // namespace vibeqc::dft::cuda_xc_detail
