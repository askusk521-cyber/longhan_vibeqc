#include "scf/gradient/hf_gradient.hpp"

namespace vibeqc::scf::gradient {
std::vector<double> analytic_forces(const integrals::IntegralData& ints, const Matrix& density,
                                    const Matrix& weighted_density,
                                    std::span<const double> two_electron) {
  const std::size_t n = ints.nbf;
  std::vector<double> forces(ints.ncoord, 0.0);
  for (std::size_t coordinate = 0; coordinate < ints.ncoord; ++coordinate) {
    const double* ds = ints.overlap_derivative.data() + coordinate * n * n;
    const double* dh = ints.hcore_derivative.data() + coordinate * n * n;
    double derivative = ints.nuclear_repulsion_derivative[coordinate];
    for (std::size_t element = 0; element < n * n; ++element) {
      derivative += density[element] * dh[element];
      derivative -= weighted_density[element] * ds[element];
    }
    derivative += two_electron[coordinate];
    forces[coordinate] = -derivative;
  }
  return forces;
}

std::vector<double> analytic_uhf_forces(const integrals::IntegralData& ints,
                                        const Matrix& alpha_density, const Matrix& beta_density,
                                        const Matrix& alpha_weighted_density,
                                        const Matrix& beta_weighted_density,
                                        std::span<const double> two_electron) {
  const std::size_t n = ints.nbf;
  std::vector<double> forces(ints.ncoord, 0.0);
  for (std::size_t coordinate = 0; coordinate < ints.ncoord; ++coordinate) {
    const double* ds = ints.overlap_derivative.data() + coordinate * n * n;
    const double* dh = ints.hcore_derivative.data() + coordinate * n * n;
    double derivative = ints.nuclear_repulsion_derivative[coordinate];
    for (std::size_t element = 0; element < n * n; ++element) {
      derivative += (alpha_density[element] + beta_density[element]) * dh[element];
      derivative -=
          (alpha_weighted_density[element] + beta_weighted_density[element]) * ds[element];
    }
    derivative += two_electron[coordinate];
    forces[coordinate] = -derivative;
  }
  return forces;
}

}  // namespace vibeqc::scf::gradient
