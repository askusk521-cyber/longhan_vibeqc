#include "scf/solver/mean_field_driver.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "runtime/resource_usage.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/gradient/hf_gradient.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/diis.hpp"
#include "scf/solver/proposal_control.hpp"

namespace vibeqc::scf::solver {
namespace {
using initial_guess::prepare_initial_density;
using initial_guess::prepare_initial_uhf_density;
using initial_guess::spin_occupations;
using reference::commutator_residual;
using reference::concatenate;
using reference::density_from_orbitals;
using reference::density_rms;
using reference::EigenResult;
using reference::electronic_energy;
using reference::energy_weighted_density;
using reference::generalized_eigen;
using reference::Matrix;
using reference::residual_rms;
using reference::split_spin_matrices;
using reference::symmetric_orthogonalizer;
using reference::uhf_electronic_energy;

template <class... Vectors>
void sample_scf_buffers(const PreparedFockPlan& plan, const Diis& diis,
                        const Vectors&... vectors) noexcept {
  if (!runtime::cpu_resource_observation.active) return;
  runtime::sample_cpu_capacity(runtime::add_capacity(
      plan.cpu_observation_capacity(),
      runtime::add_capacity(diis.numeric_capacity(), runtime::vector_capacities(vectors...))));
}

std::pair<Matrix, Matrix> build_uhf_focks(const PreparedFockPlan& plan, const Matrix& hcore,
                                          const Matrix& alpha_density, const Matrix& beta_density) {
  auto fock = assemble_fock(plan.strategy(), hcore, plan.build(alpha_density, beta_density));
  return {std::move(fock.alpha), std::move(fock.beta)};
}

Matrix build_fock(const PreparedFockPlan& plan, const Matrix& hcore, const Matrix& density) {
  return assemble_fock(plan.strategy(), hcore, plan.build(density)).alpha;
}

void finalize_scf(const PreparedFockPlan& plan, const integrals::IntegralData& ints,
                  const Matrix& orthogonalizer, std::size_t occupied, Matrix& density,
                  bool compute_forces, ScfResult& result) {
  const std::size_t n = ints.nbf;
  Matrix final_fock = build_fock(plan, ints.hcore, density);
  EigenResult orbitals = generalized_eigen(final_fock, orthogonalizer, n);
  density = density_from_orbitals(orbitals.vectors, n, occupied);
  final_fock = build_fock(plan, ints.hcore, density);
  result.energy = electronic_energy(density, ints.hcore, final_fock) + ints.nuclear_repulsion;
  if (compute_forces) {
    const Matrix weighted = energy_weighted_density(orbitals.vectors, orbitals.values, n, occupied);
    result.forces =
        gradient::analytic_forces(ints, density, weighted, plan.energy_derivative(density));
  }
  result.density = density;
}

void finalize_uhf(const PreparedFockPlan& plan, const integrals::IntegralData& ints,
                  const Matrix& orthogonalizer, std::size_t alpha_occupied,
                  std::size_t beta_occupied, Matrix& alpha_density, Matrix& beta_density,
                  bool compute_forces, ScfResult& result) {
  const std::size_t n = ints.nbf;
  auto [alpha_fock, beta_fock] = build_uhf_focks(plan, ints.hcore, alpha_density, beta_density);
  EigenResult alpha_orbitals = generalized_eigen(alpha_fock, orthogonalizer, n);
  EigenResult beta_orbitals = generalized_eigen(beta_fock, orthogonalizer, n);
  alpha_density = density_from_orbitals(alpha_orbitals.vectors, n, alpha_occupied, 1.0);
  beta_density = density_from_orbitals(beta_orbitals.vectors, n, beta_occupied, 1.0);
  std::tie(alpha_fock, beta_fock) = build_uhf_focks(plan, ints.hcore, alpha_density, beta_density);
  result.energy =
      uhf_electronic_energy(alpha_density, beta_density, ints.hcore, alpha_fock, beta_fock) +
      ints.nuclear_repulsion;
  if (compute_forces) {
    const Matrix alpha_weighted = energy_weighted_density(
        alpha_orbitals.vectors, alpha_orbitals.values, n, alpha_occupied, 1.0);
    const Matrix beta_weighted =
        energy_weighted_density(beta_orbitals.vectors, beta_orbitals.values, n, beta_occupied, 1.0);
    result.forces = gradient::analytic_uhf_forces(
        ints, alpha_density, beta_density, alpha_weighted, beta_weighted,
        plan.energy_derivative(alpha_density, beta_density));
  }
  result.density = concatenate(alpha_density, beta_density);
}

}  // namespace

ScfResult run_rhf_host_plan(const core::System& system, const ScfOptions& options,
                            const integrals::IntegralData& ints, const PreparedFockPlan& plan,
                            const std::vector<double>* initial_density) {
  const std::size_t n = ints.nbf;
  const std::size_t occupied = static_cast<std::size_t>(system.electron_count / 2);
  if (occupied > n) {
    throw std::runtime_error("basis has fewer orbitals than occupied electron pairs");
  }
  const Matrix orthogonalizer = symmetric_orthogonalizer(ints.overlap, n);
  EigenResult orbitals;
  Matrix density =
      prepare_initial_density(system, ints, orthogonalizer, occupied, initial_density, orbitals);
  if (options.strict_initial_density && initial_density) {
    validate_seed(ints.overlap, *initial_density, n, {static_cast<unsigned>(system.electron_count)},
                  2.0);
    density = *initial_density;
  }
  Diis diis(options.diis_history);
  const auto generation = new_scf_generation(options);
  unsigned proposal_failures = 0;

  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  double previous_energy = std::numeric_limits<double>::infinity();
  for (unsigned iteration = 1; iteration <= options.max_iterations; ++iteration) {
    ++result.fock_builds;
    const Matrix fock = build_fock(plan, ints.hcore, density);
    const double energy = electronic_energy(density, ints.hcore, fock) + ints.nuclear_repulsion;
    const Matrix residual = commutator_residual(fock, density, ints.overlap, n);
    const Matrix effective_fock = diis.update(fock, residual);
    orbitals = generalized_eigen(effective_fock, orthogonalizer, n);
    Matrix next_density = density_from_orbitals(orbitals.vectors, n, occupied);

    sample_scf_buffers(plan, diis, orthogonalizer, density, fock, residual, effective_fock,
                       orbitals.values, orbitals.vectors, next_density);

    result.iterations = iteration;
    result.energy = energy;
    result.energy_change = std::isfinite(previous_energy) ? std::abs(energy - previous_energy)
                                                          : std::numeric_limits<double>::infinity();
    result.density_rms = density_rms(next_density, density);
    const bool terminal = iteration > 1 && result.energy_change < options.energy_tolerance &&
                          result.density_rms < options.density_tolerance &&
                          (!options.hooks || !options.hooks->propose ||
                           residual_rms(residual) < options.density_tolerance);
    if (options.hooks) {
      next_density = safeguarded_update(
          options, generation, iteration, ints.overlap, density, fock, residual,
          std::move(next_density), {static_cast<unsigned>(system.electron_count)}, 2.0, result,
          diis, proposal_failures, terminal, [&](const Matrix& trial) {
            const Matrix trial_fock = build_fock(plan, ints.hcore, trial);
            return std::make_pair(
                electronic_energy(trial, ints.hcore, trial_fock) + ints.nuclear_repulsion,
                commutator_residual(trial_fock, trial, ints.overlap, n));
          });
    }
    if (terminal) {
      density = std::move(next_density);
      result.converged = true;
      break;
    }
    previous_energy = energy;
    density = std::move(next_density);
  }

  if (!result.converged) {
    // Failed traces retain the last iterate, never a converged reference.
    result.density = density;
    return result;
  }
  result.fock_builds += 2;  // Physical rebuilds performed by finalization.

  // Rebuild and diagonalize the un-extrapolated converged Fock matrix. The
  // resulting orbitals define the energy-weighted density in the Pulay term.
  finalize_scf(plan, ints, orthogonalizer, occupied, density, options.compute_forces, result);
  return result;
}

ScfResult run_uhf_host_plan(const core::System& system, const ScfOptions& options,
                            const integrals::IntegralData& ints, const PreparedFockPlan& plan,
                            const std::vector<double>* initial_density) {
  const std::size_t n = ints.nbf;
  const auto [alpha_occupied, beta_occupied] = spin_occupations(system);
  if (alpha_occupied > n || beta_occupied > n) {
    throw std::runtime_error("basis has fewer orbitals than required UHF spin occupations");
  }
  const Matrix orthogonalizer = symmetric_orthogonalizer(ints.overlap, n);
  EigenResult alpha_orbitals;
  EigenResult beta_orbitals;
  auto [alpha_density, beta_density] =
      prepare_initial_uhf_density(ints, orthogonalizer, alpha_occupied, beta_occupied,
                                  initial_density, alpha_orbitals, beta_orbitals);
  if (options.strict_initial_density && initial_density) {
    validate_seed(ints.overlap, *initial_density, n,
                  {static_cast<unsigned>(alpha_occupied), static_cast<unsigned>(beta_occupied)},
                  1.0);
    std::tie(alpha_density, beta_density) = split_spin_matrices(*initial_density, n * n);
  }
  Diis diis(options.diis_history);
  const auto generation = new_scf_generation(options);
  unsigned proposal_failures = 0;

  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  double previous_energy = std::numeric_limits<double>::infinity();
  for (unsigned iteration = 1; iteration <= options.max_iterations; ++iteration) {
    ++result.fock_builds;
    auto [alpha_fock, beta_fock] = build_uhf_focks(plan, ints.hcore, alpha_density, beta_density);
    const double energy =
        uhf_electronic_energy(alpha_density, beta_density, ints.hcore, alpha_fock, beta_fock) +
        ints.nuclear_repulsion;
    const Matrix alpha_residual = commutator_residual(alpha_fock, alpha_density, ints.overlap, n);
    const Matrix beta_residual = commutator_residual(beta_fock, beta_density, ints.overlap, n);
    const Matrix physical_fock = concatenate(alpha_fock, beta_fock);
    const Matrix physical_residual = concatenate(alpha_residual, beta_residual);
    const Matrix effective_joined = diis.update(physical_fock, physical_residual);
    std::tie(alpha_fock, beta_fock) = split_spin_matrices(effective_joined, n * n);
    alpha_orbitals = generalized_eigen(alpha_fock, orthogonalizer, n);
    beta_orbitals = generalized_eigen(beta_fock, orthogonalizer, n);
    Matrix next_alpha = density_from_orbitals(alpha_orbitals.vectors, n, alpha_occupied, 1.0);
    Matrix next_beta = density_from_orbitals(beta_orbitals.vectors, n, beta_occupied, 1.0);

    sample_scf_buffers(plan, diis, orthogonalizer, alpha_density, beta_density, alpha_fock,
                       beta_fock, alpha_residual, beta_residual, physical_fock, physical_residual,
                       effective_joined, alpha_orbitals.values, alpha_orbitals.vectors,
                       beta_orbitals.values, beta_orbitals.vectors, next_alpha, next_beta);

    result.iterations = iteration;
    result.energy = energy;
    result.energy_change = std::isfinite(previous_energy) ? std::abs(energy - previous_energy)
                                                          : std::numeric_limits<double>::infinity();
    result.density_rms =
        density_rms(concatenate(next_alpha, next_beta), concatenate(alpha_density, beta_density));
    const bool terminal = iteration > 1 && result.energy_change < options.energy_tolerance &&
                          result.density_rms < options.density_tolerance &&
                          (!options.hooks || !options.hooks->propose ||
                           residual_rms(physical_residual) < options.density_tolerance);
    if (options.hooks) {
      const Matrix next = safeguarded_update(
          options, generation, iteration, ints.overlap, concatenate(alpha_density, beta_density),
          physical_fock, physical_residual, concatenate(next_alpha, next_beta),
          {static_cast<unsigned>(alpha_occupied), static_cast<unsigned>(beta_occupied)}, 1.0,
          result, diis, proposal_failures, terminal, [&](const Matrix& trial) {
            const auto [a, b] = split_spin_matrices(trial, n * n);
            const auto [fa, fb] = build_uhf_focks(plan, ints.hcore, a, b);
            return std::make_pair(
                uhf_electronic_energy(a, b, ints.hcore, fa, fb) + ints.nuclear_repulsion,
                concatenate(commutator_residual(fa, a, ints.overlap, n),
                            commutator_residual(fb, b, ints.overlap, n)));
          });
      std::tie(next_alpha, next_beta) = split_spin_matrices(next, n * n);
    }
    if (terminal) {
      alpha_density = std::move(next_alpha);
      beta_density = std::move(next_beta);
      result.converged = true;
      break;
    }
    previous_energy = energy;
    alpha_density = std::move(next_alpha);
    beta_density = std::move(next_beta);
  }

  if (!result.converged) {
    // Failed traces retain the last iterate, never a converged reference.
    result.density = concatenate(alpha_density, beta_density);
    return result;
  }
  result.fock_builds += 2;  // Physical rebuilds performed by finalization.

  // As in RHF, rebuild from the un-extrapolated converged spin Fock matrices
  // before forming orbital-weighted Pulay densities and analytic forces.
  finalize_uhf(plan, ints, orthogonalizer, alpha_occupied, beta_occupied, alpha_density,
               beta_density, options.compute_forces, result);
  return result;
}

}  // namespace vibeqc::scf::solver
