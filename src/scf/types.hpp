#ifndef VIBEQC_SCF_TYPES_HPP
#define VIBEQC_SCF_TYPES_HPP

#include <memory>
#include <vector>

namespace vibeqc::scf {

/** Numerical controls shared by the implemented mean-field solvers. */
struct ScfOptions {
  unsigned max_iterations{100};
  unsigned diis_history{8};
  double energy_tolerance{1.0e-10};
  double density_tolerance{1.0e-8};
  double screening_tolerance{1.0e-12};
  /** Select the DF solver; direct four-center remains the default. */
  vibeqc_density_fitting_mode density_fitting_mode{VIBEQC_DENSITY_FITTING_NONE};
  /** Relative cutoff used when factoring the auxiliary Coulomb metric. */
  double density_fitting_relative_threshold{1.0e-10};
  /** Byte budget for bounded DF plan/integral work; zero means implementation default. */
  std::size_t density_fitting_memory_budget_bytes{};
  /** Correlated energy consumers require values-only, bounded direct RHF.
   * Defaults preserve
   * the existing HF energy/force paths. */
  bool export_physical_reference{false};
  std::size_t reference_memory_budget_bytes{};
};

/** Owned physical canonical RHF state; C[mu,p] is row-major on every backend.
 * Never published
 * for a failed/noncanonical SCF state. All arrays are detached
 * from the HF arena.
 * Geometry/basis/generation ownership belongs to its plan. */
struct PhysicalReference {
  std::size_t nbf{};
  std::size_t nocc{};
  std::vector<double> overlap, hcore, fock, coefficients, orbital_energies, density;
  double energy{};
  double commutator_residual{};
  double canonical_density_drift{};
  double eigen_residual{};
  std::size_t numeric_capacity_bytes{};
};

/** Internal mean-field result, including state retained for warm starts. */
struct ScfResult {
  double energy{};
  std::vector<double> forces;
  // RHF stores one N x N AO density; UHF stores alpha then beta matrices.
  // The state is explicit and remains private to prepared execution plans.
  std::vector<double> density;
  unsigned iterations{};
  double energy_change{};
  double density_rms{};
  bool converged{};
  bool initial_density_used{};
  std::shared_ptr<const PhysicalReference> reference;
};

}  // namespace vibeqc::scf

#endif
