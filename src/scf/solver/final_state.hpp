#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "scf/density_factor.hpp"
#include "scf/fock_build.hpp"
#include "scf/initial_guess/eigen_operation.hpp"
#include "scf/reference/linalg.hpp"
#include "scf/solver/eigen_frame.hpp"

namespace vibeqc::scf::solver {
/** Immutable prepared-source identity plus one solve and determinant state.
 * `factor.basis` binds ordered basis/geometry/representation/device ownership;
 * `factor.reference` identifies the source item, as in occupied RI-K. The
 * separate solve epoch prevents iteration counters restarting on warm replay
 * from authorizing a previous solve's frame. Model/policy and spin occupations
 * are compared by value, never inferred from AO dimensions. */
struct FinalStateIdentity {
  DensityFactorIdentity factor;
  std::uint64_t solve_epoch{};
  ResolvedFockBuild model;
  std::vector<std::size_t> occupied;
  bool operator==(const FinalStateIdentity&) const = default;
};

/** Detached candidate columns. Their physical Fock may have been evaluated at
 * an older D generation; acceptance always checks against a freshly evaluated
 * current Fock. Effective/DIIS/shifted-origin frames require correction. */
struct FinalFrameCandidate {
  FinalStateIdentity identity;
  std::uint64_t fock_density_generation{};
  bool physical_origin{};
  std::vector<reference::EigenResult> spins;
};

/** A physical evaluation at exactly the tagged density. Defaults fail closed.
 * Providers must evaluate the passed D on their immutable target model; an
 * old cached or extrapolated Fock cannot be relabeled as this evaluation. */
struct PhysicalFockFrame {
  FinalStateIdentity identity;
  bool physical{};
  std::vector<reference::Matrix> spins;
};

struct FinalStateLimits {
  double density_tolerance{1e-8};
  double energy_tolerance{1e-10};
  // Zero permits validation only; requests above 64 are invalid. This is a
  // bounded final correction, not another unbounded SCF driver.
  unsigned maximum_corrections{4};
};

struct FinalStateDiagnostic {
  double maximum_commutator{}, density_rms{}, maximum_trace_error{}, maximum_idempotency_error{};
  double energy{}, energy_change{};
  std::vector<EigenFrameDiagnostic> eigenframes;
};

/** Check current F, C/epsilon and D without diagonalizing or repairing inputs.
 * The physical-reference 1e-8 absolute gates remain caps; a tighter requested
 * density tolerance also applies to density drift and physical commutator.
 * Energy is recomputed at the supplied D for the resolved quadratic J/K model,
 * with the established RHF/UHF weights. This is not a KS/XC energy functional.
 * Failed validation never authorizes W, occupied factors or force output. */
bool validate_final_state(const FinalStateIdentity& current, const reference::Matrix& overlap,
                          const reference::Matrix& hcore, double nuclear_energy,
                          const std::vector<reference::Matrix>& density,
                          const PhysicalFockFrame& fock, const FinalFrameCandidate& orbitals,
                          const FinalStateLimits& limits, FinalStateDiagnostic& diagnostic,
                          std::string& detail);

/** Owned, validated output. W is empty unless requested after strict checks;
 * no cached W is accepted as input. Consumers must preserve this identity when
 * constructing factors or exporting a physical reference. */
struct VerifiedFinalState {
  FinalStateIdentity identity;
  std::vector<reference::Matrix> density, fock, weighted_density;
  std::vector<reference::EigenResult> orbitals;
  FinalStateDiagnostic diagnostic;
};

enum class FinalStateStatus {
  Success,
  InvalidInput,
  NumericalFailure,
  ProviderFailure,
  OutOfMemory
};
struct FinalStateSelection {
  FinalStateStatus status{FinalStateStatus::NumericalFailure};
  std::string detail;
  unsigned fock_evaluations{}, eigen_solves{}, density_updates{}, candidate_rejections{};
  bool reused{};
  std::optional<VerifiedFinalState> state;
};

using PhysicalFockOperation = std::function<PhysicalFockFrame(
    const FinalStateIdentity&, const std::vector<reference::Matrix>&)>;

/** Always evaluate physical F[D_returned] before selecting a candidate. A
 * bounded correction solves that F, projects D, advances both determinant
 * generations, evaluates the new F and checks again, including energy change.
 * `force_rebuild` exercises the actual correction path, even for a valid
 * candidate. Callbacks are synchronous/borrowed; exceptions propagate as an
 * explicit failed selection, never as a reference fallback. No successful
 * state survives exhausted correction, invalid provider output or allocation
 * failure. Public C/Python result layouts are unaffected by this internal API. */
FinalStateSelection select_final_state(
    FinalStateIdentity current, const reference::Matrix& overlap, const reference::Matrix& hcore,
    const reference::Matrix& orthogonalizer, double nuclear_energy,
    std::vector<reference::Matrix> density, const FinalFrameCandidate* candidate,
    const PhysicalFockOperation& evaluate, const initial_guess::EigenOperation& eigen,
    const FinalStateLimits& limits, bool compute_weighted_density, bool force_rebuild = false);
}  // namespace vibeqc::scf::solver
