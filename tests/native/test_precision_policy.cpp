#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/rhf.hpp"
#include "vibeqc/vibeqc.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void require_close(double actual, double expected, double tolerance, const char* message) {
  require(std::abs(actual - expected) <= tolerance, message);
}

/** RAII environment setter that restores the previous value on scope exit. */
class ScopedEnv {
 public:
  explicit ScopedEnv(const char* name, const char* value) : name_(name) {
    if (value == nullptr) {
      previous_ = std::getenv(name_);
      unsetenv(name_);
    } else {
      previous_ = std::getenv(name_);
      setenv(name_, value, 1);
    }
  }
  ~ScopedEnv() {
    if (previous_ == nullptr)
      unsetenv(name_);
    else
      setenv(name_, previous_, 1);
  }
  const char* name_;
  const char* previous_;
};

using Threshold = std::optional<double>;

/**
 * Tolerance-derived \p auto threshold: 1e4 * energy_tolerance, floored by the
 * screening tolerance. The 1e-10 default resolves to the legacy measured-accurate
 * 1e-6 anchor; tighter targets resolve smaller and, below the floor, to no mixed
 * route at all (the operator stays FP64).
 */
void verify_auto_threshold_derivation() {
  const double screen = 1.0e-12;
  const Threshold default_target =
      vibeqc::scf::cuda_policy::auto_mixed_precision_fock_threshold(1.0e-10, screen);
  require(default_target.has_value(), "default 1e-10 target should derive a mixed route");
  require_close(*default_target, 1.0e-6, 1.0e-12,
                "default target anchors the 1e-6 legacy threshold");

  const Threshold looser =
      vibeqc::scf::cuda_policy::auto_mixed_precision_fock_threshold(1.0e-9, screen);
  require(looser.has_value() && *looser > *default_target,
          "looser target derives a larger threshold");
  require_close(*looser, 1.0e-5, 1.0e-13, "1e-9 target derives 1e-5");

  const Threshold tighter =
      vibeqc::scf::cuda_policy::auto_mixed_precision_fock_threshold(1.0e-11, screen);
  require(tighter.has_value() && *tighter < *default_target,
          "tighter target derives a smaller threshold");
  require_close(*tighter, 1.0e-7, 1.0e-14, "1e-11 target derives 1e-7");

  // 1e4 * 1e-17 = 1e-13 <= screen (1e-12): the requested accuracy is tight enough
  // that per-tile FP32 rounding could reach the target, so the operator stays FP64.
  const Threshold too_tight =
      vibeqc::scf::cuda_policy::auto_mixed_precision_fock_threshold(1.0e-17, screen);
  require(!too_tight.has_value(), "sub-floor target collapses to the FP64 operator");

  const Threshold non_positive =
      vibeqc::scf::cuda_policy::auto_mixed_precision_fock_threshold(0.0, screen);
  require(!non_positive.has_value(), "non-positive energy tolerance derives no mixed route");
  const Threshold inf = vibeqc::scf::cuda_policy::auto_mixed_precision_fock_threshold(
      std::numeric_limits<double>::infinity(), screen);
  require(!inf.has_value(), "non-finite energy tolerance derives no mixed route");
}

/** Explicit FP64 keeps the pure double path; even a numeric legacy env cannot relax it. */
void verify_fp64_strict() {
  const double screen = 1.0e-12;
  const double energy = 1.0e-10;
  const Threshold clean = vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_threshold(
      std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_FP64), energy, screen);
  require(!clean.has_value(), "explicit FP64 resolves to the pure double path");

  // A numeric diagnostic override is present but FP64 must ignore it.
  ScopedEnv override("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "5e-7");
  const Threshold with_override = vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_threshold(
      std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_FP64), energy, screen);
  require(!with_override.has_value(), "FP64 is not relaxed by the legacy diagnostic override");
}

/** AUTO derives from tolerances, but an explicit numeric env acts as a hard override. */
void verify_auto_with_legacy_override() {
  const double screen = 1.0e-12;
  const double energy = 1.0e-10;

  ScopedEnv numeric("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "2e-7");
  const Threshold overridden = vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_threshold(
      std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_AUTO), energy, screen);
  require(overridden.has_value(), "AUTO with a numeric override resolves a mixed route");
  require_close(*overridden, 2.0e-7, 1.0e-14, "numeric override wins over the derived value");

  // The 'auto' / '0' / 'none' spellings are not numeric overrides; they fall back
  // to the derived threshold.
  ScopedEnv auto_spelling("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "auto");
  const Threshold derived = vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_threshold(
      std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_AUTO), energy, screen);
  require(derived.has_value() &&
              std::abs(*derived -
                       vibeqc::scf::cuda_policy::auto_mixed_precision_fock_threshold(energy, screen)
                           .value()) <= 1.0e-16,
          "AUTO with the 'auto' spelling derives from the tolerances");

  // A numeric override at or below the screening floor is rejected.
  ScopedEnv below_floor("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "5e-13");
  const Threshold below = vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_threshold(
      std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_AUTO), energy, screen);
  require(below.has_value() && std::abs(*below - 1.0e-6) <= 1.0e-12,
          "a sub-floor numeric override falls back to the derived threshold");
}

/** A nullopt mode preserves the legacy diagnostic switch verbatim. */
void verify_nullopt_legacy_parity() {
  const double screen = 1.0e-12;
  const double energy = 1.0e-10;
  std::optional<vibeqc_precision_mode> nullopt;

  ScopedEnv absent("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "0");
  require(!vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_threshold(nullopt, energy, screen)
               .has_value(),
          "absent/0 legacy switch keeps the default FP64 path");

  ScopedEnv legacy_auto("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "auto");
  const Threshold legacy_auto_threshold =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_threshold(nullopt, energy, screen);
  require(legacy_auto_threshold.has_value(), "'auto' legacy switch enables the mixed route");
  require_close(*legacy_auto_threshold, 1.0e-6, 1.0e-12,
                "'auto' legacy switch uses the default 1e-6");

  ScopedEnv legacy_numeric("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "3e-7");
  const Threshold legacy_numeric_threshold =
      vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_threshold(nullopt, energy, screen);
  require(legacy_numeric_threshold.has_value() &&
              std::abs(*legacy_numeric_threshold - 3.0e-7) <= 1.0e-14,
          "numeric legacy switch passes through verbatim");

  ScopedEnv legacy_too_small("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "5e-13");
  require(!vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_threshold(nullopt, energy, screen)
               .has_value(),
          "a numeric legacy switch at or below the floor keeps the FP64 path");

  ScopedEnv legacy_garbage("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "bogus");
  require(!vibeqc::scf::cuda_policy::resolve_mixed_precision_fock_threshold(nullopt, energy, screen)
               .has_value(),
          "an invalid legacy switch keeps the FP64 path rather than relaxing it");
}

/** A converged-fock reuse RMS scales with the density tolerance. */
void verify_converged_fock_reuse_rms() {
  require_close(vibeqc::scf::cuda_policy::converged_fock_reuse_density_rms(1.0e-12), 1.0e-12,
                1.0e-16, "tight density tolerance keeps the tight reuse RMS");
  require_close(vibeqc::scf::cuda_policy::converged_fock_reuse_density_rms(1.0e-9), 2.0e-9, 1.0e-16,
                "expanded density tolerance widens the reuse RMS");
}

/** Minimal closed-shell H2 used to pin CPU provenance end to end. */
vibeqc::core::System h2_system() {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0, 0, {{1.5, 1.0}, {0.4, 0.5}}},
      {1, 0, {{1.5, 1.0}, {0.4, 0.5}}},
  };
  system.charge = 0;
  system.multiplicity = 1;
  system.basis_representation = VIBEQC_BASIS_SPHERICAL;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          ("H2 system normalization failed: " + detail).c_str());
  return system;
}

/**
 * The host-plan path always runs the FP64 Fock, so effective_bits stays 64 while
 * requested_mode reports the caller's policy. This pins the honest-provenance fix
 * at the native level: an \p auto request that collapses to FP64 on CPU must still
 * say it was asked for \p auto.
 */
vibeqc::scf::ScfResult run_cpu_rhf(std::optional<vibeqc_precision_mode> precision_mode) {
  const vibeqc::core::System system = h2_system();
  vibeqc::scf::ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1.0e-10;
  options.density_tolerance = 1.0e-8;
  options.screening_tolerance = 1.0e-12;
  options.precision_mode = precision_mode;
  return vibeqc::scf::run_rhf(system, options, nullptr);
}

void verify_cpu_provenance() {
  const vibeqc::scf::ScfResult fp64 =
      run_cpu_rhf(std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_FP64));
  require(fp64.converged, "CPU FP64 RHF should converge");
  require(fp64.precision.requested_mode == VIBEQC_PRECISION_FP64,
          "explicit FP64 reports FP64 as the requested mode");
  require(fp64.precision.effective_bits == 64, "CPU FP64 runs at 64 bits");
  require(!fp64.precision.strict_refinement_applied, "CPU FP64 applies no mixed refinement");

  const vibeqc::scf::ScfResult auto_result =
      run_cpu_rhf(std::optional<vibeqc_precision_mode>(VIBEQC_PRECISION_AUTO));
  require(auto_result.converged, "CPU auto RHF should converge");
  require(auto_result.precision.requested_mode == VIBEQC_PRECISION_AUTO,
          "CPU auto reports the requested auto policy (not the collapsed FP64)");
  require(auto_result.precision.effective_bits == 64,
          "CPU auto still runs the FP64 Fock (mixed route is CUDA-only)");

  // The two policies are computationally identical on CPU, so the energies match
  // and only the requested-mode provenance differs.
  require_close(auto_result.energy, fp64.energy, 1.0e-12, "CPU auto and FP64 give the same energy");
  require(auto_result.precision.requested_mode != fp64.precision.requested_mode,
          "provenance distinguishes the explicit fp64 from an auto request");

  // A nullopt mode keeps the struct default (fp64) on the host path.
  const vibeqc::scf::ScfResult nullopt_result = run_cpu_rhf(std::nullopt);
  require(nullopt_result.converged, "CPU nullopt RHF should converge");
  require(nullopt_result.precision.requested_mode == VIBEQC_PRECISION_FP64,
          "absent precision mode keeps the FP64 requested-mode default");
}

}  // namespace

int main() {
  try {
    verify_auto_threshold_derivation();
    verify_fp64_strict();
    verify_auto_with_legacy_override();
    verify_nullopt_legacy_parity();
    verify_converged_fock_reuse_rms();
    verify_cpu_provenance();
    std::cout << "validated precision policy controller and CPU provenance\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
