#ifndef VIBEQC_SCF_CUDA_RHF_POLICY_HPP
#define VIBEQC_SCF_CUDA_RHF_POLICY_HPP

#include <cstdint>
#include <optional>

#include "vibeqc/vibeqc.h"

namespace vibeqc::scf::cuda_policy {

/** Runtime policy switches kept in a host-only translation unit. */
bool reuse_converged_fock_requested() noexcept;
std::optional<double> configured_mixed_precision_fock_threshold(
    double screening_tolerance) noexcept;
/**
 * Outcome of the budget-aware admission for the public \p auto policy.
 *
 * \p eligible_tiles is the mixed-capable tile census of the exact prepared
 * topology: the active Fock tiles in the high-angular-order shell classes that
 * may run in FP32. A zero census is never admitted because the accumulated
 * bound cannot be evaluated without it.
 */
struct AutoMixedPrecisionAdmission {
  /** The mixed iterative Fock may run for this reference. */
  bool admitted{false};
  /** Resolved contribution cutoff for FP32 tiles; zero when refused. */
  double threshold{0.0};
  /** Error the policy reserved for the iterative operator out of the target. */
  double reserved_error{0.0};
  /** Mixed-capable tile census the accumulated bound was evaluated against. */
  double eligible_tiles{0.0};
};
/**
 * Budget-aware admission for the public \p auto policy.
 *
 * One global contribution cutoff is not an error budget: any number of
 * individually eligible FP32 tiles can accumulate. This evaluates the
 * worst-case accumulated bound `eps32 * cutoff * eligible_tiles` against the
 * error the policy reserves for the iterative operator out of the requested
 * energy tolerance and resolves the largest cutoff that bound certifies. A
 * cutoff at or below the screening floor admits no mixed tile, so the operator
 * stays FP64. Because the census is the exact per-shell-class active tile count
 * of this reference, a larger or lower-symmetry tile census tightens the cutoff
 * instead of silently accumulating error.
 */
AutoMixedPrecisionAdmission admit_auto_mixed_precision_fock(double energy_tolerance,
                                                            double screening_tolerance,
                                                            double eligible_tiles) noexcept;
/** Complete resolution of the requested precision policy, including its audit. */
struct MixedPrecisionFockPolicy {
  /** Resolved FP32 tile cutoff; empty keeps the FP64 operator. */
  std::optional<double> threshold;
  /** The cutoff came from the certified accumulated-error budget. */
  bool budget_certified{false};
  /** Error reserved for the iterative operator; zero when uncertified. */
  double reserved_error{0.0};
  /** Mixed-capable tile census the budget was evaluated against. */
  double eligible_tiles{0.0};
};
/**
 * Resolve the mixed-precision Fock policy from the requested public policy and
 * the tolerances. \p nullopt preserves the legacy
 * VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD diagnostic switch verbatim. An explicit
 * \p VIBEQC_PRECISION_FP64 keeps the pure double path (the environment cannot
 * relax it). An explicit \p VIBEQC_PRECISION_AUTO derives the threshold from the
 * budget-aware admission above, with an explicit numeric environment value
 * acting as a hard diagnostic override that bypasses the budget.
 */
MixedPrecisionFockPolicy resolve_mixed_precision_fock_policy(
    std::optional<vibeqc_precision_mode> precision_mode, double energy_tolerance,
    double screening_tolerance, double eligible_tiles) noexcept;
bool graph_native_eigensolver_override_requested() noexcept;
bool xsyev_probe_skip_diagnostic_requested() noexcept;
bool bounded_direct_streaming_override_requested() noexcept;
bool bounded_direct_count_diagnostic_requested() noexcept;
bool bounded_direct_aot_only_diagnostic_requested() noexcept;
bool bounded_direct_fock_only_diagnostic_requested() noexcept;
bool bounded_fock_class_timing_requested() noexcept;
bool direct_tile_validation_requested() noexcept;
double converged_fock_reuse_density_rms(double density_tolerance) noexcept;
bool force_density_product_screening_requested() noexcept;
bool resident_ppps_bra_requested() noexcept;
bool ppps_signature_bucketing_requested() noexcept;
bool psps_signature_bucketing_requested() noexcept;
bool ppss_signature_bucketing_requested() noexcept;
unsigned ppps_resident_block_threads_requested() noexcept;
bool one_electron_force_scalar_requested() noexcept;
/** 0: one AO pair per thread; 1: one shell pair per warp. */
unsigned one_electron_value_mapping_requested() noexcept;
/** Generated derivative candidates are opt-in and read at each force execution. */
bool generated_one_electron_derivatives_requested() noexcept;
/** 0: AO threads; 1: shell-pair warp lanes; 2: deterministic serial diagnostics. */
unsigned one_electron_derivative_mapping_requested() noexcept;
bool resident_psss_bra_requested() noexcept;
/** Generated weighted primitive candidate; frozen into a prepared bucket. */
bool generated_psss_weighted_requested() noexcept;

/** 0: cooperative dense elements; 1: deterministic serial traversal. */
unsigned df_derivative_mapping_requested() noexcept;
/** 0: contiguous auxiliary outputs; 1: AO components; 2: primitive lanes. */
unsigned df_value_mapping_requested() noexcept;

}  // namespace vibeqc::scf::cuda_policy

#endif
