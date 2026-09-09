#include <stdexcept>

#include "scf/mean_field.hpp"

namespace vibeqc::scf {

ScfResult run_fock_strategy(const core::System& system, const core::System* auxiliary,
                            const ScfOptions& options, int device_id,
                            const std::vector<double>* initial_density) {
  if (!options.resolved_fock_build)
    throw std::invalid_argument("Fock execution requires a resolved strategy");
  const auto& strategy = *options.resolved_fock_build;
  validate_resolved_fock_build(strategy);
  if (strategy.screening_tolerance != options.screening_tolerance ||
      (options.compute_forces && strategy.spec.derivative_order != 1) ||
      (strategy.metric_relative_threshold != 0.0 &&
       strategy.metric_relative_threshold != options.density_fitting_relative_threshold))
    throw std::invalid_argument("Fock strategy disagrees with execution controls");
  if (strategy.backend == FockBackend::Cpu)
    return run_cpu_fock_strategy(system, auxiliary, options, initial_density);
  if (strategy.schedule == FockSchedule::CudaIndependent || strategy.spec.derivative_order == 0)
    return run_cuda_independent_fock_strategy(system, auxiliary, options, device_id,
                                              initial_density);

  // Retain the established resident/streamed CUDA HF solver and its fused
  // Fock/force schedules. Backend choice never authorizes a fitted Hamiltonian.
  const bool unrestricted = strategy.spec.spin == FockSpin::Unrestricted;
  if (strategy.legacy_density_fitting) {
    const auto& source = auxiliary ? *auxiliary : system;
    return unrestricted
               ? run_uhf_density_fitting_cuda(system, source, options, device_id, initial_density)
               : run_rhf_density_fitting_cuda(system, source, options, device_id, initial_density);
  }
  require_exact_direct_strategy(strategy, strategy.spec.spin, FockBackend::Cuda);
  return unrestricted ? run_uhf_cuda(system, options, device_id, initial_density)
                      : run_rhf_cuda(system, options, device_id, initial_density);
}

}  // namespace vibeqc::scf
