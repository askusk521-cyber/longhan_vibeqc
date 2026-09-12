// Private implementation fragment included once inside vibeqc::scf, after
// CUDA bucket helpers. Keeps the owned-reference entry and failure propagation
// separate from the large kernel translation unit; no independent runtime.
#pragma once

ScfResult run_rhf_cuda(const core::System& system, const ScfOptions& options, int device_id,
                       const std::vector<double>* initial_density) {
  hf_cuda_failure_detail.clear();
  const std::vector<core::System> systems{system};
  const std::vector<const std::vector<double>*> initial_densities{initial_density};
  std::vector<RhfBucketItem> result =
      run_rhf_cuda_bucket(systems, options, initial_densities, device_id);
  if (result.empty()) throw std::runtime_error("CUDA RHF returned no result");
  if (options.export_physical_reference && result.front().status == VIBEQC_STATUS_NUMERICAL_FAILURE)
    throw std::runtime_error("CUDA RHF physical reference has a numerical failure");
  if (options.export_physical_reference && result.front().status == VIBEQC_STATUS_OUT_OF_MEMORY)
    throw std::length_error("CUDA RHF reference exceeds memory budget or allocation failed");
  if (result.front().status == VIBEQC_STATUS_CUDA_ERROR ||
      result.front().status == VIBEQC_STATUS_OUT_OF_MEMORY) {
    throw std::runtime_error("CUDA RHF execution failed: " + hf_cuda_failure_detail);
  }
  return std::move(result.front().scf);
}
