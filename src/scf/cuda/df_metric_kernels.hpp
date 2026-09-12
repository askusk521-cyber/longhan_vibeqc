#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::scf::cuda_df {

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_symmetrize_metrics_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::size_t dimension, double* metrics);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_scale_eigenvectors_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::size_t matrix_elements,
                                      std::size_t dimension, const double* eigenvectors,
                                      const double* scales, double* scaled_eigenvectors);

}  // namespace vibeqc::scf::cuda_df
