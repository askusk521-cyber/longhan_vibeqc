#pragma once

#include <cuda_runtime.h>

#include "scf/cuda/integral_limits.hpp"

// The host allocates three axis tables; the force kernels borrow this exact
// layout in dynamic shared memory. Keep the layout independent of recurrences.

namespace vibeqc::scf::cuda_execution {

/** Compact Hermite workspace including one raised quantum on either center. */
struct OneElectronDerivativeHermiteCoefficients {
  static constexpr unsigned kIDimension = kMaximumAngularMomentum + 2;
  static constexpr unsigned kJDimension = kMaximumAngularMomentum + 2;
  static constexpr unsigned kTDimension = 2 * kMaximumAngularMomentum + 4;
  double data[kIDimension * kJDimension * kTDimension];

  __device__ double& at(unsigned i, unsigned j, unsigned t) {
    return data[(i * kJDimension + j) * kTDimension + t];
  }

  __device__ double at(unsigned i, unsigned j, unsigned t) const {
    return data[(i * kJDimension + j) * kTDimension + t];
  }
};

}  // namespace vibeqc::scf::cuda_execution
