#pragma once

#include <stdexcept>

namespace vibeqc::runtime {

/** Backend failures are distinct from scientific nonfinite results. This
 * host-only type crosses method/API boundaries without requiring CUDA headers
 * in CPU builds or inferring a failure category from its message text. */
struct CudaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace vibeqc::runtime
