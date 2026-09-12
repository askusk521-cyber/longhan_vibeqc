#pragma once

#include <cstddef>
#include <limits>

namespace vibeqc::scf::cuda_execution {

/** Overflow-checked arithmetic for layout and topology planning. Failure leaves the caller
 * responsible for rejecting the plan. */
inline bool checked_multiply(std::size_t first, std::size_t second, std::size_t& result) {
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  result = first * second;
  return true;
}

inline bool checked_add(std::size_t first, std::size_t second, std::size_t& result) {
  if (second > std::numeric_limits<std::size_t>::max() - first) return false;
  result = first + second;
  return true;
}

}  // namespace vibeqc::scf::cuda_execution
