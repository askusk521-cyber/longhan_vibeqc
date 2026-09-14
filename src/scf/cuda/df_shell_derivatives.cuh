#pragma once

#include "scf/cuda/df_derivatives.cuh"

namespace vibeqc::scf {
/** Compact angular-class shell lists; storage grows with shells, never triples.
 * AO offsets are in the original public Cartesian/spherical order. A host
 * owner may narrow each class list to the shells intersecting an active panel.
 */
struct DfShellBasisView {
  DfDerivativeBasisView basis;
  const std::int32_t* shell_ids{};
  const std::int64_t* ao_offsets{};
  std::size_t begin[4]{}, count[4]{};
};

/** Launch the generated s/p prototype on a complete auxiliary-major AO panel.
 * All pointers belong to the caller and remain live through its stream drain.
 * Optional device counters record visited/nonzero shell triples, nonzero
 * public weights, executed primitive products, and Cartesian component work.
 * They are diagnostic atomics and must be disabled in promotion timings.
 */
cudaError_t launch_df_shell_derivative_panel(DfShellBasisView orbital, DfShellBasisView auxiliary,
                                             const double* positions, std::size_t auxiliary_begin,
                                             std::size_t auxiliary_count, const double* weights,
                                             double* gradient, unsigned long long* counters,
                                             cudaStream_t stream);
}  // namespace vibeqc::scf
