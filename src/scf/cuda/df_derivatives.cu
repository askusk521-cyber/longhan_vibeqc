#include <limits>

#include "generated_df_derivatives.cuh"
#include "molecule/basis.hpp"
#include "scf/cuda/df_derivatives.cuh"
namespace vibeqc::scf {
namespace {
namespace generated = generated_df_derivatives;
constexpr std::size_t terms = molecule::kMaximumAoExpansionTerms;
__device__ generated::Angular angular(DfDerivativeBasisView basis, std::size_t term) {
  const auto* a = basis.term_angular + 3 * term;
  return {a[0], a[1], a[2]};
}
__device__ generated::Vec3 position(const double* r, std::size_t atom) {
  return {r[3 * atom], r[3 * atom + 1], r[3 * atom + 2]};
}
__device__ void add(double* output, std::size_t atom, generated::Vec3 g, double weight) {
  atomicAdd(output + 3 * atom, weight * g.x);
  atomicAdd(output + 3 * atom + 1, weight * g.y);
  atomicAdd(output + 3 * atom + 2, weight * g.z);
}
/** All mathematical-center contributions survive until physical atom accumulation. */
__device__ void contract(DfDerivativeBasisView o, DfDerivativeBasisView x, const double* positions,
                         unsigned kind, std::size_t element, double weight, double* gradient) {
  if (weight == 0) return;
  const std::size_t p = element % x.nbf;
  const std::size_t i = kind ? element / x.nbf : element / x.nbf / o.nbf;
  const std::size_t j = kind ? 0 : element / x.nbf % o.nbf;
  const auto left = kind ? x : o;
  const auto si = left.ao_shells[i], sp = x.ao_shells[p];
  const auto ai = left.shell_atoms[si], ap = x.shell_atoms[sp];
  const auto A = position(positions, ai), C = position(positions, ap);
  const auto sj = kind ? 0 : o.ao_shells[j];
  const auto aj = kind ? ai : o.shell_atoms[sj];
  const auto B = position(positions, aj);
  const auto jb = kind ? 0 : o.primitive_offsets[sj], je = kind ? 1 : o.primitive_offsets[sj + 1];
  for (auto a = left.primitive_offsets[si]; a < left.primitive_offsets[si + 1]; ++a)
    for (auto b = jb; b < je; ++b)
      for (auto c = x.primitive_offsets[sp]; c < x.primitive_offsets[sp + 1]; ++c)
        for (unsigned ti = 0; ti < left.term_counts[i]; ++ti)
          for (unsigned tj = 0; tj < (kind ? 1U : o.term_counts[j]); ++tj)
            for (unsigned tp = 0; tp < x.term_counts[p]; ++tp) {
              const auto term_i = i * terms + ti, term_j = j * terms + tj, term_p = p * terms + tp;
              const double norm = weight * left.coefficients[a] * x.coefficients[c] *
                                  left.term_coefficients[term_i] * x.term_coefficients[term_p] *
                                  (kind ? 1.0 : o.coefficients[b] * o.term_coefficients[term_j]);
              const auto result =
                  kind ? generated::metric(left.exponents[a], A, angular(left, term_i),
                                           x.exponents[c], C, angular(x, term_p))
                       : generated::three_center(left.exponents[a], A, angular(left, term_i),
                                                 o.exponents[b], B, angular(o, term_j),
                                                 x.exponents[c], C, angular(x, term_p));
              add(gradient, ai, result.first, norm);
              if (!kind) add(gradient, aj, result.second, norm);
              add(gradient, ap, result.third, norm);
            }
}
__global__ void derivative_tile(DfDerivativeBasisView o, DfDerivativeBasisView x,
                                const double* positions, unsigned kind, std::size_t offset,
                                std::size_t count, const double* weights, unsigned schedule,
                                double* gradient, std::size_t stride) {
  const auto thread = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (schedule) {
    if (thread == 0)
      for (std::size_t item = 0; item < count; ++item)
        contract(o, x, positions, kind, offset + item * stride, weights[item], gradient);
  } else if (thread < count)
    contract(o, x, positions, kind, offset + thread * stride, weights[thread], gradient);
}
}  // namespace
cudaError_t launch_df_derivative_tile(DfDerivativeBasisView o, DfDerivativeBasisView x,
                                      const double* positions, unsigned kind, std::size_t offset,
                                      std::size_t count, const double* weights, unsigned schedule,
                                      double* gradient, cudaStream_t stream, std::size_t stride) {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (!o.nbf || !x.nbf || kind > 1 || schedule > 1 || !positions || !weights || !gradient ||
      !count || o.nbf > maximum / o.nbf || o.nbf * o.nbf > maximum / x.nbf ||
      x.nbf > maximum / x.nbf)
    return cudaErrorInvalidValue;
  const auto elements = kind ? x.nbf * x.nbf : o.nbf * o.nbf * x.nbf;
  if (!stride || offset >= elements || count - 1 > (elements - 1 - offset) / stride ||
      (count - 1) / 128 >= std::numeric_limits<int>::max())
    return cudaErrorInvalidValue;
  const auto blocks = schedule ? 1U : static_cast<unsigned>((count - 1) / 128 + 1);
  derivative_tile<<<blocks, 128, 0, stream>>>(o, x, positions, kind, offset, count, weights,
                                              schedule, gradient, stride);
  return cudaPeekAtLastError();
}
}  // namespace vibeqc::scf
