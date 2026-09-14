#include <limits>

#include "generated_df_shell_derivatives.cuh"
#include "molecule/basis.hpp"
#include "scf/cuda/df_shell_derivatives.cuh"

namespace vibeqc::scf {
namespace {
namespace generated = generated_df_shell;
namespace scalar = generated_df_derivatives;
constexpr unsigned term_capacity = molecule::kMaximumAoExpansionTerms;

/** One generated subgroup owns a shell triple and contracts its public weights.
 * Sparse normalized AO expansions are applied to weights once, before any
 * primitive work. Partial auxiliary shells contribute only in-panel weights.
 * Every subgroup lane rendezvous uses its original mask, including lanes with
 * zero weights or no component. Independent groups may share a physical warp;
 * no barrier waits on another group's primitive count or panel clipping.
 * No derivative array escapes the subgroup.
 */
template <unsigned A, unsigned B, unsigned C, unsigned Variant>
__global__ void shell_panel(DfShellBasisView orbital, DfShellBasisView auxiliary,
                            const double* positions, std::size_t panel_begin,
                            std::size_t panel_count, const double* weights, double* gradient,
                            unsigned long long* counters) {
  using Math = generated::Shell<A, B, C>;
  using Schedule = generated::Schedule<A, B, C, Variant>;
  constexpr auto lanes = Schedule::lanes, groups = Schedule::groups;
  __shared__ double all_weights[groups][Math::components], all_cache[groups][3 * Math::axis_size];
  __shared__ scalar::Geometry all_geometry[groups];
  const unsigned lane = threadIdx.x % lanes, group = threadIdx.x / lanes;
  const unsigned mask = (0xffffffffU >> (32 - lanes)) << ((threadIdx.x % 32) / lanes * lanes);
  auto* cart_weights = all_weights[group];
  auto* cache = all_cache[group];
  auto& geometry = all_geometry[group];
  const auto task = std::size_t{blockIdx.x} * groups + group;
  if (task >= orbital.count[A] * orbital.count[B] * auxiliary.count[C]) return;
  const auto sc = auxiliary.shell_ids[auxiliary.begin[C] + task % auxiliary.count[C]];
  const auto pair = task / auxiliary.count[C];
  const auto sb = orbital.shell_ids[orbital.begin[B] + pair % orbital.count[B]];
  const auto sa = orbital.shell_ids[orbital.begin[A] + pair / orbital.count[B]];
  if (lane == 0 && counters) atomicAdd(counters, 1ULL);
  const auto& o = orbital.basis;
  const auto& x = auxiliary.basis;
  const auto oa = orbital.ao_offsets[sa], ob = orbital.ao_offsets[sb];
  const auto oc = auxiliary.ao_offsets[sc];
  const auto na = orbital.ao_offsets[sa + 1] - oa;
  const auto nb = orbital.ao_offsets[sb + 1] - ob;
  const auto nc = auxiliary.ao_offsets[sc + 1] - oc;
  if (oc >= panel_begin + panel_count || oc + nc <= panel_begin) return;
  for (unsigned i = lane; i < Math::components; i += lanes) cart_weights[i] = 0;
  __syncwarp(mask);
  unsigned public_work = 0;
  for (auto i = std::int64_t{lane}; i < na * nb * nc; i += lanes) {
    const auto ai = oa + i / nb / nc, bi = ob + i / nc % nb, ci = oc + i % nc;
    if (ci < panel_begin || ci - panel_begin >= panel_count) continue;
    const double weight = weights[(ci - panel_begin) * o.nbf * o.nbf + ai * o.nbf + bi];
    if (weight == 0) continue;
    ++public_work;
    for (unsigned at = 0; at < o.term_counts[ai]; ++at)
      for (unsigned bt = 0; bt < o.term_counts[bi]; ++bt)
        for (unsigned ct = 0; ct < x.term_counts[ci]; ++ct) {
          const auto ia = ai * term_capacity + at, ib = bi * term_capacity + bt;
          const auto ic = ci * term_capacity + ct;
          const auto ac = generated::cartesian_index(A, o.term_angular + 3 * ia);
          const auto bc = generated::cartesian_index(B, o.term_angular + 3 * ib);
          const auto cc = generated::cartesian_index(C, x.term_angular + 3 * ic);
          atomicAdd(
              cart_weights + (ac * Math::nb + bc) * Math::nc + cc,
              weight * o.term_coefficients[ia] * o.term_coefficients[ib] * x.term_coefficients[ic]);
        }
  }
  __syncwarp(mask);
  unsigned component_work = 0;
  for (unsigned i = lane; i < Math::components; i += lanes) component_work += cart_weights[i] != 0;
  const bool active = __any_sync(mask, component_work != 0);
  for (unsigned delta = lanes / 2; delta; delta /= 2) {
    public_work += __shfl_down_sync(mask, public_work, delta, lanes);
    component_work += __shfl_down_sync(mask, component_work, delta, lanes);
  }
  if (lane == 0 && counters) atomicAdd(counters + 2, static_cast<unsigned long long>(public_work));
  if (!active) return;
  const auto atom_a = o.shell_atoms[sa], atom_b = o.shell_atoms[sb], atom_c = x.shell_atoms[sc];
  const auto* ra = positions + 3 * atom_a;
  const auto* rb = positions + 3 * atom_b;
  const auto* rc = positions + 3 * atom_c;
  const scalar::Vec3 center_a{ra[0], ra[1], ra[2]}, center_b{rb[0], rb[1], rb[2]},
      center_c{rc[0], rc[1], rc[2]};
  double output[6]{};
  unsigned long long primitive_work = 0;
  for (auto pa = o.primitive_offsets[sa]; pa < o.primitive_offsets[sa + 1]; ++pa)
    for (auto pb = o.primitive_offsets[sb]; pb < o.primitive_offsets[sb + 1]; ++pb)
      for (auto pc = x.primitive_offsets[sc]; pc < x.primitive_offsets[sc + 1]; ++pc) {
        const double alpha = o.exponents[pa], beta = o.exponents[pb], gamma = x.exponents[pc];
        if (lane == 0) {
          scalar::prepare_geometry(alpha, center_a, beta, center_b, gamma, center_c, A + B + C,
                                   geometry);
          ++primitive_work;
        }
        __syncwarp(mask);
        Math::prepare(geometry, cache, lane, lanes);
        __syncwarp(mask);
        for (unsigned i = lane; i < Math::components; i += lanes)
          if (cart_weights[i] != 0)
            Math::accumulate(
                i, alpha, beta, geometry, cache,
                cart_weights[i] * o.coefficients[pa] * o.coefficients[pb] * x.coefficients[pc],
                output);
        // The next primitive must not overwrite moments still being consumed.
        __syncwarp(mask);
      }
  for (unsigned i = 0; i < 6; ++i)
    for (unsigned delta = lanes / 2; delta; delta /= 2)
      output[i] += __shfl_down_sync(mask, output[i], delta, lanes);
  if (lane == 0) {
    const runtime::cuda_gaussian_products::Factor factors[3]{{o, oa}, {o, ob}, {x, oc}};
    runtime::cuda_gaussian_products::scatter(factors, Math::finish(output), 1.0, gradient);
    if (counters) {
      atomicAdd(counters + 1, 1ULL);
      atomicAdd(counters + 3, primitive_work);
      atomicAdd(counters + 4, primitive_work * component_work);
    }
  }
}

template <unsigned A, unsigned B, unsigned C, unsigned Variant>
cudaError_t launch(DfShellBasisView o, DfShellBasisView x, const double* positions,
                   std::size_t begin, std::size_t count, const double* weights, double* gradient,
                   unsigned long long* counters, cudaStream_t stream) {
  if (!o.count[A] || !o.count[B] || !x.count[C]) return cudaSuccess;
  const auto maximum = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (o.count[A] > maximum / o.count[B] || o.count[A] * o.count[B] > maximum / x.count[C])
    return cudaErrorInvalidValue;
  using Schedule = generated::Schedule<A, B, C, Variant>;
  const auto tasks = o.count[A] * o.count[B] * x.count[C];
  if (tasks)
    shell_panel<A, B, C, Variant>
        <<<static_cast<unsigned>((tasks + Schedule::groups - 1) / Schedule::groups),
           Schedule::lanes * Schedule::groups, 0, stream>>>(o, x, positions, begin, count, weights,
                                                            gradient, counters);
  return cudaPeekAtLastError();
}
}  // namespace

cudaError_t launch_df_shell_derivative_panel(DfShellBasisView o, DfShellBasisView x,
                                             const double* positions, std::size_t begin,
                                             std::size_t count, const double* weights,
                                             double* gradient, unsigned long long* counters,
                                             cudaStream_t stream, bool full_domain,
                                             unsigned variant) {
  if (variant > 2) return cudaErrorInvalidValue;
  cudaError_t status = cudaSuccess;
  generated::for_each_class([&]<unsigned A, unsigned B, unsigned C>() {
    if (status != cudaSuccess || (!full_domain && (A > 1 || B > 1 || C > 1 || A + B + C == 0)))
      return;
    if (variant == 0)
      status =
          launch<A, B, C, 0>(o, x, positions, begin, count, weights, gradient, counters, stream);
    else if (variant == 1)
      status =
          launch<A, B, C, 1>(o, x, positions, begin, count, weights, gradient, counters, stream);
    else
      status =
          launch<A, B, C, 2>(o, x, positions, begin, count, weights, gradient, counters, stream);
  });
  return status;
}
}  // namespace vibeqc::scf
