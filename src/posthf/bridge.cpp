/** Private development bridge for the HF/post-HF interface (schema 1).
 * These symbols are deliberately separate from method registration and the
 * public versioned C API. A source owns a deep copy of the native system.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>

#include "api/handles.hpp"
#include "posthf/capacity.hpp"
#include "posthf/mp2_energy.hpp"
#include "posthf/raw_source.hpp"
#include "scf/density_fitting.hpp"
#include "scf/mean_field.hpp"
#include "vibeqc/vibeqc.h"

using vibeqc::posthf::RawSource;
namespace {
template <class F>
int guarded(char* error, std::size_t size, F&& fn) noexcept {
  try {
    fn();
    return 0;
  } catch (const std::exception& ex) {
    if (error && size) std::snprintf(error, size, "%s", ex.what());
    return 1;
  } catch (...) {
    if (error && size) std::snprintf(error, size, "unknown post-HF failure");
    return 1;
  }
}
vibeqc::scf::PhysicalReference supplied_reference(const RawSource& source, const double* arrays,
                                                  std::size_t elements, double energy) {
  const auto n = source.nbf();
  const auto count = vibeqc::posthf::checked_add(
      vibeqc::posthf::checked_mul(5, vibeqc::posthf::checked_mul(n, n)), n);
  if (!arrays || count != elements || !std::isfinite(energy) ||
      source.orbital().multiplicity != 1 || source.orbital().electron_count % 2)
    throw std::invalid_argument("invalid supplied reference");
  vibeqc::scf::PhysicalReference ref;
  ref.nbf = n;
  ref.nocc = source.orbital().electron_count / 2;
  ref.energy = energy;
  const double* cursor = arrays;
  for (auto* m : {&ref.overlap, &ref.hcore, &ref.fock, &ref.coefficients, &ref.density}) {
    m->assign(cursor, cursor + n * n);
    cursor += n * n;
  }
  ref.orbital_energies.assign(cursor, cursor + n);
  vibeqc::scf::validate_physical_reference(ref);
  return ref;
}
}  // namespace
extern "C" {
/** Reuse the SCF metric threshold and square symmetric inverse convention. */
VIBEQC_API int vibeqc_posthf_metric_v1(const double* metric, std::size_t n, double threshold,
                                       double* inverse_root, double* diagnostics, char* error,
                                       std::size_t size) {
  return guarded(error, size, [&] {
    if (!metric || !inverse_root || !diagnostics || !n || n > SIZE_MAX / n)
      throw std::invalid_argument("invalid post-HF metric");
    const auto factor = vibeqc::scf::factor_density_fitting_metric(
        std::vector<double>(metric, metric + n * n), n, threshold);
    std::copy(factor.inverse_square_root.begin(), factor.inverse_square_root.end(), inverse_root);
    diagnostics[0] = factor.effective_rank;
    diagnostics[1] = factor.absolute_threshold;
    diagnostics[2] = factor.condition_number;
  });
}
VIBEQC_API int vibeqc_posthf_source_create_v1(const vibeqc_system* orbital,
                                              const vibeqc_system* auxiliary, void** out,
                                              char* error, std::size_t size) {
  return guarded(error, size, [&] {
    if (!out) throw std::invalid_argument("null source output");
    *out = nullptr;
    if (!orbital) throw std::invalid_argument("null orbital system");
    *out = new RawSource(orbital->data, auxiliary ? &auxiliary->data : nullptr);
  });
}
VIBEQC_API void vibeqc_posthf_source_destroy_v1(void* source) {
  delete static_cast<RawSource*>(source);
}
VIBEQC_API int vibeqc_posthf_source_read_v1(void* source, int op, const std::size_t* begin,
                                            const std::size_t* count, double* out,
                                            std::size_t elements, char* error, std::size_t size) {
  return guarded(error, size, [&] {
    if (!source || !begin || !count) throw std::invalid_argument("null raw source request");
    std::array<std::size_t, 4> b{}, c{};
    std::copy_n(begin, 4, b.begin());
    std::copy_n(count, 4, c.begin());
    static_cast<RawSource*>(source)->read(static_cast<RawSource::Operator>(op), b, c, out,
                                          elements);
  });
}
/** Run the existing RHF implementation, exporting only independently owned
 * density and scalar diagnostics. Snapshot canonicalization is a separate
 * checked operation; an SCF failure never returns a usable density.
 */
VIBEQC_API int vibeqc_posthf_rhf_density_v1(void* source, int backend, int device,
                                            unsigned max_iterations, double tolerance, int df,
                                            double metric_threshold, double* density,
                                            std::size_t elements, double* scalars, char* error,
                                            std::size_t size) {
  return guarded(error, size, [&] {
    if (!source || !density || !scalars || (backend != 0 && backend != 1) || (df != 0 && df != 1))
      throw std::invalid_argument("invalid RHF export request");
    const auto& raw = *static_cast<RawSource*>(source);
    if (raw.orbital().multiplicity != 1 || raw.orbital().electron_count % 2)
      throw std::invalid_argument("snapshot export supports closed-shell RHF only");
    if (elements != raw.nbf() * raw.nbf())
      throw std::invalid_argument("density output size mismatch");
    vibeqc::scf::ScfOptions options;
    options.max_iterations = max_iterations;
    options.energy_tolerance = tolerance;
    options.density_tolerance = tolerance;
    options.screening_tolerance = 0;
    options.density_fitting_relative_threshold = metric_threshold;
    vibeqc::scf::ScfResult result;
    if (df) {
      result = backend
                   ? vibeqc::scf::run_rhf_density_fitting_cuda(raw.orbital(), raw.auxiliary(),
                                                               options, device)
                   : vibeqc::scf::run_rhf_density_fitting(raw.orbital(), raw.auxiliary(), options);
    } else {
      result = backend ? vibeqc::scf::run_rhf_cuda(raw.orbital(), options, device)
                       : vibeqc::scf::run_rhf(raw.orbital(), options);
    }
    if (!result.converged || result.density.size() != elements)
      throw std::runtime_error("HF failed or did not converge; no reference exported");
    std::copy(result.density.begin(), result.density.end(), density);
    scalars[0] = result.energy;
    scalars[1] = result.energy_change;
    scalars[2] = result.density_rms;
    scalars[3] = result.iterations;
  });
}

/** Private owned reference export for native consumer validation. Buffer order
 * is S,h,F,C,D
 * (each row-major n*n), then epsilon[n]. It uses the same
 * reference-only HF driver as the method
 * adapter, never the 12-AO exporter. */
VIBEQC_API int vibeqc_posthf_reference_v1(void* source, int backend, int device,
                                          unsigned iterations, double tolerance, std::size_t budget,
                                          double* arrays, std::size_t elements, double* scalars,
                                          char* error, std::size_t size) {
  return guarded(error, size, [&] {
    if (!source || !arrays || !scalars || (backend != 0 && backend != 1) || !iterations ||
        !std::isfinite(tolerance) || tolerance <= 0)
      throw std::invalid_argument("invalid bounded reference request");
    const auto& raw = *static_cast<RawSource*>(source);
    const auto n = raw.nbf();
    const auto expected = vibeqc::posthf::checked_add(
        vibeqc::posthf::checked_mul(5, vibeqc::posthf::checked_mul(n, n)), n);
    if (elements != expected) throw std::invalid_argument("reference export size mismatch");
    vibeqc::scf::ScfOptions options;
    options.max_iterations = iterations;
    options.energy_tolerance = tolerance;
    options.density_tolerance = tolerance;
    options.screening_tolerance = 0;
    options.export_physical_reference = true;
    options.reference_memory_budget_bytes = budget;
    const auto result = backend ? vibeqc::scf::run_rhf_cuda(raw.orbital(), options, device)
                                : vibeqc::scf::run_rhf(raw.orbital(), options);
    if (!result.converged || !result.reference)
      throw std::runtime_error("HF failed or bounded reference export is unsupported by backend");
    const auto& r = *result.reference;
    auto* cursor = arrays;
    for (const auto* a :
         {&r.overlap, &r.hcore, &r.fock, &r.coefficients, &r.density, &r.orbital_energies})
      cursor = std::copy(a->begin(), a->end(), cursor);
    scalars[0] = r.energy;
    scalars[1] = result.iterations;
    scalars[2] = r.commutator_residual;
    scalars[3] = r.canonical_density_drift;
    scalars[4] = r.eigen_residual;
    scalars[5] = static_cast<double>(r.numeric_capacity_bytes);
  });
}

/** Private identical-orbital validation entry to the production native consumer.
 * It is not a
 * public MP2 method or a substitute for the HF-to-MP2 route. */
VIBEQC_API int vibeqc_posthf_mp2_energy_v1(void* source, int backend, int device,
                                           const double* arrays, std::size_t elements,
                                           double hf_energy, std::size_t budget, double threshold,
                                           unsigned tile, double* out, char* error,
                                           std::size_t size) {
  return guarded(error, size, [&] {
    if (!source || !out || (backend != 0 && backend != 1))
      throw std::invalid_argument("invalid MP2 validation request");
    const auto& raw = *static_cast<RawSource*>(source);
    const auto ref = supplied_reference(raw, arrays, elements, hf_energy);
    const auto result =
        vibeqc::mp2::conventional_energy(ref, raw, budget, threshold, tile, backend == 1, device);
    out[0] = result.opposite_spin;
    out[1] = result.same_spin;
    out[2] = result.minimum_denominator;
    out[3] = result.numeric_capacity_bytes;
    out[4] = result.tiles;
  });
}
/** Explicit CG10 slot order for direct native CUDA/CPU layout verification. */
VIBEQC_API int vibeqc_posthf_mo_block_v1(void* source, int backend, int device,
                                         const double* arrays, std::size_t elements,
                                         double hf_energy, const std::size_t* shape,
                                         const std::size_t* slots, std::size_t budget, double* out,
                                         std::size_t output_elements, char* error,
                                         std::size_t size) {
  return guarded(error, size, [&] {
    if (!source || !out || !shape || !slots || (backend != 0 && backend != 1))
      throw std::invalid_argument("invalid MO validation request");
    const auto& raw = *static_cast<RawSource*>(source);
    const auto ref = supplied_reference(raw, arrays, elements, hf_energy);
    vibeqc::posthf::MOSlots request;
    std::size_t count = 1;
    for (unsigned k = 0; k < 4; ++k) {
      if (!shape[k] || shape[k] > ref.nbf)
        throw std::invalid_argument("invalid MO validation shape");
      request[k].assign(slots, slots + shape[k]);
      slots += shape[k];
      count = vibeqc::posthf::checked_mul(count, shape[k]);
    }
    if (count != output_elements) throw std::invalid_argument("MO validation output size mismatch");
    const vibeqc::posthf::NativeBlockProvider provider(raw, ref, budget);
    const auto values = provider.get(request, backend == 1, device);
    std::copy(values.begin(), values.end(), out);
  });
}
}
