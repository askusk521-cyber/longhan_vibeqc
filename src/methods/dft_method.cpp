#include "methods/dft_method.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/cuda_xc.hpp"
#include "dft/grid.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/mean_field.hpp"
#include "scf/types.hpp"

namespace vibeqc::methods::detail {
namespace {

bool is_uks(vibeqc_method method) noexcept {
  return method == VIBEQC_METHOD_LDA_UKS || method == VIBEQC_METHOD_PBE_UKS;
}

bool is_supported_dft(vibeqc_method method) noexcept {
  return method == VIBEQC_METHOD_LDA_RKS || method == VIBEQC_METHOD_PBE_RKS || is_uks(method);
}

bool field_present(const vibeqc_method_descriptor& descriptor, std::size_t offset,
                   std::size_t width) noexcept {
  return descriptor.struct_size >= offset && descriptor.struct_size - offset >= width;
}

bool valid_coordinates(const std::vector<double>& coordinates, std::size_t atom_count) {
  return coordinates.size() == 3 * atom_count &&
         std::all_of(coordinates.begin(), coordinates.end(),
                     [](double value) { return std::isfinite(value); });
}

std::vector<double> coordinates_of(const core::System& system) {
  std::vector<double> result;
  result.reserve(3 * system.atoms.size());
  for (const auto& atom : system.atoms)
    result.insert(result.end(), atom.position.begin(), atom.position.end());
  return result;
}

void apply_coordinates(core::System& system, const std::vector<double>& coordinates) {
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
    std::copy_n(coordinates.begin() + static_cast<std::ptrdiff_t>(3 * atom), 3,
                system.atoms[atom].position.begin());
}

vibeqc_status exception_status() {
  try {
    throw;
  } catch (const MethodError& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument&) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    return VIBEQC_STATUS_INTERNAL_ERROR;
  }
}

scf::ScfOptions dft_options(const vibeqc_method_descriptor& descriptor,
                            vibeqc_backend requested_backend) {
  if (!std::isfinite(descriptor.energy_tolerance) || !std::isfinite(descriptor.density_tolerance) ||
      !std::isfinite(descriptor.screening_tolerance))
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "DFT tolerances must be finite");
  scf::ScfOptions options;
  options.max_iterations = descriptor.max_iterations == 0 ? 100 : descriptor.max_iterations;
  options.diis_history = descriptor.diis_history == 0 ? 8 : descriptor.diis_history;
  options.energy_tolerance =
      descriptor.energy_tolerance > 0.0 ? descriptor.energy_tolerance : 1.0e-10;
  options.density_tolerance =
      descriptor.density_tolerance > 0.0 ? descriptor.density_tolerance : 1.0e-8;
  options.screening_tolerance =
      descriptor.screening_tolerance > 0.0 ? descriptor.screening_tolerance : 1.0e-12;
  if (field_present(descriptor, offsetof(vibeqc_method_descriptor, density_fitting_mode),
                    sizeof(descriptor.density_fitting_mode))) {
    const auto mode = descriptor.density_fitting_mode;
    if (mode != VIBEQC_DENSITY_FITTING_NONE && mode != VIBEQC_DENSITY_FITTING_CPU_REFERENCE &&
        mode != VIBEQC_DENSITY_FITTING_CUDA && mode != VIBEQC_DENSITY_FITTING_AUTO)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown density-fitting execution mode");
    if (mode != VIBEQC_DENSITY_FITTING_NONE)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        "DFT energy methods support conventional Coulomb only");
  }
  if (field_present(descriptor, offsetof(vibeqc_method_descriptor, density_fitting_auxiliary_basis),
                    sizeof(descriptor.density_fitting_auxiliary_basis)) &&
      descriptor.density_fitting_auxiliary_basis != nullptr)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "DFT energy methods do not accept an unused auxiliary basis");
  if (field_present(descriptor, offsetof(vibeqc_method_descriptor, precision_mode),
                    sizeof(descriptor.precision_mode))) {
    if (descriptor.precision_mode != VIBEQC_PRECISION_FP64 &&
        descriptor.precision_mode != VIBEQC_PRECISION_AUTO)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown floating-point precision mode");
    if (descriptor.precision_mode == VIBEQC_PRECISION_AUTO)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        "DFT energy methods support explicit FP64 precision only");
  }

  scf::FockBuildSpec fock;
  fock.spin = is_uks(descriptor.method) ? scf::FockSpin::Unrestricted : scf::FockSpin::Restricted;
  fock.derivative_order = 0;
  fock.exchange.present = false;
  const auto backend =
      requested_backend == VIBEQC_BACKEND_CUDA ? scf::FockBackend::Cuda : scf::FockBackend::Cpu;
  options.resolved_fock_build = scf::resolve_fock_build(fock, backend, options.screening_tolerance);
  options.compute_forces = false;
  return options;
}

Result adapt_result(scf::ScfResult native, vibeqc_backend backend) {
  Result result;
  result.energy = native.energy;
  result.convergence.iterations = native.iterations;
  result.convergence.energy_change = native.energy_change;
  result.convergence.residual_rms = native.density_rms;
  result.physical_residual_rms = native.physical_residual_rms;
  result.convergence.converged = native.converged;
  result.executed_backend = backend;
  result.fock_builds = native.fock_builds;
  return result;
}

class DftPreparedCalculation final : public PreparedCalculation {
 public:
  DftPreparedCalculation(Capabilities capabilities, core::System system, vibeqc_method method,
                         scf::ScfOptions options, const core::ContextState& context)
      : capabilities_(capabilities),
        system_(std::move(system)),
        method_(method),
        options_(std::move(options)),
        backend_(context.requested_backend),
        fock_(system_, nullptr, *options_.resolved_fock_build, context.device_id,
              options_.density_fitting_memory_budget_bytes),
        basis_(system_),
        grid_(system_),
        cuda_xc_(backend_ == VIBEQC_BACKEND_CUDA
                     ? std::make_unique<dft::PreparedCudaXcPlan>(
                           basis_, context.device_id, context.compute_capability_major,
                           context.compute_capability_minor,
                           method_ == VIBEQC_METHOD_PBE_RKS || method_ == VIBEQC_METHOD_PBE_UKS)
                     : nullptr) {}

  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const Capabilities& capabilities() const noexcept override { return capabilities_; }

  scf::ScfResult solve(bool compute_forces, const std::vector<double>* initial_density = nullptr) {
    const bool pbe = method_ == VIBEQC_METHOD_PBE_RKS || method_ == VIBEQC_METHOD_PBE_UKS;
    const bool uks = is_uks(method_);
    const char* method_name = pbe ? "PBE" : "LDA";
    if (compute_forces) {
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        std::string(method_name) + (uks ? " UKS" : " RKS") +
                            " nuclear gradients are tracked separately in issue #163");
    }
    if (backend_ == VIBEQC_BACKEND_CUDA) {
      if (method_ == VIBEQC_METHOD_PBE_RKS)
        return scf::run_pbe_rks_cuda(fock_, basis_, grid_, *cuda_xc_, options_, initial_density);
      if (method_ == VIBEQC_METHOD_LDA_UKS)
        return scf::run_lda_uks_cuda(fock_, basis_, grid_, *cuda_xc_, options_, initial_density);
      if (method_ == VIBEQC_METHOD_PBE_UKS)
        return scf::run_pbe_uks_cuda(fock_, basis_, grid_, *cuda_xc_, options_, initial_density);
      return scf::run_lda_rks_cuda(fock_, basis_, grid_, *cuda_xc_, options_, initial_density);
    }
    if (method_ == VIBEQC_METHOD_PBE_RKS)
      return scf::run_pbe_rks(fock_, basis_, grid_, options_, initial_density);
    if (method_ == VIBEQC_METHOD_LDA_UKS)
      return scf::run_lda_uks(fock_, basis_, grid_, options_, initial_density);
    if (method_ == VIBEQC_METHOD_PBE_UKS)
      return scf::run_pbe_uks(fock_, basis_, grid_, options_, initial_density);
    return scf::run_lda_rks(fock_, basis_, grid_, options_, initial_density);
  }

  Result execute(bool compute_forces) override {
    return adapt_result(solve(compute_forces), backend_);
  }

 private:
  Capabilities capabilities_;
  core::System system_;
  vibeqc_method method_{};
  scf::ScfOptions options_;
  vibeqc_backend backend_{VIBEQC_BACKEND_CPU_REFERENCE};
  scf::PreparedFockPlan fock_;
  dft::AoBasis basis_;
  dft::MolecularGrid grid_;
  std::unique_ptr<dft::PreparedCudaXcPlan> cuda_xc_;
};

class DftPreparedBatch final : public PreparedBatch {
 public:
  DftPreparedBatch(Capabilities capabilities, core::ContextState& context,
                   std::vector<core::System> systems, vibeqc_method method, scf::ScfOptions options,
                   vibeqc_batch_flags flags)
      : capabilities_(capabilities),
        context_(&context),
        method_(method),
        options_(std::move(options)),
        warm_starts_enabled_((flags & VIBEQC_BATCH_ENABLE_WARM_STARTS) != 0) {
    constexpr vibeqc_batch_flags supported = VIBEQC_BATCH_ENABLE_WARM_STARTS;
    if ((flags & ~supported) != 0)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unsupported DFT batch flag");
    items_.reserve(systems.size());
    std::map<std::tuple<std::size_t, int, unsigned, std::size_t>, std::size_t> buckets;
    for (auto& system : systems) {
      std::size_t primitives = 0;
      for (const auto& shell : system.shells) primitives += shell.primitives.size();
      const auto key = std::tuple{molecule::ao_count(system), system.electron_count,
                                  system.multiplicity, primitives};
      const auto [entry, inserted] = buckets.emplace(key, buckets.size());
      (void)inserted;
      Item item;
      item.system = std::move(system);
      item.bucket_id = entry->second;
      items_.push_back(std::move(item));
    }
  }

  std::size_t size() const noexcept override { return items_.size(); }

  std::vector<BatchItemResult> execute(const Coordinates& coordinates,
                                       bool compute_forces = true) override {
    if (!coordinates.empty() && coordinates.size() != items_.size())
      throw std::invalid_argument("DFT batch coordinate list does not match system count");
    std::vector<BatchItemResult> results(items_.size());
    for (std::size_t index = 0; index < items_.size(); ++index) {
      auto& item = items_[index];
      auto& output = results[index];
      output.bucket_id = item.bucket_id;
      output.calculation.executed_backend = context_->requested_backend;
      if (compute_forces) {
        output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
        continue;
      }
      core::System execution_system = item.system;
      if (!coordinates.empty() && coordinates[index]) {
        if (!valid_coordinates(*coordinates[index], execution_system.atoms.size())) {
          output.status = VIBEQC_STATUS_INVALID_ARGUMENT;
          continue;
        }
        apply_coordinates(execution_system, *coordinates[index]);
      }
      const auto current_coordinates = coordinates_of(execution_system);
      if (item.warm && item.warm->coordinates != current_coordinates) item.warm.reset();
      const bool use_warm = warm_starts_enabled_ && item.warm.has_value();
      output.warm_start_used = use_warm;
      try {
        if (!item.calculation || item.prepared_coordinates != current_coordinates) {
          item.calculation = std::make_unique<DftPreparedCalculation>(
              capabilities_, execution_system, method_, options_, *context_);
          item.prepared_coordinates = current_coordinates;
        }
        const auto evaluate = [&](const std::vector<double>* seed) {
          return item.calculation->solve(false, seed);
        };
        scf::ScfResult native = evaluate(use_warm ? &item.warm->density : nullptr);
        if (use_warm && !native.converged) {
          output.warm_start_fallback = true;
          native = evaluate(nullptr);
        }
        output.status = native.converged ? VIBEQC_STATUS_SUCCESS : VIBEQC_STATUS_NOT_CONVERGED;
        if (output.status == VIBEQC_STATUS_SUCCESS && warm_starts_enabled_ &&
            warm_start_updates_enabled_)
          retain(item, current_coordinates, native);
        output.calculation = adapt_result(std::move(native), context_->requested_backend);
      } catch (...) {
        if (use_warm) {
          try {
            output.warm_start_fallback = true;
            auto native = item.calculation->solve(false, nullptr);
            output.status = native.converged ? VIBEQC_STATUS_SUCCESS : VIBEQC_STATUS_NOT_CONVERGED;
            if (output.status == VIBEQC_STATUS_SUCCESS && warm_starts_enabled_ &&
                warm_start_updates_enabled_)
              retain(item, current_coordinates, native);
            output.calculation = adapt_result(std::move(native), context_->requested_backend);
            continue;
          } catch (...) {
            output.status = exception_status();
          }
        } else {
          output.status = exception_status();
        }
      }
    }
    return results;
  }

  void clear_warm_starts() override {
    for (auto& item : items_) item.warm.reset();
  }

  std::size_t warm_density_size(std::size_t index) const override {
    const auto& system = items_.at(index).system;
    const auto n = molecule::ao_count(system);
    const std::size_t spins = is_uks(method_) ? 2 : 1;
    if (!n || n > std::numeric_limits<std::size_t>::max() / n / spins)
      throw std::invalid_argument("DFT warm density dimensions overflow");
    return spins * n * n;
  }

  const std::optional<scf::HfWarmState>& warm_state(std::size_t index) const override {
    return items_.at(index).warm;
  }

  void restore_warm_states(std::vector<std::optional<scf::HfWarmState>> states) override {
    if (!warm_starts_enabled_ || states.size() != items_.size())
      throw std::invalid_argument("checkpoint restore requires a matching warm-enabled DFT batch");
    for (std::size_t index = 0; index < states.size(); ++index) {
      if (!states[index]) continue;
      const auto& state = *states[index];
      if (state.density.size() != warm_density_size(index) ||
          !valid_coordinates(state.coordinates, items_[index].system.atoms.size()) ||
          state.iterations < 0 || !std::isfinite(state.energy) ||
          !std::isfinite(state.energy_change) || !std::isfinite(state.density_rms) ||
          state.density_rms < 0.0)
        throw std::invalid_argument("invalid DFT checkpoint state dimensions or diagnostics");
      auto source = items_[index].system;
      apply_coordinates(source, state.coordinates);
      scf::validate_hf_warm_density(source, method_, state.density);
    }
    for (std::size_t index = 0; index < states.size(); ++index)
      if (states[index]) items_[index].warm.swap(states[index]);
  }

  void set_warm_start_updates(bool enabled) override { warm_start_updates_enabled_ = enabled; }

  std::optional<std::vector<DirectShellClassProfileEntry>> last_direct_shell_class_profile()
      const override {
    return std::nullopt;
  }
  std::optional<DirectPppsQueueProfile> last_direct_ppps_queue_profile() const override {
    return std::nullopt;
  }
  std::vector<EigensolverDiagnostic> last_eigensolver_diagnostics() const override { return {}; }
  std::vector<scf::CudaDensityFittingMetricDiagnostic> last_density_fitting_metric_diagnostics()
      const override {
    return {};
  }
  std::vector<InactiveEigensolverProfileEntry> last_inactive_eigensolver_profile() const override {
    return {};
  }

 private:
  struct Item {
    core::System system;
    std::size_t bucket_id{};
    std::vector<double> prepared_coordinates;
    std::unique_ptr<DftPreparedCalculation> calculation;
    std::optional<scf::HfWarmState> warm;
  };

  static void retain(Item& item, const std::vector<double>& coordinates,
                     const scf::ScfResult& result) {
    scf::HfWarmState state;
    state.density = result.density;
    state.coordinates = coordinates;
    state.energy = result.energy;
    state.energy_change = result.energy_change;
    state.density_rms = result.density_rms;
    state.iterations = static_cast<int>(result.iterations);
    item.warm = std::move(state);
  }

  Capabilities capabilities_;
  core::ContextState* context_{};
  vibeqc_method method_{};
  scf::ScfOptions options_;
  bool warm_starts_enabled_{};
  bool warm_start_updates_enabled_{true};
  std::vector<Item> items_;
};

}  // namespace

vibeqc_status validate_dft_system(vibeqc_method method, const core::System& system,
                                  std::string& detail) {
  if (!is_supported_dft(method)) {
    detail = "requested DFT method is reserved but not implemented";
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  const char* functional =
      method == VIBEQC_METHOD_PBE_RKS || method == VIBEQC_METHOD_PBE_UKS ? "PBE" : "LDA";
  if (!is_uks(method)) {
    if (system.electron_count > 0 && system.electron_count % 2 == 0 && system.multiplicity == 1)
      return VIBEQC_STATUS_SUCCESS;
    detail = std::string(functional) +
             " RKS requires a positive even electron count and spin multiplicity 1";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const int spin_excess = static_cast<int>(system.multiplicity) - 1;
  if (system.electron_count > 0 && spin_excess >= 0 && spin_excess <= system.electron_count &&
      (system.electron_count - spin_excess) % 2 == 0)
    return VIBEQC_STATUS_SUCCESS;
  detail = std::string(functional) +
           " UKS requires electron count and multiplicity to define integer nonnegative spin "
           "occupations";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

std::unique_ptr<PreparedCalculation> prepare_dft_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const vibeqc_method_descriptor& descriptor) {
  if (context.requested_backend != VIBEQC_BACKEND_CPU_REFERENCE &&
      context.requested_backend != VIBEQC_BACKEND_CUDA)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown DFT execution backend");
  if (!is_supported_dft(descriptor.method))
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "requested DFT method is reserved but not implemented");
  return std::make_unique<DftPreparedCalculation>(
      capabilities, system, descriptor.method, dft_options(descriptor, context.requested_backend),
      context);
}

std::unique_ptr<PreparedBatch> prepare_dft_batch(const Capabilities& capabilities,
                                                 core::ContextState& context,
                                                 std::vector<core::System> systems,
                                                 const vibeqc_method_descriptor& descriptor,
                                                 vibeqc_batch_flags flags) {
  if (systems.empty())
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "DFT batch requires at least one system");
  if (context.requested_backend != VIBEQC_BACKEND_CPU_REFERENCE &&
      context.requested_backend != VIBEQC_BACKEND_CUDA)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown DFT execution backend");
  return std::make_unique<DftPreparedBatch>(
      capabilities, context, std::move(systems), descriptor.method,
      dft_options(descriptor, context.requested_backend), flags);
}

}  // namespace vibeqc::methods::detail
