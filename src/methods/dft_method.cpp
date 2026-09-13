#include "methods/dft_method.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/mean_field.hpp"
#include "scf/types.hpp"

#if VIBEQC_HAS_CUDA
#include "dft/cuda_ks.hpp"
#endif

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

scf::ScfOptions dft_options(const vibeqc_method_descriptor& descriptor, vibeqc_backend backend) {
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
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "DFT supports conventional Coulomb only");
  }
  if (field_present(descriptor, offsetof(vibeqc_method_descriptor, density_fitting_auxiliary_basis),
                    sizeof(descriptor.density_fitting_auxiliary_basis)) &&
      descriptor.density_fitting_auxiliary_basis != nullptr)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "DFT does not accept an unused auxiliary basis");
  if (field_present(descriptor, offsetof(vibeqc_method_descriptor, precision_mode),
                    sizeof(descriptor.precision_mode))) {
    if (descriptor.precision_mode != VIBEQC_PRECISION_FP64 &&
        descriptor.precision_mode != VIBEQC_PRECISION_AUTO)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown floating-point precision mode");
    if (descriptor.precision_mode == VIBEQC_PRECISION_AUTO)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "DFT supports explicit FP64 precision only");
  }

  scf::FockBuildSpec fock;
  fock.spin =
      (descriptor.method == VIBEQC_METHOD_LDA_UKS || descriptor.method == VIBEQC_METHOD_PBE_UKS)
          ? scf::FockSpin::Unrestricted
          : scf::FockSpin::Restricted;
  fock.derivative_order = 0;
  fock.exchange.present = false;
  options.resolved_fock_build = scf::resolve_fock_build(
      fock, backend == VIBEQC_BACKEND_CUDA ? scf::FockBackend::Cuda : scf::FockBackend::Cpu,
      options.screening_tolerance);
  options.compute_forces = false;
  return options;
}

Result adapt_result(const scf::ScfResult& native, vibeqc_backend backend) {
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

class KsPreparedCalculation final : public PreparedCalculation {
 public:
  KsPreparedCalculation(Capabilities capabilities, core::System system, vibeqc_method method,
                        scf::ScfOptions options, vibeqc_backend backend, int device)
      : capabilities_(capabilities),
        system_(std::move(system)),
        method_(method),
        options_(std::move(options)),
        backend_(backend),
        fock_(system_, nullptr, *options_.resolved_fock_build, device),
        basis_(system_),
        grid_(system_) {
#if VIBEQC_HAS_CUDA
    if (backend_ == VIBEQC_BACKEND_CUDA)
      cuda_ = std::make_unique<dft::CudaKsPlan>(
          fock_, basis_, grid_, options_,
          method_ == VIBEQC_METHOD_PBE_RKS || method_ == VIBEQC_METHOD_PBE_UKS);
#endif
  }

  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const Capabilities& capabilities() const noexcept override { return capabilities_; }
  const core::System& system() const noexcept { return system_; }

  /** Explicit output/rebuild export. Ordinary CUDA replays keep this on device. */
  std::vector<double> warm_density() {
#if VIBEQC_HAS_CUDA
    if (cuda_) return cuda_->warm_density();
#endif
    return warm_;
  }

  void clear_warm_start() noexcept {
    warm_.clear();
#if VIBEQC_HAS_CUDA
    if (cuda_) cuda_->clear_warm_start();
#endif
  }

#if VIBEQC_HAS_CUDA
  dft::CudaKsPlan* cuda_plan() noexcept { return cuda_.get(); }
#endif

  Result execute(bool compute_forces) override {
    const char* method_name =
        (method_ == VIBEQC_METHOD_PBE_RKS || method_ == VIBEQC_METHOD_PBE_UKS) ? "PBE" : "LDA";
    if (compute_forces) {
      throw MethodError(
          VIBEQC_STATUS_NOT_IMPLEMENTED,
          std::string(method_name) + " KS nuclear gradients are tracked separately in issue #163");
    }
    return adapt_result(run(nullptr, true, true), backend_);
  }

  /** Single-system and native batch paths share the same scientific owner. */
  scf::ScfResult run(const std::vector<double>* initial_density, bool reuse_warm,
                     bool update_warm) {
#if VIBEQC_HAS_CUDA
    if (cuda_) {
      // Native iterations read only scalar diagnostics. The public energy
      // result does not require a final AO matrix download; warm D stays resident.
      cuda_->set_warm_start_updates(update_warm);
      auto native = cuda_->run(initial_density, reuse_warm, false);
      if (cuda_->failed())
        throw MethodError(VIBEQC_STATUS_NUMERICAL_FAILURE, "CUDA KS physical evaluation failed");
      return native;
    }
#endif
    const auto* seed =
        initial_density ? initial_density : (reuse_warm && !warm_.empty() ? &warm_ : nullptr);
    scf::ScfResult native;
    if (method_ == VIBEQC_METHOD_LDA_UKS || method_ == VIBEQC_METHOD_PBE_UKS)
      native = scf::run_uks(fock_, basis_, grid_, options_, method_ == VIBEQC_METHOD_PBE_UKS, seed);
    else if (method_ == VIBEQC_METHOD_PBE_RKS)
      native = scf::run_pbe_rks(fock_, basis_, grid_, options_, seed);
    else
      native = scf::run_lda_rks(fock_, basis_, grid_, options_, seed);
    // This owner has immutable model/geometry/spin identity. Only successful
    // executions may replace its compatible last-good density; DIIS is fresh.
    if (native.converged && update_warm) warm_ = std::move(native.density);
    return native;
  }

 private:
  Capabilities capabilities_;
  core::System system_;
  vibeqc_method method_{};
  scf::ScfOptions options_;
  vibeqc_backend backend_;
  scf::PreparedFockPlan fock_;
  dft::AoBasis basis_;
  dft::MolecularGrid grid_;
  std::vector<double> warm_;
#if VIBEQC_HAS_CUDA
  std::unique_ptr<dft::CudaKsPlan> cuda_;
#endif
};

std::vector<double> positions(const core::System& system) {
  std::vector<double> out;
  out.reserve(3 * system.atoms.size());
  for (const auto& atom : system.atoms)
    out.insert(out.end(), atom.position.begin(), atom.position.end());
  return out;
}

bool valid_positions(const std::vector<double>& coordinates, const core::System& system) {
  return coordinates.size() == 3 * system.atoms.size() &&
         std::all_of(coordinates.begin(), coordinates.end(),
                     [](double value) { return std::isfinite(value); });
}

void set_positions(core::System& system, const std::vector<double>& coordinates) {
  for (std::size_t i = 0; i < system.atoms.size(); ++i)
    std::copy_n(coordinates.begin() + 3 * i, 3, system.atoms[i].position.begin());
}

vibeqc_status item_exception_status() {
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

/** Independent native KS owners, with ordinary-stream round-robin CUDA work.
 * Geometry is rebuilt per item, while model/basis/charge/spin remain immutable.
 * No HF graph or Python calculation loop participates in this schedule. */
class KsPreparedBatch final : public PreparedBatch {
 public:
  KsPreparedBatch(Capabilities capabilities, std::vector<core::System> systems,
                  vibeqc_method method, scf::ScfOptions options, vibeqc_backend backend, int device,
                  bool warm_enabled)
      : capabilities_(capabilities),
        systems_(std::move(systems)),
        method_(method),
        options_(std::move(options)),
        backend_(backend),
        device_(device),
        warm_enabled_(warm_enabled),
        items_(systems_.size()) {
    for (std::size_t i = 0; i < size(); ++i) items_[i].plan = make_plan(systems_[i]);
  }

  std::size_t size() const noexcept override { return systems_.size(); }

  std::vector<BatchItemResult> execute(const Coordinates& coordinates,
                                       bool compute_forces) override {
    if (compute_forces)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        "KS nuclear gradients are tracked separately in issue #163");
    if (!coordinates.empty() && coordinates.size() != size())
      throw std::invalid_argument("KS batch coordinates do not match system count");
    std::vector<BatchItemResult> results(size());
    std::vector<bool> ready(size(), false);
    // Allocate source-geometry metadata before launching any item. The success
    // path can then publish its last-good identity without a coordinate copy.
    std::vector<scf::HfWarmState> candidates(size());
    for (std::size_t i = 0; i < size(); ++i) {
      auto& result = results[i];
      result.bucket_id = i;  // One ordinary stream/owner per stable input slot.
      result.calculation.executed_backend = backend_;
      result.calculation.energy = std::numeric_limits<double>::quiet_NaN();
      try {
        auto target = systems_[i];
        if (!coordinates.empty() && coordinates[i]) {
          if (!valid_positions(*coordinates[i], target))
            throw std::invalid_argument("invalid KS batch item coordinates");
          set_positions(target, *coordinates[i]);
        }
        auto& item = items_[i];
        candidates[i].coordinates = positions(target);
        if (!item.plan || positions(item.plan->system()) != candidates[i].coordinates) {
          // Preserve the last GOOD seed before freeing its device owner. This
          // explicit rebuild download is never part of routine SCF iterations.
          materialize_warm(i);
          item.plan.reset();
          item.resident_warm = false;
          item.plan = make_plan(target);
        }
        result.warm_start_used = warm_enabled_ && item.warm.has_value();
        ready[i] = true;
      } catch (...) {
        result.status = item_exception_status();
      }
    }

    const auto finish = [&](std::size_t i, const scf::ScfResult& native) {
      auto& result = results[i];
      result.calculation = adapt_result(native, backend_);
      result.status = native.converged ? VIBEQC_STATUS_SUCCESS : VIBEQC_STATUS_NOT_CONVERGED;
      if (native.converged && warm_enabled_ && warm_updates_) {
        auto& state = candidates[i];
        state.energy = native.energy;
        state.energy_change = native.energy_change;
        state.density_rms = native.density_rms;
        state.iterations = native.iterations;
        items_[i].warm = std::move(state);
        items_[i].resident_warm = true;
      }
    };

    // A rejected/nonconverged warm solve gets one cold retry. CUDA retries
    // retain the same per-item scheduler; a failed neighbor never halts it.
    for (unsigned attempt = 0; attempt < 2; ++attempt) {
      std::vector<bool> running(size(), false);
      for (std::size_t i = 0; i < size(); ++i) {
        auto& result = results[i];
        if (!ready[i] ||
            (attempt && (!result.warm_start_used || result.status == VIBEQC_STATUS_SUCCESS)))
          continue;
        if (attempt) result.warm_start_fallback = true;
        auto& item = items_[i];
        const bool reuse = !attempt && result.warm_start_used;
        const auto* seed = reuse && !item.resident_warm ? &item.warm->density : nullptr;
        try {
#if VIBEQC_HAS_CUDA
          if (auto* cuda = item.plan->cuda_plan()) {
            cuda->set_warm_start_updates(warm_enabled_ && warm_updates_);
            cuda->begin(seed, reuse && item.resident_warm);
            running[i] = true;
            continue;
          }
#endif
          finish(i,
                 item.plan->run(seed, reuse && item.resident_warm, warm_enabled_ && warm_updates_));
        } catch (...) {
          result.status = item_exception_status();
        }
      }
#if VIBEQC_HAS_CUDA
      while (std::any_of(running.begin(), running.end(), [](bool value) { return value; })) {
        // Submit ALL active streams before synchronizing any scalar record.
        for (std::size_t i = 0; i < size(); ++i) {
          if (!running[i]) continue;
          try {
            items_[i].plan->cuda_plan()->enqueue_iteration();
          } catch (...) {
            results[i].status = item_exception_status();
            running[i] = false;
          }
        }
        for (std::size_t i = 0; i < size(); ++i) {
          if (!running[i]) continue;
          try {
            auto* cuda = items_[i].plan->cuda_plan();
            if (cuda->finish_iteration()) continue;
            running[i] = false;
            if (cuda->failed())
              throw MethodError(VIBEQC_STATUS_NUMERICAL_FAILURE,
                                "CUDA KS physical evaluation failed");
            finish(i, cuda->result(false));
          } catch (...) {
            results[i].status = item_exception_status();
            running[i] = false;
          }
        }
      }
#endif
    }
    return results;
  }

  void clear_warm_starts() override {
    for (auto& item : items_) {
      item.warm.reset();
      item.resident_warm = false;
      if (item.plan) item.plan->clear_warm_start();
    }
  }

  std::size_t warm_density_size(std::size_t index) const override {
    const auto n = molecule::ao_count(systems_.at(index));
    const std::size_t spins = is_uks(method_) ? 2 : 1;
    if (!n || n > std::numeric_limits<std::size_t>::max() / n / spins / sizeof(double))
      throw std::invalid_argument("KS warm density dimensions overflow");
    return spins * n * n;
  }

  const std::optional<scf::HfWarmState>& warm_state(std::size_t index) const override {
    materialize_warm(index);
    return items_.at(index).warm;
  }

  void restore_warm_states(std::vector<std::optional<scf::HfWarmState>> states) override {
    if (!warm_enabled_ || states.size() != size())
      throw std::invalid_argument("KS seed restore requires a matching warm-enabled batch");
    for (std::size_t i = 0; i < size(); ++i) {
      if (!states[i]) continue;
      const auto& state = *states[i];
      if (state.density.size() != warm_density_size(i) ||
          !valid_positions(state.coordinates, systems_[i]) || state.iterations < 0 ||
          !std::isfinite(state.energy) || !std::isfinite(state.energy_change) ||
          !std::isfinite(state.density_rms) || state.density_rms < 0)
        throw std::invalid_argument("invalid KS seed dimensions or diagnostics");
      auto source = systems_[i];
      set_positions(source, state.coordinates);
      // This common validation reads only source S and checks the shared
      // spin-density convention. It performs no HF Fock/energy evaluation.
      scf::validate_hf_warm_density(source, is_uks(method_) ? VIBEQC_METHOD_UHF : VIBEQC_METHOD_RHF,
                                    state.density);
    }
    // All source-metric validation precedes the no-throw commit. Missing
    // entries preserve neighbors, including their resident density ownership.
    for (std::size_t i = 0; i < size(); ++i) {
      if (!states[i]) continue;
      auto& item = items_[i];
      item.warm.swap(states[i]);
      item.resident_warm = false;
      if (item.plan) item.plan->clear_warm_start();
    }
  }

  void set_warm_start_updates(bool enabled) override { warm_updates_ = enabled; }

  // These profiles describe HF graph/provider layouts, not this method's
  // ordinary-stream schedule. Absence is explicit at the common interface.
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
    std::unique_ptr<KsPreparedCalculation> plan;
    // Density is materialized only for explicit output, import, or rebuilding
    // an owner. Empty density with resident_warm=true is a valid lazy snapshot.
    mutable std::optional<scf::HfWarmState> warm;
    bool resident_warm{};
  };
  std::unique_ptr<KsPreparedCalculation> make_plan(const core::System& system) const {
    return std::make_unique<KsPreparedCalculation>(capabilities_, system, method_, options_,
                                                   backend_, device_);
  }
  void materialize_warm(std::size_t i) const {
    const auto& item = items_.at(i);
    if (item.warm && item.warm->density.empty() && item.resident_warm)
      item.warm->density = item.plan->warm_density();
  }

  Capabilities capabilities_;
  std::vector<core::System> systems_;
  vibeqc_method method_;
  scf::ScfOptions options_;
  vibeqc_backend backend_;
  int device_;
  bool warm_enabled_, warm_updates_{true};
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
#if !VIBEQC_HAS_CUDA
  if (context.requested_backend == VIBEQC_BACKEND_CUDA)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "DFT CUDA backend is not built");
#endif
  if (descriptor.method != VIBEQC_METHOD_LDA_RKS && descriptor.method != VIBEQC_METHOD_PBE_RKS &&
      descriptor.method != VIBEQC_METHOD_LDA_UKS && descriptor.method != VIBEQC_METHOD_PBE_UKS)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "requested DFT method is reserved but not implemented");
  return std::make_unique<KsPreparedCalculation>(capabilities, system, descriptor.method,
                                                 dft_options(descriptor, context.requested_backend),
                                                 context.requested_backend, context.device_id);
}

std::unique_ptr<PreparedBatch> prepare_dft_batch(const Capabilities& capabilities,
                                                 core::ContextState& context,
                                                 std::vector<core::System> systems,
                                                 const vibeqc_method_descriptor& descriptor,
                                                 vibeqc_batch_flags flags) {
  if ((flags & ~VIBEQC_BATCH_ENABLE_WARM_STARTS) != 0)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "KS batches support warm starts but not HF-specific profiling flags");
#if !VIBEQC_HAS_CUDA
  if (context.requested_backend == VIBEQC_BACKEND_CUDA)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "DFT CUDA backend is not built");
#endif
  return std::make_unique<KsPreparedBatch>(capabilities, std::move(systems), descriptor.method,
                                           dft_options(descriptor, context.requested_backend),
                                           context.requested_backend, context.device_id,
                                           (flags & VIBEQC_BATCH_ENABLE_WARM_STARTS) != 0);
}

}  // namespace vibeqc::methods::detail
