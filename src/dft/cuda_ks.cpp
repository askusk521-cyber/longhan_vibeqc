#include "dft/cuda_ks.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include "dft/cuda_ks_kernels.hpp"
#include "dft/cuda_xc.hpp"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda/eigensolver.hpp"
#include "scf/cuda/scf_constants.hpp"
#include "scf/cuda/scf_density_kernels.hpp"
#include "scf/cuda/scf_diis_kernels.hpp"
#include "scf/cuda/scf_matrix_kernels.hpp"
#include "scf/cuda_direct_jk_device.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/proposal_control.hpp"
#include "vibeqc/vibeqc.hpp"

#if defined(VIBEQC_TEST_HOOKS)
namespace {
// One-shot injection uses the real status mapper without poisoning the CUDA
// context, allowing the public API to verify explicit recovery and seed reuse.
thread_local bool fail_next_ks_runtime = false;
}  // namespace
extern "C" void ks_cuda_fail_next_runtime_for_test_v1() { fail_next_ks_runtime = true; }
#endif

namespace vibeqc::dft {
namespace {
using namespace scf::cuda_execution;
void check(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess)
    throw vibeqc::Error(VIBEQC_STATUS_CUDA_ERROR, cudaGetErrorString(status));
}
void check(vibeqc_status status, const std::string& detail) {
  if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status == VIBEQC_STATUS_CUDA_ERROR) throw vibeqc::Error(status, detail);
  if (status == VIBEQC_STATUS_INVALID_ARGUMENT) throw std::invalid_argument(detail);
  if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}
std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::overflow_error("CUDA KS storage overflow");
  return a * b;
}
std::size_t sum(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::overflow_error("CUDA KS storage overflow");
  return a + b;
}
/** Numeric arena view shared by allocation and metadata-only planning. */
struct KsStateStorage {
  double *hcore{}, *overlap{}, *x{}, *j{}, *density{}, *proposal{}, *warm{}, *fock{}, *residual{},
      *tmp1{}, *tmp2{}, *effective{}, *fock_history{}, *residual_history{}, *gram{}, *weights{},
      *eigenvalues{};
  std::int32_t* occupied{};
  std::uint8_t *enabled{}, *spin_enabled{};
  std::uint32_t *history_count{}, *history_head{};
  int *solver_info{}, *jk_error{};
  cuda_ks_detail::Scalars* scalars{};
  /** The dry run and actual partition share one checked, typed layout. All
   * persistent and phase-local numeric buffers are explicitly charged. */
  std::size_t partition(std::size_t n, unsigned spins, unsigned history, void* storage) {
    const auto matrix = product(n, n), elements = product(spins, matrix);
    std::size_t bytes = 0;
    const auto reserve = [&](auto*& pointer, std::size_t count) {
      using T = std::remove_pointer_t<std::remove_reference_t<decltype(pointer)>>;
      const auto remainder = bytes % alignof(T);
      if (remainder) bytes = sum(bytes, alignof(T) - remainder);
      pointer = storage ? reinterpret_cast<T*>(static_cast<char*>(storage) + bytes) : nullptr;
      bytes = sum(bytes, product(count, sizeof(T)));
    };
    for (auto** pointer : {&hcore, &overlap, &x, &j}) reserve(*pointer, matrix);
    for (auto** pointer : {&density, &proposal, &warm, &fock, &residual, &tmp1, &tmp2, &effective})
      reserve(*pointer, elements);
    reserve(fock_history, product(history, elements));
    reserve(residual_history, product(history, elements));
    reserve(gram, product(history + 1, history + 1));
    reserve(weights, history + 1);
    reserve(eigenvalues, product(spins, n));
    reserve(occupied, spins);
    reserve(enabled, 1);
    reserve(spin_enabled, spins);
    reserve(history_count, 1);
    reserve(history_head, 1);
    reserve(solver_info, spins);
    reserve(jk_error, 1);
    reserve(scalars, 1);
    return bytes;
  }
};
}  // namespace

std::size_t cuda_ks_state_bytes(std::size_t n, unsigned spins, unsigned history) {
  if (!n || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      (spins != 1 && spins != 2) || history > 64)
    throw std::invalid_argument("invalid CUDA KS resource shape");
  KsStateStorage layout;
  return layout.partition(n, spins, std::max(1U, history), nullptr);
}

struct CudaKsPlan::Impl : KsStateStorage {
  const scf::PreparedFockPlan& provider;
  scf::ScfOptions options;
  scf::CudaDirectJkPlan* direct{};
  cudaStream_t stream{};
  int device{};
  std::size_t n{}, matrix{}, elements{};
  unsigned spins{}, history{};
  std::array<std::size_t, 2> occupations{};
  std::vector<double> orthogonalizer, cold_density;
  CudaKsResources resource;
  CudaKsTransfers movement;
  void *arena{}, *xc_arena{};
  std::unique_ptr<CudaXcPlan> xc;
  scf::ScfResult output;
  bool is_active{}, is_pending{}, is_failed{}, warm_ready{}, started{};
  bool warm_updates{true};
  bool stabilize_occupations{};
  std::uint64_t generation{};
  double previous_energy{std::numeric_limits<double>::infinity()};

  void current_device() const {
    // Prepared owners select their bound device on every entry, as the common
    // Fock provider does. Another context may have changed this thread's device
    // between calls; borrowed XC views still enforce their own device identity.
    check(cudaSetDevice(device));
  }

  std::vector<double> seed(const std::vector<double>* input) const {
    using namespace scf::reference;
    const auto& ints = provider.one_electron();
    if (options.strict_initial_density && input) {
      const std::vector<unsigned> counts =
          spins == 2 ? std::vector<unsigned>{static_cast<unsigned>(occupations[0]),
                                             static_cast<unsigned>(occupations[1])}
                     : std::vector<unsigned>{static_cast<unsigned>(occupations[0])};
      scf::solver::validate_seed(ints.overlap, *input, n, counts, spins == 2 ? 1.0 : 2.0);
      return *input;
    }
    std::optional<EigenResult> a, b;
    if (spins == 2) {
      const auto pair = scf::initial_guess::prepare_initial_uhf_density(
          ints, orthogonalizer, occupations[0], occupations[1], input, a, b);
      return concatenate(pair.first, pair.second);
    }
    return scf::initial_guess::prepare_initial_density(provider.system(), ints, orthogonalizer,
                                                       occupations[0], input, a);
  }

  Impl(const scf::PreparedFockPlan& plan, const AoBasis& basis, const MolecularGrid& grid,
       const scf::ScfOptions& control, bool pbe, std::size_t tile)
      : provider(plan), options(control) {
    const auto& strategy = provider.strategy();
    scf::validate_resolved_fock_build(strategy);
    if (strategy.backend != scf::FockBackend::Cuda || strategy.spec.derivative_order != 0 ||
        !strategy.spec.coulomb.present || strategy.spec.coulomb.coefficient != 1.0 ||
        strategy.spec.coulomb.approximation != scf::FockApproximation::Exact ||
        strategy.spec.exchange.present || !(direct = provider.cuda_direct_source()))
      throw std::invalid_argument(
          "CUDA KS requires the prepared conventional Coulomb-only strategy");
    if (options.compute_forces || options.hooks || options.export_physical_reference ||
        options.xc_density_route != XcDensityRoute::DensityMatrix ||
        (options.precision_mode && *options.precision_mode != VIBEQC_PRECISION_FP64))
      throw std::invalid_argument("CUDA KS supports FP64 density-matrix energy execution only");
    if (!options.max_iterations || !std::isfinite(options.energy_tolerance) ||
        !std::isfinite(options.density_tolerance) || options.energy_tolerance <= 0.0 ||
        options.density_tolerance <= 0.0 || options.diis_history > 64)
      throw std::invalid_argument("invalid CUDA KS convergence or DIIS controls");
    if (!provider.matches_system(grid.system()) ||
        basis.packed != AoBasis(provider.system()).packed)
      throw std::invalid_argument(
          "CUDA KS refuses a stale geometry, basis, charge or spin binding");
    n = provider.one_electron().nbf;
    if (!n || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) || basis.nao != n)
      throw std::invalid_argument("invalid CUDA KS AO dimension");
    matrix = product(n, n);
    spins = strategy.spec.spin == scf::FockSpin::Unrestricted ? 2 : 1;
    elements = product(spins, matrix);
    const auto counts = scf::initial_guess::spin_occupations(provider.system());
    occupations = {counts.first, counts.second};
    if (!provider.system().electron_count || occupations[0] > n || occupations[1] > n ||
        (spins == 1 && (occupations[0] != occupations[1] || provider.system().multiplicity != 1)))
      throw std::invalid_argument("CUDA KS occupations do not match the spin/orbital space");
    // A valid overlap does not imply finite physical data: unlike two equal
    // H centers, coincident O/H centers can retain a nonsingular AO metric
    // while nuclear repulsion is infinite. Fail before staging a cold seed.
    const auto& integrals = provider.one_electron();
    const auto finite = [](double value) { return std::isfinite(value); };
    if (!finite(integrals.nuclear_repulsion) ||
        !std::all_of(integrals.overlap.begin(), integrals.overlap.end(), finite) ||
        !std::all_of(integrals.hcore.begin(), integrals.hcore.end(), finite))
      throw std::runtime_error("nonfinite CUDA KS one-electron or nuclear energy");
    history = std::max(1U, options.diis_history);
    device = scf::cuda_direct_jk_device(direct);
    stream = scf::cuda_direct_jk_stream(direct);
    current_device();
    orthogonalizer = scf::reference::symmetric_orthogonalizer(provider.one_electron().overlap, n);
    cold_density = seed(nullptr);
    resource.state_device_bytes = partition(n, spins, history, nullptr);
    resource.xc_device_bytes = cuda_xc_layout(basis, grid, pbe, spins == 2, tile).device_bytes;
    resource.provider_device_bytes = provider.diagnostic().device_bytes;
    output.dft_diagnostic.history.reserve(options.max_iterations);
    resource.retained_host_numeric_bytes =
        (orthogonalizer.capacity() + cold_density.capacity()) * sizeof(double) +
        output.dft_diagnostic.history.capacity() * sizeof(ScfIteration);
    try {
      check(runtime::resource_cuda_malloc(&arena, resource.state_device_bytes));
      partition(n, spins, history, arena);
      check(runtime::resource_cuda_malloc(&xc_arena, resource.xc_device_bytes));
      check(cudaMemsetAsync(arena, 0, resource.state_device_bytes, stream));
      const auto upload = [&](void* destination, const void* source, std::size_t bytes) {
        check(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
        movement.setup_h2d_bytes += bytes;
      };
      upload(hcore, provider.one_electron().hcore.data(), matrix * sizeof(double));
      upload(overlap, provider.one_electron().overlap.data(), matrix * sizeof(double));
      upload(x, orthogonalizer.data(), matrix * sizeof(double));
      const std::int32_t spin_counts[]{static_cast<std::int32_t>(occupations[0]),
                                       static_cast<std::int32_t>(occupations[1])};
      const std::uint8_t selected[]{static_cast<std::uint8_t>(occupations[0] > 0),
                                    static_cast<std::uint8_t>(occupations[1] > 0)};
      upload(occupied, spin_counts, spins * sizeof(std::int32_t));
      upload(spin_enabled, selected, spins * sizeof(std::uint8_t));
      // XC setup drains this same stream, including the small stack inputs.
      xc = std::make_unique<CudaXcPlan>(basis, grid, pbe, spins == 2, tile, xc_arena,
                                        resource.xc_device_bytes, stream);
    } catch (...) {
      cleanup();
      throw;
    }
  }

  void cleanup() noexcept {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (stream) cudaStreamSynchronize(stream);
    xc.reset();
    if (xc_arena) runtime::resource_cuda_free(xc_arena);
    if (arena) runtime::resource_cuda_free(arena);
    xc_arena = arena = nullptr;
    cudaSetDevice(previous);
  }
  ~Impl() { cleanup(); }

  void begin(const std::vector<double>* input, bool reuse_warm) {
    current_device();
#if defined(VIBEQC_TEST_HOOKS)
    if (fail_next_ks_runtime) {
      fail_next_ks_runtime = false;
      check(cudaErrorUnknown);
    }
#endif
    if (is_pending) throw std::logic_error("cannot replace a pending CUDA KS iteration");
    const bool use_warm = !input && reuse_warm && warm_ready;
    std::vector<double> prepared;
    if (input) prepared = seed(input);  // Validate before replacing current state.
    auto retained_history = std::move(output.dft_diagnostic.history);
    retained_history.clear();
    output = {};
    output.dft_diagnostic.history = std::move(retained_history);
    output.dft_diagnostic.occupations = occupations;
    output.dft_diagnostic.grid_points = xc->layout().npoint;
    output.dft_diagnostic.tile_points = xc->layout().tile_points;
    output.dft_diagnostic.ao_order = xc->layout().pbe ? 1 : 0;
    output.initial_density_used = input != nullptr || use_warm;
    is_active = false;
    started = true;
    is_failed = false;
    stabilize_occupations = false;
    try {
      check(cudaMemsetAsync(history_count, 0, sizeof(*history_count), stream));
      check(cudaMemsetAsync(history_head, 0, sizeof(*history_head), stream));
      check(cudaMemsetAsync(enabled, 1, sizeof(*enabled), stream));
      if (use_warm) {
        check(cudaMemcpyAsync(density, warm, elements * sizeof(double), cudaMemcpyDeviceToDevice,
                              stream));
      } else {
        const auto& initial = input ? prepared : cold_density;
        check(cudaMemcpyAsync(density, initial.data(), elements * sizeof(double),
                              cudaMemcpyHostToDevice, stream));
        // Explicit initial-guess staging, never an iteration matrix transfer.
        check(cudaStreamSynchronize(stream));
        movement.density_h2d_bytes += elements * sizeof(double);
        ++movement.synchronizations;
      }
    } catch (...) {
      cudaStreamSynchronize(stream);
      is_failed = true;
      throw;
    }
    previous_energy = std::numeric_limits<double>::infinity();
    is_active = true;
  }

  void enqueue() {
    current_device();
    if (!is_active || is_pending) throw std::logic_error("CUDA KS iteration state is not ready");
    if (generation == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("CUDA KS density generation exhausted");
    is_pending = true;  // Any partial CUDA submission is drained on failure.
    try {
      std::string detail;
      check(scf::enqueue_cuda_direct_jk_device(direct, provider.strategy().spec, density,
                                               spins == 2 ? density + matrix : nullptr, matrix, j,
                                               nullptr, nullptr, jk_error, detail),
            detail);
      xc->enqueue(density, elements, ++generation);
      const auto potential = xc->view(generation);
      cuda_ks_detail::assemble_fock(stream, n, spins, hcore, j, potential.potential, fock);
      check(cudaGetLastError());
      const auto blocks = static_cast<unsigned>((elements + 127) / 128);
      const auto multiply = [&](const double* a, bool a_spin, bool transpose, const double* b,
                                bool b_spin, double* c) {
        launch_spin_matrix_product_kernel(blocks, 128, 0, stream, 1, spins, n, a, a_spin, transpose,
                                          b, b_spin, enabled, c);
        check(cudaGetLastError());
      };
      // Physical residual is FDS-SDF, using the unchanged CURRENT density.
      multiply(fock, true, false, density, true, tmp1);
      multiply(tmp1, true, false, overlap, false, residual);
      multiply(overlap, false, false, density, true, tmp1);
      multiply(tmp1, true, false, fock, true, tmp2);
      launch_subtract_matrix_batches_kernel(blocks, 128, 0, stream, 1, spins, n, tmp2, enabled,
                                            residual);
      check(cudaGetLastError());
      launch_update_diis_kernel(1, 32, 0, stream, 1, n, spins, history, fock, residual, enabled,
                                fock_history, residual_history, gram, weights, history_count,
                                history_head, effective, true);
      check(cudaGetLastError());
      if (stabilize_occupations) {
        // Match the CPU stationary-cycle policy. The unit-occupation virtual
        // projector is S-SDS for each spin. Shift only the DIIS proposal;
        // physical F/D/residual and the history above remain unmodified.
        multiply(overlap, false, false, density, true, tmp1);
        multiply(tmp1, true, false, overlap, false, tmp2);
        cuda_ks_detail::stabilize_uks_proposal(stream, n, overlap, tmp2, effective);
        check(cudaGetLastError());
        ++movement.occupation_stabilized_proposals;
      }
      multiply(effective, true, false, x, false, tmp1);
      multiply(x, false, true, tmp1, true, tmp2);
      EigensolverResources solver{};
      solver.stream_ = stream;
      // Both families are existing native solvers and work on ordinary
      // streams. The larger family has no fixed AO bound or opaque workspace.
      const auto family = n <= kSmallEigensolverLimit ? scf::CudaEigensolverFamily::small_native
                                                      : scf::CudaEigensolverFamily::graph_native;
      check(launch_solver(solver, family, n, spins, tmp2, effective, eigenvalues, 0, solver_info,
                          spin_enabled),
            "CUDA KS eigensolver launch failed");
      multiply(x, false, false, tmp2, true, tmp1);
      if (spins == 1)
        launch_build_density_kernel(blocks, 128, 0, stream, 1, n, occupied, tmp1, enabled,
                                    proposal);
      else
        launch_build_spin_density_kernel(blocks, 128, 0, stream, 1, spins, n, occupied, tmp1,
                                         enabled, proposal);
      check(cudaGetLastError());
      cuda_ks_detail::diagnostics(stream, n, spins, density, proposal, residual, hcore, overlap, j,
                                  potential.totals, potential.error, jk_error, solver_info,
                                  scalars);
      check(cudaGetLastError());
    } catch (...) {
      cudaStreamSynchronize(stream);
      ++movement.synchronizations;
      is_pending = is_active = false;
      is_failed = true;
      throw;
    }
  }

  bool finish() {
    current_device();
    if (!is_pending) throw std::logic_error("no pending CUDA KS iteration");
    cuda_ks_detail::Scalars physical{};
    try {
      check(cudaMemcpyAsync(&physical, scalars, sizeof(physical), cudaMemcpyDeviceToHost, stream));
      check(cudaStreamSynchronize(stream));
    } catch (...) {
      cudaStreamSynchronize(stream);
      is_pending = is_active = false;
      is_failed = true;
      throw;
    }
    movement.scalar_d2h_bytes += sizeof(physical);
    ++movement.synchronizations;
    ++movement.iterations;
    is_pending = false;
    ++output.iterations;
    ++output.fock_builds;
    auto& diagnostic = output.dft_diagnostic;
    diagnostic.components = {provider.one_electron().nuclear_repulsion, physical.one_electron,
                             physical.hartree, physical.xc};
    diagnostic.physical_residual = physical.residual;
    diagnostic.electrons = {physical.electrons[0], physical.electrons[1]};
    diagnostic.density_change = physical.density_change;
    output.physical_residual_rms = physical.residual_rms;
    output.energy = diagnostic.components.total();
    output.energy_change = std::abs(output.energy - previous_energy);
    output.density_rms = physical.density_rms;
    diagnostic.history.push_back({output.iterations, diagnostic.components, output.energy_change,
                                  physical.density_change, physical.residual, diagnostic.electrons,
                                  stabilize_occupations});
    // The kernel validates electronic components; their host-side sum with
    // the nuclear term must also be finite before any convergence/cache gate.
    is_failed = physical.failure != 0 || !std::isfinite(output.energy);
    for (unsigned s = 0; s < 2; ++s)
      if (std::abs(physical.electrons[s] - occupations[s]) > 1e-8) is_failed = true;
    if (is_failed) {
      is_active = false;
      return false;
    }
    // A stationary physical state can still alternate integer occupations.
    // Enable the same 0.1-Eh proposal shift as CPU UKS only after both physical
    // gates pass. A subsequent density-change gate must still pass to finish.
    if (spins == 2 && output.iterations > 1 && output.energy_change < options.energy_tolerance &&
        physical.residual < std::min(1e-9, options.density_tolerance) &&
        physical.density_change >= options.density_tolerance)
      stabilize_occupations = true;
    output.converged = output.iterations > 1 && output.energy_change < options.energy_tolerance &&
                       physical.density_change < options.density_tolerance &&
                       physical.residual < std::min(1e-9, options.density_tolerance);
    is_active = !output.converged && output.iterations < options.max_iterations;
    try {
      if (output.converged && warm_updates) {
        // E, F, residual and retained D all belong to this same generation.
        // A failed/unfinished solve can never overwrite the last-good cache.
        check(cudaMemcpyAsync(warm, density, elements * sizeof(double), cudaMemcpyDeviceToDevice,
                              stream));
        warm_ready = true;
      } else if (is_active) {
        check(cudaMemcpyAsync(density, proposal, elements * sizeof(double),
                              cudaMemcpyDeviceToDevice, stream));
      }
    } catch (...) {
      cudaStreamSynchronize(stream);
      is_active = false;
      is_failed = true;
      output.converged = false;
      throw;
    }
    previous_energy = output.energy;
    return is_active;
  }

  std::vector<double> download(const double* source) {
    current_device();
    std::vector<double> data(elements);
    try {
      check(cudaMemcpyAsync(data.data(), source, elements * sizeof(double), cudaMemcpyDeviceToHost,
                            stream));
      check(cudaStreamSynchronize(stream));
    } catch (...) {
      cudaStreamSynchronize(stream);
      throw;
    }
    movement.matrix_d2h_bytes += elements * sizeof(double);
    ++movement.synchronizations;
    return data;
  }
};

CudaKsPlan::CudaKsPlan(const scf::PreparedFockPlan& fock, const AoBasis& basis,
                       const MolecularGrid& grid, const scf::ScfOptions& options, bool pbe,
                       std::size_t tile_points)
    : impl_(std::make_unique<Impl>(fock, basis, grid, options, pbe, tile_points)) {}
CudaKsPlan::~CudaKsPlan() = default;
void CudaKsPlan::begin(const std::vector<double>* seed, bool reuse_warm) {
  impl_->begin(seed, reuse_warm);
}
bool CudaKsPlan::active() const noexcept { return impl_->is_active; }
bool CudaKsPlan::pending() const noexcept { return impl_->is_pending; }
bool CudaKsPlan::failed() const noexcept { return impl_->is_failed; }
void CudaKsPlan::enqueue_iteration() { impl_->enqueue(); }
bool CudaKsPlan::finish_iteration() { return impl_->finish(); }
scf::ScfResult CudaKsPlan::result(bool export_density) {
  if (!impl_->started || impl_->is_active || impl_->is_pending)
    throw std::logic_error("CUDA KS result is not terminal");
  auto result = impl_->output;
  if (export_density) result.density = impl_->download(impl_->density);
  return result;
}
scf::ScfResult CudaKsPlan::run(const std::vector<double>* seed, bool reuse_warm,
                               bool export_density) {
  begin(seed, reuse_warm);
  while (active()) {
    enqueue_iteration();
    finish_iteration();
  }
  return result(export_density);
}
std::vector<double> CudaKsPlan::warm_density() {
  if (impl_->is_pending)
    throw std::logic_error("cannot export warm state during a pending iteration");
  return impl_->warm_ready ? impl_->download(impl_->warm) : std::vector<double>{};
}
void CudaKsPlan::set_warm_start_updates(bool enabled) noexcept { impl_->warm_updates = enabled; }
void CudaKsPlan::clear_warm_start() noexcept { impl_->warm_ready = false; }
const CudaKsResources& CudaKsPlan::resources() const noexcept { return impl_->resource; }
CudaKsTransfers CudaKsPlan::transfers() const noexcept {
  auto out = impl_->movement;
  const auto& xc = impl_->xc->transfers();
  out.setup_h2d_bytes += xc.setup_h2d_bytes;
  out.synchronizations += xc.synchronizations;
  return out;
}
}  // namespace vibeqc::dft
