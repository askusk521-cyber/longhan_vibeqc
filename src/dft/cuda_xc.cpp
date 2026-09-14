#include "dft/cuda_xc.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "vibeqc/vibeqc.hpp"

#if VIBEQC_HAS_CUDA
#include "dft/grid_task_view.cuh"

extern "C" {
int grid_cuda_create_v3(int device, int major, int minor, const std::size_t* dimensions,
                        const double* basis, std::size_t capacity, unsigned order,
                        std::size_t expected_bytes, std::size_t active_capacity,
                        const std::size_t* orbital_capacity, std::size_t orbital_tile,
                        unsigned feature_mask, void** output, char* error, std::size_t size);
void grid_cuda_destroy_v1(void* pointer);
int grid_cuda_density_v1(void* pointer, const double* density, std::size_t elements, char* error,
                         std::size_t size);
int grid_cuda_run_selected_v1(void* pointer, const double* points, std::size_t npoint, int features,
                              const std::size_t* ao_ids, std::size_t active, double* feature_output,
                              double* jet_output, char* error, std::size_t size);
int grid_cuda_view_v1(void* pointer, vibeqc::dft::GridTaskView* output, char* error,
                      std::size_t size);
int grid_cuda_xc_v2(void* pointer, std::uint64_t generation, int pbe, int restricted, int interior,
                    const double* weights, std::size_t npoint, double* integrals, char* error,
                    std::size_t size);
int grid_cuda_scatter_v1(void* pointer, std::uint64_t generation, const double* host_local,
                         int reset, double* host_global, char* error, std::size_t size);
}
#endif

namespace vibeqc::dft {
namespace {

#if VIBEQC_HAS_CUDA
constexpr std::size_t kErrorSize = 512;

void checked(int status, const std::array<char, kErrorSize>& error, const char* operation) {
  if (status == 0) return;
  const std::string detail(error.data());
  if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  const std::string message = std::string(operation) + " failed" +
                              (detail.empty() ? std::string{} : ": " + detail);
  if (status == VIBEQC_STATUS_CUDA_ERROR)
    throw vibeqc::Error(VIBEQC_STATUS_CUDA_ERROR, message);
  throw std::runtime_error(message);
}
#endif

}  // namespace

struct PreparedCudaXcPlan::Impl {
  std::size_t nao{};
  std::size_t tile_points{};
  bool pbe{};
  std::vector<std::size_t> ao_ids;
  std::mutex mutex;
#if VIBEQC_HAS_CUDA
  void* handle{};
#endif

  Impl(const AoBasis& basis, int device_id, int major, int minor, bool use_pbe,
       std::size_t capacity)
      : nao(basis.nao), tile_points(capacity), pbe(use_pbe), ao_ids(nao) {
    if (!nao || !tile_points) throw std::invalid_argument("invalid CUDA XC plan dimensions");
    std::iota(ao_ids.begin(), ao_ids.end(), std::size_t{0});
#if VIBEQC_HAS_CUDA
    const std::array<std::size_t, 3> dimensions{basis.natom, basis.nprimitive, basis.nao};
    std::array<char, kErrorSize> error{};
    checked(grid_cuda_create_v3(device_id, major, minor, dimensions.data(), basis.packed.data(),
                                tile_points, pbe ? 1U : 0U, 0, nao, nullptr, 0, pbe ? 3U : 1U,
                                &handle, error.data(), error.size()),
            error, "CUDA XC preparation");
#else
    (void)basis;
    (void)device_id;
    (void)major;
    (void)minor;
    throw std::runtime_error("CUDA XC preparation requires a CUDA-enabled build");
#endif
  }

  ~Impl() {
#if VIBEQC_HAS_CUDA
    grid_cuda_destroy_v1(handle);
#endif
  }

#if VIBEQC_HAS_CUDA
  SpinXcIntegral evaluate(const MolecularGrid& grid, const std::vector<double>& alpha,
                          const std::vector<double>& beta, bool restricted) {
    if (alpha.size() != nao * nao || beta.size() != nao * nao)
      throw std::invalid_argument("CUDA XC density dimensions do not match the AO basis");
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<double> spin_density;
    spin_density.reserve(2 * nao * nao);
    spin_density.insert(spin_density.end(), alpha.begin(), alpha.end());
    spin_density.insert(spin_density.end(), beta.begin(), beta.end());
    std::array<char, kErrorSize> error{};
    checked(grid_cuda_density_v1(handle, spin_density.data(), spin_density.size(), error.data(),
                                 error.size()),
            error, "CUDA XC density upload");

    SpinXcIntegral result;
    result.potential[0].resize(nao * nao);
    result.potential[1].resize(nao * nao);
    std::vector<double> joined_potential(2 * nao * nao);
    for (std::size_t begin = 0; begin < grid.point_count(); begin += tile_points) {
      const std::size_t count = std::min(tile_points, grid.point_count() - begin);
      error.fill(0);
      checked(grid_cuda_run_selected_v1(handle, grid.points().data() + 3 * begin, count, 1,
                                        ao_ids.data(), nao, nullptr, nullptr, error.data(),
                                        error.size()),
              error, "CUDA XC grid tile");
      GridTaskView view{};
      error.fill(0);
      checked(grid_cuda_view_v1(handle, &view, error.data(), error.size()), error,
              "CUDA XC grid view");
      std::array<double, 3> integrals{};
      error.fill(0);
      checked(grid_cuda_xc_v2(handle, view.generation, pbe ? 1 : 0, restricted ? 1 : 0, 0,
                              grid.weights().data() + begin, count, integrals.data(), error.data(),
                              error.size()),
              error, "CUDA XC contraction");
      result.energy += integrals[0];
      result.electrons[0] += integrals[1];
      result.electrons[1] += integrals[2];
      error.fill(0);
      checked(grid_cuda_scatter_v1(
                  handle, view.generation, nullptr, 0,
                  begin + count == grid.point_count() ? joined_potential.data() : nullptr,
                  error.data(), error.size()),
              error, "CUDA XC potential scatter");
      result.points += count;
    }
    std::copy_n(joined_potential.begin(), nao * nao, result.potential[0].begin());
    std::copy_n(joined_potential.begin() + static_cast<std::ptrdiff_t>(nao * nao), nao * nao,
                result.potential[1].begin());
    return result;
  }
#endif
};

PreparedCudaXcPlan::PreparedCudaXcPlan(const AoBasis& basis, int device_id,
                                       int compute_capability_major, int compute_capability_minor,
                                       bool pbe, std::size_t tile_points)
    : impl_(std::make_unique<Impl>(basis, device_id, compute_capability_major,
                                   compute_capability_minor, pbe, tile_points)) {}
PreparedCudaXcPlan::~PreparedCudaXcPlan() = default;

SpinXcIntegral PreparedCudaXcPlan::evaluate_uks(const MolecularGrid& grid,
                                                const std::vector<double>& alpha_density,
                                                const std::vector<double>& beta_density) {
#if VIBEQC_HAS_CUDA
  return impl_->evaluate(grid, alpha_density, beta_density, false);
#else
  (void)grid;
  (void)alpha_density;
  (void)beta_density;
  throw std::runtime_error("CUDA XC execution requires a CUDA-enabled build");
#endif
}

XcIntegral PreparedCudaXcPlan::evaluate_rks(const MolecularGrid& grid,
                                            const std::vector<double>& density) {
  if (density.size() != impl_->nao * impl_->nao)
    throw std::invalid_argument("CUDA RKS density dimensions do not match the AO basis");
#if VIBEQC_HAS_CUDA
  std::vector<double> spin_density(density.size());
  std::transform(density.begin(), density.end(), spin_density.begin(),
                 [](double value) { return 0.5 * value; });
  auto spin = impl_->evaluate(grid, spin_density, spin_density, true);
  XcIntegral result;
  result.energy = spin.energy;
  result.electrons = spin.electrons[0] + spin.electrons[1];
  result.points = spin.points;
  result.potential.resize(density.size());
  for (std::size_t i = 0; i < result.potential.size(); ++i)
    result.potential[i] = 0.5 * (spin.potential[0][i] + spin.potential[1][i]);
  result.density_diagnostic.requested = XcDensityRoute::DensityMatrix;
  result.density_diagnostic.executed = XcDensityRoute::DensityMatrix;
  result.density_diagnostic.npoint = grid.point_count();
  result.density_diagnostic.active_ao = impl_->nao;
  result.density_diagnostic.max_tile_points = std::min(impl_->tile_points, grid.point_count());
  result.density_diagnostic.ingredient_mask = impl_->pbe ? 3U : 1U;
  result.density_diagnostic.borrowed_density_bytes = density.size() * sizeof(double);
  result.density_diagnostic.owned_numeric_bytes =
      (2 * density.size() + result.potential.size()) * sizeof(double);
  return result;
#else
  (void)grid;
  throw std::runtime_error("CUDA XC execution requires a CUDA-enabled build");
#endif
}

}  // namespace vibeqc::dft
