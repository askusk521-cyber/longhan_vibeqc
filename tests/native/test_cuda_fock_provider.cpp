#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "scf/cuda_density_fitting.hpp"
#include "scf/density_fitting.hpp"

namespace {
using namespace vibeqc::scf;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
struct DeviceMatrix {
  double* pointer{};
  explicit DeviceMatrix(const std::vector<double>& input) {
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), input.size() * sizeof(double)));
    const auto status =
        cudaMemcpy(pointer, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      cudaFree(pointer);
      check(status);
    }
  }
  ~DeviceMatrix() { cudaFree(pointer); }
  DeviceMatrix(const DeviceMatrix&) = delete;
  DeviceMatrix& operator=(const DeviceMatrix&) = delete;
  void verify(const std::vector<double>& expected) const {
    std::vector<double> actual(expected.size());
    check(
        cudaMemcpy(actual.data(), pointer, actual.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < actual.size(); ++i)
      require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 3e-12,
              "independent device DF matrix differs from CPU");
  }
};

void device_selection() {
  const std::vector<double> metric{2.0, 0.1, 0.1, 1.3};
  const std::vector<double> tensor{1.3, 0.2, 0.3, -0.1, 0.3, -0.1, 0.8, 0.6};
  const std::vector<double> a{1.2, 0.31, -0.07, 0.7}, b{0.1, -0.05, 0.13, 0.4};
  const auto transformed = orthonormalize_density_fitting_three_center(
      tensor, 2, factor_density_fitting_metric(metric, 2));
  const auto rhf = build_density_fitting_rhf_jk(transformed, a);
  const auto uhf = build_density_fitting_uhf_jk(transformed, a, b);
  for (const auto layout : {FockMatrixLayout::RowMajor, FockMatrixLayout::ColumnMajor})
    for (std::size_t tile : {0U, 1U}) {
      CudaDensityFittingJkPlan* raw{};
      std::string detail;
      std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
      require(create_cuda_density_fitting_jk_plan_tiled(0, 1, 2, 2, metric, tensor, 1e-10, tile,
                                                        tile, &raw, diagnostics,
                                                        detail) == VIBEQC_STATUS_SUCCESS,
              detail.c_str());
      std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)>
          plan(raw, &destroy_cuda_density_fitting_jk_plan);
      auto input_a = a, input_b = b;
      if (layout == FockMatrixLayout::ColumnMajor) {
        std::swap(input_a[1], input_a[2]);
        std::swap(input_b[1], input_b[2]);
      }
      DeviceMatrix da(input_a), db(input_b);
      for (bool j : {false, true})
        for (bool k : {false, true}) {
          const JkTermSelection terms{j, k};
          const std::vector<double> sentinel(4, 123.0);
          DeviceMatrix dj(sentinel), dka(sentinel), dkb(sentinel);
          require(execute_cuda_density_fitting_rhf_jk_device(
                      plan.get(), da.pointer, j ? dj.pointer : nullptr, k ? dka.pointer : nullptr,
                      detail, terms, layout) == VIBEQC_STATUS_SUCCESS,
                  detail.c_str());
          check(cudaDeviceSynchronize());  // Test boundary observes the plan's nonblocking stream.
          dj.verify(j ? rhf.coulomb : sentinel);
          dka.verify(k ? rhf.exchange : sentinel);
          // Give unselected outputs valid sentinels this time: the service must
          // neither write them nor assume non-null means a term was requested.
          require(execute_cuda_density_fitting_uhf_jk_device(
                      plan.get(), da.pointer, db.pointer, dj.pointer, dka.pointer, dkb.pointer,
                      detail, terms, layout) == VIBEQC_STATUS_SUCCESS,
                  detail.c_str());
          check(cudaDeviceSynchronize());
          dj.verify(j ? uhf.coulomb : sentinel);
          dka.verify(k ? uhf.alpha_exchange : sentinel);
          dkb.verify(k ? uhf.beta_exchange : sentinel);
        }
      require(execute_cuda_density_fitting_rhf_jk_device(plan.get(), da.pointer, nullptr, nullptr,
                                                         detail, {true, false}) ==
                  VIBEQC_STATUS_INVALID_ARGUMENT,
              "missing selected device J accepted");
      require(execute_cuda_density_fitting_uhf_jk_device(
                  plan.get(), da.pointer, db.pointer, nullptr, nullptr, nullptr, detail,
                  {false, true}) == VIBEQC_STATUS_INVALID_ARGUMENT,
              "missing selected device K accepted");
      da.verify(input_a);
      db.verify(input_b);
    }
}
}  // namespace
int main() {
  try {
    device_selection();
    std::cout << "CUDA DF independent device J/K: resident, streamed, absent outputs PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
