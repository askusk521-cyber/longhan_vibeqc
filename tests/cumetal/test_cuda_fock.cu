#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// Compile the production CUDA translation unit directly so this smoke test
// exercises VibeQC's real kernel and host-side CUDA runtime calls without
// teaching the main CMake build about CuMetal.
#include "src/scf/cuda_fock.cu"

namespace {

std::vector<double> reference_fock(std::size_t n, const std::vector<double>& hcore,
                                   const std::vector<double>& eri,
                                   const std::vector<double>& density) {
  const std::size_t matrix_size = n * n;
  std::vector<double> result(matrix_size, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double coulomb = 0.0;
      double exchange = 0.0;
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
          const double pkl = density[k * n + l];
          const std::size_t coulomb_index = ((i * n + j) * n + k) * n + l;
          const std::size_t exchange_index = ((i * n + k) * n + j) * n + l;
          coulomb += pkl * eri[coulomb_index];
          exchange += pkl * eri[exchange_index];
        }
      }
      result[i * n + j] = hcore[i * n + j] + coulomb - 0.5 * exchange;
    }
  }
  return result;
}

}  // namespace

int main() {
  constexpr std::size_t batch_size = 1;
  constexpr std::size_t n = 2;
  const std::vector<double> hcore = {1.0, 0.2, 0.2, 0.8};
  const std::vector<double> density = {1.1, 0.3, 0.3, 0.9};
  const std::vector<double> eri = {
      0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
      0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
  };

  vibeqc::scf::CudaFockBucketHandle* handle = nullptr;
  std::string detail;
  vibeqc_status status =
      vibeqc::scf::create_cuda_fock_bucket(0, batch_size, n, hcore, eri, &handle, detail);
  if (status != VIBEQC_STATUS_SUCCESS) {
    std::fprintf(stderr, "FAIL: create_cuda_fock_bucket: status=%d detail=%s\n",
                 static_cast<int>(status), detail.c_str());
    return 1;
  }

  std::vector<double> actual(n * n, 0.0);
  status = vibeqc::scf::execute_cuda_fock_bucket(handle, density, actual, detail);
  vibeqc::scf::destroy_cuda_fock_bucket(handle);
  if (status != VIBEQC_STATUS_SUCCESS) {
    std::fprintf(stderr, "FAIL: execute_cuda_fock_bucket: status=%d detail=%s\n",
                 static_cast<int>(status), detail.c_str());
    return 2;
  }

  const std::vector<double> expected = reference_fock(n, hcore, eri, density);
  double max_error = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::fabs(actual[i] - expected[i]);
    if (error > max_error) max_error = error;
  }

  // CuMetal currently emulates FP64 on Metal, so this gate is intentionally a
  // functional CUDA-semantics smoke test rather than VibeQC's strict FP64
  // numerical-accuracy test.
  if (max_error > 1.0e-7) {
    std::fprintf(stderr, "FAIL: CuMetal Fock mismatch, max_error=%.12e\n", max_error);
    for (std::size_t i = 0; i < actual.size(); ++i) {
      std::fprintf(stderr, "  %zu: got=%.12e expected=%.12e\n", i, actual[i], expected[i]);
    }
    return 3;
  }

  std::printf("PASS: VibeQC CUDA Fock kernel ran through CuMetal; max_error=%.12e\n", max_error);
  return 0;
}
