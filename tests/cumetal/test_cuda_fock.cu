#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

// Keep this smoke kernel externally visible. CuMetal's native-AOT executable
// path currently fails to link CUDA host stubs for kernels with anonymous-
// namespace linkage, which is how src/scf/cuda_fock.cu intentionally hides its
// production kernel. The arithmetic below mirrors rhf_fock_bucket_kernel so the
// macOS job can still exercise the same FP64 CUDA semantics without changing
// production linkage solely for a compatibility test.
extern "C" __global__ void vibeqc_cumetal_rhf_fock_kernel(
    std::size_t batch_size, std::size_t n, const double* hcore, const double* eri,
    const double* density, double* fock) {
  const std::size_t matrix_size = n * n;
  const std::size_t eri_size = matrix_size * matrix_size;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= batch_size * matrix_size) return;

  const std::size_t system = element / matrix_size;
  const std::size_t local = element % matrix_size;
  const std::size_t i = local / n;
  const std::size_t j = local % n;
  const double* system_density = density + system * matrix_size;
  const double* system_eri = eri + system * eri_size;
  double coulomb = 0.0;
  double exchange = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < n; ++l) {
      const double pkl = system_density[k * n + l];
      const std::size_t coulomb_index = ((i * n + j) * n + k) * n + l;
      const std::size_t exchange_index = ((i * n + k) * n + j) * n + l;
      coulomb += pkl * system_eri[coulomb_index];
      exchange += pkl * system_eri[exchange_index];
    }
  }
  fock[element] = hcore[element] + coulomb - 0.5 * exchange;
}

namespace {

bool check_cuda(cudaError_t error, const char* expression) {
  if (error == cudaSuccess) return true;
  std::fprintf(stderr, "FAIL: %s: %s\n", expression, cudaGetErrorString(error));
  return false;
}

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
  constexpr std::size_t matrix_size = n * n;
  constexpr std::size_t eri_size = matrix_size * matrix_size;
  constexpr unsigned threads = 256;

  const std::vector<double> hcore = {1.0, 0.2, 0.2, 0.8};
  const std::vector<double> density = {1.1, 0.3, 0.3, 0.9};
  const std::vector<double> eri = {
      0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
      0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
  };
  std::vector<double> actual(matrix_size, 0.0);

  cudaStream_t stream = nullptr;
  double* device_hcore = nullptr;
  double* device_eri = nullptr;
  double* device_density = nullptr;
  double* device_fock = nullptr;
  bool ok = check_cuda(cudaSetDevice(0), "cudaSetDevice(0)") &&
            check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags") &&
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_hcore),
                                  matrix_size * sizeof(double)),
                       "cudaMalloc(hcore)") &&
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_eri), eri_size * sizeof(double)),
                       "cudaMalloc(eri)") &&
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_density),
                                  matrix_size * sizeof(double)),
                       "cudaMalloc(density)") &&
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_fock),
                                  matrix_size * sizeof(double)),
                       "cudaMalloc(fock)");
  if (!ok) return 1;

  ok = check_cuda(cudaMemcpyAsync(device_hcore, hcore.data(), matrix_size * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(hcore)") &&
       check_cuda(cudaMemcpyAsync(device_eri, eri.data(), eri_size * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(eri)") &&
       check_cuda(cudaMemcpyAsync(device_density, density.data(), matrix_size * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(density)");
  if (!ok) return 2;

  const unsigned blocks =
      static_cast<unsigned>((batch_size * matrix_size + threads - 1) / threads);
  vibeqc_cumetal_rhf_fock_kernel<<<blocks, threads, 0, stream>>>(
      batch_size, n, device_hcore, device_eri, device_density, device_fock);
  ok = check_cuda(cudaGetLastError(), "vibeqc_cumetal_rhf_fock_kernel launch") &&
       check_cuda(cudaMemcpyAsync(actual.data(), device_fock, matrix_size * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync(fock)") &&
       check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  if (!ok) return 3;

  cudaFree(device_hcore);
  cudaFree(device_eri);
  cudaFree(device_density);
  cudaFree(device_fock);
  cudaStreamDestroy(stream);

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
    return 4;
  }

  std::printf("PASS: VibeQC CUDA Fock semantics ran through CuMetal; max_error=%.12e\n", max_error);
  return 0;
}
