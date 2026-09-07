#include <cublas_v2.h>
#include <cub/block/block_scan.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

// CuMetal's current Apple-Silicon FP64 path is semantic emulation and is far
// too slow for a pull-request gate.  Keep the PR runtime checks in FP32 so they
// validate CUDA indexing, synchronization, atomics and provider wiring without
// pretending to validate VibeQC's production FP64 numerics.
extern "C" __global__ void vibeqc_cumetal_rhf_fock_fp32_kernel(
    std::size_t batch_size, std::size_t n, const float* hcore, const float* eri,
    const float* density, float* fock) {
  const std::size_t matrix_size = n * n;
  const std::size_t eri_size = matrix_size * matrix_size;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= batch_size * matrix_size) return;

  const std::size_t system = element / matrix_size;
  const std::size_t local = element % matrix_size;
  const std::size_t i = local / n;
  const std::size_t j = local % n;
  const float* system_density = density + system * matrix_size;
  const float* system_eri = eri + system * eri_size;
  float coulomb = 0.0F;
  float exchange = 0.0F;
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < n; ++l) {
      const float pkl = system_density[k * n + l];
      const std::size_t coulomb_index = ((i * n + j) * n + k) * n + l;
      const std::size_t exchange_index = ((i * n + k) * n + j) * n + l;
      coulomb += pkl * system_eri[coulomb_index];
      exchange += pkl * system_eri[exchange_index];
    }
  }
  fock[element] = hcore[element] + coulomb - 0.5F * exchange;
}

extern "C" __global__ void vibeqc_cumetal_block_scan_kernel(const int* input,
                                                             int* exclusive,
                                                             int* aggregate) {
  using BlockScan = cub::BlockScan<int, 32>;
  __shared__ typename BlockScan::TempStorage storage;
  int prefix = 0;
  int total = 0;
  BlockScan(storage).ExclusiveSum(input[threadIdx.x], prefix, total);
  exclusive[threadIdx.x] = prefix;
  if (threadIdx.x == 0) *aggregate = total;
}

extern "C" __global__ void vibeqc_cumetal_atomic_kernel(float* output) {
  atomicAdd(output, 1.0F);
}

namespace {

bool check_cuda(cudaError_t error, const char* expression) {
  if (error == cudaSuccess) return true;
  std::fprintf(stderr, "FAIL: %s: %s\n", expression, cudaGetErrorString(error));
  return false;
}

bool check_cublas(cublasStatus_t status, const char* expression) {
  if (status == CUBLAS_STATUS_SUCCESS) return true;
  std::fprintf(stderr, "FAIL: %s: cuBLAS status %d\n", expression,
               static_cast<int>(status));
  return false;
}

std::vector<float> reference_fock(std::size_t n, const std::vector<float>& hcore,
                                  const std::vector<float>& eri,
                                  const std::vector<float>& density) {
  const std::size_t matrix_size = n * n;
  std::vector<float> result(matrix_size, 0.0F);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float coulomb = 0.0F;
      float exchange = 0.0F;
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
          const float pkl = density[k * n + l];
          const std::size_t coulomb_index = ((i * n + j) * n + k) * n + l;
          const std::size_t exchange_index = ((i * n + k) * n + j) * n + l;
          coulomb += pkl * eri[coulomb_index];
          exchange += pkl * eri[exchange_index];
        }
      }
      result[i * n + j] = hcore[i * n + j] + coulomb - 0.5F * exchange;
    }
  }
  return result;
}

bool run_fock_contract(cudaStream_t stream) {
  constexpr std::size_t n = 2;
  constexpr std::size_t matrix_size = n * n;
  constexpr std::size_t eri_size = matrix_size * matrix_size;
  const std::vector<float> hcore = {1.0F, 0.2F, 0.2F, 0.8F};
  const std::vector<float> density = {1.1F, 0.3F, 0.3F, 0.9F};
  const std::vector<float> eri = {
      0.01F, 0.02F, 0.03F, 0.04F, 0.05F, 0.06F, 0.07F, 0.08F,
      0.09F, 0.10F, 0.11F, 0.12F, 0.13F, 0.14F, 0.15F, 0.16F,
  };
  std::vector<float> actual(matrix_size, 0.0F);

  float* device_hcore = nullptr;
  float* device_eri = nullptr;
  float* device_density = nullptr;
  float* device_fock = nullptr;
  bool ok =
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_hcore),
                            matrix_size * sizeof(float)),
                 "cudaMalloc(hcore)") &&
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_eri),
                            eri_size * sizeof(float)),
                 "cudaMalloc(eri)") &&
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_density),
                            matrix_size * sizeof(float)),
                 "cudaMalloc(density)") &&
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_fock),
                            matrix_size * sizeof(float)),
                 "cudaMalloc(fock)");
  if (!ok) return false;

  ok = check_cuda(cudaMemcpyAsync(device_hcore, hcore.data(),
                                  matrix_size * sizeof(float),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(hcore)") &&
       check_cuda(cudaMemcpyAsync(device_eri, eri.data(), eri_size * sizeof(float),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(eri)") &&
       check_cuda(cudaMemcpyAsync(device_density, density.data(),
                                  matrix_size * sizeof(float),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(density)");
  if (ok) {
    vibeqc_cumetal_rhf_fock_fp32_kernel<<<1, 32, 0, stream>>>(
        1, n, device_hcore, device_eri, device_density, device_fock);
    ok = check_cuda(cudaGetLastError(), "FP32 Fock launch") &&
         check_cuda(cudaMemcpyAsync(actual.data(), device_fock,
                                    matrix_size * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync(fock)") &&
         check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(Fock)");
  }

  cudaFree(device_hcore);
  cudaFree(device_eri);
  cudaFree(device_density);
  cudaFree(device_fock);
  if (!ok) return false;

  const auto expected = reference_fock(n, hcore, eri, density);
  float max_error = 0.0F;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
  }
  if (max_error > 2.0e-6F) {
    std::fprintf(stderr, "FAIL: FP32 Fock mismatch, max_error=%.9e\n",
                 static_cast<double>(max_error));
    return false;
  }
  std::printf("PASS: FP32 Fock indexing/assembly max_error=%.9e\n",
              static_cast<double>(max_error));
  return true;
}

bool run_block_scan_contract(cudaStream_t stream) {
  constexpr int threads = 32;
  std::vector<int> input(threads);
  std::vector<int> output(threads, -1);
  for (int i = 0; i < threads; ++i) input[i] = i + 1;
  int aggregate = -1;

  int* device_input = nullptr;
  int* device_output = nullptr;
  int* device_aggregate = nullptr;
  bool ok =
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_input),
                            threads * sizeof(int)),
                 "cudaMalloc(scan input)") &&
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output),
                            threads * sizeof(int)),
                 "cudaMalloc(scan output)") &&
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_aggregate), sizeof(int)),
                 "cudaMalloc(scan aggregate)");
  if (!ok) return false;

  ok = check_cuda(cudaMemcpyAsync(device_input, input.data(), threads * sizeof(int),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(scan input)");
  if (ok) {
    vibeqc_cumetal_block_scan_kernel<<<1, threads, 0, stream>>>(
        device_input, device_output, device_aggregate);
    ok = check_cuda(cudaGetLastError(), "BlockScan launch") &&
         check_cuda(cudaMemcpyAsync(output.data(), device_output,
                                    threads * sizeof(int),
                                    cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync(scan output)") &&
         check_cuda(cudaMemcpyAsync(&aggregate, device_aggregate, sizeof(int),
                                    cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync(scan aggregate)") &&
         check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(BlockScan)");
  }

  cudaFree(device_input);
  cudaFree(device_output);
  cudaFree(device_aggregate);
  if (!ok) return false;

  int running = 0;
  for (int i = 0; i < threads; ++i) {
    if (output[i] != running) {
      std::fprintf(stderr, "FAIL: BlockScan[%d]=%d expected=%d\n", i,
                   output[i], running);
      return false;
    }
    running += input[i];
  }
  if (aggregate != running) {
    std::fprintf(stderr, "FAIL: BlockScan aggregate=%d expected=%d\n",
                 aggregate, running);
    return false;
  }
  std::printf("PASS: CUB BlockScan cooperative semantics\n");
  return true;
}

bool run_atomic_contract(cudaStream_t stream) {
  float* device_value = nullptr;
  float value = 0.0F;
  bool ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_value), sizeof(float)),
                       "cudaMalloc(atomic)") &&
            check_cuda(cudaMemsetAsync(device_value, 0, sizeof(float), stream),
                       "cudaMemsetAsync(atomic)");
  if (ok) {
    vibeqc_cumetal_atomic_kernel<<<1, 32, 0, stream>>>(device_value);
    ok = check_cuda(cudaGetLastError(), "atomicAdd launch") &&
         check_cuda(cudaMemcpyAsync(&value, device_value, sizeof(float),
                                    cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync(atomic)") &&
         check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(atomic)");
  }
  cudaFree(device_value);
  if (!ok) return false;
  if (std::fabs(value - 32.0F) > 1.0e-5F) {
    std::fprintf(stderr, "FAIL: atomicAdd result=%.7g expected=32\n",
                 static_cast<double>(value));
    return false;
  }
  std::printf("PASS: FP32 atomicAdd semantics\n");
  return true;
}

bool run_cublas_contract(cudaStream_t stream) {
  cublasHandle_t handle = nullptr;
  void* workspace = nullptr;
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;
  const std::vector<float> a = {1.0F, 3.0F, 2.0F, 4.0F};
  const std::vector<float> identity = {1.0F, 0.0F, 0.0F, 1.0F};
  std::vector<float> c(4, 0.0F);
  const float alpha = 1.0F;
  const float beta = 0.0F;

  bool ok = check_cublas(cublasCreate(&handle), "cublasCreate") &&
            check_cublas(cublasSetStream(handle, stream), "cublasSetStream") &&
            check_cuda(cudaMalloc(&workspace, 4096), "cudaMalloc(cuBLAS workspace)") &&
            check_cublas(cublasSetWorkspace(handle, workspace, 4096),
                         "cublasSetWorkspace") &&
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_a),
                                  4 * sizeof(float)),
                       "cudaMalloc(SGEMM A)") &&
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_b),
                                  4 * sizeof(float)),
                       "cudaMalloc(SGEMM B)") &&
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_c),
                                  4 * sizeof(float)),
                       "cudaMalloc(SGEMM C)");
  if (ok) {
    ok = check_cuda(cudaMemcpyAsync(device_a, a.data(), 4 * sizeof(float),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync(SGEMM A)") &&
         check_cuda(cudaMemcpyAsync(device_b, identity.data(), 4 * sizeof(float),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync(SGEMM B)") &&
         check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2,
                                  &alpha, device_a, 2, device_b, 2, &beta,
                                  device_c, 2),
                      "cublasSgemm") &&
         check_cuda(cudaMemcpyAsync(c.data(), device_c, 4 * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync(SGEMM C)") &&
         check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(SGEMM)");
  }

  cudaFree(workspace);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  if (handle != nullptr) cublasDestroy(handle);
  if (!ok) return false;

  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(c[i] - a[i]) > 1.0e-5F) {
      std::fprintf(stderr, "FAIL: SGEMM[%zu]=%.7g expected=%.7g\n", i,
                   static_cast<double>(c[i]), static_cast<double>(a[i]));
      return false;
    }
  }
  std::printf("PASS: cuBLAS workspace + FP32 SGEMM\n");
  return true;
}

}  // namespace

int main() {
  if (!check_cuda(cudaSetDevice(0), "cudaSetDevice(0)")) return 1;

  cudaDeviceProp properties{};
  if (!check_cuda(cudaGetDeviceProperties(&properties, 0),
                  "cudaGetDeviceProperties")) {
    return 2;
  }
  if (properties.maxThreadsPerMultiProcessor <= 0 ||
      properties.maxBlocksPerMultiProcessor <= 0 ||
      properties.regsPerMultiprocessor <= 0) {
    std::fprintf(stderr,
                 "FAIL: invalid occupancy properties threads=%d blocks=%d regs=%d\n",
                 properties.maxThreadsPerMultiProcessor,
                 properties.maxBlocksPerMultiProcessor,
                 properties.regsPerMultiprocessor);
    return 3;
  }
  std::printf("PASS: cudaDeviceProp occupancy compatibility\n");

  cudaStream_t stream = nullptr;
  if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags")) {
    return 4;
  }

  const bool ok = run_fock_contract(stream) && run_block_scan_contract(stream) &&
                  run_atomic_contract(stream) && run_cublas_contract(stream);
  cudaStreamDestroy(stream);
  if (!ok) return 5;

  std::printf("PASS: CuMetal FP32 CUDA logic contracts\n");
  return 0;
}
