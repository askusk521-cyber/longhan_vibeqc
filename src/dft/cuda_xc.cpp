#include "dft/cuda_xc.hpp"

#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>

#include "tensor/cuda_error.hpp"
#include "vibeqc/vibeqc.hpp"

#if defined(VIBEQC_TEST_HOOKS)
namespace {
thread_local cudaError_t fail_next_xc_status = cudaSuccess;
}  // namespace
extern "C" void xc_cuda_fail_next_runtime_for_test_v1() { fail_next_xc_status = cudaErrorUnknown; }
extern "C" void xc_cuda_fail_next_allocation_for_test_v1() {
  fail_next_xc_status = cudaErrorMemoryAllocation;
}
#endif

namespace vibeqc::dft {
namespace {
void check(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess)
    throw vibeqc::Error(VIBEQC_STATUS_CUDA_ERROR, cudaGetErrorString(status));
}
std::size_t multiply(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::overflow_error("CUDA XC storage overflow");
  return a * b;
}
std::size_t add(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::overflow_error("CUDA XC storage overflow");
  return a + b;
}
void device_pointer(const void* pointer, int device) {
  if (!pointer) throw std::invalid_argument("null CUDA XC device buffer");
  cudaPointerAttributes attributes{};
  check(cudaPointerGetAttributes(&attributes, pointer));
  if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
    throw std::invalid_argument("CUDA XC requires a device buffer on the current device");
}
}  // namespace

CudaXcLayout cuda_xc_layout(const AoBasis& basis, const MolecularGrid& grid, bool pbe,
                            bool unrestricted, std::size_t tile_points) {
  // Equal dimensions alone cannot bind a grid to its current geometry/basis.
  const AoBasis grid_basis(grid.system());
  if (basis.nao != grid_basis.nao || basis.natom != grid_basis.natom ||
      basis.nprimitive != grid_basis.nprimitive || basis.packed != grid_basis.packed)
    throw std::invalid_argument("CUDA XC grid/basis identity mismatch");
  return cuda_xc_layout_shape(basis.natom, basis.nprimitive, basis.nao, grid.point_count(), pbe,
                              unrestricted, tile_points);
}

CudaXcLayout cuda_xc_layout_shape(std::size_t atoms, std::size_t primitives, std::size_t nao,
                                  std::size_t points, bool pbe, bool unrestricted,
                                  std::size_t tile_points) {
  if (!atoms || !primitives || !nao || !points || !tile_points || tile_points > INT_MAX ||
      atoms > INT_MAX || primitives > INT_MAX || nao > INT_MAX)
    throw std::invalid_argument("invalid CUDA XC resource shape");
  const auto packed = add(add(multiply(3, atoms), multiply(2, primitives)), multiply(16, nao));
  CudaXcLayout out{
      atoms,         primitives, nao, points, std::min(tile_points, points), unrestricted ? 2U : 1U,
      pbe ? 4U : 1U, packed,     0,   pbe};
  std::size_t elements = add(out.packed_elements, multiply(4, out.npoint));
  const auto panel = multiply(out.tile_points, out.nao);
  elements = add(elements, multiply(out.jets + out.spins, panel));
  elements = add(elements, multiply(2 * out.spins * out.jets + 3, out.tile_points));
  elements = add(elements, multiply(out.spins, multiply(out.nao, out.nao)));
  elements = add(elements, 3);
  // The final double-sized slot aligns the numerical-error integer and makes
  // the exact byte request independent of host struct padding.
  out.device_bytes = multiply(add(elements, 1), sizeof(double));
  return out;
}

CudaXcPlan::CudaXcPlan(const AoBasis& basis, const MolecularGrid& grid, bool pbe, bool unrestricted,
                       std::size_t tile_points, void* arena, std::size_t arena_bytes,
                       cudaStream_t stream)
    : layout_(cuda_xc_layout(basis, grid, pbe, unrestricted, tile_points)),
      arena_(arena),
      stream_(stream) {
  if (arena_bytes < layout_.device_bytes ||
      reinterpret_cast<std::uintptr_t>(arena) % alignof(double))
    throw std::invalid_argument("CUDA XC arena is too small or misaligned");
  check(cudaGetDevice(&device_));
  device_pointer(arena, device_);
  const auto& l = layout_;
  basis_ = static_cast<double*>(arena);
  points_ = basis_ + l.packed_elements;
  weights_ = points_ + 3 * l.npoint;
  ao_ = weights_ + l.npoint;
  work_ = ao_ + l.jets * l.tile_points * l.nao;
  features_ = work_ + l.spins * l.tile_points * l.nao;
  coefficients_ = features_ + l.spins * l.jets * l.tile_points;
  point_totals_ = coefficients_ + l.spins * l.jets * l.tile_points;
  potential_ = point_totals_ + 3 * l.tile_points;
  totals_ = potential_ + l.spins * l.nao * l.nao;
  error_ = reinterpret_cast<int*>(totals_ + 3);
  try {
    check(cudaMemcpyAsync(basis_, basis.packed.data(), l.packed_elements * sizeof(double),
                          cudaMemcpyHostToDevice, stream_));
    check(cudaMemcpyAsync(points_, grid.points().data(), 3 * l.npoint * sizeof(double),
                          cudaMemcpyHostToDevice, stream_));
    check(cudaMemcpyAsync(weights_, grid.weights().data(), l.npoint * sizeof(double),
                          cudaMemcpyHostToDevice, stream_));
    // Complete setup before releasing borrowed host quadrature/basis inputs.
    check(cudaStreamSynchronize(stream_));
  } catch (...) {
    cudaStreamSynchronize(stream_);
    throw;
  }
  transfers_.setup_h2d_bytes = (l.packed_elements + 4 * l.npoint) * sizeof(double);
  transfers_.synchronizations = 1;
}

CudaXcPlan::~CudaXcPlan() {
  int previous = 0;
  cudaGetDevice(&previous);
  cudaSetDevice(device_);
  cudaStreamSynchronize(stream_);
  cudaSetDevice(previous);
}

void CudaXcPlan::check_device() const {
  int current = -1;
  check(cudaGetDevice(&current));
  if (current != device_) throw std::invalid_argument("CUDA XC current device changed");
}

void CudaXcPlan::enqueue(const double* density, std::size_t elements, std::uint64_t generation) {
  check_device();
  const auto count = layout_.spins * layout_.nao * layout_.nao;
  if (elements != count || !generation || generation <= submitted_generation_)
    throw std::invalid_argument("CUDA XC density size or generation is stale");
  device_pointer(density, device_);
  const auto input = reinterpret_cast<std::uintptr_t>(density);
  const auto arena = reinterpret_cast<std::uintptr_t>(arena_);
  const auto input_bytes = multiply(count, sizeof(double));
  if (input < add(arena, layout_.device_bytes) && arena < add(input, input_bytes))
    throw std::invalid_argument("CUDA XC density aliases its workspace");
  // Invalidate a previously exported view even when a subsequent launch fails.
  generation_ = 0;
  submitted_generation_ = generation;
  try {
#if defined(VIBEQC_TEST_HOOKS)
    // Exercise the generated executor's real exception types without leaving a
    // failed CUDA context behind; explicit replay must retain the last-good seed.
    const auto injected = fail_next_xc_status;
    fail_next_xc_status = cudaSuccess;
    vibeqc_tensor::cuda_check(injected);
#endif
    cuda_xc_detail::enqueue(layout_, stream_, basis_, points_, weights_, density, ao_, work_,
                            features_, coefficients_, point_totals_, potential_, totals_, error_);
  } catch (const vibeqc_tensor::DeviceAllocationError&) {
    // The generated executor has a separate exception vocabulary. Translate at
    // this native owner boundary so both single-point and batch APIs preserve it.
    throw std::bad_alloc();
  } catch (const vibeqc_tensor::DeviceRuntimeError& error) {
    throw vibeqc::Error(VIBEQC_STATUS_CUDA_ERROR, error.what());
  }
  generation_ = generation;
  ++transfers_.evaluations;
}

CudaXcView CudaXcPlan::view(std::uint64_t generation) const {
  check_device();
  if (!generation || generation != generation_)
    throw std::invalid_argument("CUDA XC result generation is stale");
  return {generation, layout_.nao, layout_.spins, potential_, totals_, error_, stream_};
}

CudaXcScalars CudaXcPlan::read_scalars(std::uint64_t generation) {
  const auto result = view(generation);
  double values[3]{};
  CudaXcScalars out;
  try {
    check(cudaMemcpyAsync(values, result.totals, sizeof(values), cudaMemcpyDeviceToHost, stream_));
    check(cudaMemcpyAsync(&out.error, result.error, sizeof(out.error), cudaMemcpyDeviceToHost,
                          stream_));
    check(cudaStreamSynchronize(stream_));
  } catch (...) {
    cudaStreamSynchronize(stream_);
    throw;
  }
  out.energy = values[0];
  out.electrons = {values[1], values[2]};
  transfers_.output_d2h_bytes += sizeof(values) + sizeof(out.error);
  ++transfers_.synchronizations;
  return out;
}

std::vector<double> CudaXcPlan::download_potential(std::uint64_t generation) {
  const auto result = view(generation);
  std::vector<double> output(layout_.spins * layout_.nao * layout_.nao);
  try {
    check(cudaMemcpyAsync(output.data(), result.potential, output.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, stream_));
    check(cudaStreamSynchronize(stream_));
  } catch (...) {
    cudaStreamSynchronize(stream_);
    throw;
  }
  transfers_.output_d2h_bytes += output.size() * sizeof(double);
  ++transfers_.synchronizations;
  return output;
}
}  // namespace vibeqc::dft
