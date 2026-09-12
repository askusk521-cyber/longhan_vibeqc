#include <math_constants.h>

#include <cmath>

#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/scf_diis_kernels.hpp"

namespace vibeqc::scf::cuda_execution {

__global__ void update_diis_kernel(std::int32_t batch_size, std::int32_t nbf,
                                   std::int32_t matrices_per_system, std::uint32_t history_capacity,
                                   const double* fock, const double* residual,
                                   const std::uint8_t* active, double* fock_history,
                                   double* residual_history, double* linear_system,
                                   double* coefficients, std::uint32_t* history_count,
                                   std::uint32_t* history_head, double* effective_fock) {
  // One warp owns one system.  History vectors and the O(N^2) residual-dot
  // products are distributed across lanes, while the small dense DIIS solve
  // remains in lane zero.  This preserves the original dot-product order for
  // each B-matrix entry and avoids the old single-thread N^2 bottleneck.
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t vector_size = matrix_size * static_cast<std::size_t>(matrices_per_system);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * vector_size;
  if (history_capacity < 2) {
    for (std::size_t element = threadIdx.x; element < vector_size; element += blockDim.x) {
      effective_fock[matrix_offset + element] = fock[matrix_offset + element];
    }
    return;
  }

  const std::size_t history_stride = static_cast<std::size_t>(history_capacity) * vector_size;
  std::uint32_t slot = 0;
  if (threadIdx.x == 0) slot = history_head[system];
  slot = __shfl_sync(0xffffffffU, slot, 0);
  const std::size_t slot_offset = static_cast<std::size_t>(system) * history_stride +
                                  static_cast<std::size_t>(slot) * vector_size;
  for (std::size_t element = threadIdx.x; element < vector_size; element += blockDim.x) {
    fock_history[slot_offset + element] = fock[matrix_offset + element];
    residual_history[slot_offset + element] = residual[matrix_offset + element];
  }
  __syncwarp();
  std::uint32_t count = 0;
  if (threadIdx.x == 0) {
    count = history_count[system] < history_capacity ? history_count[system] + 1 : history_capacity;
    history_count[system] = count;
    history_head[system] = (slot + 1) % history_capacity;
  }
  count = __shfl_sync(0xffffffffU, count, 0);
  if (count < 2) {
    for (std::size_t element = threadIdx.x; element < vector_size; element += blockDim.x) {
      effective_fock[matrix_offset + element] = fock[matrix_offset + element];
    }
    return;
  }

  const std::uint32_t dimension = count + 1;
  const std::size_t system_stride =
      static_cast<std::size_t>(history_capacity + 1) * (history_capacity + 1);
  double* matrix = linear_system + static_cast<std::size_t>(system) * system_stride;
  double* rhs = coefficients + static_cast<std::size_t>(system) * (history_capacity + 1);
  const std::size_t linear_elements = static_cast<std::size_t>(dimension) * dimension;
  for (std::size_t element = threadIdx.x; element < linear_elements; element += blockDim.x) {
    matrix[element] = 0.0;
  }
  for (std::uint32_t row = threadIdx.x; row < dimension; row += blockDim.x) {
    rhs[row] = row == count ? -1.0 : 0.0;
  }
  __syncwarp();
  const std::size_t dot_count = static_cast<std::size_t>(count) * count;
  for (std::size_t pair = threadIdx.x; pair < dot_count; pair += blockDim.x) {
    const std::uint32_t row = static_cast<std::uint32_t>(pair / count);
    const std::uint32_t column = static_cast<std::uint32_t>(pair % count);
    const std::size_t row_offset = static_cast<std::size_t>(system) * history_stride +
                                   static_cast<std::size_t>(row) * vector_size;
    const std::size_t column_offset = static_cast<std::size_t>(system) * history_stride +
                                      static_cast<std::size_t>(column) * vector_size;
    double dot = 0.0;
    for (std::size_t element = 0; element < vector_size; ++element) {
      dot += residual_history[row_offset + element] * residual_history[column_offset + element];
    }
    matrix[static_cast<std::size_t>(row) * dimension + column] = dot;
  }
  __syncwarp();
  if (threadIdx.x == 0) {
    for (std::uint32_t row = 0; row < count; ++row) {
      matrix[static_cast<std::size_t>(row) * dimension + count] = -1.0;
      matrix[static_cast<std::size_t>(count) * dimension + row] = -1.0;
    }
  }
  __syncwarp();

  int nonsingular = 1;
  if (threadIdx.x == 0) {
    for (std::uint32_t column = 0; column < dimension; ++column) {
      std::uint32_t pivot = column;
      for (std::uint32_t row = column + 1; row < dimension; ++row) {
        if (fabs(matrix[static_cast<std::size_t>(row) * dimension + column]) >
            fabs(matrix[static_cast<std::size_t>(pivot) * dimension + column])) {
          pivot = row;
        }
      }
      const double diagonal = matrix[static_cast<std::size_t>(pivot) * dimension + column];
      if (fabs(diagonal) < 1.0e-14) {
        nonsingular = 0;
        break;
      }
      if (pivot != column) {
        for (std::uint32_t item = 0; item < dimension; ++item) {
          const std::size_t first = static_cast<std::size_t>(column) * dimension + item;
          const std::size_t second = static_cast<std::size_t>(pivot) * dimension + item;
          const double swap = matrix[first];
          matrix[first] = matrix[second];
          matrix[second] = swap;
        }
        const double swap = rhs[column];
        rhs[column] = rhs[pivot];
        rhs[pivot] = swap;
      }
      const double scale = matrix[static_cast<std::size_t>(column) * dimension + column];
      for (std::uint32_t item = column; item < dimension; ++item) {
        matrix[static_cast<std::size_t>(column) * dimension + item] /= scale;
      }
      rhs[column] /= scale;
      for (std::uint32_t row = 0; row < dimension; ++row) {
        if (row == column) continue;
        const double factor = matrix[static_cast<std::size_t>(row) * dimension + column];
        for (std::uint32_t item = column; item < dimension; ++item) {
          matrix[static_cast<std::size_t>(row) * dimension + item] -=
              factor * matrix[static_cast<std::size_t>(column) * dimension + item];
        }
        rhs[row] -= factor * rhs[column];
      }
    }
  }
  __syncwarp();
  // The solve is lane-zero-only; broadcast its success flag before any lane
  // decides whether it should form the extrapolated Fock matrix.
  nonsingular = __shfl_sync(0xffffffffU, nonsingular, 0);

  for (std::size_t element = threadIdx.x; element < vector_size; element += blockDim.x) {
    double value = fock[matrix_offset + element];
    if (nonsingular) {
      value = 0.0;
      for (std::uint32_t item = 0; item < count; ++item) {
        const std::size_t item_offset = static_cast<std::size_t>(system) * history_stride +
                                        static_cast<std::size_t>(item) * vector_size;
        value += rhs[item] * fock_history[item_offset + element];
      }
    }
    effective_fock[matrix_offset + element] = value;
  }
}

void launch_update_diis_kernel(dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
                               std::int32_t batch_size, std::int32_t nbf,
                               std::int32_t matrices_per_system, std::uint32_t history_capacity,
                               const double* fock, const double* residual,
                               const std::uint8_t* active, double* fock_history,
                               double* residual_history, double* linear_system,
                               double* coefficients, std::uint32_t* history_count,
                               std::uint32_t* history_head, double* effective_fock) {
  update_diis_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, matrices_per_system, history_capacity, fock, residual, active, fock_history,
      residual_history, linear_system, coefficients, history_count, history_head, effective_fock);
}

}  // namespace vibeqc::scf::cuda_execution
