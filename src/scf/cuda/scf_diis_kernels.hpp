#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::scf::cuda_execution {

/** Preserve launch geometry, stream and per-item state routing. */
void launch_update_diis_kernel(dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
                               std::int32_t batch_size, std::int32_t nbf,
                               std::int32_t matrices_per_system, std::uint32_t history_capacity,
                               const double* fock, const double* residual,
                               const std::uint8_t* active, double* fock_history,
                               double* residual_history, double* linear_system,
                               double* coefficients, std::uint32_t* history_count,
                               std::uint32_t* history_head, double* effective_fock);

}  // namespace vibeqc::scf::cuda_execution
