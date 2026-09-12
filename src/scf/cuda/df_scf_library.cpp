#include "scf/cuda/df_scf_library.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"

namespace vibeqc::scf::cuda_df {

vibeqc_status scf_gemm(CudaDensityFittingJkPlan& plan, bool transpose_left, std::size_t batch_size,
                       std::size_t nbf, const double* left, const double* right, double* output,
                       std::string& detail) {
  const double one = 1.0;
  const double zero = 0.0;
  const std::size_t matrix_elements = nbf * nbf;
  const cublasStatus_t status = cublasDgemmStridedBatched(
      plan.blas, transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(nbf),
      static_cast<int>(nbf), static_cast<int>(nbf), &one, left, static_cast<int>(nbf),
      static_cast<long long>(matrix_elements), right, static_cast<int>(nbf),
      static_cast<long long>(matrix_elements), &zero, output, static_cast<int>(nbf),
      static_cast<long long>(matrix_elements), static_cast<int>(batch_size));
  return status == CUBLAS_STATUS_SUCCESS
             ? VIBEQC_STATUS_SUCCESS
             : blas_failure(status, "CUDA DF device matrix product", detail);
}

vibeqc_status setup_device_solver(CudaDensityFittingJkPlan& plan, std::size_t nbf,
                                  std::size_t batch_size, double* eigensystem, double* eigenvalues,
                                  DeviceSolver& solver, std::string& detail) {
  cusolverStatus_t status = cusolverDnCreate(&solver.handle);
  if (status == CUSOLVER_STATUS_SUCCESS) {
    status = cusolverDnSetStream(solver.handle, plan.stream);
  }
  solver.xsyev = nbf > 32;
  if (status == CUSOLVER_STATUS_SUCCESS && !solver.xsyev) {
    status = cusolverDnCreateSyevjInfo(&solver.jacobi);
    if (status == CUSOLVER_STATUS_SUCCESS) {
      status = cusolverDnXsyevjSetTolerance(solver.jacobi, 1.0e-13);
    }
    if (status == CUSOLVER_STATUS_SUCCESS) {
      status = cusolverDnXsyevjSetMaxSweeps(solver.jacobi, 100);
    }
    if (status == CUSOLVER_STATUS_SUCCESS) {
      status = cusolverDnXsyevjSetSortEig(solver.jacobi, 1);
    }
  } else if (status == CUSOLVER_STATUS_SUCCESS) {
    status = cusolverDnCreateParams(&solver.parameters);
  }
  if (status != CUSOLVER_STATUS_SUCCESS) {
    return solver_failure(status, "initialize CUDA DF SCF eigensolver", detail);
  }
  if (!solver.xsyev) {
    status = cusolverDnDsyevjBatched_bufferSize(
        solver.handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, static_cast<int>(nbf),
        eigensystem, static_cast<int>(nbf), eigenvalues, &solver.lwork, solver.jacobi,
        static_cast<int>(batch_size));
    if (status != CUSOLVER_STATUS_SUCCESS || solver.lwork <= 0) {
      return solver_failure(
          status == CUSOLVER_STATUS_SUCCESS ? CUSOLVER_STATUS_INTERNAL_ERROR : status,
          "size CUDA DF SCF eigensolver", detail);
    }
    return allocate_device(reinterpret_cast<void**>(&solver.workspace),
                           static_cast<std::size_t>(solver.lwork) * sizeof(double),
                           "allocate CUDA DF SCF eigensolver workspace", detail);
  }
  std::size_t device_bytes = 0;
  std::size_t host_bytes = 0;
  status = cusolverDnXsyevBatched_bufferSize(
      solver.handle, solver.parameters, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      static_cast<int>(nbf), CUDA_R_64F, eigensystem, static_cast<int>(nbf), CUDA_R_64F,
      eigenvalues, CUDA_R_64F, &device_bytes, &host_bytes, static_cast<int>(batch_size));
  if (status != CUSOLVER_STATUS_SUCCESS || device_bytes == 0) {
    return solver_failure(
        status == CUSOLVER_STATUS_SUCCESS ? CUSOLVER_STATUS_INTERNAL_ERROR : status,
        "size CUDA DF SCF generic eigensolver", detail);
  }
  solver.workspace_bytes = device_bytes;
  solver.host_workspace_bytes = host_bytes;
  vibeqc_status allocation =
      allocate_device(reinterpret_cast<void**>(&solver.workspace), device_bytes,
                      "allocate CUDA DF SCF generic eigensolver workspace", detail);
  if (allocation != VIBEQC_STATUS_SUCCESS) return allocation;
  if (host_bytes != 0) {
    solver.host_workspace = std::malloc(host_bytes);
    if (solver.host_workspace == nullptr) {
      detail = "host allocation for CUDA DF SCF generic eigensolver failed";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status solve_device_batch(DeviceSolver& solver, std::size_t nbf, std::size_t batch_size,
                                 double* eigensystem, double* eigenvalues, int* info,
                                 std::string& detail) {
  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
  if (!solver.xsyev) {
    status = cusolverDnDsyevjBatched(
        solver.handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, static_cast<int>(nbf),
        eigensystem, static_cast<int>(nbf), eigenvalues, solver.workspace, solver.lwork, info,
        solver.jacobi, static_cast<int>(batch_size));
  } else {
    status = cusolverDnXsyevBatched(
        solver.handle, solver.parameters, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        static_cast<int>(nbf), CUDA_R_64F, eigensystem, static_cast<int>(nbf), CUDA_R_64F,
        eigenvalues, CUDA_R_64F, solver.workspace, solver.workspace_bytes, solver.host_workspace,
        solver.host_workspace_bytes, info, static_cast<int>(batch_size));
  }
  return status == CUSOLVER_STATUS_SUCCESS
             ? VIBEQC_STATUS_SUCCESS
             : solver_failure(status, "CUDA DF SCF eigensolve", detail);
}

}  // namespace vibeqc::scf::cuda_df
