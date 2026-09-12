#pragma once

#include "scf/cuda/df_scf_state.hpp"
#include "scf/cuda_density_fitting.hpp"

namespace vibeqc::scf::cuda_df {

/** cuBLAS/cuSOLVER integration for retained DF SCF state.
 * Borrow plan streams and own solver workspace setup independently of J/K selection.
 */
vibeqc_status scf_gemm(CudaDensityFittingJkPlan& plan, bool transpose_left, std::size_t batch_size,
                       std::size_t nbf, const double* left, const double* right, double* output,
                       std::string& detail);

vibeqc_status setup_device_solver(CudaDensityFittingJkPlan& plan, std::size_t nbf,
                                  std::size_t batch_size, double* eigensystem, double* eigenvalues,
                                  DeviceSolver& solver, std::string& detail);

vibeqc_status solve_device_batch(DeviceSolver& solver, std::size_t nbf, std::size_t batch_size,
                                 double* eigensystem, double* eigenvalues, int* info,
                                 std::string& detail);

}  // namespace vibeqc::scf::cuda_df
