#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::scf::cuda_df {

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_assemble_rhf_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                     cudaStream_t stream, std::size_t elements, const double* hcore,
                                     const double* coulomb, const double* exchange, double* fock);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_assemble_uhf_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                     cudaStream_t stream, std::size_t elements, const double* hcore,
                                     const double* coulomb, const double* alpha_exchange,
                                     const double* beta_exchange, double* alpha_fock,
                                     double* beta_fock);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_build_device_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                        cudaStream_t stream, std::size_t batch_size,
                                        std::size_t nbf, const std::int32_t* occupied,
                                        const double* coefficients, double occupation_weight,
                                        double* density);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_compute_device_energy_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::size_t batch_size,
                                         std::size_t nbf, const double* density,
                                         const double* hcore, const double* fock,
                                         const double* nuclear_repulsion, double* energy);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_compute_device_uhf_energy_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                             cudaStream_t stream, std::size_t batch_size,
                                             std::size_t nbf, const double* alpha_density,
                                             const double* beta_density, const double* hcore,
                                             const double* alpha_fock, const double* beta_fock,
                                             const double* nuclear_repulsion, double* energy);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_update_device_convergence_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::size_t batch_size,
    std::size_t nbf, double energy_tolerance, double density_tolerance, const double* energy,
    double* previous_energy, const double* next_density, double* density, std::uint8_t* active,
    std::uint8_t* converged, std::uint32_t* iterations, double* energy_change, double* density_rms);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_tail_cuda_density_fitting_scf_graph_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    std::uint32_t maximum_iterations, const std::uint8_t* active, const std::uint32_t* iterations);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_update_device_uhf_convergence_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::size_t batch_size,
    std::size_t nbf, double energy_tolerance, double density_tolerance, const double* energy,
    double* previous_energy, const double* next_alpha, const double* next_beta,
    double* alpha_density, double* beta_density, std::uint8_t* active, std::uint8_t* converged,
    std::uint32_t* iterations, double* energy_change, double* density_rms);

}  // namespace vibeqc::scf::cuda_df
