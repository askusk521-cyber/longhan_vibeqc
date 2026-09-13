#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

/** Evaluate one already-prepared local CUDA grid task with the shared
 * LDA/PBE point algebra. Features and AO jets stay on the grid owner's
 * stream; only weights are uploaded and the three scalar integrals are
 * returned. The local spin potentials remain in the borrowed task buffer and
 * are consumed by grid_cuda_scatter_v1.
 */
int grid_cuda_xc_v1(void* pointer, std::uint64_t generation, int pbe, const double* weights,
                    std::size_t npoint, double* integrals, char* error, std::size_t size);
}
