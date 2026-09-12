#pragma once

namespace vibeqc::scf::cuda_execution {

// Matrix reductions use one complete warp per system. Keep this independent
// from the generic capture-safe launch width so tuning other kernels cannot
// silently drop reductions from additional warps.
constexpr unsigned kMatrixReductionThreads = 32;
// External warm densities require an O(N^2) symmetry and metric-trace pass
// before they can enter a captured SCF replay. One block owns each system so
// batch-size-one production runs can spread that setup scan across the GPU.
constexpr unsigned kWarmDensityThreads = 256;
static_assert(kWarmDensityThreads % 32 == 0);
// Direct J/K scatters millions of independently evaluated AO quartets through
// FP64 atomics. Their nondeterministic accumulation order changes the total
// energy by a small number of representable values even after the density is
// stationary. Add only a machine-precision-scaled comparison guard; the
// requested absolute tolerance remains the dominant term for ordinary cases.
constexpr double kDirectFockEnergyRoundoffFactor = 16.0;
constexpr double kDoubleMachineEpsilon = 2.2204460492503131e-16;

}  // namespace vibeqc::scf::cuda_execution
