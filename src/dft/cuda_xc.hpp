#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "dft/xc.hpp"

namespace vibeqc::dft {

/** Prepared native CUDA AO/grid/XC owner for one immutable basis.
 *
 * SCF control and the returned matrices remain host-side. Each evaluation
 * uploads the current two-spin density, evaluates bounded point tiles and
 * downloads the accumulated two-spin XC potential. The plan is reusable but
 * serializes complete evaluations so tile leases cannot interleave.
 */
class PreparedCudaXcPlan {
 public:
  PreparedCudaXcPlan(const AoBasis& basis, int device_id, int compute_capability_major,
                     int compute_capability_minor, bool pbe, std::size_t tile_points = 256);
  ~PreparedCudaXcPlan();
  PreparedCudaXcPlan(const PreparedCudaXcPlan&) = delete;
  PreparedCudaXcPlan& operator=(const PreparedCudaXcPlan&) = delete;

  XcIntegral evaluate_rks(const MolecularGrid& grid, const std::vector<double>& density);
  SpinXcIntegral evaluate_uks(const MolecularGrid& grid, const std::vector<double>& alpha_density,
                              const std::vector<double>& beta_density);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vibeqc::dft
