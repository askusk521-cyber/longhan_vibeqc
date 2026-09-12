#pragma once

#include "scf/cuda_density_fitting.hpp"

namespace vibeqc::scf::cuda_df {

/** Build mathematical J/K into plan-owned device buffers on plan.stream.
 * The same interfaces implement resident, host-backed and source-backed tiles.
 */
vibeqc_status build_coulomb(CudaDensityFittingJkPlan& plan, const double* density,
                            std::string& detail);

vibeqc_status build_exchange(CudaDensityFittingJkPlan& plan, const double* density,
                             double* exchange, std::string& detail,
                             bool density_is_column_major = false);

}  // namespace vibeqc::scf::cuda_df
