#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

namespace vibeqc::dft {

/** Physical energy terms, never the half-trace of an XC-containing Fock. */
struct EnergyComponents {
  double nuclear{}, one_electron{}, hartree{}, xc{};
  double total() const noexcept { return nuclear + one_electron + hartree + xc; }
};

/** A physical evaluation and its proposed density change. Each spin residual
 * and density update must pass separately; an empty spin cannot dilute RMS. */
struct ScfIteration {
  unsigned iteration{};
  EnergyComponents components;
  double energy_change{}, density_change{}, physical_residual{};
  std::array<double, 2> electrons{};
};

/** Method-owned diagnostics for one solve; history is reset for every replay.
 * Electron counts use Tr(D_s S), independently of quadrature electron counts.
 */
struct ScfDiagnostic {
  std::array<std::size_t, 2> occupations{};
  double physical_residual{std::numeric_limits<double>::infinity()};
  EnergyComponents components;
  std::vector<ScfIteration> history;
};
}  // namespace vibeqc::dft
