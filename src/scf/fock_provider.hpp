#ifndef VIBEQC_SCF_FOCK_PROVIDER_HPP
#define VIBEQC_SCF_FOCK_PROVIDER_HPP

#include <optional>

#include "scf/density_fitting.hpp"
#include "scf/fock_build.hpp"

namespace vibeqc::scf {

/** Typed, non-owning view of one immutable CPU provider's prepared integrals.
 * The enclosing geometry cache owns the data and must outlive this view and
 * every plan using it. Replacing geometry, AO representation, auxiliary basis,
 * or the metric cutoff requires rebinding the view. No integral tensor is
 * copied and no independent cache is introduced at this boundary.
 */
class CpuFockProviderView {
 public:
  explicit CpuFockProviderView(const integrals::IntegralData& exact);
  explicit CpuFockProviderView(const DensityFittingScfData& fitted);
  // Do not allow a view of a temporary integral owner.
  CpuFockProviderView(integrals::IntegralData&&) = delete;
  CpuFockProviderView(DensityFittingScfData&&) = delete;

  FockApproximation approximation() const;
  std::size_t nbf() const;
  std::size_t ncoord() const;
  bool operator==(const CpuFockProviderView&) const = default;

 private:
  friend class CpuFockPlanView;
  void validate(const ResolvedFockBuild& strategy) const;
  DirectJkMatrices build(FockBuildSpec spec, const std::vector<double>& density,
                         const std::vector<double>& beta) const;
  std::vector<double> derivative(FockBuildSpec spec, const std::vector<double>& density,
                                 const std::vector<double>& beta) const;
  const integrals::IntegralData* exact_{};
  const DensityFittingScfData* fitted_{};
};

/** Independent J/K binding with preflight of both providers before execution.
 * Compatible callers may share a single provider for both terms; its combined
 * J/K and response services then execute once. Absent terms require no source.
 * The plan returns results by value, so a failed provider cannot publish a
 * partial Fock or change a neighboring item's density/warm state.
 */
class CpuFockPlanView {
 public:
  CpuFockPlanView(ResolvedFockBuild strategy, std::size_t nbf, std::size_t ncoord,
                  std::optional<CpuFockProviderView> coulomb = {},
                  std::optional<CpuFockProviderView> exchange = {});
  const ResolvedFockBuild& strategy() const { return strategy_; }
  DirectJkMatrices build(const std::vector<double>& density,
                         const std::vector<double>& beta = {}) const;
  /** Fixed-density two-electron gradient only; Pulay/one-electron/nuclear
   * assembly remains in the method. Uses exactly the value-side providers. */
  std::vector<double> energy_derivative(const std::vector<double>& density,
                                        const std::vector<double>& beta = {}) const;

 private:
  void validate_density(const std::vector<double>& density, const std::vector<double>& beta) const;
  ResolvedFockBuild strategy_;
  std::size_t nbf_{}, ncoord_{};
  std::optional<CpuFockProviderView> coulomb_, exchange_;
};

}  // namespace vibeqc::scf
#endif
