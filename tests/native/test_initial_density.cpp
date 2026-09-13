#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/reference/observation.hpp"

namespace {
using namespace vibeqc;
using namespace scf::initial_guess;
using namespace scf::reference;

unsigned solves{};
std::size_t begin(const char* name, std::size_t) noexcept {
  if (std::string_view(name) == "reference_eigensolve") ++solves;
  return 0;
}
void end(std::size_t, int) noexcept {}
const observation::Observer observer{begin, end};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void close(const Matrix& actual, const Matrix& expected) {
  require(actual.size() == expected.size(), "matrix shape changed");
  for (std::size_t i = 0; i < actual.size(); ++i)
    require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 1e-13,
            "initial-density convention changed");
}
template <class Function>
void invalid(Function&& function) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("invalid initial density was accepted");
}

void check_initial_density_contract() {
  // Diagonal, nonidentity S gives analytically known eigenpairs. Expected
  // densities below are independent of the production preparation helpers.
  integrals::IntegralData ints;
  ints.nbf = 3;
  ints.overlap = {2, 0, 0, 0, 1, 0, 0, 0, .5};
  ints.hcore = {-2, 0, 0, 0, -.5, 0, 0, 0, .25};
  const Matrix x{std::sqrt(.5), 0, 0, 0, 1, 0, 0, 0, std::sqrt(2.)};
  core::System system;
  system.electron_count = 2;
  std::optional<EigenResult> a, b;
  solves = 0;
  const auto cold = prepare_initial_density(system, ints, x, 1, nullptr, a);
  require(solves == 1 && a, "cold RHF must construct its core frame once");
  close(cold, {1, 0, 0, 0, 0, 0, 0, 0, 0});
  const auto [alpha, beta] = prepare_initial_uhf_density(ints, x, 2, 1, nullptr, a, b);
  require(solves == 2 && a && b, "cold UHF must share one core solve");
  close(alpha, {.5, 0, 0, 0, 1, 0, 0, 0, 0});
  // Historical 45-degree beta HOMO/LUMO mixing remains a cold-only choice.
  close(beta, {.25, std::sqrt(.125), 0, std::sqrt(.125), .5, 0, 0, 0, 0});

  const Matrix hcore = ints.hcore;
  std::fill(ints.hcore.begin(), ints.hcore.end(), std::numeric_limits<double>::quiet_NaN());
  const Matrix raw{2, .2, 0, .4, 0, 0, 0, 0, 0};
  solves = 0;
  close(prepare_initial_density(system, ints, x, 1, &raw, a), {1, .15, 0, .15, 0, 0, 0, 0, 0});
  require(solves == 0 && !a, "warm RHF must not compute or retain a core frame");
  Matrix spin = raw;
  spin.insert(spin.end(), raw.begin(), raw.end());
  auto [warm_alpha, warm_beta] = prepare_initial_uhf_density(ints, x, 2, 0, &spin, a, b);
  close(warm_alpha, {1, .15, 0, .15, 0, 0, 0, 0, 0});
  close(warm_beta, Matrix(9, 0));
  require(solves == 0 && !a && !b, "warm empty-spin UHF must clear both core frames");

  // A consumer needing orbitals requests them explicitly, after validating D.
  ints.hcore = hcore;
  prepare_initial_uhf_density(ints, x, 2, 1, &spin, a, b, InitialOrbitalRequest::RequireCoreFrame);
  require(solves == 1 && a && b, "explicit warm frame request did not solve");
  close(a->vectors, b->vectors);
  close(density_from_orbitals(b->vectors, 3, 1, 1), {.5, 0, 0, 0, 0, 0, 0, 0, 0});
  prepare_initial_density(system, ints, x, 1, &raw, a, InitialOrbitalRequest::RequireCoreFrame);
  require(solves == 2 && a, "explicit RHF warm frame request did not solve");

  for (const Matrix bad :
       {Matrix(2, 1), Matrix(9, 0), Matrix(9, std::numeric_limits<double>::infinity())}) {
    a = EigenResult{{1}, {1}};
    solves = 0;
    invalid([&] { prepare_initial_density(system, ints, x, 1, &bad, a); });
    require(solves == 0 && !a, "failed RHF validation leaked a frame or solved hcore");
  }
  for (const Matrix bad :
       {Matrix(2, 1), Matrix(18, 0), Matrix(18, std::numeric_limits<double>::quiet_NaN())}) {
    a = b = EigenResult{{1}, {1}};
    solves = 0;
    invalid([&] { prepare_initial_uhf_density(ints, x, 2, 1, &bad, a, b); });
    require(solves == 0 && !a && !b, "failed UHF validation leaked a frame or solved hcore");
  }
  // A failed item cannot poison a subsequent independent cold request.
  prepare_initial_density(system, ints, x, 1, nullptr, a);
  require(solves == 1 && a, "cold retry did not rebuild a valid frame");
}
}  // namespace

int main() {
  observation::active = &observer;
  try {
    check_initial_density_contract();
    observation::active = nullptr;
    return 0;
  } catch (const std::exception& error) {
    observation::active = nullptr;
    std::cerr << error.what() << '\n';
    return 1;
  }
}
