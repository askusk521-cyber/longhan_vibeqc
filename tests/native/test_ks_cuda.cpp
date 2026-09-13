#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "dft/cuda_ks.hpp"
#include "dft/xc.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/mean_field.hpp"

namespace {
using namespace vibeqc;
using scf::reference::Matrix;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
core::System hydrogens(unsigned count, bool restricted, double shift = 0.0) {
  core::System system;
  system.multiplicity = restricted ? 1 : 2;
  system.charge = !restricted && count == 2 ? 1 : 0;
  for (unsigned i = 0; i < count; ++i) {
    system.atoms.push_back({1, {0.15 * i * i, 0.13 * i, (1.5 + shift) * i}});
    system.shells.push_back(
        {i,
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}});
  }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          detail.c_str());
  return system;
}
scf::ResolvedFockBuild strategy(bool restricted, scf::FockBackend backend) {
  scf::FockBuildSpec spec;
  spec.spin = restricted ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted;
  spec.exchange.present = false;
  spec.derivative_order = 0;
  return scf::resolve_fock_build(spec, backend, 1e-12);
}

/** Independently rebuild the retained density with CPU integrals/XC. This
 * catches a converged flag or energy belonging to the preceding generation. */
void physical_check(const scf::PreparedFockPlan& cpu, const dft::AoBasis& basis,
                    const dft::MolecularGrid& grid, bool pbe, const scf::ScfResult& result) {
  using namespace scf::reference;
  const auto n = basis.nao, elements = n * n;
  const bool uks = cpu.strategy().spec.spin == scf::FockSpin::Unrestricted;
  Matrix a(result.density.begin(), result.density.begin() + elements), b;
  if (uks) b.assign(result.density.begin() + elements, result.density.end());
  const auto jk = cpu.build(a, b);
  auto fock = scf::assemble_fock(cpu.strategy(), cpu.one_electron().hcore, jk);
  double xc_energy;
  if (uks) {
    const auto xc = pbe ? dft::integrate_pbe_uks(basis, grid, a, b)
                        : dft::integrate_lda_xc_pw_uks(basis, grid, a, b);
    xc_energy = xc.energy;
    for (std::size_t i = 0; i < elements; ++i) {
      fock.alpha[i] += xc.potential[0][i];
      fock.beta[i] += xc.potential[1][i];
    }
  } else {
    const auto xc = pbe ? dft::integrate_pbe_rks_with_tail(basis, grid, a)
                        : dft::integrate_lda_xc_pw_rks(basis, grid, a);
    xc_energy = xc.energy;
    for (std::size_t i = 0; i < elements; ++i) fock.alpha[i] += xc.potential[i];
  }
  const auto& ints = cpu.one_electron();
  const double energy = ints.nuclear_repulsion + dot(a, ints.hcore) +
                        (uks ? dot(b, ints.hcore) : 0.0) +
                        scf::contract_fock_energy(cpu.strategy(), jk, a, b) + xc_energy;
  const auto residual_a = commutator_residual(fock.alpha, a, ints.overlap, n);
  const auto residual_b = uks ? commutator_residual(fock.beta, b, ints.overlap, n) : Matrix{};
  const double residual = std::max(residual_rms(residual_a), uks ? residual_rms(residual_b) : 0.0);
  const double public_residual =
      residual_rms(uks ? concatenate(residual_a, residual_b) : residual_a);
  require(std::abs(energy - result.energy) < 2e-11, "CUDA E and returned D are inconsistent");
  require(std::abs(residual - result.dft_diagnostic.physical_residual) < 2e-11,
          "CUDA physical residual belongs to another state");
  require(std::abs(public_residual - result.physical_residual_rms) < 2e-11,
          "CUDA public RMS no longer combines the physical spin-matrix entries");
  if (result.converged)
    require(residual < 1e-9, "CUDA reported convergence above the physical gate");
  require(std::abs(energy - (result.energy + xc_energy)) > 0.05,
          "CUDA endpoint gate does not detect XC double counting");
}

/** OH exercises the stationary integer-occupation cycle from #305 on CUDA.
 * Rebuild every returned physical quantity with the unshifted CPU operator. */
void run_hydroxyl(bool pbe) {
  core::System system;
  system.multiplicity = 2;
  system.atoms = {{8, {0, 0, 0}}, {1, {0, 0, 1.8}}};
  system.shells = {
      {0,
       0,
       {{130.7093214, 0.1543289673}, {23.80886605, 0.5353281423}, {6.443608313, 0.4446345422}}},
      {0,
       0,
       {{5.033151319, -0.09996722919}, {1.169596125, 0.3995128261}, {0.38038896, 0.7001154689}}},
      {0, 1, {{5.033151319, 0.155916275}, {1.169596125, 0.6076837186}, {0.38038896, 0.3919573931}}},
      {1,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}}};
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          detail.c_str());
  const dft::AoBasis basis(system);
  const dft::MolecularGrid grid(system);
  const scf::PreparedFockPlan cpu(system, nullptr, strategy(false, scf::FockBackend::Cpu));
  const scf::PreparedFockPlan gpu(system, nullptr, strategy(false, scf::FockBackend::Cuda), 0);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.max_iterations = 200;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  dft::CudaKsPlan plan(gpu, basis, grid, options, pbe);
  const auto cold = plan.run(nullptr, false, false);
  require(cold.converged && !plan.failed(), "CUDA OH occupation cycle did not converge");
  require(plan.transfers().matrix_d2h_bytes == 0,
          "CUDA occupation stabilization exported iteration matrices");
  physical_check(cpu, basis, grid, pbe, plan.result());
  const auto& history = cold.dft_diagnostic.history;
  const auto cycle = std::find_if(history.begin(), history.end(), [&](const auto& item) {
    return item.iteration > 1 && item.energy_change < options.energy_tolerance &&
           item.physical_residual < options.density_tolerance &&
           item.density_change >= options.density_tolerance;
  });
  if (!pbe) require(cycle != history.end(), "OH regression did not exercise the occupation cycle");
  if (cycle != history.end()) {
    require(cycle->iteration < cold.iterations,
            "stationary energy bypassed the subsequent density-change gate");
    require(plan.transfers().occupation_stabilized_proposals > 0,
            "CUDA stationary cycle did not apply the CPU-compatible proposal policy");
  }
  for (const auto& item : history)
    require(std::abs(item.electrons[0] - 5) < 1e-10 && std::abs(item.electrons[1] - 4) < 1e-10,
            "occupation stabilization changed the requested spin populations");
  const auto warm = plan.run();
  require(
      warm.converged && warm.initial_density_used && std::abs(warm.energy - cold.energy) < 1e-10,
      "CUDA OH resident replay lost its physical endpoint");
  physical_check(cpu, basis, grid, pbe, warm);
  const auto restarted = plan.run(nullptr, false);
  require(restarted.converged && !restarted.initial_density_used &&
              std::abs(restarted.energy - cold.energy) < 1e-10,
          "CUDA OH cold restart retained stale proposal control");
  physical_check(cpu, basis, grid, pbe, restarted);
  std::cout << "KS OH pbe=" << pbe << " iterations=" << cold.iterations << '\n';
}

void run_case(unsigned atoms, bool restricted, bool pbe) {
  const auto system = hydrogens(atoms, restricted);
  const dft::AoBasis basis(system);
  const dft::GridSpec grid_spec{1, 24, 12, 24, 3, 1e-12};
  const dft::MolecularGrid grid(system, grid_spec);
  const scf::PreparedFockPlan cpu(system, nullptr, strategy(restricted, scf::FockBackend::Cpu));
  const scf::PreparedFockPlan gpu(system, nullptr, strategy(restricted, scf::FockBackend::Cuda), 0);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  options.max_iterations = 150;
  dft::CudaKsPlan plan(gpu, basis, grid, options, pbe, 257);
  plan.begin(nullptr, false);
  while (plan.active()) {
    plan.enqueue_iteration();
    require(plan.pending(), "CUDA iteration did not retain pending state");
    plan.finish_iteration();
  }
  require(plan.transfers().matrix_d2h_bytes == 0, "CUDA SCF staged an iteration matrix");
  const auto result = plan.result();
  if (!result.converged || plan.failed()) {
    std::cerr << "failed atoms=" << atoms << " restricted=" << restricted << " pbe=" << pbe
              << " iter=" << result.iterations
              << " residual=" << result.dft_diagnostic.physical_residual
              << " density=" << result.density_rms << '\n';
    throw std::runtime_error("native CUDA KS did not converge");
  }
  const auto reference = !restricted ? scf::run_uks(cpu, basis, grid, options, pbe)
                         : pbe       ? scf::run_pbe_rks(cpu, basis, grid, options)
                                     : scf::run_lda_rks(cpu, basis, grid, options);
  require(reference.converged && std::abs(reference.energy - result.energy) < 1e-10,
          "CPU/CUDA SCF endpoints disagree");
  physical_check(cpu, basis, grid, pbe, result);
  require(result.iterations == result.dft_diagnostic.history.size(),
          "missing CUDA iteration history");
  const auto before = plan.transfers();
  const auto warm = plan.run();
  const auto after = plan.transfers();
  require(warm.converged && warm.initial_density_used && warm.iterations <= result.iterations &&
              std::abs(warm.energy - result.energy) < 1e-11 &&
              before.density_h2d_bytes == after.density_h2d_bytes,
          "unchanged-geometry replay did not reuse resident warm density");
  const auto energy_only = plan.run(nullptr, true, false);
  require(energy_only.converged && energy_only.density.empty() &&
              plan.transfers().matrix_d2h_bytes == after.matrix_d2h_bytes,
          "energy-only CUDA KS exported a final density matrix");
  const auto frozen_density = plan.warm_density();
  plan.set_warm_start_updates(false);
  const auto frozen_before = plan.transfers();
  const auto frozen_result = plan.run(nullptr, false, false);
  require(frozen_result.converged && !frozen_result.initial_density_used &&
              plan.transfers().matrix_d2h_bytes == frozen_before.matrix_d2h_bytes,
          "frozen cold solve performed an implicit density export");
  require(plan.warm_density() == frozen_density,
          "successful frozen solve replaced the resident last-good density");
  plan.clear_warm_start();
  require(plan.warm_density().empty(), "cleared CUDA seed remains visible");
  require(plan.run(nullptr, true, false).converged && plan.warm_density().empty(),
          "frozen CUDA owner established a new seed");
  plan.set_warm_start_updates(true);
  require(plan.run(nullptr, false, false).converged && !plan.warm_density().empty(),
          "unfrozen CUDA owner failed to establish a seed");

  if (atoms > 1) {
    auto invalid = result.density;
    std::fill(invalid.begin(), invalid.begin() + atoms * atoms, 0.0);
    invalid[0] = -1.0;
    invalid[atoms * atoms - 1] = 2.0;
    const auto failed = plan.run(&invalid);
    require(plan.failed() && !failed.converged, "invalid grid density did not fail the CUDA item");
    const auto recovered = plan.run();
    require(recovered.converged && std::abs(recovered.energy - result.energy) < 1e-11,
            "failed CUDA item replaced its last-good warm state");
    const auto moved = hydrogens(atoms, restricted, 0.2);
    const scf::PreparedFockPlan new_gpu(moved, nullptr,
                                        strategy(restricted, scf::FockBackend::Cuda), 0);
    const dft::AoBasis new_basis(moved);
    const dft::MolecularGrid new_grid(moved, grid_spec);
    bool stale = false;
    try {
      dft::CudaKsPlan wrong(new_gpu, new_basis, grid, options, pbe);
    } catch (const std::invalid_argument&) {
      stale = true;
    }
    require(stale, "CUDA SCF accepted a same-shape old grid");
    dft::CudaKsPlan changed(new_gpu, new_basis, new_grid, options, pbe);
    auto seed = plan.warm_density();
    for (auto& value : seed) value *= 1.3;
    const auto moved_warm = changed.run(&seed), moved_cold = changed.run(nullptr, false);
    require(moved_warm.converged && moved_cold.converged &&
                std::abs(moved_warm.energy - moved_cold.energy) < 1e-10,
            "changed-geometry warm normalization changed the endpoint");
  }
  options.max_iterations = 1;
  dft::CudaKsPlan unfinished(gpu, basis, grid, options, pbe);
  const auto limited = unfinished.run();
  require(!limited.converged && !unfinished.failed(), "iteration limit misreported its status");
  require(unfinished.warm_density().empty(), "unfinished solve published a good warm state");
  physical_check(cpu, basis, grid, pbe, limited);

  // Exact arena request is charged through the existing #203 device ledger.
  auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
  ledger->limit = plan.resources().state_device_bytes + plan.resources().xc_device_bytes;
  ledger->device = 0;
  const auto previous = runtime::active_device_resource_ledger;
  runtime::active_device_resource_ledger = ledger;
  try {
    {
      dft::CudaKsPlan measured(gpu, basis, grid, options, pbe, 257);
      require(ledger->live ==
                  measured.resources().state_device_bytes + measured.resources().xc_device_bytes,
              "CUDA KS resource request omits explicit allocations");
    }
    require(ledger->live == 0, "CUDA KS retained charged memory after destruction");
  } catch (...) {
    runtime::active_device_resource_ledger = previous;
    throw;
  }
  runtime::active_device_resource_ledger = previous;
  std::cout << "KS atoms=" << atoms << " restricted=" << restricted << " pbe=" << pbe
            << " iterations=" << result.iterations
            << " residual=" << result.dft_diagnostic.physical_residual << '\n';
}
}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  try {
    for (bool pbe : {false, true}) {
      run_case(2, true, pbe);
      run_hydroxyl(pbe);
      for (unsigned atoms : {1U, 2U, 3U}) run_case(atoms, false, pbe);
    }
    std::cout << "Native CUDA KS SCF, physical-state, warm/failure/resource gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
