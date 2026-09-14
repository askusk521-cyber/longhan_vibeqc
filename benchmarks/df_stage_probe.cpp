/** Bounded setup and fixed-density J/dense-K/occupied-K diagnosis for #308.
 * The input freezes physical geometry/basis and converged canonical orbitals;
 * preparation is timed separately. A Python driver records source/library/input
 * hashes. Execute only under a finite Slurm allocation, with tracing disabled
 * for endpoint measurements and a separate traced invocation for components.
 */
#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "runtime/df_progress_trace.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
void check(vibeqc_status status, const std::string& detail) {
  if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    using namespace vibeqc::scf;
    std::cout << std::unitbuf;
    Dl_info library_info{};
    if (!dladdr(reinterpret_cast<void*>(&vibeqc_get_source_identity), &library_info))
      throw std::runtime_error("cannot identify the loaded native library");
    std::cout << "{\"operation\":\"identity\",\"source_identity\":"
              << std::quoted(vibeqc_get_source_identity())
              << ",\"library\":" << std::quoted(library_info.dli_fname) << "}\n";
    vibeqc::runtime::df_progress::Scope endpoint("fixed_density_probe");
    if (!std::getenv("SLURM_JOB_ID") || argc != 7)
      throw std::runtime_error(
          "usage inside Slurm: probe INPUT REPEATS AUX_TILE AO_PAIRS ARRAYS MODE");
    const auto repeats = std::stoul(argv[2]);
    if (!repeats || repeats > 1000) throw std::runtime_error("invalid repeat count");
    std::ifstream input(argv[1]);
    std::string magic;
    std::size_t atoms = 0, shells = 0, rank = 0;
    int representation = 0;
    input >> magic >> atoms >> shells >> representation >> rank;
    if (magic != "vibeqc-stage-v1" || !atoms || !shells)
      throw std::runtime_error("invalid occupied probe input");
    vibeqc::core::System system;
    system.basis_representation = static_cast<vibeqc_basis_representation>(representation);
    system.atoms.resize(atoms);
    system.shells.resize(shells);
    for (auto& atom : system.atoms)
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    for (auto& shell : system.shells) {
      std::size_t primitives = 0;
      input >> shell.atom_index >> shell.angular_momentum >> primitives;
      shell.primitives.resize(primitives);
      for (auto& primitive : shell.primitives) input >> primitive.exponent >> primitive.coefficient;
    }
    std::string detail;
    check(vibeqc::molecule::validate_and_normalize(system, detail), detail);
    const auto nbf = vibeqc::molecule::ao_count(system);
    if (rank > nbf) throw std::runtime_error("occupied rank exceeds AO dimension");
    std::vector<double> coefficients(nbf * rank), occupations(rank, 2.0);
    for (auto& value : coefficients) input >> value;
    if (!input) throw std::runtime_error("truncated occupied probe input");
    // The input overlap is independently evaluated by libcint in exactly the
    // supplied shell order. Qualify conventions before using the imported D.
    std::vector<double> reference_overlap(nbf * nbf);
    for (auto& value : reference_overlap) input >> value;
    if (!input) throw std::runtime_error("missing independent overlap");
    vibeqc::integrals::IntegralData cartesian;
    check(build_cuda_one_electron_integrals(0, system, cartesian, detail, false, false), detail);
    const auto one_electron = vibeqc::integrals::transform_integrals(cartesian, system);
    double overlap_error = 0, orthogonality_error = 0;
    std::vector<double> sc(nbf * rank, 0);
    for (std::size_t i = 0; i < nbf; ++i)
      for (std::size_t j = 0; j < nbf; ++j) {
        overlap_error = std::max(overlap_error, std::abs(one_electron.overlap[i * nbf + j] -
                                                         reference_overlap[i * nbf + j]));
        for (std::size_t a = 0; a < rank; ++a)
          sc[i * rank + a] += one_electron.overlap[i * nbf + j] * coefficients[j * rank + a];
      }
    double electrons = 0;
    for (std::size_t a = 0; a < rank; ++a)
      for (std::size_t b = 0; b < rank; ++b) {
        double value = 0;
        for (std::size_t i = 0; i < nbf; ++i)
          value += coefficients[i * rank + a] * sc[i * rank + b];
        orthogonality_error = std::max(orthogonality_error, std::abs(value - (a == b ? 1.0 : 0.0)));
        if (a == b) electrons += 2 * value;
      }
    if (!std::isfinite(overlap_error) || overlap_error > 1e-10 ||
        !std::isfinite(orthogonality_error) || orthogonality_error > 1e-9)
      throw std::runtime_error("imported independent orbital/overlap qualification failed");
    std::cout << std::setprecision(17)
              << "{\"operation\":\"seed_qualification\",\"overlap_error\":" << overlap_error
              << ",\"orthogonality_error\":" << orthogonality_error
              << ",\"electron_trace\":" << electrons << "}\n";
    const auto start = Clock::now();
    CudaDensityFittingIntegralSource* source = nullptr;
    std::vector<double> metrics;
    std::size_t source_n = 0, naux = 0;
    check(create_cuda_density_fitting_integral_source(0, {system}, {system}, &source, metrics,
                                                      source_n, naux, detail),
          detail);
    CudaDensityFittingJkPlan* raw = nullptr;
    std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
    const auto aux_tile = std::stoul(argv[3]);
    const auto pairs = std::stoul(argv[4]);
    const auto status = create_cuda_density_fitting_jk_plan_from_source(
        0, &source, 1, nbf, naux, metrics, 1e-10, aux_tile ? aux_tile : naux,
        pairs ? pairs : nbf * nbf, &raw, diagnostics, detail);
    destroy_cuda_density_fitting_integral_source(source);
    check(status, detail);
    std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)> plan(
        raw, destroy_cuda_density_fitting_jk_plan);
    const double setup_seconds = elapsed(start);
    const auto identity = cuda_density_fitting_factor_identity(plan.get(), 0, 1, 1);
    const OccupiedDensityFactor factor(identity, DensityFactorSpin::Restricted, nbf, coefficients,
                                       occupations);
    const CudaOccupiedDensityInput factors[]{{&factor, identity}};
    std::vector<double> density(factor.density().begin(), factor.density().end());
    std::vector<double> unused, dense, occupied, coulomb;
    std::vector<std::uint8_t> selected;
    const auto run = [&](bool compressed) {
      if (compressed) {
        check(execute_cuda_density_fitting_occupied_exchange(plan.get(), density,
                                                             DensityFactorSpin::Restricted, factors,
                                                             occupied, selected, detail),
              detail);
        if (selected != std::vector<std::uint8_t>{1})
          throw std::runtime_error("fixed-K factor unexpectedly fell back");
      } else {
        check(execute_cuda_density_fitting_rhf_jk(plan.get(), density, unused, dense, detail,
                                                  {false, true}),
              detail);
      }
    };
    std::cout << std::setprecision(17);
    std::cout << "{\"operation\":\"setup\",\"seconds\":" << setup_seconds << ",\"nbf\":" << nbf
              << ",\"naux\":" << naux << ",\"rank\":" << rank
              << ",\"value_plan_resident_bytes\":" << diagnostics[0].device_resident_bytes
              << ",\"value_plan_peak_bytes\":" << diagnostics[0].peak_device_bytes
              << ",\"streamed\":" << (diagnostics[0].streamed ? "true" : "false") << "}\n";
    const std::string mode = argv[6];
    std::vector<std::string> operations;
    if (mode == "all")
      operations = {"j", "dense", "occupied"};
    else if (mode == "j" || mode == "dense" || mode == "occupied")
      operations = {mode};
    else if (mode != "setup")
      throw std::runtime_error("invalid stage operation");
    // One prepared plan serves every requested fixed-D operation. A narrow
    // K-only bounded probe can reach the suspect panel before paying for J.
    for (std::size_t repeat = 0; repeat <= repeats; ++repeat) {
      for (const auto& operation : operations) {
        const auto begin = Clock::now();
        if (operation == "j")
          check(execute_cuda_density_fitting_rhf_jk(plan.get(), density, coulomb, unused, detail,
                                                    {true, false}),
                detail);
        else
          run(operation == "occupied");
        std::cout << "{\"operation\":" << std::quoted(operation) << ",\"sample\":" << repeat
                  << ",\"seconds\":" << elapsed(begin) << "}\n";
      }
    }
    if (!dense.empty() && !occupied.empty()) {
      double error = 0;
      for (std::size_t i = 0; i < dense.size(); ++i) {
        if (!std::isfinite(dense[i]) || !std::isfinite(occupied[i]))
          throw std::runtime_error("nonfinite fixed-K output");
        error = std::max(error, std::abs(dense[i] - occupied[i]));
      }
      if (error > 1e-9) throw std::runtime_error("occupied/dense K gate failed");
      std::cout << "{\"operation\":\"dense_occupied_comparison\",\"maximum_k_error\":" << error
                << "}\n";
    }
    static_assert(std::endian::native == std::endian::little);
    std::ofstream arrays(argv[5], std::ios::binary);
    // Exact float64 little-endian row-major D, followed by requested J,
    // dense K and occupied K. Unrequested matrices have zero elements.
    for (const auto* values : {&density, &coulomb, &dense, &occupied})
      arrays.write(reinterpret_cast<const char*>(values->data()),
                   static_cast<std::streamsize>(values->size() * sizeof(double)));
    if (!arrays) throw std::runtime_error("failed to retain fixed-K arrays");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
