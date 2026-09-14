#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "vibeqc/vibeqc.h"

#if VIBEQC_HAS_CUDA
extern "C" void grid_cuda_fail_next_allocation_for_test_v1();
#endif

namespace {
// Fail exactly one grid-coordinate allocation after warm-state import. This
// executable-only interposition exercises real constructor unwinding without
// adding fault controls to the production API or exhausting machine memory.
thread_local std::size_t fail_allocation_bytes = 0;
}  // namespace

void* operator new(std::size_t bytes) {
  if (bytes && bytes == fail_allocation_bytes) {
    fail_allocation_bytes = 0;
    throw std::bad_alloc();
  }
  if (void* pointer = std::malloc(bytes ? bytes : 1)) return pointer;
  throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Fixture {
  vibeqc_context* context{};
  vibeqc_system* system{};

  explicit Fixture(vibeqc_backend backend = VIBEQC_BACKEND_CPU_REFERENCE, int charge = 0,
                   std::uint32_t multiplicity = 1) {
    vibeqc_context_descriptor context_descriptor{sizeof(vibeqc_context_descriptor),
                                                 VIBEQC_ABI_VERSION, 0, backend};
    require(vibeqc_context_create(&context_descriptor, &context) == VIBEQC_STATUS_SUCCESS,
            "DFT context creation failed");
    system = create_system(context, charge, multiplicity);
  }

  static vibeqc_system* create_system(vibeqc_context* context, int charge = 0,
                                      std::uint32_t multiplicity = 1) {
    const std::array<vibeqc_atom, 2> atoms{{
        {1, 0.0, 0.0, -0.7},
        {1, 0.0, 0.0, 0.7},
    }};
    const std::array<vibeqc_primitive, 6> primitives{{
        {3.425250914, 0.1543289673},
        {0.6239137298, 0.5353281423},
        {0.168855404, 0.4446345422},
        {3.425250914, 0.1543289673},
        {0.6239137298, 0.5353281423},
        {0.168855404, 0.4446345422},
    }};
    const std::array<vibeqc_shell, 2> shells{{{0, 0, 0, 3}, {1, 0, 3, 3}}};
    vibeqc_system_descriptor descriptor{sizeof(vibeqc_system_descriptor),
                                        VIBEQC_ABI_VERSION,
                                        atoms.data(),
                                        static_cast<uint32_t>(atoms.size()),
                                        shells.data(),
                                        static_cast<uint32_t>(shells.size()),
                                        primitives.data(),
                                        static_cast<uint32_t>(primitives.size()),
                                        charge,
                                        multiplicity,
                                        VIBEQC_BASIS_CARTESIAN};
    vibeqc_system* created = nullptr;
    require(vibeqc_system_create(context, &descriptor, &created) == VIBEQC_STATUS_SUCCESS,
            "DFT system creation failed");
    return created;
  }

  ~Fixture() {
    vibeqc_system_destroy(system);
    vibeqc_context_destroy(context);
  }
};

vibeqc_method_descriptor lda_method() {
  return {sizeof(vibeqc_method_descriptor),
          VIBEQC_ABI_VERSION,
          VIBEQC_METHOD_LDA_RKS,
          200,
          8,
          1.0e-12,
          1.0e-10,
          1.0e-12,
          VIBEQC_DENSITY_FITTING_NONE,
          nullptr,
          1.0e-10,
          0};
}

void warm_preparation_failure(bool retained_plan) {
  Fixture fixture;
  auto method = lda_method();
  vibeqc_system* systems[]{fixture.system, fixture.system};
  vibeqc_batch *source = nullptr, *target = nullptr;
  require(vibeqc_batch_prepare(fixture.context, systems, 2, &method,
                               VIBEQC_BATCH_ENABLE_WARM_STARTS, &source) == VIBEQC_STATUS_SUCCESS,
          "warm failure source preparation failed");
  std::unique_ptr<vibeqc_batch, decltype(&vibeqc_batch_destroy)> source_owner(
      source, &vibeqc_batch_destroy);
  require(vibeqc_batch_prepare(fixture.context, systems, 2, &method,
                               VIBEQC_BATCH_ENABLE_WARM_STARTS, &target) == VIBEQC_STATUS_SUCCESS,
          "warm failure target preparation failed");
  std::unique_ptr<vibeqc_batch, decltype(&vibeqc_batch_destroy)> target_owner(
      target, &vibeqc_batch_destroy);
  const std::array<double, 6> changed{0.0, 0.0, -0.9, 0.0, 0.0, 0.9};
  std::array<vibeqc_batch_input_descriptor, 2> inputs;
  for (auto& input : inputs)
    input = {sizeof(input), VIBEQC_ABI_VERSION, changed.data(), changed.size()};
  std::array<vibeqc_batch_item_result_descriptor, 2> results{};
  const auto execute = [&](vibeqc_batch* batch, bool moved) {
    for (auto& result : results) {
      result = {};
      result.struct_size = sizeof(result);
      result.abi_version = VIBEQC_ABI_VERSION;
    }
    return vibeqc_batch_execute(batch, moved ? inputs.data() : nullptr, moved ? inputs.size() : 0,
                                results.data(), results.size());
  };
  require(execute(source, true) == VIBEQC_STATUS_SUCCESS && results[0].converged,
          "warm failure source did not converge");
  const double expected = results[0].energy;
  if (retained_plan)
    require(execute(target, false) == VIBEQC_STATUS_SUCCESS && results[0].converged,
            "warm failure target's original geometry did not converge");
  std::array<std::array<double, 4>, 2> densities{};
  std::array<std::array<double, 6>, 2> coordinates{};
  std::array<vibeqc_hf_warm_state, 2> seeds{};
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    seeds[i].struct_size = sizeof(seeds[i]);
    seeds[i].abi_version = VIBEQC_ABI_VERSION;
    seeds[i].density = densities[i].data();
    seeds[i].density_count = densities[i].size();
    seeds[i].coordinates = coordinates[i].data();
    seeds[i].coordinate_count = coordinates[i].size();
    require(vibeqc_batch_get_hf_warm_state(source, i, &seeds[i]) == VIBEQC_STATUS_SUCCESS &&
                seeds[i].present,
            "warm failure seed export failed");
  }
  require(vibeqc_batch_restore_hf_warm_states(target, seeds.data(), seeds.size()) ==
              VIBEQC_STATUS_SUCCESS,
          "warm failure seed import failed");
  // H2's default quadrature has two atoms times 48*16*32 points. Its xyz
  // vector is allocated inside preparation, after the imported seed is ready.
  fail_allocation_bytes = 3 * 2 * 48 * 16 * 32 * sizeof(double);
  const auto status = execute(target, true);
  const bool injected = fail_allocation_bytes == 0;
  fail_allocation_bytes = 0;
  require(injected, "warm preparation allocation fault was not reached");
  require(status == VIBEQC_STATUS_SUCCESS && results[0].status == VIBEQC_STATUS_OUT_OF_MEMORY &&
              !results[0].converged && !results[0].warm_start_fallback,
          "failed warm preparation retried a missing or stale calculation");
  require(results[1].status == VIBEQC_STATUS_SUCCESS && results[1].warm_start_used &&
              std::abs(results[1].energy - expected) < 2e-10,
          "warm preparation failure corrupted its neighbor");
  require(execute(target, true) == VIBEQC_STATUS_SUCCESS && results[0].converged &&
              results[0].warm_start_used && std::abs(results[0].energy - expected) < 2e-10,
          "warm preparation failure lost the imported seed or target geometry");
}

}  // namespace

int main() {
  try {
    warm_preparation_failure(false);
    warm_preparation_failure(true);
    vibeqc_method_capabilities_descriptor capabilities{
        sizeof(vibeqc_method_capabilities_descriptor), VIBEQC_ABI_VERSION, 0, 0, 0, 0, 0};
    require(vibeqc_method_get_capabilities(VIBEQC_METHOD_LDA_RKS, &capabilities) ==
                VIBEQC_STATUS_SUCCESS,
            "LDA RKS capability query failed");
    require(capabilities.family == VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL &&
                capabilities.supported_properties == VIBEQC_PROPERTY_ENERGY &&
                capabilities.available == 1 && capabilities.supports_batch == 1,
            "LDA RKS capabilities are incorrect");
    require(vibeqc_method_get_capabilities(VIBEQC_METHOD_PBE_RKS, &capabilities) ==
                    VIBEQC_STATUS_SUCCESS &&
                capabilities.family == VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL &&
                capabilities.supported_properties == VIBEQC_PROPERTY_ENERGY &&
                capabilities.available == 1 && capabilities.supports_batch == 1,
            "PBE RKS capabilities are incorrect");
    for (vibeqc_method method : {VIBEQC_METHOD_LDA_UKS, VIBEQC_METHOD_PBE_UKS}) {
      require(vibeqc_method_get_capabilities(method, &capabilities) == VIBEQC_STATUS_SUCCESS &&
                  capabilities.family == VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL &&
                  capabilities.supported_properties == VIBEQC_PROPERTY_ENERGY &&
                  capabilities.available == 1 && capabilities.supports_batch == 1,
              "UKS capabilities are incorrect");
    }

    Fixture fixture;
    auto method = lda_method();
    vibeqc_calculation* calculation = nullptr;
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_SUCCESS,
            "LDA RKS preparation failed");
    vibeqc_result_descriptor result{
        sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
        VIBEQC_BACKEND_CPU_REFERENCE};
    require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_SUCCESS &&
                result.converged == 1 && std::isfinite(result.energy) &&
                result.executed_backend == VIBEQC_BACKEND_CPU_REFERENCE,
            "LDA RKS energy-only execution failed");
    require(std::abs(result.energy - (-1.121017859421488)) < 2.0e-12,
            "LDA RKS H2 regression energy changed");

    std::array<double, 6> forces{};
    result.forces = forces.data();
    result.force_count = static_cast<uint32_t>(forces.size());
    require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_NOT_IMPLEMENTED,
            "LDA RKS force request was not rejected");
    const char* detail = vibeqc_context_get_last_detail(fixture.context);
    require(detail != nullptr && std::string(detail).find("issue #163") != std::string::npos,
            "LDA RKS force rejection omitted its capability boundary");
    vibeqc_calculation_destroy(calculation);

    method = lda_method();
    method.method = VIBEQC_METHOD_PBE_RKS;
    calculation = nullptr;
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_SUCCESS,
            "PBE RKS preparation failed");
    result = {sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
              VIBEQC_BACKEND_CPU_REFERENCE};
    require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_SUCCESS &&
                result.converged == 1 && std::isfinite(result.energy) &&
                result.executed_backend == VIBEQC_BACKEND_CPU_REFERENCE,
            "PBE RKS energy-only execution failed");
    require(std::abs(result.energy - (-1.1520643753396715)) < 2.0e-12,
            "PBE RKS H2 implementation regression energy changed");
    std::cout << std::setprecision(17) << "PBE RKS H2 energy: " << result.energy << "\n";
    vibeqc_calculation_destroy(calculation);

    method.density_fitting_mode = VIBEQC_DENSITY_FITTING_CPU_REFERENCE;
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_NOT_IMPLEMENTED,
            "LDA RKS accepted density fitting");

    method = lda_method();
    method.density_fitting_mode = static_cast<vibeqc_density_fitting_mode>(99);
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "LDA RKS misclassified an unknown density-fitting mode");

    for (double vibeqc_method_descriptor::* field : {
             &vibeqc_method_descriptor::energy_tolerance,
             &vibeqc_method_descriptor::density_tolerance,
             &vibeqc_method_descriptor::screening_tolerance,
         }) {
      method = lda_method();
      method.*field = std::numeric_limits<double>::quiet_NaN();
      require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                  VIBEQC_STATUS_INVALID_ARGUMENT,
              "LDA RKS accepted a NaN tolerance");
    }

    method = lda_method();
    method.precision_mode = VIBEQC_PRECISION_AUTO;
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_NOT_IMPLEMENTED,
            "LDA RKS silently accepted automatic precision");
    method.precision_mode = static_cast<vibeqc_precision_mode>(99);
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "LDA RKS misclassified an unknown precision mode");

    for (const auto [charge, multiplicity] :
         {std::pair{1, std::uint32_t{2}}, std::pair{0, std::uint32_t{3}}}) {
      Fixture invalid_spin(VIBEQC_BACKEND_CPU_REFERENCE, charge, multiplicity);
      method = lda_method();
      calculation = nullptr;
      require(vibeqc_calculation_prepare(invalid_spin.context, invalid_spin.system, &method,
                                         &calculation) == VIBEQC_STATUS_INVALID_ARGUMENT &&
                  calculation == nullptr,
              "LDA RKS accepted an odd-electron or non-singlet system");
      detail = vibeqc_context_get_last_detail(invalid_spin.context);
      require(detail != nullptr && std::string(detail).find("multiplicity 1") != std::string::npos,
              "LDA RKS spin rejection omitted its closed-shell boundary");
    }

    Fixture open_shell(VIBEQC_BACKEND_CPU_REFERENCE, -1, 2);
    method = lda_method();
    method.method = VIBEQC_METHOD_LDA_UKS;
    calculation = nullptr;
    require(vibeqc_calculation_prepare(open_shell.context, open_shell.system, &method,
                                       &calculation) == VIBEQC_STATUS_SUCCESS &&
                calculation != nullptr,
            "LDA UKS preparation rejected a valid doublet");
    result = {sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
              VIBEQC_BACKEND_CPU_REFERENCE};
    const vibeqc_status uks_status = vibeqc_calculation_execute(calculation, &result);
    if (uks_status != VIBEQC_STATUS_SUCCESS) {
      const char* uks_detail = vibeqc_context_get_last_detail(open_shell.context);
      throw std::runtime_error(std::string("LDA UKS execution failed: ") +
                               (uks_detail == nullptr ? "no detail" : uks_detail));
    }
    require(result.converged == 1 && std::isfinite(result.energy) &&
                std::isfinite(result.density_rms) && result.density_rms < 1.0e-8 &&
                result.executed_backend == VIBEQC_BACKEND_CPU_REFERENCE,
            "LDA UKS energy-only result is invalid");
    vibeqc_calculation_destroy(calculation);

    method.method = VIBEQC_METHOD_PBE_UKS;
    calculation = nullptr;
    require(vibeqc_calculation_prepare(open_shell.context, open_shell.system, &method,
                                       &calculation) == VIBEQC_STATUS_SUCCESS &&
                calculation != nullptr,
            "PBE UKS preparation rejected a valid doublet");
    result = {sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
              VIBEQC_BACKEND_CPU_REFERENCE};
    const vibeqc_status pbe_uks_status = vibeqc_calculation_execute(calculation, &result);
    if (pbe_uks_status != VIBEQC_STATUS_SUCCESS) {
      const char* pbe_uks_detail = vibeqc_context_get_last_detail(open_shell.context);
      throw std::runtime_error(std::string("PBE UKS execution failed: ") +
                               (pbe_uks_detail == nullptr ? "no detail" : pbe_uks_detail));
    }
    require(result.converged == 1 && std::isfinite(result.energy) &&
                std::isfinite(result.density_rms) && result.density_rms < 1.0e-8,
            "PBE UKS energy-only result is invalid");
    vibeqc_calculation_destroy(calculation);

    method = lda_method();
    method.max_iterations = 1;
    calculation = nullptr;
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_SUCCESS,
            "one-iteration LDA RKS preparation failed");
    vibeqc_result_descriptor unconverged{
        sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
        VIBEQC_BACKEND_CPU_REFERENCE};
    require(vibeqc_calculation_execute(calculation, &unconverged) == VIBEQC_STATUS_NOT_CONVERGED &&
                unconverged.converged == 0 && unconverged.iterations == 1 &&
                std::isfinite(unconverged.energy),
            "LDA RKS nonconvergence status or diagnostics are incorrect");
    vibeqc_calculation_destroy(calculation);

    vibeqc_system* systems[]{fixture.system};
    method = lda_method();
    vibeqc_batch* batch = nullptr;
    require(vibeqc_batch_prepare(fixture.context, systems, 1, &method, 0, &batch) ==
                    VIBEQC_STATUS_SUCCESS &&
                batch != nullptr,
            "LDA RKS prepared batch registration failed");
    vibeqc_batch_destroy(batch);

#if VIBEQC_HAS_CUDA
    vibeqc_context_descriptor cuda_descriptor{sizeof(vibeqc_context_descriptor), VIBEQC_ABI_VERSION,
                                              0, VIBEQC_BACKEND_CUDA};
    vibeqc_context* cuda_context = nullptr;
    if (vibeqc_context_create(&cuda_descriptor, &cuda_context) == VIBEQC_STATUS_SUCCESS) {
      vibeqc_system* cuda_system = Fixture::create_system(cuda_context);
      vibeqc_calculation* cuda_calculation = nullptr;
#if VIBEQC_HAS_CUDA
      grid_cuda_fail_next_allocation_for_test_v1();
      require(vibeqc_calculation_prepare(cuda_context, cuda_system, &method, &cuda_calculation) ==
                      VIBEQC_STATUS_OUT_OF_MEMORY &&
                  cuda_calculation == nullptr,
              "CUDA grid allocation failure lost its out-of-memory status");
#endif
      require(vibeqc_calculation_prepare(cuda_context, cuda_system, &method, &cuda_calculation) ==
                      VIBEQC_STATUS_SUCCESS &&
                  cuda_calculation != nullptr,
              "LDA RKS CUDA preparation failed");
      vibeqc_calculation_destroy(cuda_calculation);
      vibeqc_system_destroy(cuda_system);
      vibeqc_context_destroy(cuda_context);
    }
#endif
    std::cout << "LDA RKS public CPU energy-only contract passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
