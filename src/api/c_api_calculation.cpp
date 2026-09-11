#include <algorithm>
#include <memory>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "methods/method.hpp"
#include "vibeqc/vibeqc.h"

extern "C" {
const char* vibeqc_context_last_error(const vibeqc_context* context) {
  if (!context) return "invalid context";
  try {
    std::lock_guard<std::recursive_mutex> lock(context->mutex);
    thread_local std::string copy;
    copy = context->last_detail;
    return copy.c_str();
  } catch (...) {
    return "context error text unavailable";
  }
}

vibeqc_status vibeqc_calculation_prepare(vibeqc_context* context, const vibeqc_system* system,
                                         const vibeqc_method_descriptor* descriptor,
                                         vibeqc_calculation** calculation) {
  if (context == nullptr || system == nullptr || descriptor == nullptr || calculation == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  *calculation = nullptr;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  context->last_detail.clear();
  if (!vibeqc::api::valid_method_descriptor(descriptor)) {
    return VIBEQC_STATUS_ABI_MISMATCH;
  }
  try {
    auto candidate = std::make_unique<vibeqc_calculation>();
    candidate->context = context;
    candidate->plan =
        vibeqc::methods::prepare_calculation(context->state, system->data, *descriptor);
    *calculation = candidate.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&context->last_detail);
  }
}

void vibeqc_calculation_destroy(vibeqc_calculation* calculation) { delete calculation; }

vibeqc_status vibeqc_calculation_execute(vibeqc_calculation* calculation,
                                         vibeqc_result_descriptor* output) {
  if (calculation == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  calculation->context->last_detail.clear();
  calculation->plan->invalidate_result();
  if (!output) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (!vibeqc::api::valid_descriptor(output)) {
    return VIBEQC_STATUS_ABI_MISMATCH;
  }
  const bool omit_forces = output->forces == nullptr && output->force_count == 0;
  if (output->forces == nullptr && !omit_forces) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (!omit_forces && output->force_count < calculation->plan->atom_count() * 3) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (!omit_forces &&
      !(calculation->plan->capabilities().supported_properties & VIBEQC_PROPERTY_FORCES)) {
    calculation->context->last_detail = "requested method does not implement forces";
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }

  try {
    vibeqc::methods::Result native = calculation->plan->execute();
    output->energy = native.energy;
    output->iterations = native.convergence.iterations;
    output->energy_change = native.convergence.energy_change;
    output->density_rms = native.convergence.residual_rms;
    output->converged = native.convergence.converged ? 1 : 0;
    output->executed_backend = native.executed_backend;
    if (!native.convergence.converged) {
      return VIBEQC_STATUS_NOT_CONVERGED;
    }
    if (!omit_forces) {
      if (native.forces.size() > output->force_count) {
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
      std::copy(native.forces.begin(), native.forces.end(), output->forces);
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&calculation->context->last_detail);
  }
}

vibeqc_status vibeqc_calculation_get_correlation_diagnostic(
    const vibeqc_calculation* calculation, vibeqc_correlation_diagnostic* diagnostic) {
  if (!calculation || !diagnostic) return VIBEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  if (!vibeqc::api::valid_descriptor(diagnostic)) return VIBEQC_STATUS_ABI_MISMATCH;
  const auto value = calculation->plan->correlation_diagnostic();
  if (!value) return VIBEQC_STATUS_NOT_IMPLEMENTED;
  *diagnostic = *value;
  return VIBEQC_STATUS_SUCCESS;
}

}  // extern "C"
