#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/coulomb_auxiliary.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/one_electron_force_workspace.hpp"

// Retained first derivatives of nuclear attraction. The force consumer
// supplies the same compact Hermite workspace for each primitive pair.
namespace vibeqc::scf::cuda_execution {

/** Fill every coefficient required by first derivatives of one shell pair. */
__device__ inline void fill_one_electron_derivative_hermite(
    unsigned maximum_i, unsigned maximum_j, double product, double center_a, double center_b,
    double alpha, double beta, OneElectronDerivativeHermiteCoefficients& coefficients) {
  for (unsigned item = 0; item < OneElectronDerivativeHermiteCoefficients::kIDimension *
                                     OneElectronDerivativeHermiteCoefficients::kJDimension *
                                     OneElectronDerivativeHermiteCoefficients::kTDimension;
       ++item) {
    coefficients.data[item] = 0.0;
  }
  const double exponent = alpha + beta;
  const double reduced = alpha * beta / exponent;
  const double difference = center_a - center_b;
  coefficients.at(0, 0, 0) = exp(-reduced * difference * difference);
  const double product_first = product - center_a;
  const double product_second = product - center_b;
  const double inverse_two_exponent = 0.5 / exponent;
  for (unsigned i = 0; i <= maximum_i; ++i) {
    for (unsigned j = 0; j <= maximum_j; ++j) {
      if (i == 0 && j == 0) continue;
      if (i > 0) {
        coefficients.at(i, j, 0) =
            product_first * coefficients.at(i - 1, j, 0) + coefficients.at(i - 1, j, 1);
      } else {
        coefficients.at(i, j, 0) =
            product_second * coefficients.at(i, j - 1, 0) + coefficients.at(i, j - 1, 1);
      }
      for (unsigned t = 1; t <= i + j; ++t) {
        if (i > 0) {
          coefficients.at(i, j, t) = product_first * coefficients.at(i - 1, j, t) +
                                     inverse_two_exponent * coefficients.at(i - 1, j, t - 1) +
                                     static_cast<double>(t + 1) * coefficients.at(i - 1, j, t + 1);
        } else {
          coefficients.at(i, j, t) = product_second * coefficients.at(i, j - 1, t) +
                                     inverse_two_exponent * coefficients.at(i, j - 1, t - 1) +
                                     static_cast<double>(t + 1) * coefficients.at(i, j - 1, t + 1);
        }
      }
    }
  }
}

template <unsigned MaximumAngular>
__device__ inline __noinline__ void
primitive_nuclear_attraction_cartesian_atom_gradient_from_hermite(
    const DeviceBatch& batch, double alpha, const Angular& angular_first, double beta,
    const Angular& angular_second, std::int64_t atom, double exponent, const Vec3<double>& product,
    const OneElectronDerivativeHermiteCoefficients* coefficients, double (&first_gradient)[3],
    double (&second_gradient)[3]) {
  static_assert(MaximumAngular <= 2 * kMaximumAngularMomentum + 1);
  CoulombAuxiliary<double, MaximumAngular> auxiliary;
  fill_coulomb<MaximumAngular>(exponent, product, atom_position<double>(batch, atom, -1),
                               auxiliary);
  for (unsigned axis = 0; axis < 3; ++axis) {
    first_gradient[axis] = 0.0;
    second_gradient[axis] = 0.0;
  }
  const unsigned x_limit = angular_first.x + angular_second.x + 1;
  const unsigned y_limit = angular_first.y + angular_second.y + 1;
  const unsigned z_limit = angular_first.z + angular_second.z + 1;
  const double attraction_scale =
      -static_cast<double>(batch.atomic_numbers[atom]) * (2.0 * kPi / exponent);
  for (unsigned t = 0; t <= x_limit; ++t) {
    for (unsigned u = 0; u <= y_limit; ++u) {
      for (unsigned v = 0; v <= z_limit; ++v) {
        if (t + u + v > MaximumAngular) continue;
        const unsigned orders[3] = {t, u, v};
        double base[3];
        double first_derivative[3];
        double second_derivative[3];
        for (int axis = 0; axis < 3; ++axis) {
          const unsigned first_power = angular_axis(angular_first, axis);
          const unsigned second_power = angular_axis(angular_second, axis);
          const unsigned order = orders[axis];
          base[axis] = coefficients[axis].at(first_power, second_power, order);
          first_derivative[axis] =
              2.0 * alpha * coefficients[axis].at(first_power + 1, second_power, order);
          if (first_power > 0) {
            first_derivative[axis] -= static_cast<double>(first_power) *
                                      coefficients[axis].at(first_power - 1, second_power, order);
          }
          second_derivative[axis] =
              2.0 * beta * coefficients[axis].at(first_power, second_power + 1, order);
          if (second_power > 0) {
            second_derivative[axis] -= static_cast<double>(second_power) *
                                       coefficients[axis].at(first_power, second_power - 1, order);
          }
        }
        const double coulomb = attraction_scale * auxiliary.at(0, t, u, v);
        first_gradient[0] += first_derivative[0] * base[1] * base[2] * coulomb;
        first_gradient[1] += base[0] * first_derivative[1] * base[2] * coulomb;
        first_gradient[2] += base[0] * base[1] * first_derivative[2] * coulomb;
        second_gradient[0] += second_derivative[0] * base[1] * base[2] * coulomb;
        second_gradient[1] += base[0] * second_derivative[1] * base[2] * coulomb;
        second_gradient[2] += base[0] * base[1] * second_derivative[2] * coulomb;
      }
    }
  }
}

/**
 * Evaluate both basis-center gradients of one nucleus's attraction integral.
 *
 * All raised/lowered Cartesian components share one Hermite workspace, one
 * Boys sequence, and one Coulomb auxiliary recurrence. The separate
 * precomputed entry point lets a cooperative AO-pair worker build the Hermite
 * coefficients once and reuse them across all nuclear centers in its warp.
 */
template <unsigned MaximumAngular>
__device__ inline __noinline__ void primitive_nuclear_attraction_cartesian_atom_gradient(
    const DeviceBatch& batch, double alpha, const Vec3<double>& first, const Angular& angular_first,
    double beta, const Vec3<double>& second, const Angular& angular_second, std::int64_t atom,
    double (&first_gradient)[3], double (&second_gradient)[3]) {
  static_assert(MaximumAngular <= 2 * kMaximumAngularMomentum + 1);
  const double exponent = alpha + beta;
  const Vec3<double> product = product_center(alpha, first, beta, second);
  OneElectronDerivativeHermiteCoefficients coefficients[3];
  for (int axis = 0; axis < 3; ++axis) {
    fill_one_electron_derivative_hermite(angular_axis(angular_first, axis) + 1,
                                         angular_axis(angular_second, axis) + 1,
                                         vec_axis(product, axis), vec_axis(first, axis),
                                         vec_axis(second, axis), alpha, beta, coefficients[axis]);
  }
  primitive_nuclear_attraction_cartesian_atom_gradient_from_hermite<MaximumAngular>(
      batch, alpha, angular_first, beta, angular_second, atom, exponent, product, coefficients,
      first_gradient, second_gradient);
}

}  // namespace vibeqc::scf::cuda_execution
