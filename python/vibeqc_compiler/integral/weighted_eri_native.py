"""Complete native primitive binding for the existing weighted Hermite/AD DAG.

This module emits scalar callables only. Bounded record uploads, public-weight
pullbacks, contraction/scatter and compilation caches remain in their existing
consumer/runtime owners. Explicit radial inputs are frozen in every callable.
"""

from __future__ import annotations

from .range_separation import CoulombKernelFamily
from .weighted_eri import WeightedEriKernel
from .weighted_eri_cuda import emit_weighted_eri_header


def emit_weighted_eri_primitive_header(
    functions: tuple[tuple[WeightedEriKernel, str], ...], *, backend="cuda"
) -> str:
    """Emit CPU/CUDA values and all twelve derivatives for packed weight subsets.

    Each ``name_primitive`` consumes four positive exponents, twelve Bohr
    coordinates, and weights in the kernel's component_indices order. Coefficients
    and Cartesian/public normalization are already in the weights. A failed
    geometry or nonfinite result leaves the caller's output unchanged.

    The operator family and exact normalized omega are embedded in the source;
    they cannot be changed by replaying a compiled callable with new geometry.
    No screening, omega derivative, Hessian or range-separated DF is implied.
    """
    source = emit_weighted_eri_header(functions, backend=backend, packed_weights=True)
    qualifier = "__device__ __forceinline__" if backend == "cuda" else "inline"
    lines = [
        "#ifndef VIBEQC_GENERATED_WEIGHTED_ERI_PRIMITIVE_HPP",
        "#define VIBEQC_GENERATED_WEIGHTED_ERI_PRIMITIVE_HPP",
        source,
        '#include "integrals/eri_geometry.hpp"',
        "namespace vibeqc::scf::generated_weighted_eri {",
    ]
    native_ranges = {
        CoulombKernelFamily.FULL_RANGE: "Full",
        CoulombKernelFamily.LONG_RANGE: "Long",
        CoulombKernelFamily.SHORT_RANGE: "Short",
    }
    for kernel, name in functions:
        radial = kernel.integral.operator.coulomb_kernel
        lines.extend(
            [
                f"{qualifier} bool {name}_primitive(const double* exponents, const double* centers,",
                "    const double* component_weights, Gradient& output) {",
                "  if (!component_weights) return false;",
                f"  for (unsigned i = 0; i < {len(kernel.component_indices)}; ++i)",
                "    if (!std::isfinite(component_weights[i])) return false;",
                "  Geometry geometry{};",
                "  if (!vibeqc::integrals::make_eri_geometry(exponents, centers,",
                f"      {kernel.integral.maximum_coulomb_order}, vibeqc::integrals::CoulombRange::{native_ranges[radial.family]},",
                f"      {radial.omega.hex()}, geometry)) return false;",
                f"  const auto candidate = {name}(geometry, component_weights);",
                "  if (!std::isfinite(candidate.value)) return false;",
                "  for (unsigned center = 0; center < 4; ++center)",
                "    for (unsigned axis = 0; axis < 3; ++axis)",
                "      if (!std::isfinite(candidate.center[center][axis])) return false;",
                "  output = candidate;",
                "  return true;",
                "}",
            ]
        )
    lines.append("}  // namespace vibeqc::scf::generated_weighted_eri")
    lines.append("#endif")
    return "\n".join(lines) + "\n"
