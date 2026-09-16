"""Analytic molecular Hessians for VibeQC (issue #180).

Slice A of issue #180 delivers a tiny dense analytic RHF Hessian together with
the numerical oracle used to verify it. The package is organised so that the
oracle and the analytic assembly stay separately checkable:

* :mod:`numerical` owns the finite-difference-of-analytic-gradient oracle. It
  depends only on first derivatives, so it is independent of the analytic
  assembly in the useful direction.
* :mod:`weights` owns the density folding the generated second integral
  derivatives need, checked against the energy it is meant to reproduce rather
  than stated only in prose.
* :mod:`assembly` owns the frozen-density skeleton (the four second-derivative
  components, response boundary left explicit).
* :mod:`response` owns the nuclear-perturbation right-hand side in the shared
  metric gauge.
* :mod:`assemble` (PR A2) is a self-contained **complete** analytic Hessian
  reference: it re-derives every integral (and its coordinate derivatives) by
  fresh-molecule finite differences and adds the full-response-space 1st-order
  CPHF relaxation on top of the frozen skeleton. It is an independent oracle
  for the pieces above, checked against the numerical oracle and PySCF.

The term-by-term dependency graph, the provider boundaries and the sign
conventions are documented in ``docs/hessian.md``.
"""

from .assemble import (
    System,
    build_mol,
    fd_hessian,
    hessian_components,
    hessian_total,
)
from .assembly import assemble_frozen_skeleton, validate_hessian_component
from .numerical import (
    forces_to_gradient,
    hessian_difference,
    hessian_symmetry_error,
    hessian_translation_error,
    numerical_hessian,
)
from .response import build_rhf_nuclear_rhs, metric_density_response_mo
from .weights import two_electron_energy, two_electron_weight, weight_energy

__all__ = [
    "System",
    "assemble_frozen_skeleton",
    "build_mol",
    "build_rhf_nuclear_rhs",
    "fd_hessian",
    "forces_to_gradient",
    "hessian_components",
    "hessian_difference",
    "hessian_symmetry_error",
    "hessian_total",
    "hessian_translation_error",
    "metric_density_response_mo",
    "numerical_hessian",
    "two_electron_energy",
    "two_electron_weight",
    "validate_hessian_component",
    "weight_energy",
]
