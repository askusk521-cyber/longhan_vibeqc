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
* Later slice-A work adds the nuclear-perturbation right-hand side and the
  component-wise assembly built on the ``tools.vibeqc_response`` operator and the
  generated second integral derivatives.

The term-by-term dependency graph, the provider boundaries and the sign
conventions are documented in ``docs/hessian.md``.
"""

from .assembly import assemble_frozen_skeleton, validate_hessian_component
from .response import build_rhf_nuclear_rhs, metric_density_response_mo
from .numerical import (
    forces_to_gradient,
    hessian_difference,
    hessian_symmetry_error,
    hessian_translation_error,
    numerical_hessian,
)
from .weights import two_electron_energy, two_electron_weight, weight_energy

__all__ = [
    "assemble_frozen_skeleton",
    "build_rhf_nuclear_rhs",
    "metric_density_response_mo",
    "forces_to_gradient",
    "hessian_difference",
    "hessian_symmetry_error",
    "hessian_translation_error",
    "numerical_hessian",
    "two_electron_energy",
    "two_electron_weight",
    "validate_hessian_component",
    "weight_energy",
]
