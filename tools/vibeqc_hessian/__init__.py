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
* :mod:`reference` is the self-contained **semi-numerical** RHF Hessian: it
  re-derives every integral and its coordinate derivatives by fresh-molecule
  finite differences and solves the full-response-space 1st-order CPHF. It is
  an independent oracle for the production path, checked against PySCF.
* :mod:`analytic` is the **complete analytic integration**: the same term map
  consumed through the shared #178 second-integral providers and #179 response
  operator, checked against the reference oracle and PySCF's analytic Hessian.

The term-by-term dependency graph, the provider boundaries and the sign
conventions are documented in ``docs/hessian.md``.
"""

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

# The reference and analytic Hessian modules pull in PySCF and the generated
# #178 kernels, so they are imported lazily (PEP 562) to keep the package's
# eager import surface free of those dependencies.
_LAZY = {
    # reference.py (semi-numerical oracle)
    "System": "reference",
    "build_mol": "reference",
    "fd_hessian": "reference",
    "hessian_components": "reference",
    "hessian_total": "reference",
    # analytic.py (complete analytic integration through #178/#179)
    "analytic_hessian": "analytic",
    "build_reference": "analytic",
    "cphf_relaxation": "analytic",
    "fixture_mol": "analytic",
    "nuclear_closed_form": "analytic",
    "provider_components": "analytic",
}

__all__ = [
    "System",
    "analytic_hessian",
    "assemble_frozen_skeleton",
    "build_mol",
    "build_reference",
    "build_rhf_nuclear_rhs",
    "cphf_relaxation",
    "fd_hessian",
    "fixture_mol",
    "forces_to_gradient",
    "hessian_components",
    "hessian_difference",
    "hessian_symmetry_error",
    "hessian_total",
    "hessian_translation_error",
    "metric_density_response_mo",
    "nuclear_closed_form",
    "numerical_hessian",
    "provider_components",
    "two_electron_energy",
    "two_electron_weight",
    "validate_hessian_component",
    "weight_energy",
]


def __getattr__(name):
    module = _LAZY.get(name)
    if module is not None:
        import importlib

        return getattr(importlib.import_module(f".{module}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
