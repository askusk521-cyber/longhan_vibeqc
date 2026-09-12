"""Numerical Hessian oracle checks for issue #180 slice A (step 2).

The oracle is verified in two independent ways:

* internally, against the invariance identities every exact Hessian must
  satisfy -- raw symmetry and the translation zero mode -- and against its own
  multi-step convergence;
* externally, against PySCF's analytic RHF Hessian for the same geometry.

The PySCF comparison is the one that can catch a systematically wrong analytic
gradient, which the invariance identities alone cannot: a gradient that is
wrong in a geometry-independent way still produces a symmetric,
translation-invariant Hessian.

All coordinates are in **Bohr**, which is what VibeQC's native layer expects.
All checks run on the raw, unsymmetrized array; symmetrizing first would hide
exactly the errors these tests exist to find.

Cost
----
One analytic gradient costs about 11 ms for H2, 3.1 s for water and 95 s for
the 18-AO d/f system on the CPU reference path, and a three-step Hessian needs
``2 * 3N`` gradients per step. H2 and water therefore run in the default suite;
the d/f case is gated behind ``VIBEQC_HESSIAN_SLOW=1`` because a full run takes
tens of minutes rather than seconds.
"""

import os
from itertools import pairwise

import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell

from tools.vibeqc_hessian import (
    forces_to_gradient,
    hessian_difference,
    hessian_symmetry_error,
    hessian_translation_error,
    numerical_hessian,
)

# H2 at a geometry that is not the equilibrium bond length: an exact Hessian is
# symmetric and translation-invariant at any geometry, so using an off-minimum
# point additionally keeps the test from passing on a stationary-point accident.
H2 = {
    "symbols": ("H", "H"),
    "coordinates": np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]),
    "charge": 0,
    "multiplicity": 1,
}

# The familiar 0.958 A / 104.5 degree water structure converted to Bohr
# (O-H = 1.8102 bohr). Entering the Angstrom numbers directly would place the
# hydrogens at 0.507 A -- a compressed, pathological geometry whose SCF is both
# slower and much harder to converge.
WATER = {
    "symbols": ("O", "H", "H"),
    "coordinates": np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.43052268, 1.10926924],
            [0.0, -1.43052268, 1.10926924],
        ]
    ),
    "charge": 0,
    "multiplicity": 1,
}

# An 18-AO system built from explicit s/d/f shells on the helium centre. The
# off-centre d and f functions make higher-angular-momentum derivative paths
# participate, which is where an angular-momentum mistake in the gradient would
# show up. Mirrors the fixture in tests/python/test_calculator.py.
HEH_DF = {
    "symbols": ("He", "H"),
    "coordinates": np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]),
    "charge": 1,
    "multiplicity": 1,
    "basis": (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, 2, (Primitive(0.8, 1.0),)),
        Shell(0, 3, (Primitive(0.6, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    ),
}

STEPS = (1e-2, 3e-3, 1e-3)
FAST_CASES = ("h2", "water")
SLOW_CASES = ("heh_df",)

# A residual below this fraction of the Hessian scale is at the roundoff floor:
# it is indistinguishable from zero at double precision and carries no
# information about the truncation order.
ROUNDOFF_RELATIVE = 1.0e-9


def _case(name):
    if name in SLOW_CASES and os.environ.get("VIBEQC_HESSIAN_SLOW") != "1":
        pytest.skip(
            "set VIBEQC_HESSIAN_SLOW=1 to run the 18-AO d/f case; "
            "it costs about 95 s per analytic gradient"
        )
    return {"h2": H2, "water": WATER, "heh_df": HEH_DF}[name]


# One three-step report per case, shared by every check below. Recomputing the
# curve per test would multiply the SCF count for no extra evidence, and sharing
# it also guarantees the invariance, convergence and external-reference checks
# all look at exactly the same numbers.
_REPORTS = {}


def _report(name):
    if name not in _REPORTS:
        case = _case(name)
        gradient, settings = _calculator_gradient(case)
        _REPORTS[name] = numerical_hessian(
            gradient, case["coordinates"], settings=settings, steps=STEPS
        )
    return _REPORTS[name]


def _calculator_gradient(case):
    """Build a gradient callback that solves each displaced geometry cold.

    Every evaluation constructs a fresh native system through ``singlepoint``,
    so no warm start can bias one displacement relative to another. The
    tolerances are tight enough that the SCF residual does not dominate the
    finite-difference error.
    """
    calculator = Calculator(
        method="rhf",
        basis=case.get("basis", "sto-3g"),
        device="cpu",
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    )
    settings = {
        "method": "rhf",
        "device": "cpu",
        "energy_tolerance": 1.0e-12,
        "density_tolerance": 1.0e-10,
        "charge": case["charge"],
        "multiplicity": case["multiplicity"],
    }

    def gradient(coordinates, policy):
        atoms = [
            (symbol, tuple(float(value) for value in row))
            for symbol, row in zip(case["symbols"], coordinates)
        ]
        result = calculator.singlepoint(
            atoms,
            charge=policy["charge"],
            multiplicity=policy["multiplicity"],
        )
        assert result.forces is not None
        return forces_to_gradient(result.forces)

    return gradient, settings


def _pyscf_hessian(case):
    """Return PySCF's analytic RHF Hessian in the oracle's axis order.

    Two conventions have to be reconciled, and both are silent if they are
    wrong -- they produce a plausible-looking matrix rather than an error:

    ``unit="bohr"`` is essential, because VibeQC works in Bohr while PySCF's
    default input unit is Angstrom.

    PySCF returns the Hessian shaped ``(natom, natom, 3, 3)`` -- atom indices
    first, then the Cartesian axes -- whereas the oracle uses
    ``(natom, 3, natom, 3)`` with each atom's axes adjacent. The transpose
    below performs that permutation; a plain ``reshape`` would scramble the
    axes rather than reorder them.
    """
    gto = pytest.importorskip("pyscf.gto")
    scf = pytest.importorskip("pyscf.scf")
    atom = [
        [symbol, [float(value) for value in row]]
        for symbol, row in zip(case["symbols"], case["coordinates"])
    ]
    mol = gto.M(
        atom=atom,
        basis=case.get("basis", "sto-3g"),
        charge=case["charge"],
        spin=case["multiplicity"] - 1,
        unit="bohr",
        verbose=0,
    )
    mean_field = scf.RHF(mol)
    mean_field.conv_tol = 1.0e-12
    mean_field.run()
    values = np.asarray(mean_field.Hessian().kernel())
    assert values.shape == (len(case["symbols"]),) * 2 + (3, 3), values.shape
    return values.transpose(0, 2, 1, 3)


@pytest.mark.parametrize("name", FAST_CASES + SLOW_CASES)
def test_gradient_is_deterministic(name):
    """The same geometry must produce bit-identical gradients.

    A finite-difference Hessian differences gradients taken at *different*
    geometries, so any non-determinism at a fixed geometry would appear
    directly as Hessian noise and would make every downstream bound
    meaningless.
    """

    case = _case(name)
    gradient, settings = _calculator_gradient(case)
    first = gradient(case["coordinates"], settings)
    second = gradient(case["coordinates"], settings)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("name", FAST_CASES + SLOW_CASES)
def test_invariance_residuals_follow_the_second_order_law(name):
    """The only departure from exact invariance is O(h^2) truncation.

    An exact Hessian satisfies ``H == H^T`` and ``sum_a H[a, c, b, d] == 0``
    exactly. A central difference of exact gradients violates them only through
    its truncation term, which falls as ``h^2``. Asserting that law is stronger
    than an absolute tolerance: a sign or factor defect does not shrink with the
    step, and a defect in the differencing itself scales differently, so neither
    can satisfy a second-order decrease.

    The law is only asserted where it is informative. A system whose residual is
    already at the roundoff floor (H2 collapses to ~1e-16, essentially an exact
    gradient) has nothing left to shrink, so those steps are checked against the
    absolute floor instead.
    """

    case = _case(name)
    report = _report(name)
    samples = report["samples"]
    assert len(samples) == 3

    scale = max(1.0, max(sample["max_absolute"] for sample in samples))

    for coarse, fine in pairwise(samples):
        ratio = coarse["step_bohr"] / fine["step_bohr"]
        for key in ("symmetry_error", "translation_error"):
            coarse_relative = coarse[key] / scale
            fine_relative = fine[key] / scale
            # A residual that is already at the roundoff floor carries no
            # convergence information: there is nothing left to shrink, and
            # requiring a decrease there would only assert that the noise
            # happens to be monotone. Only the truncation-limited regime is
            # informative, so it is the regime the law is asserted in.
            if coarse_relative <= ROUNDOFF_RELATIVE:
                continue
            # Allow the residual to fall as slowly as half the ideal h^2 rate,
            # so the assertion is about the order of convergence rather than
            # about an exact constant.
            envelope = max(ROUNDOFF_RELATIVE, coarse_relative / (0.5 * ratio**2))
            assert fine_relative <= envelope, (
                f"{name}: {key} fell from {coarse_relative} to {fine_relative} "
                f"relative to a Hessian scale of {scale} for a step ratio of "
                f"{ratio}, which is not second order"
            )

    finest = samples[-1]
    assert finest["symmetry_error"] < 1.0e-3 * scale
    assert finest["translation_error"] < 1.0e-3 * scale

    shape = (len(case["symbols"]), 3, len(case["symbols"]), 3)
    for sample in samples:
        hessian = np.asarray(sample["hessian"])
        assert hessian.shape == shape
        assert np.isfinite(hessian).all()


@pytest.mark.parametrize("name", FAST_CASES)
def test_hessian_values_converge_with_step_size(name):
    """The Hessian itself stabilizes as the step shrinks.

    The invariance residuals above only constrain the antisymmetric part and
    the translation sum. This check constrains the values: the two smallest
    steps must agree more closely than either agrees with the largest, so the
    reported matrix is not still moving with the step. No step is selected as
    the answer and no absolute tolerance is claimed.
    """

    report = _report(name)
    coarse, middle, fine = (np.asarray(s["hessian"]) for s in report["samples"])
    coarse_gap = hessian_difference(coarse, fine)["max_absolute_error"]
    middle_gap = hessian_difference(middle, fine)["max_absolute_error"]
    assert middle_gap < coarse_gap


@pytest.mark.parametrize("name", FAST_CASES + SLOW_CASES)
def test_hessian_matches_pyscf_analytic_reference(name):
    """The oracle agrees with an independent analytic Hessian.

    PySCF assembles the Hessian by an entirely separate route, so agreement
    here is evidence about the analytic gradient itself rather than about the
    oracle's internal consistency. Every step size is compared, not just the
    best one: agreement that held at a single step would suggest a coincidence
    between the step and the truncation error rather than a correct gradient.
    """

    case = _case(name)
    report = _report(name)
    reference = _pyscf_hessian(case)
    scale = max(1.0, float(np.max(np.abs(reference))))

    for sample in report["samples"]:
        actual = np.asarray(sample["hessian"])
        difference = hessian_difference(actual, reference)
        # The dominant error at these steps is the O(h^2) truncation term, so
        # the bound is set from the observed convergence in relative form.
        assert difference["max_absolute_error"] < 5.0e-4 * scale, (
            f"{name} at step {sample['step_bohr']}: max error "
            f"{difference['max_absolute_error']} vs scale {scale}"
        )


def test_numerical_hessian_requires_three_distinct_steps():
    """The oracle refuses a single-step request instead of implying a verdict."""

    gradient, settings = _calculator_gradient(H2)
    for steps in ((1e-3,), (1e-3, 1e-3), (1e-3, 0.0), (1e-3, -1e-3)):
        with pytest.raises(ValueError):
            numerical_hessian(
                gradient, H2["coordinates"], settings=settings, steps=steps
            )


def test_numerical_hessian_rejects_malformed_coordinates():
    """Coordinate shape and finiteness are validated before any solve is run."""

    gradient, settings = _calculator_gradient(H2)
    with pytest.raises(ValueError):
        numerical_hessian(gradient, np.zeros((2, 2)), settings=settings)
    with pytest.raises(ValueError):
        numerical_hessian(
            gradient,
            np.array([[0.0, 0.0, np.nan], [0.0, 0.0, 0.7]]),
            settings=settings,
        )


def test_forces_to_gradient_negates_and_copies():
    """The force/gradient sign boundary is a single named conversion."""

    forces = np.array([[1.0, -2.0, 3.0]])
    gradient = forces_to_gradient(forces)
    np.testing.assert_array_equal(gradient, -forces)
    assert gradient.dtype == np.float64
    gradient[0, 0] = 99.0
    assert forces[0, 0] == 1.0


def test_hessian_error_helpers_report_the_whole_distribution():
    """Symmetry and translation helpers see raw arrays, including asymmetry."""

    asymmetric = np.zeros((1, 3, 1, 3))
    asymmetric[0, 0, 0, 1] = 1.0
    asymmetric[0, 1, 0, 0] = -1.0
    # The two entries are transposes of each other, so the symmetry check sees
    # them as a pair and reports 2.0. They land in different components of the
    # translation sum, so they do not cancel there -- the two helpers detect
    # genuinely different defects and must not be treated as interchangeable.
    assert hessian_symmetry_error(asymmetric) == pytest.approx(2.0)
    assert hessian_translation_error(asymmetric) == pytest.approx(1.0)

    difference = hessian_difference(np.ones((1, 3, 1, 3)), np.zeros((1, 3, 1, 3)))
    assert difference["max_absolute_error"] == pytest.approx(1.0)
    assert difference["rms_error"] == pytest.approx(1.0)
    assert difference["shape"] == [1, 3, 1, 3]
