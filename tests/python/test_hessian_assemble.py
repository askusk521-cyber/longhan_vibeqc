"""Test the component-wise analytic RHF Hessian (PR A2).

Every test is self-contained: integrals and their coordinate derivatives are
re-derived by fresh-molecule finite differences, so no intor-cache state
leaks between cases. The oracle is a total-energy FD Hessian at h=1e-4.

Geometries:
  * H2 STO-3G, off-minimum (2 AO, 1 occ, 1 virt)
  * water STO-3G (6 AO, 4 occ, 2 virt) — multi-virtual CPHF stress test
"""

import numpy as np
import pytest

from tools.vibeqc_hessian.assemble import (
    System,
    build_mol,
    fd_hessian,
    hessian_components,
    hessian_total,
)

# --- fixtures (STO-3G, Bohr) -----------------------------------------------
_S = [
    [
        0,
        (3.425250914, 0.1543289673),
        (0.6239137298, 0.5353281423),
        (0.168855404, 0.4446345422),
    ]
]
_P = [
    [
        1,
        (18.5950316763, 0.0499684015),
        (5.6203025291, 0.1505027689),
        (1.3720603452, 0.4117903918),
        (0.3434943771, 0.4106611356),
    ],
    [
        2,
        (18.5950316763, 0.0499684015),
        (5.6203025291, 0.1505027689),
        (1.3720603452, 0.4117903918),
        (0.3434943771, 0.4106611356),
    ],
]
# STO-3G O (second row, 5 AOs): 2s, 2p, 2d
_O = [
    [
        0,
        (18.5950316763, 0.0499684015),
        (5.6203025291, 0.1505027689),
        (1.3720603452, 0.4117903918),
        (0.3434943771, 0.4106611356),
    ],
    [
        1,
        (7.1153375795, 0.0102428309),
        (2.0218099978, 0.0314286582),
        (0.5271803969, 0.0814495192),
        (0.1703101032, 0.1646776769),
        (0.0628267133, 0.2917579986),
    ],
    [2, (0.0628267133, 0.1506938663), (0.0351485943, 0.2198166377)],
]


def _h2_mol():
    return build_mol(
        [(1, [0.0, 0.0, 0.0]), (1, [0.1, 0.2, 1.4])],
        {"H0": list(_S), "H1": list(_S)},
        charge=0,
        spin=0,
    )


def _water_mol():
    return build_mol(
        [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.958, 0.587]), (1, [0.0, -0.958, 0.587])],
        {"O0": list(_O), "H1": list(_S), "H2": list(_S)},
        charge=0,
        spin=0,
    )


# --- tests ----------------------------------------------------------------


@pytest.mark.parametrize(
    "mol_fn, label, fd_tol",
    [
        (_h2_mol, "h2", 5e-6),
        (_water_mol, "water", 5e-4),
    ],
)
def test_analytic_matches_fd(mol_fn, label, fd_tol):
    mol = mol_fn()
    s = System(mol)
    s.derive()
    H = hessian_total(s)
    H_fd = fd_hessian(mol, h=1e-4)
    diff = np.abs(H - H_fd).max()
    print(f"[{label}] max|analytic - FD| = {diff:.3e}")
    assert diff < fd_tol, f"{label}: analytic differs from FD by {diff:.3e}"


@pytest.mark.parametrize(
    "mol_fn, label, sym_tol",
    [
        (_h2_mol, "h2", 1e-12),
        (_water_mol, "water", 1e-10),
    ],
)
def test_symmetry(mol_fn, label, sym_tol):
    mol = mol_fn()
    s = System(mol)
    s.derive()
    H = hessian_total(s)
    diff = np.abs(H - H.transpose(1, 0, 3, 2)).max()
    print(f"[{label}] sym viol = {diff:.3e}")
    assert diff < sym_tol, f"{label}: asymmetry {diff:.3e}"


@pytest.mark.parametrize(
    "mol_fn, label, tr_tol",
    [
        (_h2_mol, "h2", 1e-6),
        (_water_mol, "water", 1e-4),
    ],
)
def test_translation_invariance(mol_fn, label, tr_tol):
    mol = mol_fn()
    s = System(mol)
    s.derive()
    H = hessian_total(s).transpose(0, 2, 1, 3).reshape(3 * mol.natm, 3 * mol.natm)
    row = np.abs(H.sum(axis=1)).max()
    col = np.abs(H.sum(axis=0)).max()
    print(f"[{label}] translation: row={row:.3e} col={col:.3e}")
    assert row < tr_tol and col < tr_tol, f"{label}: translation row={row:.3e}"


@pytest.mark.parametrize(
    "mol_fn, label",
    [
        (_h2_mol, "h2"),
        (_water_mol, "water"),
    ],
)
def test_component_sum_equals_total(mol_fn, label):
    mol = mol_fn()
    s = System(mol)
    s.derive()
    comps = hessian_components(s)
    total_from_parts = (
        comps["nuclear"]
        + comps["core"]
        + comps["pulay"]
        + comps["two_electron"]
        + comps["relaxation"]
    )
    diff = np.abs(total_from_parts - comps["total"]).max()
    print(f"[{label}] component sum vs total: {diff:.3e}")
    assert diff < 1e-10, f"{label}: component sum off by {diff:.3e}"


@pytest.mark.parametrize(
    "mol_fn, label, key",
    [
        (_h2_mol, "h2", "relaxation"),
        (_h2_mol, "h2", "pulay"),
        (_h2_mol, "h2", "two_electron"),
        (_h2_mol, "h2", "nuclear"),
        (_water_mol, "water", "relaxation"),
        (_water_mol, "water", "pulay"),
        (_water_mol, "water", "two_electron"),
        (_water_mol, "water", "nuclear"),
    ],
)
def test_component_nonzero(mol_fn, label, key):
    """Each component must be non-trivial (negative case: removing it breaks)."""
    mol = mol_fn()
    s = System(mol)
    s.derive()
    comps = hessian_components(s)
    mag = np.abs(comps[key]).max()
    print(f"[{label}] |{key}| max = {mag:.5f}")
    assert mag > 1e-4, f"{label}: {key} is trivially small ({mag:.3e})"


@pytest.mark.parametrize(
    "mol_fn, label, tol",
    [
        (_h2_mol, "h2", 5e-5),
        (_water_mol, "water", 5e-3),
    ],
)
def test_frozen_density_wrong(mol_fn, label, tol):
    """Negative case: removing orbital relaxation must shift the Hessian
    by more than the analytic-vs-FD error."""
    mol = mol_fn()
    s = System(mol)
    s.derive()
    H_full = hessian_total(s)
    H_frozen = hessian_total(s, with_relax=False)
    H_fd = fd_hessian(mol, h=1e-4)
    shift = np.abs(H_full - H_frozen).max()
    fd_err = np.abs(H_full - H_fd).max()
    print(f"[{label}] frozen-dens shift={shift:.4e}  fd_err={fd_err:.4e}")
    assert shift > 3 * fd_err, (
        f"{label}: frozen density barely changes the Hessian "
        f"(shift={shift:.3e} vs fd_err={fd_err:.3e})"
    )
