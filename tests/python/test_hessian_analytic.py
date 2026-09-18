"""Complete analytic RHF Hessian through the shared #178/#179 layers (PR A2).

PR #414 (A1) shipped the component formula as a *reference-only* oracle and
was merged with "full analytic integration remains open."  These tests cover the bounded diagnostic integration, not a native endpoint.  They check, for the same molecules A1 already exercises, that:

* the #178 generated second-integral providers reproduce the frozen-skeleton
  components (core, pulay, two_electron) that A1's finite-difference oracle
  derives;
* the electronic relaxation solved through #179's shared matrix-free
  ``RHFResponseOperator`` and its true-residual GMRES matches the dense
  full-response-space CPHF that A1's reference uses;
* the shared operator identity ``A = D (I + F_vv^T)`` holds (the algebra that
  turns A1's dense block system into #179's single matrix-free solve);
* the assembled total matches PySCF's analytic RHF Hessian as an end-to-end cross-check.

The water fixtures mirror the reference ``test_hessian_reference.py`` so the reference
and the integrated path are checked on identical molecules.  The ``water_sdf``
12-AO d-shell case is gated behind ``VIBEQC_HESSIAN_SLOW=1`` for the
two-electron component only: its ERI second-derivative provider is Python-
looped over the 5^4 shell quartets and the dddd quartet alone is 6^4 AO
components.  The #179 relaxation on that same case is fast and runs in the
default suite.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from tools.vibeqc_hessian.analytic import (
    System,
    analytic_hessian,
    cphf_relaxation,
    fixture_mol,
    hessian_components,
    nuclear_closed_form,
    provider_components,
)
from tools.vibeqc_hessian.reference import (
    _first_order_mo1_e1,
    _first_order_mo1_e1_vir_only,
    _wof,
    fd_hessian,
    hessian_total,
)
from tools.vibeqc_hessian.reference import (
    h1ao as ref_h1ao,
)

SLOW_CASES = {"water_sdf"}
COMP_KEYS = ("nuclear", "core", "pulay", "two_electron", "relaxation")


def _sys(name):
    mol = fixture_mol(name)
    mol.build()
    s = System(mol)
    s.derive()
    return s


# One derived system + FD oracle per case, shared across tests so the slow
# work happens once and every check looks at identical numbers.
_CACHE = {}


def _shared(name, *, need_providers):
    if name not in _CACHE or _CACHE[name][1] != need_providers:
        s = _sys(name)
        fd = hessian_components(s)
        comp = provider_components(s) if need_providers else None
        _CACHE[name] = (s, need_providers, fd, comp)
    return _CACHE[name][0], _CACHE[name][2], _CACHE[name][3]


def _gate(name):
    if name in SLOW_CASES and os.environ.get("VIBEQC_HESSIAN_SLOW") != "1":
        pytest.skip(
            "set VIBEQC_HESSIAN_SLOW=1 to run the 12-AO d-shell ERI "
            "second-derivative provider (longest pole of this slice)"
        )


# ---------------------------------------------------------------------------
# 1. the shared #179 operator identity: A = D (I + F_vv^T)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["h2", "water", "water_sdf"])
def test_operator_identity(name):
    """#179's matrix-free Jacobian A equals D (I + F_vv^T) exactly.

    This is the algebra that lets A1's dense full-response-space block system
    be solved as a single matrix-free CPHF through the shared operator.
    """
    s = _sys(name)
    C, eps = s.C, s.eps
    nocc, nmo = s.nocc, s.nmo
    occ, virt = s.occ, s.virt
    mocc = C[:, occ]
    e_i, e_a = eps[occ], eps[virt]
    e_ai = 1.0 / (e_a[:, None] - e_i[None, :])

    # A1's dense Jacobian F on the full (nmo, nocc) response space
    def F(mo1):
        dm = C @ (2 * mo1) @ mocc.T
        dm = dm + dm.T
        v = C.T @ _wof(s.ERI, dm) @ mocc
        out = v.copy()
        out[virt, :] *= e_ai
        out[occ, :] = 0
        return out

    Mat = np.zeros((nmo * nocc, nmo * nocc))
    for col in range(nmo * nocc):
        mo1 = np.zeros((nmo, nocc))
        mo1.ravel()[col] = 1.0
        Mat[:, col] = F(mo1).ravel()
    nvirt = len(virt)
    virt_rows = [mo * nocc + i for mo in virt for i in range(nocc)]
    # F_vv in assemble's (mo=i-major) flatten is (a, i) order: index a*nocc + i
    F_vv_ai = Mat[np.ix_(virt_rows, virt_rows)].reshape(nvirt, nocc, nvirt, nocc)
    # #179 packs x as (i, a): reindex (a,i,a',i') -> (i,a,i',a')
    F_vv_ia = F_vv_ai.transpose(1, 0, 3, 2).reshape(nocc * nvirt, nocc * nvirt)

    from tools.vibeqc_hessian.analytic import build_reference
    from tools.vibeqc_response.backends import DenseAOResponseBackend
    from tools.vibeqc_response.operators import RHFResponseOperator

    ref = build_reference(s)
    backend = DenseAOResponseBackend(s.ERI)
    op = RHFResponseOperator(RHFResponseOperator.build_problem(ref, backend), backend)
    A = op.to_dense()  # (i,a) order
    D = (e_a[None, :] - e_i[:, None]).reshape(nocc, nvirt).ravel()  # (i,a)
    expected = D[:, None] * (np.eye(nocc * nvirt) + F_vv_ia)
    err = np.abs(A - expected).max()
    print(f"[{name}] A = D(I+F_vv^T) max err = {err:.3e}")
    assert err < 1e-10, f"{name}: operator identity off by {err:.3e}"


# ---------------------------------------------------------------------------
# 2. #179 CPHF relaxation vs the A1 dense full-response-space CPHF
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["h2", "water", "water_sdf"])
def test_relaxation_matches_reference(name):
    """The #179 matrix-free relaxation equals A1's dense CPHF relaxation."""
    s = _sys(name)
    fd = hessian_components(s)  # dense CPHF relaxation lives in the A1 oracle
    relax_179 = cphf_relaxation(s)
    err = np.abs(relax_179 - fd["relaxation"]).max()
    print(f"[{name}] max|#179 relax - reference relax| = {err:.3e}")
    # The oracle finite-differences first-order AO integrals at h=1e-5; once
    # the production side is analytic, water_sdf is limited by that FD floor
    # (~4e-9 here) rather than by the response solve itself.
    assert err < 1e-8, f"{name}: #179 relaxation off by {err:.3e}"
    # the relaxation must be non-trivial (negative case)
    assert np.abs(fd["relaxation"]).max() > 1e-4


@pytest.mark.parametrize("name", ["h2", "water", "water_sdf"])
def test_relaxation_is_analytic_and_raw_symmetric(name):
    """The A2 response must not need System.derive() and must be raw-symmetric."""
    mol = fixture_mol(name)
    mol.build()
    s = System(mol)

    # Intentionally do not call s.derive(). Reintroducing s.h1/s.S1/s.ERI1
    # into cphf_relaxation therefore fails this regression immediately.
    assert not any(hasattr(s, key) for key in ("h1", "S1", "ERI1"))
    raw = cphf_relaxation(s)
    assert np.isfinite(raw).all()

    defect = np.abs(raw - raw.transpose(1, 0, 3, 2)).max()
    print(f"[{name}] raw relaxation symmetry defect = {defect:.3e}")
    assert defect < 2e-10, f"{name}: raw relaxation asymmetry {defect:.3e}"


# ---------------------------------------------------------------------------
# 3. #178 provider components vs the A1 FD oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["h2", "water"])
def test_provider_components(name):
    _s, fd, comp = _shared(name, need_providers=True)
    for key in ("core", "pulay", "two_electron"):
        err = np.abs(comp[key] - fd[key]).max()
        print(f"[{name}] {key}: max|provider - FD| = {err:.3e}")
        assert err < 5e-5, f"{name}: {key} provider off by {err:.3e}"


def test_provider_components_water_sdf():
    _gate("water_sdf")
    _s, fd, comp = _shared("water_sdf", need_providers=True)
    for key in ("core", "pulay", "two_electron"):
        err = np.abs(comp[key] - fd[key]).max()
        print(f"[water_sdf] {key}: max|provider - FD| = {err:.3e}")
        assert err < 5e-4, f"water_sdf: {key} provider off by {err:.3e}"


# ---------------------------------------------------------------------------
# 4. closed-form nuclear vs FD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["h2", "water", "water_sdf"])
def test_nuclear_closed_form(name):
    s, fd, _ = _shared(name, need_providers=False)
    nuc = nuclear_closed_form(s)
    err = np.abs(nuc - fd["nuclear"]).max()
    print(f"[{name}] nuclear: max|closed form - FD| = {err:.3e}")
    assert err < 5e-6, f"{name}: nuclear closed form off by {err:.3e}"


# ---------------------------------------------------------------------------
# 5. full integrated Hessian vs PySCF's analytic RHF Hessian as an end-to-end cross-check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["h2", "water"])
def test_total_matches_pyscf(name):
    s = _sys(name)
    comp = analytic_hessian(s)
    from pyscf import scf

    mf = scf.RHF(s.mol)
    mf.conv_tol = 1e-13
    mf.kernel()
    H_pyscf = mf.Hessian().kernel()
    err = np.abs(comp["total"] - H_pyscf).max()
    print(f"[{name}] total: max|integrated - PySCF analytic| = {err:.3e}")
    assert err < 5e-7, f"{name}: integrated Hessian off from PySCF by {err:.3e}"


def test_total_matches_pyscf_water_sdf():
    _gate("water_sdf")
    s = _sys("water_sdf")
    comp = analytic_hessian(s)
    from pyscf import scf

    mf = scf.RHF(s.mol)
    mf.conv_tol = 1e-13
    mf.kernel()
    H_pyscf = mf.Hessian().kernel()
    err = np.abs(comp["total"] - H_pyscf).max()
    print(f"[water_sdf] total: max|integrated - PySCF analytic| = {err:.3e}")
    assert err < 5e-5, f"water_sdf: integrated Hessian off from PySCF by {err:.3e}"


# ---------------------------------------------------------------------------
# 6. symmetry + translation invariance of the integrated total
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["h2", "water"])
def test_invariances(name):
    s = _sys(name)
    H = analytic_hessian(s)["total"]
    sym = np.abs(H - H.transpose(1, 0, 3, 2)).max()
    Hr = H.transpose(0, 2, 1, 3).reshape(3 * s.nat, 3 * s.nat)
    tr = max(np.abs(Hr.sum(axis=1)).max(), np.abs(Hr.sum(axis=0)).max())
    print(f"[{name}] sym = {sym:.3e}  translation = {tr:.3e}")
    assert sym < 1e-9, f"{name}: integrated Hessian not symmetric ({sym:.3e})"
    assert tr < 1e-4, f"{name}: integrated Hessian not translation invariant ({tr:.3e})"


# ---------------------------------------------------------------------------
# 7. exact elimination preserves the full dense response and Hessian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["h2", "water", "water_sdf"])
def test_reduced_response_matches_full_space(name):
    """Known occupied metric response belongs on the reduced virtual RHS."""
    mol = fixture_mol(name)
    mol.build()
    s = System(mol)
    s.derive()
    h1ao = ref_h1ao(s)
    mo1_full, e1_full = _first_order_mo1_e1(s, h1ao)
    mo1_reduced, e1_reduced = _first_order_mo1_e1_vir_only(s, h1ao)
    np.testing.assert_allclose(mo1_reduced, mo1_full, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(e1_reduced, e1_full, atol=2e-10, rtol=2e-10)
    H_full = hessian_total(s)
    H_reduced = hessian_total(s, mo1e1_fn=_first_order_mo1_e1_vir_only)
    np.testing.assert_allclose(H_reduced, H_full, atol=2e-9, rtol=2e-10)
    H_fd = fd_hessian(mol, h=1e-4)
    assert np.abs(H_full - H_fd).max() < 5e-3


@pytest.mark.parametrize("nbf", [13, 18])
@pytest.mark.parametrize("entry", [analytic_hessian, cphf_relaxation])
def test_analytic_response_domain_rejected_before_provider_work(nbf, entry):
    # A reference System may have 18 AOs, but the shared dense response
    # oracle is hard-bounded to 12. No SCF/provider work is needed to refuse.
    with pytest.raises(ValueError, match="analytic.*12 AOs"):
        entry(SimpleNamespace(nbf=nbf))


@pytest.mark.parametrize(
    "relax",
    [
        0.0,
        np.zeros((6, 6)),
        np.zeros((2, 2, 3, 3), complex),
        np.full((2, 2, 3, 3), np.nan),
        np.full((2, 2, 3, 3), np.inf),
    ],
)
def test_analytic_hessian_rejects_invalid_relaxation_before_providers(
    monkeypatch, relax
):
    def unexpected_provider(system):
        pytest.fail("invalid relaxation reached expensive provider work")

    monkeypatch.setattr(
        "tools.vibeqc_hessian.analytic.provider_components", unexpected_provider
    )
    with pytest.raises(ValueError, match="relaxation.*finite real.*shape"):
        analytic_hessian(SimpleNamespace(nbf=2, nat=2), relax=relax)
