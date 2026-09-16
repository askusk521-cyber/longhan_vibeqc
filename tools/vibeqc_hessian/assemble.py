"""Component-wise analytic RHF Hessian (issue #180, slice A, PR A2).

This is the analytic assembly that PR A1's numerical oracle verifies. It is
self-contained: every integral value and its first/second coordinate
derivatives are obtained by fresh-molecule finite differences, so there is no
shared intor-cache state and no dependency on a particular convergence path.

Structure (mirrors PySCF hess_elec + hess_nuc exactly):

    Total  = nuclear + core + Pulay + frozen-2e + relaxation
    nuclear   = En2 blocks
    core      = Tr[P0 h2]
    Pulay     = -2 Tr[s1aa W_e]  (ia==ja)  -2 Tr[s1ab W_e]  (ia!=ja)
    frozen-2e = Tr[W2 ERI2]
    relaxation = sum over atom pairs ia>=ja of
        4  Tr[h1ao[ia][x]  (mo1[ja][y] mocc^T)]
       - 4  Tr[s1ao[ia][x]  (mo1[ja][y] eps_occ mocc^T)]
       - 2  Tr[s1oo[ia][x]  mo_e1[ja][y]]

All intermediates are reconstructed from VibeQC tensors:

    s1ao[ia][x] = S1[ia*3+x]
    h1ao[ia][x] = h1[ia*3+x] + Wof(ERI1[ia*3+x], P0)
    mo1, mo_e1  = 1st-order CPHF, metric gauge (see _first_order_mo1_e1)

CPHF note. The 1st-order CPHF response is solved on the FULL (nmo, nocc)
orbital-response space -- not just the virtual-occupied block with the
occupied block frozen. PySCF's solve_withs1 iterates the occupied-column
response density C (2 mo1) mocc^T + c.c. through the full mo1 and then
refines the virtual block. For a single virtual (H2) the two formulations
coincide because the (nocc, nvir) block is 1x1; for multiple virtuals
(water: 5 occupied, 2 virtual) the full-space form is required. This was the
defect that H2 masked.

The analytic formula itself is essentially exact: water STO-3G matches
PySCF's full analytic RHF Hessian to ~1e-6 when the 2nd-derivative
integrals are differenced at a larger step (h2); at the default small step
the residual is pure O(h2^2) finite-difference truncation of the
2nd-derivative integrals, not a formula error.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto, scf

_EL = {1: "H", 2: "He", 3: "Li", 6: "C", 8: "O"}
_H1 = 1e-5  # 1st-derivative integral step
_H2 = 3e-4  # 2nd-derivative integral step (tunable; see module docstring)


def build_mol(atoms, basis, charge=0, spin=0):
    """Build a PySCF mol in Bohr / cartesian from VibeQC-style atoms+basis.

    ``atoms`` is a list of (Z, [x, y, z]); ``basis`` maps an element tag to a
    list of shells, each shell = [L, (a1, c1), (a2, c2), ...].
    """
    atom, bas = [], {}
    for ai, (z, xyz) in enumerate(atoms):
        tag = f"{_EL.get(z, 'Z' + str(z))}{ai}"
        atom.append([tag, [float(v) for v in xyz]])
        bas[tag] = basis[tag]
    return gto.M(
        atom=atom,
        basis=bas,
        charge=charge,
        spin=spin,
        unit="Bohr",
        cart=True,
        verbose=0,
    )


def _mol_at(mol0, coords):
    """Rebuild ``mol0``'s geometry with new cartesian coordinates (Bohr)."""
    atom = [[a[0], [float(v) for v in row]] for a, row in zip(mol0.atom, coords)]
    return gto.M(
        atom=atom,
        basis=mol0.basis,
        charge=mol0.charge,
        spin=mol0.spin,
        unit="Bohr",
        cart=True,
        verbose=0,
    )


class System:
    """Converged RHF state + coordinate FD derivatives of every integral."""

    def __init__(self, mol, h1=_H1, h2=_H2):
        self.mol = mol
        self.nat = mol.natm
        self.nd = 3 * mol.natm
        self.h = h2
        self.h1_ = h1
        self.coords = np.array(mol.atom_coords(), float)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-13
        mf.max_cycle = 400
        mf.kernel()
        self.C = mf.mo_coeff
        self.eps = mf.mo_energy
        self.nocc = int(np.sum(mf.mo_occ > 0.5))
        self.nmo = self.C.shape[1]
        self.nbf = self.C.shape[0]
        self.P0 = 2 * self.C[:, : self.nocc] @ self.C[:, : self.nocc].T
        self.S0 = mol.intor("int1e_ovlp_cart")
        self.ERI = mol.intor("int2e_cart", aosym=1)
        self.Z = np.array(mol.atom_charges())
        self.occ = np.arange(self.nocc)
        self.virt = np.arange(self.nocc, self.nmo)

    def _h_ao(self, c):
        m = _mol_at(self.mol, c)
        return m.intor("int1e_kin_cart") + m.intor("int1e_nuc")

    def _s_ao(self, c):
        return _mol_at(self.mol, c).intor("int1e_ovlp_cart")

    def _eri(self, c):
        return _mol_at(self.mol, c).intor("int2e_cart", aosym=1)

    def _enuc(self, c):
        c = np.asarray(c)
        e = 0.0
        for a in range(self.nat):
            for b in range(a + 1, self.nat):
                e += self.Z[a] * self.Z[b] / np.linalg.norm(c[a] - c[b])
        return e

    def _d1(self, F, i):
        h = self.h1_
        b = self.coords
        cp = b.copy()
        cp.flat[i] += h
        cm = b.copy()
        cm.flat[i] -= h
        return (F(cp) - F(cm)) / (2 * h)

    def _d2(self, F, i, j):
        h = self.h
        b = self.coords
        if i == j:
            cp = b.copy()
            cp.flat[i] += h
            cm = b.copy()
            cm.flat[i] -= h
            return (F(cp) - 2 * F(b.copy()) + F(cm)) / h**2
        c1 = b.copy()
        c1.flat[i] += h
        c1.flat[j] += h
        c2 = b.copy()
        c2.flat[i] += h
        c2.flat[j] -= h
        c3 = b.copy()
        c3.flat[i] -= h
        c3.flat[j] += h
        c4 = b.copy()
        c4.flat[i] -= h
        c4.flat[j] -= h
        return (F(c1) - F(c2) - F(c3) + F(c4)) / (4 * h**2)

    def _sym(self, d):
        for i in range(self.nd):
            for j in range(i):
                d[(i, j)] = d[(j, i)]
        return d

    def derive(self):
        nd = self.nd
        self.h1 = {i: self._d1(self._h_ao, i) for i in range(nd)}
        self.h2 = self._sym(
            {
                (i, j): self._d2(self._h_ao, i, j)
                for i in range(nd)
                for j in range(i, nd)
            }
        )
        self.S1 = {i: self._d1(self._s_ao, i) for i in range(nd)}
        self.S2 = self._sym(
            {
                (i, j): self._d2(self._s_ao, i, j)
                for i in range(nd)
                for j in range(i, nd)
            }
        )
        self.ERI1 = {i: self._d1(self._eri, i) for i in range(nd)}
        self.ERI2 = self._sym(
            {(i, j): self._d2(self._eri, i, j) for i in range(nd) for j in range(i, nd)}
        )
        self.En2 = self._sym(
            {
                (i, j): self._d2(self._enuc, i, j)
                for i in range(nd)
                for j in range(i, nd)
            }
        )

    def Wof(self, eri, P):
        return _wof(eri, P)


def _wof(eri, P):
    return np.einsum("mnls,ls->mn", eri, P) - 0.5 * np.einsum("mlns,ls->mn", eri, P)


def _s2_block(s, ia, ja):
    return np.array(
        [[s.S2[(ia * 3 + x, ja * 3 + y)] for y in range(3)] for x in range(3)]
    )


def _h2_block(s, ia, ja):
    return np.array(
        [[s.h2[(ia * 3 + x, ja * 3 + y)] for y in range(3)] for x in range(3)]
    )


def _eri2_block(s, ia, ja, W2):
    return np.array(
        [
            [
                np.einsum("uvls,uvls->", W2, s.ERI2[(ia * 3 + x, ja * 3 + y)])
                for y in range(3)
            ]
            for x in range(3)
        ]
    )


def _en2_block(s, ia, ja):
    return np.array(
        [[s.En2[(ia * 3 + x, ja * 3 + y)] for y in range(3)] for x in range(3)]
    )


def h1ao(s):
    C = s.C
    P0 = s.P0
    nat = s.nat
    nbf = C.shape[0]
    H = np.zeros((nat, 3, nbf, nbf))
    for ia in range(nat):
        for x in range(3):
            R = ia * 3 + x
            H[ia, x] = s.h1[R] + _wof(s.ERI1[R], P0)
    return H


def _first_order_mo1_e1(s, h1ao):
    """AO-space mo1[ia][x] (nbf, nocc) and mo_e1[ia][x] (nocc, nocc).

    Dense replica of PySCF solve_withs1 on the full (nmo, nocc) response:
    Krylov iteration over the occupied-column induced density, then a final
    refinement of the virtual block. Metric gauge (occ-occ = -S1_mo/2).
    """
    C = s.C
    eps = s.eps
    nocc = s.nocc
    occ = s.occ
    virt = s.virt
    mocc = C[:, occ]
    nat = s.nat
    nbf = C.shape[0]
    nmo = s.nmo
    e_i = eps[occ]
    e_a = eps[virt]
    e_ai = 1 / (e_a[:, None] - e_i[None, :])  # (nvir, nocc)
    mo1s = np.zeros((nat, 3, nbf, nocc))
    e1s = np.zeros((nat, 3, nocc, nocc))
    for ia in range(nat):
        for x in range(3):
            R = ia * 3 + x
            h1_mo = C.T @ h1ao[ia, x] @ mocc  # (nmo, nocc)
            s1_mo = C.T @ s.S1[R] @ mocc
            hs0 = h1_mo - s1_mo * e_i[None, :]
            mo1base = hs0.copy()
            mo1base[virt, :] = -hs0[virt, :] * e_ai
            mo1base[occ, :] = -s1_mo[occ, :] * 0.5

            def F(mo1):
                dm = C @ (2 * mo1) @ mocc.T
                dm = dm + dm.T
                v = C.T @ _wof(s.ERI, dm) @ mocc
                out = v.copy()
                out[virt, :] *= e_ai
                out[occ, :] = 0
                return out

            dim = nmo * nocc
            Mat = np.zeros((dim, dim))
            for col in range(dim):
                mo1 = np.zeros((nmo, nocc))
                mo1.ravel()[col] = 1.0
                Mat[:, col] = F(mo1).ravel()
            mo1 = np.linalg.solve(np.eye(dim) + Mat, mo1base.ravel()).reshape(nmo, nocc)
            mo1[occ, :] = mo1base[occ, :]
            dm = C @ (2 * mo1) @ mocc.T
            dm = dm + dm.T
            fvind_full = C.T @ _wof(s.ERI, dm) @ mocc
            hs = hs0 + fvind_full
            mo1[virt, :] = hs[virt, :] / (e_i[None, :] - e_a[:, None])
            mo1[occ, :] = mo1base[occ, :]
            mo1s[ia, x] = C @ mo1  # AO-space (nbf, nocc)
            e1s[ia, x] = hs[occ, :] + mo1[occ, :] * (e_i[:, None] - e_i)
    return mo1s, e1s


def hessian_total(
    s, with_relax=True, with_pulay=True, with_2e=True, with_nuc=True, with_core=True
):
    """Component-separated analytic RHF Hessian, shaped (nat, nat, 3, 3).

    Returns the full Hessian; component isolation is available by toggling the
    ``with_*`` flags (see the negative-case tests).
    """
    C = s.C
    P0 = s.P0
    eps = s.eps
    nocc = s.nocc
    occ = s.occ
    nat = s.nat
    nbf = C.shape[0]
    mocc = C[:, occ]
    mol = s.mol
    W_e = sum(2 * eps[k] * np.outer(C[:, k], C[:, k]) for k in range(nocc))
    W2 = 0.5 * np.einsum("uv,ls->uvls", P0, P0) - 0.25 * np.einsum(
        "ul,vs->uvls", P0, P0
    )
    aoslices = mol.aoslice_by_atom()

    h1ao = np.zeros((nat, 3, nbf, nbf))
    for ia in range(nat):
        for x in range(3):
            R = ia * 3 + x
            h1ao[ia, x] = s.h1[R] + _wof(s.ERI1[R], P0)

    mo1s, e1s = _first_order_mo1_e1(s, h1ao)

    H = np.zeros((nat, nat, 3, 3))
    for i0, ia in enumerate(range(nat)):
        p0, p1 = aoslices[ia][2:]
        if with_pulay:
            s1aa = _s2_block(s, ia, ia)
            H[i0, i0] -= np.einsum("xypq,pq->xy", s1aa[:, :, p0:p1], W_e[p0:p1]) * 2
        for j0, ja in enumerate(range(i0 + 1)):
            q0, q1 = aoslices[ja][2:]
            if with_pulay:
                s1ab = _s2_block(s, ia, ja)
                H[i0, j0] -= (
                    np.einsum(
                        "xypq,pq->xy", s1ab[:, :, p0:p1, q0:q1], W_e[p0:p1, q0:q1]
                    )
                    * 2
                )
            if with_core:
                H[i0, j0] += np.einsum("xypq,pq->xy", _h2_block(s, ia, ja), P0)
            if with_2e:
                H[i0, j0] += _eri2_block(s, ia, ja, W2)
            if with_relax:
                s1ao = np.stack([s.S1[ia * 3 + x] for x in range(3)])
                s1oo = np.einsum("xpq,pi,qj->xij", s1ao, mocc, mocc)
                for x in range(3):
                    for y in range(3):
                        dm1 = mo1s[ja, y] @ mocc.T
                        dm1e = (mo1s[ja, y] * eps[occ][None, :]) @ mocc.T
                        H[i0, j0][x, y] += np.einsum("pq,pq->", h1ao[ia, x], dm1) * 4
                        H[i0, j0][x, y] -= np.einsum("pq,pq->", s1ao[x], dm1e) * 4
                        H[i0, j0][x, y] -= np.einsum("pq,pq->", s1oo[x], e1s[ja, y]) * 2
            if with_nuc:
                H[i0, j0] += _en2_block(s, ia, ja)
        for j0 in range(i0):
            H[j0, i0] = H[i0, j0].T
    return H


def hessian_components(s):
    """Return each component separately for negative-case isolation.

    Keys: nuclear, core, pulay, two_electron, relaxation. Their sum equals
    hessian_total(s) with every flag on.
    """
    full = hessian_total(s)
    nuc = hessian_total(
        s,
        with_relax=False,
        with_pulay=False,
        with_2e=False,
        with_core=False,
        with_nuc=True,
    )
    core = hessian_total(
        s,
        with_relax=False,
        with_pulay=False,
        with_2e=False,
        with_nuc=False,
        with_core=True,
    )
    pulay = hessian_total(
        s,
        with_relax=False,
        with_2e=False,
        with_core=False,
        with_nuc=False,
        with_pulay=True,
    )
    two_e = hessian_total(
        s,
        with_relax=False,
        with_pulay=False,
        with_core=False,
        with_nuc=False,
        with_2e=True,
    )
    relax = full - (nuc + core + pulay + two_e)
    return {
        "nuclear": nuc,
        "core": core,
        "pulay": pulay,
        "two_electron": two_e,
        "relaxation": relax,
        "total": full,
    }


def fd_hessian(mol, h=1e-4):
    """Total-energy FD Hessian (central), shaped (nat, nat, 3, 3)."""
    nat = mol.natm
    nd = 3 * nat
    labels = [a[0] for a in mol.atom]
    basis = mol.basis
    charge = mol.charge
    spin = mol.spin

    def mol_at(c):
        atom = list(zip(labels, [list(map(float, x)) for x in c]))
        return gto.M(
            atom=atom,
            basis=basis,
            charge=charge,
            spin=spin,
            unit="Bohr",
            cart=True,
            verbose=0,
        )

    def E(m):
        mf = scf.RHF(m)
        mf.kernel()
        return mf.e_tot

    c0 = np.array(mol.atom_coords())
    H = np.zeros((nd, nd))
    E0 = E(mol_at(c0))
    for i in range(nd):
        cp = c0.copy()
        cp.flat[i] += h
        cm = c0.copy()
        cm.flat[i] -= h
        H[i, i] = (E(mol_at(cp)) - 2 * E0 + E(mol_at(cm))) / h**2
    for i in range(nd):
        for j in range(i + 1, nd):
            pp = c0.copy()
            pp.flat[i] += h
            pp.flat[j] += h
            pm = c0.copy()
            pm.flat[i] += h
            pm.flat[j] -= h
            mp = c0.copy()
            mp.flat[i] -= h
            mp.flat[j] += h
            mm = c0.copy()
            mm.flat[i] -= h
            mm.flat[j] -= h
            H[i, j] = (
                E(mol_at(pp)) - E(mol_at(pm)) - E(mol_at(mp)) + E(mol_at(mm))
            ) / (4 * h**2)
            H[j, i] = H[i, j]
    return H.reshape(nat, 3, nat, 3).transpose(0, 2, 1, 3)


__all__ = [
    "System",
    "build_mol",
    "fd_hessian",
    "h1ao",
    "hessian_components",
    "hessian_total",
]
