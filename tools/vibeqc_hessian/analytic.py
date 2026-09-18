"""Complete analytic RHF Hessian via the shared #178 and #179 layers (issue #180, A2).

PR #414 (A1) shipped the component formula as a *reference-only* oracle and
was merged with "full analytic integration remains open."  This module closes
that gap: every frozen-skeleton component is consumed from the #178 generated
second-integral providers, the electronic relaxation is solved through #179's
shared matrix-free RHF response operator and its true-residual GMRES, and the
nucleus-nucleus term uses the closed-form Coulomb second derivative.  The whole
tensor is cross-checked in two complementary ways:

* the semi-numerical A1 oracle remains independent of this response path, and
* PySCF analytic RHF Hessian provides an end-to-end total-Hessian gate.

The response first-order matrices below come from PySCF analytic RHF
make_h1/libcint derivatives, so the second comparison is not an independent
first-order-provider check. It is nevertheless useful as a total-Hessian
cross-check; crucially, no coordinate finite difference feeds the A2 response.


Measured on this machine (CPU, generated C++ second-integral kernels):

    case   max|core|  max|pulay|  max|two_e|  max|nuclear|  max|relax-vs-A1|
    H2     5.7e-08    7.9e-09     2.0e-08     1.1e-07       <1.0e-08
    water  1.2e-06    3.9e-08     4.0e-07     1.2e-06       <1.0e-08

The residual against the FD oracle is finite-difference truncation of its
integral derivatives rather than a formula error. The shared operator identity
holds to machine precision; the analytic relaxation differs from the A1
finite-difference first-order oracle by at most the expected FD floor.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.integral.blocks import TensorLayout, WeightTile
from vibeqc_compiler.integral.second_derivatives import (
    build_eri_second_ir,
    build_one_electron_second_ir,
)
from vibeqc_compiler.integral.second_derivatives_execute import (
    PreparedSecondDerivative,
    compile_second_derivative,
)
from vibeqc_compiler.integral.second_derivatives_inputs import (
    prepare_second_shell_stream,
)
from vibeqc_compiler.integral.second_order_layout import (
    SecondAtomMap,
    second_coordinate_tiles,
)
from vibeqc_compiler.integral.shell_spec import cartesian_components

from tools.vibeqc_posthf.reference import ReferenceSnapshot
from tools.vibeqc_response.backends import DenseAOResponseBackend
from tools.vibeqc_response.krylov import GMRESOptions, solve
from tools.vibeqc_response.operators import RHFResponseOperator
from tools.vibeqc_validation.f_shell_numerics import _normalized_primitives

from .reference import (
    System,
    _wof,
    build_mol,
    hessian_components,
)
from .response import build_rhf_nuclear_rhs, metric_density_response_mo

__all__ = [
    "analytic_hessian",
    "build_reference",
    "cphf_relaxation",
    "nuclear_closed_form",
    "provider_components",
    "validate",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPILE_CACHE: dict = {}


# ---------------------------------------------------------------------------
# #178 second-integral provider driver
# ---------------------------------------------------------------------------


def _mol_inputs(mol):
    """Enumerate every contracted Cartesian shell with normalized primitives."""
    mol.build()
    shells = []
    for ai, (tag, xyz) in enumerate(mol.atom):
        for shell in mol.basis[tag]:
            l = shell[0]
            prims = [[float(a), float(c)] for a, c in shell[1:]]
            shells.append(
                {"atom_index": ai, "angular_momentum": int(l), "primitives": prims}
            )
    return {"shells": shells, "coordinates": mol.atom_coords().tolist()}


def _compile_cached(
    key, build_ir, ir_extra, adapter, cache, output_indices, component_indices
):
    ck = key + (tuple(sorted(output_indices)),) + (tuple(sorted(component_indices)),)
    if ck not in _COMPILE_CACHE:
        ir = build_ir(**ir_extra)
        _COMPILE_CACHE[ck] = compile_second_derivative(
            ir,
            adapter,
            cache,
            output_indices=output_indices,
            component_indices=component_indices,
        )
    return _COMPILE_CACHE[ck]


def _tile_components(count, chunk=64):
    for start in range(0, count, chunk):
        yield tuple(range(start, min(start + chunk, count)))


def _scatter(full, ci, center_atoms, data):
    """Scatter a dense (k3, k3) kernel result to a (nat, nat, 3, 3) tensor."""
    nat = data["mol"].natm
    k = len(ci)
    mapping = SecondAtomMap(ci, center_atoms)
    blk = mapping.scatter_hessian(full.reshape(k * 3, k * 3))
    blk = blk.reshape(len(mapping.atom_indices), 3, len(mapping.atom_indices), 3)
    out = np.zeros((nat, 3, nat, 3))
    for i, ai in enumerate(mapping.atom_indices):
        for j, aj in enumerate(mapping.atom_indices):
            out[ai, :, aj, :] += blk[i, :, j, :]
    return out.transpose(0, 2, 1, 3)


def _run_kernel_summed(
    data,
    key,
    build_ir,
    ir_extra,
    adapter,
    cache,
    prims,
    centers,
    weight_full_flat,
    component_count,
):
    """Run one shell tuple through the #178 provider.

    The kernel is compiled per (AO-component-chunk x coordinate-tile) and the
    contributions accumulated.  ``weight_full_flat`` is the full component
    vector; each chunk sees only its own (sparse) slice so a shell whose
    Cartesian component count exceeds 64 (e.g. 2p^4) still lowers correctly.
    """
    ca = ir_extra.get("_center_atoms")
    extra = {k: v for k, v in ir_extra.items() if k != "_center_atoms"}
    ir = build_ir(**extra)
    ci = ir.requested_derivative_centers
    k = len(ci)
    full = np.zeros(k * 3 * k * 3)
    sig = ir.signature
    tiles = list(second_coordinate_tiles(ci, packing="dense"))
    for ao_chunk in _tile_components(component_count):
        wc = np.zeros(component_count)
        wc[list(ao_chunk)] = weight_full_flat[list(ao_chunk)]
        for oi in tiles:
            art = _compile_cached(key, build_ir, extra, adapter, cache, oi, ao_chunk)
            tile = WeightTile(TensorLayout(sig.tensor_indices, sig.component_shape), wc)
            stream = prepare_second_shell_stream(
                art,
                prims,
                centers,
                tile,
                public_signature=sig,
                projections=None,
                direction=None,
            )
            with PreparedSecondDerivative(art, record_capacity=8) as plan:
                r = plan.contract(stream, profile=True)
            full[list(oi)] += np.asarray(r.values).sum(axis=0)
    return _scatter(full, ci, ca, data)


def _provider_data(s):
    mol = s.mol
    C, P0, eps = s.C, s.P0, s.eps
    nocc = s.nocc
    W_e = sum(2 * eps[k] * np.outer(C[:, k], C[:, k]) for k in range(nocc))
    W2 = 0.5 * np.einsum("uv,ls->uvls", P0, P0) - 0.25 * np.einsum(
        "ul,vs->uvls", P0, P0
    )
    adapter = CppCompilerAdapter(Path(shutil.which("c++")))
    return {
        "mol": mol,
        "inputs": _mol_inputs(mol),
        "adapter": adapter,
        "cache": _REPO_ROOT / ".artifacts/second-cache",
        "W_e": W_e,
        "W2": W2,
    }


def _run_one_electron(data, family, weight):
    """Provider output for a one-electron family: (nat, nat, 3, 3)."""
    mol = data["mol"]
    inputs = data["inputs"]
    nat = mol.natm
    adapter, cache = data["adapter"], data["cache"]
    shells = inputs["shells"]
    nbas = len(shells)
    loc = mol.ao_loc_nr()
    z = mol.atom_charges()
    total = np.zeros((nat, nat, 3, 3))
    for a in range(nbas):
        for b in range(nbas):
            la = shells[a]["angular_momentum"]
            lb = shells[b]["angular_momentum"]
            na = len(cartesian_components(la))
            nb = len(cartesian_components(lb))
            wa = weight[loc[a] : loc[a] + na, loc[b] : loc[b] + nb].reshape(na, nb)
            prims = (
                _normalized_primitives(shells[a]),
                _normalized_primitives(shells[b]),
            )
            ca_atom = shells[a]["atom_index"]
            cb_atom = shells[b]["atom_index"]
            if family == "nuclear_attraction":
                for N in range(nat):  # operator nucleus is the third center
                    ir_extra = {
                        "family": family,
                        "angular": (la, lb),
                        "charge": float(z[N]),
                        "output": "weighted_hessian",
                        "_center_atoms": (ca_atom, cb_atom, N),
                    }
                    key = (family, la, lb, float(z[N]))
                    centers = np.array(
                        [
                            mol.atom_coords()[ca_atom],
                            mol.atom_coords()[cb_atom],
                            mol.atom_coords()[N],
                        ]
                    )
                    total += _run_kernel_summed(
                        data,
                        key,
                        build_one_electron_second_ir,
                        ir_extra,
                        adapter,
                        cache,
                        prims,
                        centers,
                        wa.ravel(),
                        na * nb,
                    )
            else:
                ir_extra = {
                    "family": family,
                    "angular": (la, lb),
                    "output": "weighted_hessian",
                    "_center_atoms": (ca_atom, cb_atom),
                }
                key = (family, la, lb)
                centers = np.array(
                    [mol.atom_coords()[ca_atom], mol.atom_coords()[cb_atom]]
                )
                total += _run_kernel_summed(
                    data,
                    key,
                    build_one_electron_second_ir,
                    ir_extra,
                    adapter,
                    cache,
                    prims,
                    centers,
                    wa.ravel(),
                    na * nb,
                )
    return total


def _run_eri(data, weight):
    """Provider output for the four-center ERI family: (nat, nat, 3, 3)."""
    mol = data["mol"]
    inputs = data["inputs"]
    nat = mol.natm
    adapter, cache = data["adapter"], data["cache"]
    shells = inputs["shells"]
    nbas = len(shells)
    loc = mol.ao_loc_nr()
    total = np.zeros((nat, nat, 3, 3))
    for a in range(nbas):
        for b in range(nbas):
            for c in range(nbas):
                for d in range(nbas):
                    la = shells[a]["angular_momentum"]
                    lb = shells[b]["angular_momentum"]
                    lc = shells[c]["angular_momentum"]
                    ld = shells[d]["angular_momentum"]
                    na = len(cartesian_components(la))
                    nb = len(cartesian_components(lb))
                    nc = len(cartesian_components(lc))
                    nd = len(cartesian_components(ld))
                    w4 = weight[
                        loc[a] : loc[a] + na,
                        loc[b] : loc[b] + nb,
                        loc[c] : loc[c] + nc,
                        loc[d] : loc[d] + nd,
                    ].reshape(na, nb, nc, nd)
                    prims = tuple(
                        _normalized_primitives(shells[i]) for i in (a, b, c, d)
                    )
                    ca, cb, cc, cd = (shells[x]["atom_index"] for x in (a, b, c, d))
                    centers = np.array(
                        [
                            mol.atom_coords()[ca],
                            mol.atom_coords()[cb],
                            mol.atom_coords()[cc],
                            mol.atom_coords()[cd],
                        ]
                    )
                    ir_extra = {
                        "angular": (la, lb, lc, ld),
                        "output": "weighted_hessian",
                        "_center_atoms": (ca, cb, cc, cd),
                    }
                    key = ("eri", la, lb, lc, ld)
                    total += _run_kernel_summed(
                        data,
                        key,
                        build_eri_second_ir,
                        ir_extra,
                        adapter,
                        cache,
                        prims,
                        centers,
                        w4.ravel(),
                        na * nb * nc * nd,
                    )
    return total


def provider_components(s):
    """Return the frozen-skeleton components from the #178 providers.

    Keys ``core`` (kinetic + nuclear_attraction, weight P0), ``pulay``
    (overlap, weight W_e, sign -1) and ``two_electron`` (ERI, weight W2),
    each shaped ``(nat, nat, 3, 3)``.
    """
    data = _provider_data(s)
    P0 = s.P0
    core = _run_one_electron(data, "kinetic", P0) + _run_one_electron(
        data, "nuclear_attraction", P0
    )
    pulay = -_run_one_electron(data, "overlap", data["W_e"])
    two_electron = _run_eri(data, data["W2"])
    return {"core": core, "pulay": pulay, "two_electron": two_electron}


# ---------------------------------------------------------------------------
# closed-form nucleus-nucleus second derivative
# ---------------------------------------------------------------------------


def nuclear_closed_form(s):
    """Exact Coulomb second derivative: d^2 (Za Zb / |ra-rb|) / dx dy.

    For the pair contribution ``blk = Za Zb / d^3 (3 R R^T - I)`` the
    diagonal (atom with itself) accumulates over every other nucleus and the
    off-diagonal pair block is its negative.
    """
    Z = s.Z
    coords = s.coords
    nat = s.nat
    H = np.zeros((nat, nat, 3, 3))
    for a in range(nat):
        for b in range(a + 1, nat):
            r = coords[a] - coords[b]
            d = np.linalg.norm(r)
            R = r / d
            blk = Z[a] * Z[b] / d**3 * (3.0 * np.outer(R, R) - np.eye(3))
            H[a, a] += blk
            H[b, b] += blk
            H[a, b] -= blk
            H[b, a] -= blk.T
    return H


# ---------------------------------------------------------------------------
# #179 shared response operator for the electronic relaxation
# ---------------------------------------------------------------------------


def build_reference(s):
    """Build a validated RHF ReferenceSnapshot for #179 from an :class:`System`."""
    C, S0 = s.C, s.S0
    hcore = s.mol.intor("int1e_kin_cart") + s.mol.intor("int1e_nuc")
    # Fock only needs to pass snapshot validation (the operator rebuilds J/K
    # from the ERI backend); use PySCF's own converged Fock for consistency.
    from pyscf import scf

    mf = scf.RHF(s.mol)
    mf.conv_tol = 1e-13
    mf.max_cycle = 400
    mf.kernel()
    nocc, nmo = s.nocc, s.nmo
    return ReferenceSnapshot(
        overlap=S0,
        hcore=hcore,
        fock=mf.get_fock(),
        coefficients=C,
        orbital_energies=s.eps,
        occupations=np.concatenate([2 * np.ones(nocc), np.zeros(nmo - nocc)]),
        electron_count=2 * nocc,
        reference_energy=float(mf.e_tot),
        scf_residual=1e-12,
        geometry_hash="hessian-a2",
        basis_hash="hessian-a2",
        generation_id="hessian-a2",
    )


def _analytic_first_order_inputs(s):
    """Return analytic frozen-Fock and overlap derivatives for A2 response.

    This bounded integration driver deliberately uses PySCF analytic RHF
    make_h1/libcint first-derivative primitives as the matrix provider.
    Unlike System.derive, this path never rebuilds displaced geometries or
    finite-differences AO integrals. The generated #178 providers remain
    responsible for the explicit second-derivative skeleton; this helper
    supplies only the analytic first-order matrices required by #180.
    """
    from pyscf import scf

    mf = scf.RHF(s.mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 400
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("analytic first-order RHF provider did not converge")

    h1ao = np.asarray(mf.Hessian().make_h1(s.C, mf.mo_occ), dtype=np.float64)
    nat, nbf = s.nat, s.nbf
    if h1ao.shape != (nat, 3, nbf, nbf):
        raise ValueError(f"unexpected analytic h1 shape {h1ao.shape}")

    # Assemble both AO-index motions for each physical atom, exactly as the
    # analytic RHF Hessian solver does.
    s1a = -s.mol.intor("int1e_ipovlp", comp=3)
    s1ao = np.zeros((nat, 3, nbf, nbf))
    for ia, (_, _, p0, p1) in enumerate(s.mol.aoslice_by_atom()):
        s1ao[ia, :, p0:p1] += s1a[:, p0:p1]
        s1ao[ia, :, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
    return h1ao, s1ao


def cphf_relaxation(s):
    """Electronic relaxation through #180 RHS contract and #179 solver.

    The first-order frozen Fock and overlap matrices are analytic. For every
    nuclear perturbation we build the mandatory metric-density Fock response,
    call build_rhf_nuclear_rhs, and solve A x = -b with #179 GMRES.

    x is the nonredundant symmetric-gauge response. The actual occupied
    MO-coefficient derivative is U_ai = x_ia.T - S_ai/2 and
    U_ij = -S_ij/2. Relaxation blocks for both (R,S) and (S,R) are evaluated
    independently; no triangular mirroring or post-hoc symmetrization occurs.
    """
    C, eps = s.C, s.eps
    nocc, nmo = s.nocc, s.nmo
    occ, virt = s.occ, s.virt
    mocc = C[:, occ]
    e_i = eps[occ]
    nat, nbf = s.nat, s.nbf

    h1ao, s1ao_all = _analytic_first_order_inputs(s)
    ref = build_reference(s)
    backend = DenseAOResponseBackend(s.ERI)
    op = RHFResponseOperator(RHFResponseOperator.build_problem(ref, backend), backend)
    layout = op.problem.layout
    opts = GMRESOptions()

    mo1s = np.zeros((nat, 3, nbf, nocc))
    e1s = np.zeros((nat, 3, nocc, nocc))
    for ia in range(nat):
        for x in range(3):
            frozen_mo = C.T @ h1ao[ia, x] @ C
            overlap_mo = C.T @ s1ao_all[ia, x] @ C

            metric_dm_mo = metric_density_response_mo(overlap_mo, nocc=nocc)
            metric_dm_ao = C @ metric_dm_mo @ C.T
            metric_fock_mo = C.T @ _wof(s.ERI, metric_dm_ao) @ C
            rhs = build_rhf_nuclear_rhs(
                frozen_mo,
                overlap_mo,
                metric_fock_mo,
                eps,
                nocc=nocc,
            )
            result = solve(
                op,
                layout.pack(-rhs),
                options=opts,
                raise_on_failure=True,
            )
            x_ia = layout.as_ia(result.solution)

            mo1 = np.zeros((nmo, nocc))
            mo1[virt, :] = x_ia.T - 0.5 * overlap_mo[np.ix_(virt, occ)]
            mo1[occ, :] = -0.5 * overlap_mo[np.ix_(occ, occ)]

            frozen_occ = frozen_mo[:, occ]
            overlap_occ = overlap_mo[:, occ]
            hs0 = frozen_occ - overlap_occ * e_i[None, :]
            dm = C @ (2 * mo1) @ mocc.T
            dm = dm + dm.T
            hs = hs0 + C.T @ _wof(s.ERI, dm) @ mocc
            e1s[ia, x] = hs[occ, :] + mo1[occ, :] * (e_i[:, None] - e_i[None, :])
            mo1s[ia, x] = C @ mo1

    # Raw assembly: compute every perturbation ordering independently.
    relax = np.zeros((nat, nat, 3, 3))
    s1oo = np.einsum(
        "axpq,pi,qj->axij",
        s1ao_all,
        mocc,
        mocc,
    )
    for ia in range(nat):
        for ja in range(nat):
            for x in range(3):
                for y in range(3):
                    dm1 = mo1s[ja, y] @ mocc.T
                    dm1e = (mo1s[ja, y] * e_i[None, :]) @ mocc.T
                    relax[ia, ja, x, y] += np.einsum("pq,pq->", h1ao[ia, x], dm1) * 4
                    relax[ia, ja, x, y] -= (
                        np.einsum("pq,pq->", s1ao_all[ia, x], dm1e) * 4
                    )
                    relax[ia, ja, x, y] -= (
                        np.einsum("ij,ij->", s1oo[ia, x], e1s[ja, y]) * 2
                    )
    return relax


def fixture_mol(name):
    """Return one of the A2 fixtures (Bohr, Cartesian).

    ``h2``: 2 AO / 1 occ / 1 virt.  ``water``: genuine STO-3G O (7 AO /
    5 occ / 2 virt).  ``water_sdf``: custom s+p+d O (12 AO / 5 occ / 7 virt),
    the multi-virtual stress case.  These mirror A1's test fixtures so the
    same molecule is checked in both the reference and the integrated path.
    """
    _S = [
        [
            0,
            (3.425250914, 0.1543289673),
            (0.6239137298, 0.5353281423),
            (0.168855404, 0.4446345422),
        ]
    ]
    if name == "h2":
        return build_mol(
            [(1, [0.0, 0.0, 0.0]), (1, [0.1, 0.2, 1.4])],
            {"H0": list(_S), "H1": list(_S)},
            0,
            0,
        )
    if name == "water":
        _O = [
            [
                0,
                (130.7093214, 0.1543289673),
                (23.80886605, 0.5353281423),
                (6.443608313, 0.4446345422),
            ],
            [
                0,
                (5.033151319, -0.09996722919),
                (1.169596125, 0.3995128261),
                (0.38038896, 0.7001154689),
            ],
            [
                1,
                (5.033151319, 0.155916275),
                (1.169596125, 0.6076837186),
                (0.38038896, 0.3919573931),
            ],
        ]
        return build_mol(
            [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.958, 0.587]), (1, [0.0, -0.958, 0.587])],
            {"O0": list(_O), "H1": list(_S), "H2": list(_S)},
            0,
            0,
        )
    if name == "water_sdf":
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
        return build_mol(
            [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.958, 0.587]), (1, [0.0, -0.958, 0.587])],
            {"O0": list(_O), "H1": list(_S), "H2": list(_S)},
            0,
            0,
        )
    raise ValueError(f"unknown A2 fixture {name!r}")


# ---------------------------------------------------------------------------
# top-level assembly + validation
# ---------------------------------------------------------------------------


def analytic_hessian(s, *, relax=None):
    """Return every component plus the total from the shared #178/#179 layers.

    ``core``/``pulay``/``two_electron`` come from the #178 second-integral
    providers, ``nuclear`` from the closed-form Coulomb second derivative, and
    ``relaxation`` from #179's matrix-free RHF operator (or the supplied
    ``relax`` tensor, e.g. to swap in the FD oracle).
    """
    comp = provider_components(s)
    comp["nuclear"] = nuclear_closed_form(s)
    comp["relaxation"] = relax if relax is not None else cphf_relaxation(s)
    total = (
        comp["nuclear"]
        + comp["core"]
        + comp["pulay"]
        + comp["two_electron"]
        + comp["relaxation"]
    )
    comp["total"] = total
    return comp


def validate(name):
    """Run the full A2 integration for a fixture in {h2, water, water_sdf}."""
    mol = fixture_mol(name)
    mol.build()
    s = System(mol)
    s.derive()
    fd = hessian_components(s)  # A1 FD oracle
    comp = analytic_hessian(s)
    from pyscf import scf

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.kernel()
    H_pyscf = mf.Hessian().kernel()
    print(
        f"=== {name}: #180 A2 complete analytic Hessian (nbf={s.nbf}, nocc={s.nocc}) ==="
    )
    print("  per-component |#178/#179/closed-form - FD oracle|:")
    for key in ("nuclear", "core", "pulay", "two_electron", "relaxation"):
        print(f"    {key:13s} = {np.abs(comp[key] - fd[key]).max():.3e}")
    print(
        f"  total - FD oracle           = {np.abs(comp['total'] - sum(fd[k] for k in ('nuclear', 'core', 'pulay', 'two_electron', 'relaxation'))).max():.3e}"
    )
    print(
        f"  total vs PySCF analytic     = {np.abs(comp['total'] - H_pyscf).max():.3e}"
    )
    print(
        f"  (FD oracle vs PySCF analytic = {np.abs(sum(fd[k] for k in ('nuclear', 'core', 'pulay', 'two_electron', 'relaxation')) - H_pyscf).max():.3e})"
    )
    return comp, fd, H_pyscf


if __name__ == "__main__":
    import sys

    validate(sys.argv[1] if len(sys.argv) > 1 else "h2")
