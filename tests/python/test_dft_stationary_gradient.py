from dataclasses import replace

import numpy as np
import pytest
from vibeqc._dft_gradient import (
    StableGridMotion,
    StationaryDerivativeContract,
    StationaryKsIdentity,
    StationaryKsState,
    bind_generated_xc_geometry,
)
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture


def identity(method="pbe-rks"):
    return StationaryKsIdentity(
        method=method,
        model_identity="model-pbe-rks-v1",
        geometry_identity="geometry-1",
        basis_identity="basis-1",
        overlap_identity="overlap-1",
        grid_identity="grid-1",
        topology_identity="topology-1",
        functional_identity="functional-pbe-unpolarized-v1",
        regularization_identity="regularization-1",
        provider_identity="direct-j-cpu-v1",
        owner=17,
        solve_epoch=3,
        density_generation=7,
        fock_generation=9,
        orbital_generation=11,
    )


def state(method="pbe-rks", occupations=None):
    spins = 1 if method.endswith("rks") else 2
    overlap = np.diag([1.0, 2.0])
    coefficients = np.diag([1.0, 1 / np.sqrt(2.0)])
    energies = np.array([[-0.8, 0.5]] * spins)
    if occupations is None:
        occupations = (
            np.array([[2.0, 0.0]]) if spins == 1 else np.array([[1.0, 0.0], [1.0, 0.0]])
        )
    occupations = np.asarray(occupations, dtype=float)
    density = np.stack(
        [(coefficients * occupations[spin]) @ coefficients.T for spin in range(spins)]
    )
    weighted = np.stack(
        [
            (coefficients * (occupations[spin] * energies[spin])) @ coefficients.T
            for spin in range(spins)
        ]
    )
    fock = np.stack(
        [
            overlap @ coefficients @ np.diag(row) @ coefficients.T @ overlap
            for row in energies
        ]
    )
    return StationaryKsState(
        identity=identity(method),
        density=density,
        fock=fock,
        coefficients=np.stack([coefficients] * spins),
        orbital_energies=energies,
        occupations=occupations,
        weighted_density=weighted,
        overlap=overlap,
        physical_residual=0.0,
        successful=True,
        converged=True,
        physical=True,
    )


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
def test_stationary_contract_accepts_consistent_rks_and_uks(method):
    value = state(
        method,
        occupations=([[1.0, 0.0], [0.7, 0.2]] if method == "pbe-uks" else None),
    )
    contract = StationaryDerivativeContract(value.identity)

    assert contract.validate(value) is value
    assert contract.spin == ("unpolarized" if method.endswith("rks") else "polarized")
    assert contract.family == ("lda" if method.startswith("lda") else "gga")
    assert contract.to_payload()["force_capability"] == "unsupported"
    assert contract.identity == StationaryDerivativeContract(value.identity).identity


@pytest.mark.parametrize(
    "field",
    [
        "model_identity",
        "geometry_identity",
        "basis_identity",
        "overlap_identity",
        "grid_identity",
        "topology_identity",
        "functional_identity",
        "regularization_identity",
        "provider_identity",
        "owner",
        "solve_epoch",
        "density_generation",
        "fock_generation",
        "orbital_generation",
    ],
)
def test_stationary_contract_rejects_every_stale_identity_axis(field):
    value = state()
    old = getattr(value.identity, field)
    changed = old + 1 if isinstance(old, int) else old + "-stale"
    stale = replace(value, identity=replace(value.identity, **{field: changed}))

    with pytest.raises(ValueError, match="identity mismatch"):
        StationaryDerivativeContract(value.identity).validate(stale)


@pytest.mark.parametrize("flag", ["successful", "converged", "physical"])
def test_stationary_contract_rejects_unsuccessful_state(flag):
    value = state()
    with pytest.raises(ValueError, match="successful converged physical"):
        StationaryDerivativeContract(value.identity).validate(
            replace(value, **{flag: False})
        )


@pytest.mark.parametrize(
    "field,delta,message",
    [
        ("density", 1e-4, "density reconstruction"),
        ("weighted_density", 1e-4, "weighted-density reconstruction"),
        ("fock", 1e-4, "Fock eigen residual"),
        ("coefficients", 1e-4, "S-orthonormal"),
    ],
)
def test_stationary_contract_rejects_inconsistent_orbital_state(field, delta, message):
    value = state()
    changed = np.array(getattr(value, field), copy=True)
    changed[0, 0, 0] += delta
    with pytest.raises(ValueError, match=message):
        StationaryDerivativeContract(value.identity).validate(
            replace(value, **{field: changed})
        )


def test_stationary_contract_requires_weighted_density_and_true_residual():
    value = state()
    with pytest.raises(ValueError, match="weighted density"):
        StationaryDerivativeContract(value.identity).validate(
            replace(value, weighted_density=np.empty((0, 2, 2)))
        )
    with pytest.raises(ValueError, match="physical residual"):
        StationaryDerivativeContract(value.identity).validate(
            replace(value, physical_residual=1e-5)
        )


@pytest.mark.parametrize("method", ["b3lyp-rks", "pbe0-rks", "pbe-rhf"])
def test_stationary_contract_rejects_unsupported_method_domain(method):
    with pytest.raises(ValueError, match="LDA/PBE RKS/UKS"):
        replace(identity(), method=method)


@pytest.mark.parametrize(
    "method,name,spin",
    [
        ("lda-rks", "LDA_XC_PW", "unpolarized"),
        ("pbe-rks", "PBE", "unpolarized"),
        ("lda-uks", "LDA_XC_PW", "polarized"),
        ("pbe-uks", "PBE", "polarized"),
    ],
)
def test_generated_xc_geometry_is_bound_to_stationary_identity(method, name, spin):
    spec = functional(name, spin=spin)
    value = state(
        method,
        occupations=([[1.0, 0.0], [0.7, 0.2]] if method == "pbe-uks" else None),
    )
    value = replace(
        value,
        identity=replace(value.identity, functional_identity=spec.identity),
    )
    meta, _, grid = load_integration_fixture("h2")
    with NativeAO(**basis_arguments(meta)) as basis:
        ao_atoms = np.repeat(
            [shell.atom_index for shell in basis.shells],
            [
                (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
                for shell in basis.shells
            ],
        )
        bound = bind_generated_xc_geometry(
            StationaryDerivativeContract(value.identity),
            value,
            spec,
            basis.evaluate(grid.points, 2 if name == "PBE" else 1),
            grid.weights,
            ao_atoms=ao_atoms,
            natom=basis.natom,
            basis_identity=value.identity.basis_identity,
            geometry_identity=value.identity.geometry_identity,
            grid_identity=value.identity.grid_identity,
            topology_identity=value.identity.topology_identity,
        )

    assert bound.state_identity == value.identity
    assert bound.functional_identity == spec.identity
    assert bound.density_generation == value.identity.density_generation
    assert bound.partials.centers.shape == (basis.natom, 3)
    assert bound.partials.points.shape == grid.points.shape
    assert bound.partials.weights.shape == grid.weights.shape
    assert bound.force_capability == "unsupported"


def test_stable_motion_reports_each_xc_component_once_and_translation():
    spec = functional("PBE", spin="unpolarized")
    value = state("pbe-rks")
    value = replace(
        value,
        identity=replace(value.identity, functional_identity=spec.identity),
    )
    meta, _, grid = load_integration_fixture("h2")
    with NativeAO(**basis_arguments(meta)) as basis:
        ao_atoms = np.repeat(
            [shell.atom_index for shell in basis.shells],
            [
                (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
                for shell in basis.shells
            ],
        )
        bound = bind_generated_xc_geometry(
            StationaryDerivativeContract(value.identity),
            value,
            spec,
            basis.evaluate(grid.points, 2),
            grid.weights,
            ao_atoms=ao_atoms,
            natom=basis.natom,
            basis_identity=value.identity.basis_identity,
            geometry_identity=value.identity.geometry_identity,
            grid_identity=value.identity.grid_identity,
            topology_identity=value.identity.topology_identity,
        )
    rng = np.random.default_rng(163)
    centers = rng.normal(size=bound.partials.centers.shape)
    points = rng.normal(size=bound.partials.points.shape)
    weights = rng.normal(size=bound.partials.weights.shape)
    result = bound.directional(
        StableGridMotion(
            topology_identity=value.identity.topology_identity,
            centers=centers,
            points=points,
            weights=weights,
        )
    )
    np.testing.assert_allclose(result.center, np.sum(bound.partials.centers * centers))
    np.testing.assert_allclose(result.point, np.sum(bound.partials.points * points))
    np.testing.assert_allclose(result.weight, np.sum(bound.partials.weights * weights))
    np.testing.assert_allclose(
        result.total, result.center + result.point + result.weight
    )

    translation = np.array([0.2, -0.1, 0.3])
    rigid = bound.directional(
        StableGridMotion(
            topology_identity=value.identity.topology_identity,
            centers=np.broadcast_to(translation, bound.partials.centers.shape),
            points=np.broadcast_to(translation, bound.partials.points.shape),
            weights=np.zeros_like(bound.partials.weights),
        )
    )
    np.testing.assert_allclose(rigid.total, 0, atol=1e-12)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"topology_identity": "other"}, "topology identity"),
        ({"topology_changed": True}, "topology change"),
        ({"centers": np.zeros((1, 3))}, "center direction"),
        ({"points": np.array([[np.nan, 0.0, 0.0]])}, "point direction"),
    ],
)
def test_generated_xc_geometry_rejects_stale_invalid_or_changing_motion(
    change, message
):
    from vibeqc_compiler.xc.contractions import GeometryPartials

    value = state()
    spec = functional("PBE", spin="unpolarized")
    value = replace(
        value,
        identity=replace(value.identity, functional_identity=spec.identity),
    )
    partials = GeometryPartials(np.zeros((2, 3)), np.zeros((3, 3)), np.zeros(3))
    from vibeqc._dft_gradient import GeneratedXcGeometry

    bound = GeneratedXcGeometry(
        state_identity=value.identity,
        discrete_contract_identity="generated-contract",
        basis_identity=value.identity.basis_identity,
        geometry_identity=value.identity.geometry_identity,
        grid_identity=value.identity.grid_identity,
        topology_identity=value.identity.topology_identity,
        functional_identity=value.identity.functional_identity,
        density_generation=value.identity.density_generation,
        partials=partials,
    )
    values = {
        "topology_identity": value.identity.topology_identity,
        "topology_changed": False,
        "centers": np.zeros((2, 3)),
        "points": np.zeros((3, 3)),
        "weights": np.zeros(3),
    }
    values.update(change)
    with pytest.raises(ValueError, match=message):
        bound.directional(StableGridMotion(**values))
