from dataclasses import replace

import numpy as np
import pytest
from vibeqc._dft_gradient import (
    StationaryDerivativeContract,
    StationaryKsIdentity,
    StationaryKsState,
)


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


def state(method="pbe-rks"):
    spins = 1 if method.endswith("rks") else 2
    overlap = np.diag([1.0, 2.0])
    coefficients = np.diag([1.0, 1 / np.sqrt(2.0)])
    energies = np.array([[-0.8, 0.5]] * spins)
    occupations = (
        np.array([[2.0, 0.0]]) if spins == 1 else np.array([[1.0, 0.0], [1.0, 0.0]])
    )
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
    value = state(method)
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
