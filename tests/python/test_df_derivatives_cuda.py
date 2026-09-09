"""Fail-closed Slurm tests for external DF response weights and bounded tiles."""

import copy
import os

import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell

from tools.vibeqc_validation.df_gradient import (
    execute_df_gradient,
    reference_df_matrices,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_DF_DERIVATIVE_CUDA_TEST") != "1",
    reason="explicit Slurm DF derivative tier",
)


def fixture_inputs(representation, shared):
    """A third atom owns only auxiliary functions unless shared centers are requested."""
    common = {
        "atomic_numbers": [2, 1, 1],
        "coordinates": [[0.1, -0.2, -0.7], [0.3, 0.1, 0.8], [-0.5, 0.4, 0.2]],
        "charge": 0,
        "multiplicity": 1,
        "basis_representation": representation,
    }

    def shell(atom, angular):
        return {
            "atom_index": atom,
            "angular_momentum": angular,
            "primitives": [[0.6 + 0.2 * angular, 0.8], [1.7 + 0.1 * angular, -0.1]],
        }

    orbital = {**common, "shells": [shell(0, 0), shell(0, 1), shell(1, 0), shell(1, 2)]}
    auxiliary = {
        **copy.deepcopy(common),
        "shells": [
            shell(0, 0),
            shell(0 if shared else 2, 0),
            shell(0 if shared else 2, 2),
            shell(1 if shared else 2, 3),
        ],
    }
    return orbital, auxiliary


def calculator(inputs):
    return Calculator(
        device="cuda",
        basis_representation=inputs["basis_representation"],
        basis=[
            Shell(
                s["atom_index"],
                s["angular_momentum"],
                tuple(Primitive(*p) for p in s["primitives"]),
            )
            for s in inputs["shells"]
        ],
    )


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("shared", [False, True])
def test_arbitrary_raw_fused_and_two_budgets_with_auxiliary_motion(
    representation, shared
):
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require Slurm"
    oi, xi = fixture_inputs(representation, shared)
    o, x = calculator(oi), calculator(xi)
    atoms = [(z, tuple(r)) for z, r in zip(("He", "H", "H"), oi["coordinates"])]
    a, m, da, dm = reference_df_matrices(oi, xi)
    rng = np.random.default_rng(143)
    wa, wm = rng.normal(size=a.shape), rng.normal(size=m.shape)
    expected = np.einsum("axijp,ijp->ax", da, wa) + np.einsum("axpq,pq->ax", dm, wm)
    if not shared:
        assert np.max(np.abs(expected[-1])) > 1e-7
    records = []
    for budget in (12288, 65536):
        for schedule in (0, 1):
            actual, resources = execute_df_gradient(
                o, x, atoms, wa, wm, schedule=schedule, maximum_bytes=budget
            )
            np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-11)
            np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-10)
            assert (
                resources["host_bytes"] <= budget
                and resources["device_bytes"] <= budget
            )
            assert resources["device_to_host_bytes"] == 3 * len(atoms) * 8
            assert resources["stream_synchronizations"] == 1
            records.append(resources)
    assert records[0]["weight_tile_elements"] < wa.size
    assert records[-1]["weight_tile_elements"] == wa.size
    first, _ = execute_df_gradient(
        o, x, atoms, wa, wm, schedule=1, maximum_tile_elements=13
    )
    second, _ = execute_df_gradient(
        o, x, atoms, wa, wm, schedule=1, maximum_tile_elements=13
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, expected, atol=2e-9, rtol=2e-11)
    with pytest.raises(RuntimeError, match="budget|maximum_bytes"):
        execute_df_gradient(o, x, atoms, wa, wm, maximum_bytes=32)
    for step in (2e-4, 5e-5):
        for atom in range(3):
            energies = []
            for sign in (1, -1):
                op, xp = copy.deepcopy(oi), copy.deepcopy(xi)
                op["coordinates"][atom][2] += sign * step
                xp["coordinates"][atom][2] += sign * step
                av, mv, _, _ = reference_df_matrices(op, xp)
                energies.append(np.sum(av * wa) + np.sum(mv * wm))
            assert expected[atom, 2] == pytest.approx(
                (energies[0] - energies[1]) / (2 * step), abs=5e-6, rel=5e-6
            )


@pytest.mark.parametrize(
    "orbital_rep,auxiliary_rep",
    [("cartesian", "spherical"), ("spherical", "cartesian")],
)
def test_mixed_representations_and_permuted_auxiliary_shells(
    orbital_rep, auxiliary_rep
):
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require Slurm"
    oi, xi = fixture_inputs(orbital_rep, False)
    xi["basis_representation"] = auxiliary_rep
    xi["shells"] = [xi["shells"][i] for i in (3, 0, 2, 1)]
    o, x = calculator(oi), calculator(xi)
    atoms = [(z, tuple(r)) for z, r in zip(("He", "H", "H"), oi["coordinates"])]
    a, m, da, dm = reference_df_matrices(oi, xi)
    rng = np.random.default_rng(1431)
    wa, wm = rng.normal(size=a.shape), rng.normal(size=m.shape)
    expected = np.einsum("axijp,ijp->ax", da, wa) + np.einsum("axpq,pq->ax", dm, wm)
    actual, _ = execute_df_gradient(
        o, x, atoms, wa, wm, maximum_bytes=16384, maximum_tile_elements=19
    )
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-11)
