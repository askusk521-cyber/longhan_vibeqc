"""RHF energy selection evaluates returned D and separates reuse from rebuilding."""

import os
from contextlib import ExitStack

import numpy as np
import pytest
from vibeqc import Calculator

from benchmarks.df_component_ledger import aggregate_host, read_host_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


def calls(rows, name):
    return rows.get(name, {}).get("calls", 0)


@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("size", (1, 4))
@pytest.mark.parametrize("budget", (0, 8 << 20))
def test_energy_selection_rebuild_and_force_transitions(
    representation, size, budget, monkeypatch, tmp_path
):
    """Three independent owners exercise actual provider and reuse choices.

    The CPU DF calculation supplies an independent molecular energy/force
    reference. Source order and frozen warm seeds survive output replanning.
    Necessary strict corrections are counted, never assumed away.
    """
    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("O", (0, 0, 0)), ("H", (0, 0, 1.8)), ("H", (1.7, 0, -0.6))]
    changed = np.array([atom[1] for atom in atoms])
    changed[-1, 0] += 0.02
    changed_atoms = [
        (atom[0], tuple(xyz)) for atom, xyz in zip(atoms, changed, strict=True)
    ]
    scientific = {
        "method": "rhf",
        "basis": "def2-svp",
        "basis_representation": representation,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    modes = ("reuse", "device_rebuild", "reference_rebuild")
    with ExitStack() as stack:
        owners = {
            mode: stack.enter_context(
                Calculator(
                    **scientific,
                    device="cuda",
                    density_fitting="cuda",
                    density_fitting_memory_budget_bytes=budget,
                ).prepare_batch([atoms] * size)
            )
            for mode in modes
        }
        cpu = Calculator(**scientific, device="cpu", density_fitting="cpu")
        for step, (positions, properties) in enumerate(
            (
                (None, ("energy",)),
                (None, ("energy",)),
                ([None] * (size - 1) + [changed], ("energy",)),
                ([None] * (size - 1) + [changed], ("energy", "forces")),
                ([None] * (size - 1) + [changed], ("energy",)),
            )
        ):
            outputs = []
            for mode, owner in owners.items():
                monkeypatch.setenv(
                    "VIBEQC_DF_FORCE_FINAL_REBUILD", "0" if mode == "reuse" else "1"
                )
                monkeypatch.setenv(
                    "VIBEQC_DF_REFERENCE_FINAL_EIGEN",
                    "1" if mode == "reference_rebuild" else "0",
                )
                path = tmp_path / f"{step}-{mode}.jsonl"
                monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(path))
                try:
                    output = owner.execute(
                        positions, properties=properties, strict=True
                    )
                finally:
                    monkeypatch.delenv("VIBEQC_DF_HOST_TRACE")
                    monkeypatch.delenv("VIBEQC_DF_FORCE_FINAL_REBUILD")
                    monkeypatch.delenv("VIBEQC_DF_REFERENCE_FINAL_EIGEN")
                outputs.append(output)
                ledger = aggregate_host(read_host_trace(path))
                if properties == ("energy",):
                    phases = ledger["exclusive_phases"]
                    reference = calls(ledger["eigensolves_by_reason"], "final_fock")
                    device = calls(ledger["device_eigensolves_by_reason"], "final_fock")
                    assert calls(phases, "final_state_read") == size
                    assert (
                        calls(phases, "final_state_fock_build")
                        == size + reference + device
                    )
                    assert (
                        calls(phases, "strict_final_correction") == reference + device
                    )
                    assert (
                        calls(phases, "final_state_validation")
                        == size + reference + device
                    )
                    assert (
                        calls(phases, "final_state_reuse")
                        + calls(phases, "final_state_corrected")
                        == size
                    )
                    assert calls(phases, "final_state_weighted_density") == 0
                    assert calls(phases, "force_response") == 0
                    assert calls(ledger["eigensolves_by_reason"], "fallback") == 0
                    if mode == "reference_rebuild":
                        assert size <= reference <= 16 * size and device == 0
                    else:
                        assert reference == 0 and device <= 16 * size
                        if mode == "device_rebuild":
                            assert (
                                device >= size
                                and calls(phases, "final_state_reuse") == 0
                            )
                if step == 0:
                    owner.set_warm_start_updates(False)
            for actual in outputs[1:]:
                np.testing.assert_allclose(
                    actual.energies, outputs[0].energies, atol=1e-9, rtol=0
                )
                assert [item.iterations for item in actual.items] == [
                    item.iterations for item in outputs[0].items
                ]
                if "forces" in properties:
                    for a, b in zip(actual.items, outputs[0].items, strict=True):
                        np.testing.assert_allclose(
                            a.forces, b.forces, atol=1e-8, rtol=0
                        )
            systems = (
                [atoms] * size if step < 2 else [atoms] * (size - 1) + [changed_atoms]
            )
            with cpu.prepare_batch(systems) as oracle:
                expected = oracle.execute(strict=True, properties=properties)
            np.testing.assert_allclose(
                outputs[0].energies, expected.energies, atol=1e-9, rtol=0
            )
            if "forces" in properties:
                for actual, reference in zip(
                    outputs[0].items, expected.items, strict=True
                ):
                    np.testing.assert_allclose(
                        actual.forces, reference.forces, atol=1e-8, rtol=0
                    )
        # A failed geometry cannot leak a previous candidate into its neighbor.
        broken = changed.copy()
        broken[-1, 0] = np.nan
        failed = owners["reuse"].execute(
            [None] * (size - 1) + [broken], strict=False, properties=("energy",)
        )
        assert failed.items[-1].status != 0
        assert all(item.status == 0 for item in failed.items[:-1])
        recovered = owners["reuse"].execute(
            [None] * (size - 1) + [changed], strict=True, properties=("energy",)
        )
        np.testing.assert_allclose(
            recovered.energies, outputs[0].energies, atol=1e-9, rtol=0
        )
