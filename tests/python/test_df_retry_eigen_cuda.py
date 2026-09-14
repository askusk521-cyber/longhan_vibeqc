"""Host DIIS retries retain a qualified device eigen provider and failure limits."""

import os

import pytest
from vibeqc import Calculator

from benchmarks.df_component_ledger import aggregate_host, read_host_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("route", ("single", "batch-one", "batch-four"))
def test_diis_retry_provider_and_iteration_limit(
    method, representation, route, monkeypatch, tmp_path
):
    """One iteration forces the existing compact-to-DIIS transition.

    Compare the explicit reference diagnostic with ordinary device execution
    through both public entry points. Neither provider may turn the exhausted
    iteration limit into success; actual retry leaves must identify the chosen
    provider. This intentionally failed solve is not molecular convergence
    evidence, and the reference operation remains an independent oracle.
    """
    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("O", (0, 0, 0)), ("H", (0, 0, 1.8)), ("H", (1.7, 0, -0.6))]
    spin = int(method == "uhf")
    count = 4 if route == "batch-four" else 1
    calc = Calculator(
        method=method,
        basis="def2-svp",
        basis_representation=representation,
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=32 << 20,
        max_iterations=1,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    for reference in (True, False):
        path = tmp_path / f"retry-{reference}.jsonl"
        monkeypatch.setenv("VIBEQC_DF_REFERENCE_ITERATION_EIGEN", str(int(reference)))
        monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(path))
        try:
            if route == "single":
                with pytest.raises(RuntimeError, match="converg"):
                    calc.singlepoint(
                        atoms,
                        charge=spin,
                        multiplicity=1 + spin,
                        properties=("energy",),
                    )
            else:
                with calc.prepare_batch(
                    [atoms] * count,
                    charges=[spin] * count,
                    multiplicities=[1 + spin] * count,
                ) as batch:
                    result = batch.execute(strict=False, properties=("energy",))
                    assert result.failure_indices == tuple(range(count))
                    assert all(item.iterations == 1 for item in result.items)
        finally:
            monkeypatch.delenv("VIBEQC_DF_HOST_TRACE")
            monkeypatch.delenv("VIBEQC_DF_REFERENCE_ITERATION_EIGEN")
        components = aggregate_host(read_host_trace(path))
        expected = count * (1 + spin)
        for key, active in (
            ("eigensolves_by_reason", reference),
            ("device_eigensolves_by_reason", not reference),
        ):
            assert components[key].get("fallback", {}).get("calls", 0) == (
                expected if active else 0
            )
        if not reference:
            assert not components["eigensolves_by_reason"]
