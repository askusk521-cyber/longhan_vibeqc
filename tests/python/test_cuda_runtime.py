import os

import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell


def _cuda_tolerances() -> tuple[float, float]:
    # CuMetal emulates FP64 with paired FP32 arithmetic on current Apple Silicon.
    # Keep these tests as functional runtime gates there; the existing NVIDIA
    # CUDA regression tests retain the strict FP64 numerical tolerances.
    if os.environ.get("CUMETAL_ROOT"):
        return 2.0e-6, 2.0e-5
    return 2.0e-10, 2.0e-9


def test_cuda_minimal_rhf_matches_cpu_reference():
    """Exercise one real RHF CUDA calculation without batch/replay overhead."""

    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    basis = (
        Shell(0, 0, (Primitive(1.0, 1.0),)),
        Shell(1, 0, (Primitive(1.0, 1.0),)),
    )
    options = dict(
        basis=basis,
        energy_tolerance=1.0e-10,
        density_tolerance=1.0e-8,
    )
    reference = Calculator(device="cpu", **options).singlepoint(atoms)
    try:
        result = Calculator(device="cuda", **options).singlepoint(atoms)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    energy_atol, force_atol = _cuda_tolerances()
    assert result.executed_backend == "cuda"
    assert result.energy == pytest.approx(reference.energy, abs=energy_atol)
    assert np.allclose(result.forces, reference.forces, atol=force_atol, rtol=0.0)


def test_cuda_minimal_uhf_matches_cpu_reference():
    """Exercise the unrestricted CUDA path with a one-electron one-shell case."""

    atoms = [("H", (0.0, 0.0, 0.0))]
    basis = (Shell(0, 0, (Primitive(1.0, 1.0),)),)
    options = dict(
        method="uhf",
        basis=basis,
        energy_tolerance=1.0e-10,
        density_tolerance=1.0e-8,
    )
    reference = Calculator(device="cpu", **options).singlepoint(atoms, multiplicity=2)
    try:
        result = Calculator(device="cuda", **options).singlepoint(atoms, multiplicity=2)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    energy_atol, force_atol = _cuda_tolerances()
    assert result.executed_backend == "cuda"
    assert result.energy == pytest.approx(reference.energy, abs=energy_atol)
    assert np.allclose(result.forces, reference.forces, atol=force_atol, rtol=0.0)
