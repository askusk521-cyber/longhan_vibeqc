import os

import pytest
from vibeqc import Calculator


pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_DFT_CUDA_TEST") != "1",
    reason="opt-in native CUDA DFT gate",
)


@pytest.mark.parametrize(
    ("method", "charge", "multiplicity"),
    (
        ("lda-rks", 0, 1),
        ("pbe-rks", 0, 1),
        ("lda-uks", -1, 2),
        ("pbe-uks", -1, 2),
    ),
)
def test_native_cuda_dft_matches_independently_converged_cpu_endpoint(
    method, charge, multiplicity
):
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    options = {
        "method": method,
        "basis": "sto-3g",
        "max_iterations": 200,
        "energy_tolerance": 1.0e-12,
        "density_tolerance": 1.0e-10,
    }
    state = {
        "charge": charge,
        "multiplicity": multiplicity,
        "properties": ("energy",),
    }
    cpu = Calculator(device="cpu", **options).singlepoint(atoms, **state)
    cuda_calculator = Calculator(device="cuda", **options)
    cuda = cuda_calculator.singlepoint(atoms, **state)
    replay = cuda_calculator.singlepoint(atoms, **state)

    assert cpu.converged and cuda.converged and replay.converged
    assert cuda.executed_backend == "cuda"
    assert cuda.energy == pytest.approx(cpu.energy, abs=2.0e-9)
    assert replay.energy == pytest.approx(cuda.energy, abs=2.0e-12)
    assert cuda.forces is None


def test_cuda_dft_force_request_remains_outside_issue_162():
    calculator = Calculator(method="lda-rks", basis="sto-3g", device="cuda")
    with pytest.raises(ValueError, match="does not support properties.*forces"):
        calculator.singlepoint([("He", (0.0, 0.0, 0.0))])
