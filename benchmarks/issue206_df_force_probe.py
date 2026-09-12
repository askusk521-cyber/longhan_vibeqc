"""Measure the DF force-response cost separately from the SCF endpoint.

This is a diagnostic companion to ``issue206_df_matrix.py``.  It intentionally
uses fresh single-system calculations for each observable so the reported
energy-only versus energy-plus-force difference cannot be mistaken for a
matched warm-solve speed claim.  Run it inside a finite Slurm allocation on the
target GPU; the resulting JSON is a bottleneck ledger input for issue #206.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

try:
    from _cases import benchmark_cases
except ModuleNotFoundError:  # imported as ``benchmarks.issue206_df_force_probe``
    from benchmarks._cases import benchmark_cases
from vibeqc import Calculator

CASES = (
    "water-tetramer-def2-svp-spherical",
    "water-octamer-s4-def2-svp-spherical",
)


def _git_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_git_root(), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sample(case_name: str, properties: tuple[str, ...]) -> dict:
    case = benchmark_cases()[case_name]
    calculator = Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        max_iterations=100,
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
        screening_tolerance=1.0e-12,
        density_fitting="cuda",
        auxiliary_basis=case.vibeqc_basis,
    )
    started = time.perf_counter()
    result = calculator.singlepoint(
        case.atoms,
        charge=case.charge,
        multiplicity=case.multiplicity,
        properties=properties,
    )
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "iterations": result.iterations,
        "converged": result.converged,
        "energy_hartree": result.energy,
        "has_forces": result.forces is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, action="append", dest="cases")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path(".artifacts/issue206-force-ledger.json")
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run requires a finite Slurm allocation (SLURM_JOB_ID)")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("run requires Slurm-provided CUDA_VISIBLE_DEVICES")

    cases = args.cases or list(CASES)
    records = []
    for case_name in cases:
        for repeat in range(args.repeats):
            energy = _sample(case_name, ("energy",))
            energy_force = _sample(case_name, ("energy", "forces"))
            records.append(
                {
                    "case": case_name,
                    "repeat": repeat,
                    "energy_only": energy,
                    "energy_plus_force": energy_force,
                    "force_increment_seconds": energy_force["seconds"]
                    - energy["seconds"],
                }
            )

    payload = {
        "schema": "vibeqc.issue206.df_force_ledger",
        "version": 1,
        "source": {"git_head": _git_revision(), "repository": str(_git_root())},
        "execution": {
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "python": sys.executable,
            "platform": platform.platform(),
            "repeats": args.repeats,
            "warning": (
                "fresh single-system calculations; the difference diagnoses "
                "force cost and is not a warm-solve speed claim"
            ),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
