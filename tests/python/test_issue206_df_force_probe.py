"""Hardware-free protocol checks for the #206 force ledger."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import issue206_df_force_probe as probe


def test_probe_requires_slurm_and_cuda_visibility(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(sys, "argv", ["probe", "--output", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        probe.main()


def test_probe_writes_explicit_force_increment_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "protocol-test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    samples = iter(
        [
            {
                "seconds": 2.0,
                "iterations": 4,
                "converged": True,
                "energy_hartree": -1.0,
                "has_forces": False,
            },
            {
                "seconds": 5.5,
                "iterations": 4,
                "converged": True,
                "energy_hartree": -1.0,
                "has_forces": True,
            },
        ]
    )
    monkeypatch.setattr(probe, "_sample", lambda *args: next(samples))
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--case", probe.CASES[0], "--output", str(tmp_path / "x.json")],
    )
    probe.main()
    payload = json.loads((tmp_path / "x.json").read_text())
    record = payload["records"][0]
    assert payload["schema"] == "vibeqc.issue206.df_force_ledger"
    assert record["force_increment_seconds"] == 3.5
    assert record["energy_only"]["has_forces"] is False
    assert record["energy_plus_force"]["has_forces"] is True
