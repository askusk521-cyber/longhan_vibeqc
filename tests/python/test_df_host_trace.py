"""Actual solver counts and fault-rejecting #206 host ledger gates."""

import copy
import json
import os

import pytest

from benchmarks.df_component_ledger import (
    aggregate_host,
    read_host_trace,
    validate_host_record,
)


def host_record():
    base = {"reason": "overlap", "item": 0, "nbf": 2, "finished": True, "failed": False}
    return {
        "schema": "vibeqc.df_host_trace",
        "version": 1,
        "id": 0,
        "valid": True,
        "regions": [
            dict(base, name="endpoint", parent=-1, wall_ms=5, cpu_ms=3),
            dict(base, name="reference_eigensolve", parent=0, wall_ms=2, cpu_ms=1),
        ],
    }


def test_host_ledger_counts_actual_leaves_and_keeps_clocks_separate(tmp_path):
    record = host_record()
    path = tmp_path / "host.jsonl"
    path.write_text(json.dumps(record) + "\n")
    summary = aggregate_host(read_host_trace(path))
    assert summary["exclusive_phases"]["endpoint"] == {
        "calls": 1,
        "wall_ms": 3,
        "cpu_ms": 2,
    }
    assert summary["eigensolves_by_reason"]["overlap"] == {
        "calls": 1,
        "failed_calls": 0,
        "wall_ms": 2,
        "cpu_ms": 1,
    }
    for row in record["regions"]:
        row["cpu_ms"] = None
    assert aggregate_host([record])["exclusive_phases"]["endpoint"]["cpu_ms"] is None
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate"):
        read_host_trace(path)


@pytest.mark.parametrize("suffix", ("", "\n{"))
def test_host_trace_rejects_missing_record_terminator(tmp_path, suffix):
    """Even valid JSON must carry the writer's final record terminator."""
    path = tmp_path / "interrupted.host.jsonl"
    path.write_text(json.dumps(host_record()) + suffix)
    with pytest.raises(ValueError, match="incomplete host trace"):
        read_host_trace(path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("finished", False),
        ("wall_ms", 8),
        ("cpu_ms", float("nan")),
        ("parent", 1),
        ("nbf", 0),
        ("reason", "invented"),
    ],
)
def test_partial_or_mistimed_host_records_cannot_pass(key, value):
    record = copy.deepcopy(host_record())
    record["regions"][1][key] = value
    with pytest.raises(ValueError):
        validate_host_record(record)


@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_native_solver_calls_include_warm_preparation_and_finalization(
    tmp_path, monkeypatch, method, device
):
    """Actual leaves detect reintroduced warm guesses, retaining other solves.

    Overlap/finalization calls remain until their own measured ablations;
    removing the trace hook itself must never satisfy the zero-core-call gate.
    """
    if device == "cuda" and os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    from vibeqc import Calculator

    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    charge, spin = (0, 1) if method == "rhf" else (1, 2)
    calculator = Calculator(
        method=method,
        device=device,
        density_fitting=device,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calculator.prepare_batch(
        [atoms, atoms], charges=[charge] * 2, multiplicities=[spin] * 2
    ) as batch:
        cold = batch.execute(strict=True, properties=("energy",))
        batch.set_warm_start_updates(False)
        path = tmp_path / "warm.host.jsonl"
        monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(path))
        warm = batch.execute(strict=True, properties=("energy",))
        monkeypatch.delenv("VIBEQC_DF_HOST_TRACE")
        assert [r.energy for r in warm.items] == pytest.approx(
            [r.energy for r in cold.items], abs=1e-10
        )
        records = read_host_trace(path)
        assert any(r["regions"][0]["name"] == "batch_execute" for r in records)
        summary = aggregate_host(records)
        by_reason = summary["eigensolves_by_reason"]
        assert by_reason["overlap"]["calls"] == 2
        assert by_reason.get("core_guess", {}).get("calls", 0) == 0
        assert (
            sum(
                row["name"] == "initial_density"
                for record in records
                for row in record["regions"]
            )
            == 2
        )
        if device == "cuda":
            assert by_reason["final_fock"]["calls"] == (2 if method == "rhf" else 4)
            assert not by_reason.get("fallback", {}).get("calls", 0)
            leaves = summary["reference_eigensolves"]
            assert {row["item"] for row in leaves if row["reason"] == "overlap"} == {
                0,
                1,
            }
        before = path.read_bytes()
        batch.execute(strict=True, properties=("energy",))
        assert path.read_bytes() == before
        if device == "cuda":
            eager_path = tmp_path / "eager.host.jsonl"
            monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(eager_path))
            monkeypatch.setenv("VIBEQC_DF_EAGER_CORE_GUESS", "1")
            eager = batch.execute(strict=True, properties=("energy",))
            eager_components = aggregate_host(read_host_trace(eager_path))
            assert eager_components["eigensolves_by_reason"]["core_guess"]["calls"] == 2
            assert eager.energies == pytest.approx(warm.energies, abs=1e-10)
