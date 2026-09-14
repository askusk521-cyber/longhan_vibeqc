"""Hardware-free checks of failure retention and the Slurm launch contract."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import issue206_df_matrix as matrix


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("method", ("rhf", "uhf"))
def test_final_provider_ablation_rejects_missing_or_unexpected_solves(
    reference, method
):
    """A disabled trace hook or silent oracle fallback cannot pass promotion."""
    import copy

    from benchmarks.df_host_workloads import validate_final_eigen_counts

    count = 4 * (2 if method == "uhf" else 1)
    record = {
        "eigensolves_by_reason": {"final_fock": {"calls": count if reference else 0}},
        "device_eigensolves_by_reason": {
            "final_fock": {"calls": 0 if reference else count}
        },
    }
    validate_final_eigen_counts(
        record, batch_size=4, method=method, reference=reference
    )
    for group, name in (
        ("eigensolves_by_reason", "final_fock"),
        ("eigensolves_by_reason", "fallback"),
        ("device_eigensolves_by_reason", "final_fock"),
    ):
        changed = copy.deepcopy(record)
        changed[group].setdefault(name, {"calls": 0})["calls"] += 1
        with pytest.raises(RuntimeError, match="declared provider"):
            validate_final_eigen_counts(
                changed, batch_size=4, method=method, reference=reference
            )
    missing = copy.deepcopy(record)
    group, name = (
        ("eigensolves_by_reason", "final_fock")
        if reference
        else ("device_eigensolves_by_reason", "final_fock")
    )
    missing[group][name]["calls"] = 0
    with pytest.raises(RuntimeError, match="declared provider"):
        validate_final_eigen_counts(
            missing, batch_size=4, method=method, reference=reference
        )


@pytest.mark.parametrize("failure", ["exit", "launch", "missing_result", "gate"])
@pytest.mark.parametrize("energy_only", [False, True])
def test_matrix_retains_failures_and_finishes_remaining_cases(
    tmp_path, monkeypatch, failure, energy_only
):
    # subprocess.run is replaced throughout: these tests never execute CUDA.
    monkeypatch.setenv("SLURM_JOB_ID", "protocol-test")
    monkeypatch.setattr(matrix, "_git", lambda *args: "")
    output = tmp_path / "endpoints"
    output.mkdir()
    (output / "96ao-b1.json").write_text('{"previous_attempt": true}')
    manifest = tmp_path / "manifest.json"
    payload = matrix.manifest_payload(
        cases=matrix.MATRIX[:2],
        repeats=1,
        python=sys.executable,
        library=tmp_path / "lib.so",
        output_dir=output,
        memory_budget_bytes=32 << 20,
        energy_only=energy_only,
    )
    calls = []

    def endpoint(command, **kwargs):
        calls.append(command)
        active = json.loads(manifest.read_text())["matrix"][len(calls) - 1]
        assert active["status"] == "running" and active["command"] == command
        assert command[
            command.index("--density-fitting-memory-budget-bytes") + 1
        ] == str(32 << 20)
        assert kwargs["env"].get("CUDA_VISIBLE_DEVICES") == os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        )
        assert ("--energy-only" in command) == energy_only
        assert ("--maximum-force-error" in command) != energy_only
        assert command[command.index("--maximum-energy-error") + 1] == "1e-9"
        path = Path(command[command.index("--output") + 1])
        if len(calls) == 1:
            if failure == "launch":
                raise FileNotFoundError("missing interpreter")
            if failure in ("exit", "gate"):
                if failure == "gate":
                    path.write_text('{"gate": {"passed": false}}')
                return subprocess.CompletedProcess(command, 2, "", "endpoint failed")
            return subprocess.CompletedProcess(command, 0, "", "")
        path.write_text('{"converged": true}')
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(matrix.subprocess, "run", endpoint)
    # Exercise only the visibility preflight; no device state is changed and
    # the child launcher above is a pure Python test double.
    monkeypatch.setattr(
        matrix.os, "environ", {**os.environ, "CUDA_VISIBLE_DEVICES": "protocol-test"}
    )
    with pytest.raises(SystemExit, match="DF matrix failed"):
        matrix.run_matrix(
            payload,
            manifest_path=manifest,
            python=sys.executable,
            library=tmp_path / "lib.so",
            output_dir=output,
        )
    rows = json.loads(manifest.read_text())["matrix"]
    assert len(calls) == 2
    assert [row["status"] for row in rows] == ["failed", "passed"]
    if failure == "gate":
        assert (
            json.loads(Path(rows[0]["result"]).read_text())["gate"]["passed"] is False
        )
    else:
        assert rows[0]["result"] is None
    assert Path(rows[0]["log"]).is_file()
    assert Path(rows[1]["result"]).is_file()


def test_sbatch_spool_copy_uses_submission_checkout(tmp_path):
    root = Path(matrix.ROOT)
    spool = tmp_path / "slurm_script"
    spool.write_text((root / "run_issue206_df.slurm").read_text())
    # A harmless interpreter stub reports argv; even --run never reaches Python.
    interpreter = tmp_path / "python-stub"
    interpreter.write_text('#!/bin/bash\nprintf "%s\\n" "$PWD" "$@"\n')
    interpreter.chmod(0o755)
    environment = {
        **os.environ,
        "SLURM_SUBMIT_DIR": str(root),
        "ISSUE206_PYTHON": str(interpreter),
        "ISSUE206_OUTPUT_DIR": str(tmp_path / "results"),
    }
    environment.pop("ISSUE206_ROOT", None)
    completed = subprocess.run(
        ["bash", str(spool)],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.splitlines()[:2] == [
        str(root),
        "benchmarks/issue206_df_matrix.py",
    ]


def test_published_archive_uses_verified_repository_format():
    from tools.unpack_evidence import unpack

    assert unpack(matrix.ROOT / "benchmarks/results/issue206-df-a") == 9


@pytest.mark.parametrize(
    "field,value",
    [("iterations", 3), ("warm_start_used", False), ("warm_start_fallback", True)],
)
def test_eager_lazy_timing_rejects_iteration_or_retry_changes(field, value):
    """Equal endpoints cannot hide a different amount of SCF work."""
    from benchmarks.df_host_workloads import validate_ablation_branches

    converged = {"iterations": 2, "warm_start_used": True, "warm_start_fallback": False}
    rows = [
        {
            "workload": "unchanged-geometry",
            "selection": side,
            "diagnostics": {"convergence": [dict(converged)]},
        }
        for side in ("baseline", "candidate")
    ]
    validate_ablation_branches(rows)
    rows[1]["diagnostics"]["convergence"][0][field] = value
    with pytest.raises(ValueError, match="matching SCF"):
        validate_ablation_branches(rows)


@pytest.mark.parametrize("ablation", ("lazy-core", "overlap-cache", "combined"))
@pytest.mark.parametrize(
    "workload", ("cold-start", "unchanged-geometry", "changed-geometry")
)
@pytest.mark.parametrize("selection", ("baseline", "candidate"))
def test_preparation_ablation_rejects_wrong_actual_counts(
    ablation, workload, selection
):
    """Reinstated solves or stale changed-item cache hits must fail promotion."""
    import copy

    from benchmarks.df_host_workloads import (
        preparation_policies,
        validate_preparation_counts,
    )

    eager, rebuild = preparation_policies(ablation)[selection]
    overlap = (
        4
        if rebuild or workload == "cold-start"
        else int(workload == "changed-geometry")
    )
    core = 4 if eager or workload == "cold-start" else 0
    misses = 0 if rebuild else overlap
    hits = 0 if rebuild else 4 - misses
    components = {
        "eigensolves_by_reason": {
            "core_guess": {"calls": core},
            "overlap": {"calls": overlap},
        },
        "exclusive_phases": {
            "initial_density": {"calls": 4},
            "overlap_cache_miss": {"calls": misses},
            "overlap_cache_hit": {"calls": hits},
        },
    }

    def validate(value):
        validate_preparation_counts(
            value, batch_size=4, workload=workload, eager=eager, rebuild=rebuild
        )

    validate(components)
    for group, name in (
        ("eigensolves_by_reason", "core_guess"),
        ("eigensolves_by_reason", "overlap"),
        ("exclusive_phases", "initial_density"),
        ("exclusive_phases", "overlap_cache_hit"),
        ("exclusive_phases", "overlap_cache_miss"),
    ):
        broken = copy.deepcopy(components)
        broken[group][name]["calls"] += 1
        with pytest.raises(RuntimeError, match="declared solve/cache policy"):
            validate(broken)


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize(
    "workload,overlap,core",
    (("cold-start", 4, 4), ("energy-only", 0, 0), ("changed-geometry", 1, 0)),
)
def test_setup_provider_counts_reject_wrong_provider(
    reference, workload, overlap, core
):
    """Missing or unexpectedly substituted leaves cannot pass setup promotion."""
    import copy

    from benchmarks.df_host_workloads import validate_setup_eigen_counts

    record = {
        key: {
            "overlap": {"calls": overlap if selected else 0},
            "core_guess": {"calls": core if selected else 0},
        }
        for key, selected in (
            ("eigensolves_by_reason", reference),
            ("device_eigensolves_by_reason", not reference),
        )
    }
    validate_setup_eigen_counts(
        record, batch_size=4, workload=workload, reference=reference
    )
    for key in record:
        changed = copy.deepcopy(record)
        changed[key]["overlap"]["calls"] += 1
        with pytest.raises(RuntimeError, match="declared provider"):
            validate_setup_eigen_counts(
                changed, batch_size=4, workload=workload, reference=reference
            )
