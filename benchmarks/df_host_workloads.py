"""Host-eigensolve diagnostic workloads used by the existing #206 matrix CLI.

By default A/B configurations are identical protocol controls. The explicit
eager-core ablation restores discarded warm frames in the baseline selection.
Separate clean and profiled invocations retain setup, destruction and every
changed-geometry call. Same-model ablations retain their iteration branches;
external-engine parity remains with the existing matched #206 matrix.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import convergence_payload, scaled_geometries
from benchmarks.df_component_ledger import (
    aggregate_host,
    read_host_trace,
    trace_identity,
)
from benchmarks.issue206_df_force_probe import _source_metadata
from benchmarks.validation_gate import _cuda
from tools.vibeqc_validation.performance import assess_comparison, measure_interleaved
from tools.vibeqc_validation.schema import canonical_hash


def validate_ablation_branches(rows):
    """Reject timing promotion when either selection took different SCF work.

    A frozen seed must give one iteration/retry branch per workload and item.
    Equal energies alone do not justify an iteration-unmatched speed claim.
    """
    for workload in {row["workload"] for row in rows}:
        branches = {
            selection: {
                tuple(
                    (
                        item["iterations"],
                        item["warm_start_used"],
                        item["warm_start_fallback"],
                    )
                    for item in row["diagnostics"]["convergence"]
                )
                for row in rows
                if row["workload"] == workload and row["selection"] == selection
            }
            for selection in ("baseline", "candidate")
        }
        if (
            len(branches["baseline"]) != 1
            or branches["baseline"] != branches["candidate"]
        ):
            raise ValueError(
                "preparation timing requires matching SCF iteration/retry branches"
            )


def preparation_policies(ablation):
    """Return eager-core/rebuild-overlap switches for a causal A/B comparison.

    All three increments share the original eager/rebuilt baseline. Comparing
    the separate candidates yields baseline, lazy-only, cache-only and combined
    measurements without changing the binary or frozen SCF seed.
    """
    candidates = {
        "lazy-core": (False, True),
        "overlap-cache": (True, False),
        "combined": (False, False),
    }
    if ablation not in candidates:
        raise ValueError("unknown preparation ablation")
    return {"baseline": (True, True), "candidate": candidates[ablation]}


def validate_preparation_counts(components, *, batch_size, workload, eager, rebuild):
    """Require actual solves and matching cache scopes, including partial rebuilds.

    The changed-geometry workload moves only the last item and restores the
    original geometry before every sample. Other items must retain their X.
    """
    cold = workload == "cold-start"
    changed = workload == "changed-geometry"
    expected_core = batch_size if cold or eager else 0
    expected_overlap = batch_size if cold or rebuild else int(changed)
    expected_misses = 0 if rebuild else expected_overlap
    expected_hits = 0 if rebuild else batch_size - expected_misses
    solves = components["eigensolves_by_reason"]
    phases = components["exclusive_phases"]
    expected = (
        (solves, "core_guess", expected_core),
        (solves, "overlap", expected_overlap),
        (phases, "initial_density", batch_size),
        (phases, "overlap_cache_miss", expected_misses),
        (phases, "overlap_cache_hit", expected_hits),
    )
    if any(
        rows.get(name, {}).get("calls", 0) != count for rows, name, count in expected
    ):
        raise RuntimeError(
            "preparation ablation did not execute its declared solve/cache policy"
        )


def validate_final_eigen_counts(components, *, batch_size, method, reference):
    """Require actual finalizer leaves; a flag or omitted observer is insufficient.

    This provider-only slice retains lazy cached preparation in both selections.
    UHF has two serial spin frames, including an empty occupation channel.
    """
    expected = batch_size * (2 if method == "uhf" else 1)
    solves = components["eigensolves_by_reason"]
    phases = components["exclusive_phases"]
    if (
        solves.get("final_fock", {}).get("calls", 0) != (expected if reference else 0)
        or phases.get("device_eigensolve", {}).get("calls", 0)
        != (0 if reference else expected)
        or solves.get("fallback", {}).get("calls", 0)
    ):
        raise RuntimeError("final eigen ablation did not execute its declared provider")


def host_workloads(
    *,
    case_name,
    batch_size,
    library,
    repeats,
    memory_budget_bytes,
    energy_only,
    trace_directory=None,
    eager_core_ablation=False,
    preparation_ablation=None,
    final_eigen_ablation=False,
):
    """Measure one source-bound cold/replay/rebuild domain without hiding setup.

    Warm updates are frozen after cold convergence. Every changed sample starts
    from the original geometry, restored outside its measured region. A trace
    is collected inside that region and includes all failed attempts; its wall
    time is diagnostic and must never be compared to a clean sample as a gain.
    """
    from vibeqc import Calculator

    if repeats < 5 or batch_size < 1:
        raise ValueError(
            "at least five paired samples and a positive batch are required"
        )
    if any(os.environ.get(k) for k in ("VIBEQC_DF_TRACE", "VIBEQC_DF_HOST_TRACE")):
        raise ValueError(
            "provide trace_directory explicitly; ambient profiling is not clean timing"
        )
    if any(
        os.environ.get(k)
        for k in (
            "VIBEQC_DF_EAGER_CORE_GUESS",
            "VIBEQC_DF_REBUILD_OVERLAP",
            "VIBEQC_DF_REFERENCE_FINAL_EIGEN",
        )
    ):
        raise ValueError(
            "use an explicit ablation; ambient preparation policy is ambiguous"
        )
    if (
        sum(
            map(bool, (preparation_ablation, eager_core_ablation, final_eigen_ablation))
        )
        > 1
    ):
        raise ValueError("select one preparation or final eigen ablation")
    policies = (
        preparation_policies(preparation_ablation) if preparation_ablation else None
    )
    if eager_core_ablation:
        policies = {"baseline": (True, False), "candidate": (False, False)}
    library = Path(library).resolve(strict=True)
    source = _source_metadata(library)
    device, synchronize = _cuda()
    case = benchmark_cases()[case_name]
    systems = scaled_geometries(case.atoms, batch_size)
    properties = ("energy",) if energy_only else ("energy", "forces")
    inputs = {
        "case": case_name,
        "systems": systems,
        "method": case.method,
        "basis": case.vibeqc_basis,
        "representation": case.basis_representation,
        "charge": case.charge,
        "multiplicity": case.multiplicity,
        "auxiliary_basis": case.vibeqc_basis,
        "properties": properties,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-12,
        "metric_relative_threshold": 1e-10,
        "memory_budget_bytes": memory_budget_bytes,
        "eager_core_ablation": eager_core_ablation,
        "preparation_ablation": preparation_ablation,
        "preparation_policies": policies,
        "final_eigen_ablation": final_eigen_ablation,
        "final_eigen_policies": {
            "baseline": "cpu_reference",
            "candidate": "ordinary_xsyevd",
        }
        if final_eigen_ablation
        else None,
    }
    input_hash = canonical_hash(inputs)
    calculator = Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-12,
        density_fitting="cuda",
        auxiliary_basis=case.vibeqc_basis,
        density_fitting_relative_threshold=1e-10,
        density_fitting_memory_budget_bytes=memory_budget_bytes,
    )
    if Path(calculator._library._name).resolve() != library:
        raise RuntimeError(
            "loaded library differs from the recorded source-bound binary"
        )
    if trace_directory is not None:
        trace_directory = Path(trace_directory)
        trace_directory.mkdir(parents=True, exist_ok=False)
    sequence = 0

    def measured(workload, evaluate):
        def sample(_selection):
            nonlocal sequence
            # Both selections replay the same frozen density on one plan.
            # Diagnostic switches restore discarded work, preserving numerical
            # inputs. Separate actual-count gates reject ineffective controls.
            path = None
            if trace_directory is not None:
                path = trace_directory / f"{sequence:04d}-{workload}.jsonl"
                with path.open("x"):
                    pass
                os.environ["VIBEQC_DF_HOST_TRACE"] = str(path.resolve())
            sequence += 1
            try:
                if policies:
                    eager, rebuild = policies[_selection]
                    os.environ["VIBEQC_DF_EAGER_CORE_GUESS"] = "1" if eager else "0"
                    os.environ["VIBEQC_DF_REBUILD_OVERLAP"] = "1" if rebuild else "0"
                if final_eigen_ablation:
                    os.environ["VIBEQC_DF_REFERENCE_FINAL_EIGEN"] = (
                        "1" if _selection == "baseline" else "0"
                    )
                result = evaluate()
            finally:
                if final_eigen_ablation:
                    os.environ.pop("VIBEQC_DF_REFERENCE_FINAL_EIGEN", None)
                if policies:
                    os.environ.pop("VIBEQC_DF_EAGER_CORE_GUESS", None)
                    os.environ.pop("VIBEQC_DF_REBUILD_OVERLAP", None)
                if path is not None:
                    os.environ.pop("VIBEQC_DF_HOST_TRACE")
            if path is not None:
                components = aggregate_host(read_host_trace(path))
                if eager_core_ablation:
                    expected = (
                        batch_size
                        if _selection == "baseline" or workload == "cold-start"
                        else 0
                    )
                    actual = (
                        components["eigensolves_by_reason"]
                        .get("core_guess", {})
                        .get("calls", 0)
                    )
                    if (
                        actual != expected
                        or components["exclusive_phases"]
                        .get("initial_density", {})
                        .get("calls", 0)
                        != batch_size
                    ):
                        raise RuntimeError(
                            "eager/lazy ablation did not execute its declared core-guess policy"
                        )
                if preparation_ablation:
                    validate_preparation_counts(
                        components,
                        batch_size=batch_size,
                        workload=workload,
                        eager=eager,
                        rebuild=rebuild,
                    )
                if final_eigen_ablation:
                    validate_preparation_counts(
                        components,
                        batch_size=batch_size,
                        workload=workload,
                        eager=False,
                        rebuild=False,
                    )
                    validate_final_eigen_counts(
                        components,
                        batch_size=batch_size,
                        method=case.method,
                        reference=_selection == "baseline",
                    )
                result["host_components"] = {
                    **components,
                    "raw_trace": trace_identity(path),
                }
            return result

        return sample

    def prepare(geometries=None):
        return calculator.prepare_batch(
            systems if geometries is None else geometries,
            charges=[case.charge] * batch_size,
            multiplicities=[case.multiplicity] * batch_size,
            warm_start=True,
        )

    def execute(batch, coordinates=None):
        result = batch.execute(coordinates, strict=True, properties=properties)
        if any(item.executed_backend != "cuda" for item in result.items):
            raise RuntimeError("a CUDA workload cannot pass with a substituted backend")
        return {
            "energies": result.energies.tolist(),
            "convergence": convergence_payload(result),
            "fock_builds": [item.fock_builds for item in result.items],
            "forces": None
            if energy_only
            else [item.forces.tolist() for item in result.items],
            "metric": [
                d.to_dict() for d in batch.last_density_fitting_metric_diagnostics()
            ],
        }

    def cold():
        # Both ownership creation and destruction are within the timed call.
        with prepare() as batch:
            return execute(batch)

    rows = measure_interleaved(
        measured("cold-start", cold),
        synchronize,
        workload="cold-start",
        inputs_hash=input_hash,
        repeats=repeats,
    )
    synchronize()
    setup_start = time.perf_counter()
    batch = prepare()
    try:
        seed = execute(batch)
        batch.set_warm_start_updates(False)
        synchronize()
        setup_seconds = time.perf_counter() - setup_start
        for workload in (
            "unchanged-geometry",
            "energy-only" if energy_only else "energy-plus-force",
        ):
            rows += measure_interleaved(
                measured(workload, lambda: execute(batch)),
                synchronize,
                workload=workload,
                inputs_hash=input_hash,
                repeats=repeats,
            )
        coordinates = [np.array([atom[1] for atom in system]) for system in systems]
        changed = [xyz.copy() for xyz in coordinates]
        changed[-1][-1, 0] += 0.01
        changed_systems = [
            [(atom[0], tuple(xyz)) for atom, xyz in zip(system, positions, strict=True)]
            for system, positions in zip(systems, changed, strict=True)
        ]
        # Rebuild from independent ownership, outside measured replay, to
        # detect a stale geometry cache even when all repeated samples agree.
        with prepare(changed_systems) as changed_batch:
            changed_seed = execute(changed_batch)
        changed_hash = canonical_hash(
            {**inputs, "coordinates": [xyz.tolist() for xyz in changed]}
        )
        rows += measure_interleaved(
            measured("changed-geometry", lambda: execute(batch, changed)),
            synchronize,
            workload="changed-geometry",
            inputs_hash=changed_hash,
            repeats=repeats,
            prepare=lambda _: execute(batch, coordinates),
        )
    finally:
        synchronize()
        start = time.perf_counter()
        batch.close()
        synchronize()
        destruction_seconds = time.perf_counter() - start
    if _source_metadata(library) != source:
        raise RuntimeError("source/library changed during workload measurement")
    if policies or final_eigen_ablation:
        validate_ablation_branches(rows)
    # Same-model cold endpoints check cache/replay integrity. They are not an
    # independent scientific oracle or a substitute for the matched #206 gate.
    for row in rows:
        expected = changed_seed if row["workload"] == "changed-geometry" else seed
        if not np.allclose(
            row["diagnostics"]["energies"],
            expected["energies"],
            atol=1e-9,
            rtol=0,
        ):
            raise RuntimeError("cold/replay energy endpoint changed")
        if not energy_only and not np.allclose(
            row["diagnostics"]["forces"], expected["forces"], atol=1e-8, rtol=0
        ):
            raise RuntimeError("cold/replay force endpoint changed")
    return {
        "schema": "vibeqc.issue206.df_host_workloads",
        "version": 1,
        "source": source,
        "device": device,
        "inputs": inputs,
        "inputs_hash": input_hash,
        "cold_endpoints": {"original": seed, "changed": changed_seed},
        "profiled": trace_directory is not None,
        "samples": rows,
        "prepared_setup_seconds": setup_seconds,
        "prepared_destruction_seconds": destruction_seconds,
        "comparison": "reference versus ordinary device final eigensolve with identical lazy cached preparation"
        if final_eigen_ablation
        else f"original eager/rebuilt preparation versus {preparation_ablation} on one native library"
        if preparation_ablation
        else "eager core frame versus lazy warm initialization on one native library"
        if eager_core_ablation
        else "identical native configurations as an ABBA protocol control; no speedup claim",
        "timing_assessment": assess_comparison(rows)
        if (policies or final_eigen_ablation) and trace_directory is None
        else None,
        "limitations": [
            "This probe records actual host solves; complete device work/traffic requires the separate CUDA/Nsight ledger.",
            "Reported legacy Fock counts can omit finalizer work; actual reference-eigensolve leaves remain complete within traced scopes.",
            "External DF numerical/performance parity remains the existing matched #206 matrix gate.",
        ],
    }
