"""Recompute the paired endpoint decision from retained samples and work records.

The original 3% magnitude gate remains visible even though the user explicitly
accepted the smaller observed benefit. Five pairs support an exact enumeration
of all 5**5 paired bootstrap resamples; no random seed or outlier removal is used.
"""

import argparse
import itertools
import json
import statistics
from pathlib import Path


def percentile(values, fraction):
    """Use linear interpolation between sorted empirical quantiles."""
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def analyze(directory):
    """Check all samples independently before comparing their paired timings."""
    work = json.loads((directory / "work.json").read_text())
    results = {}
    for aos, observable in itertools.product((96, 192, 384, 768), ("forces", "energy")):
        label = f"{aos}-{observable}"
        data = json.loads((directory / f"{label}.json").read_text())
        assert len(data["samples"]) == 10 and len(data["diagnostics"]) == 2
        arms = {}
        for policy in ("auto", "candidate"):
            rows = [row for row in data["samples"] if row["policy"] == policy]
            assert [row["repeat"] for row in rows] == list(range(5))
            arms[policy] = [row["seconds"] for row in rows]
        for row in data["samples"] + data["diagnostics"]:
            assert row["maximum_energy_error"] <= 1e-9
            assert (row["maximum_force_error"] or 0) <= 1e-8
            assert all(
                c["converged"] and not c["warm_start_fallback"]
                for c in row["convergence"]
            )
            if aos >= 384:
                assert row["iterations"] == row["prime_iterations"] == [3]
        ratios = [c / a for a, c in zip(arms["auto"], arms["candidate"], strict=True)]
        bootstrap = sorted(
            statistics.median(ratios[i] for i in indices)
            for indices in itertools.product(range(5), repeat=5)
        )
        bounds = [percentile(bootstrap, p) for p in (0.025, 0.975)]
        median = {policy: statistics.median(times) for policy, times in arms.items()}
        diagnostic = {row["policy"]: row for row in data["diagnostics"]}
        per_arm = {}
        for policy, row in diagnostic.items():
            observed = work[f"{label}-{policy}"]
            values = observed["values"]
            starts = observed["scope_counts"]
            last_iteration = max(
                v["value"] for v in values if v["key"] == "device_iterations"
            )
            graph = sum(v["value"] for v in values if v["key"] == "host_graph_replay")
            assert not any(v["key"] == "graph_construction_attempt" for v in values)
            # A replay can tail-launch further iterations. Completed device
            # iterations, not host launch count, determine captured J/K work.
            captured_iterations = last_iteration if graph else 0
            counts = {
                "iterations": last_iteration,
                "host_graph_replays": graph,
                "j_builds": starts.get("ri_j", 0) + captured_iterations,
                "k_builds": starts.get("ri_k", 0)
                + starts.get("ri_k_occupied", 0)
                + captured_iterations,
                "eigensolves_including_density_factor_seed": starts.get(
                    "compact_eigensolve", 0
                )
                + captured_iterations,
                "final_fock_evaluations": next(
                    v["value"] for v in values if v["key"] == "final_fock_evaluations"
                ),
                "final_density_updates": next(
                    v["value"] for v in values if v["key"] == "final_density_updates"
                ),
                "final_candidate_rejections": next(
                    v["value"]
                    for v in values
                    if v["key"] == "final_candidate_rejections"
                ),
                "warm_start_fallback": any(
                    c["warm_start_fallback"] for c in row["convergence"]
                ),
            }
            groups = row["components"]["groups"]
            force = next(
                (g for g in groups if g["operation"] == "force_response"), None
            )
            per_arm[policy] = {
                "work": counts,
                "force_stage_seconds": row.get("force_stage_seconds"),
                "process_device_resident_bytes": row["process_device_resident_bytes"],
                "force_counter_sums": force["counter_sums"] if force else {},
                "inclusive_gpu_region_ms": observed["inclusive_gpu_region_ms"],
                "final_residual_observations": [
                    v
                    for v in values
                    if v["key"].startswith("final_")
                    and v["key"]
                    not in ("final_solve_epoch", "final_density_generation")
                ],
            }
        assert per_arm["auto"]["work"] == per_arm["candidate"]["work"]
        conserved = (
            "shell_triples_visited",
            "shell_triples_nonzero",
            "shell_primitive_products",
            "shell_cartesian_component_products",
            "shell_public_weights_consumed",
            "three_center_shell_panels",
            "raw_value_upload_bytes",
            "host_to_device_bytes",
            "device_to_host_bytes",
            "stream_synchronizations",
            "response_final_projection_reused",
        )
        for key in conserved:
            assert per_arm["auto"]["force_counter_sums"].get(key) == per_arm[
                "candidate"
            ]["force_counter_sums"].get(key), key
        results[label] = {
            "seconds": arms,
            "median_seconds": median,
            "ratio_of_medians": median["candidate"] / median["auto"],
            "paired_ratios": ratios,
            "median_paired_ratio": statistics.median(ratios),
            "paired_ratio_range": [min(ratios), max(ratios)],
            "paired_bootstrap_95_percentile_interval": bounds,
            "maximum_energy_error": max(
                row["maximum_energy_error"]
                for row in data["samples"] + data["diagnostics"]
            ),
            "maximum_force_error": max(
                (row["maximum_force_error"] or 0)
                for row in data["samples"] + data["diagnostics"]
            ),
            "diagnostics": per_arm,
        }
    original_gate = all(
        results[f"{aos}-forces"]["median_paired_ratio"] <= 0.97 for aos in (384, 768)
    )
    stable = all(
        results[f"{aos}-forces"]["paired_bootstrap_95_percentile_interval"][1] < 1
        for aos in (384, 768)
    )
    energy_ok = all(
        results[f"{aos}-energy"]["ratio_of_medians"] <= 1.03
        for aos in (96, 192, 384, 768)
    )
    return {
        "original_predeclared_3_percent_magnitude_gate_passed": original_gate,
        "paired_stability_gate_passed": stable,
        "energy_regression_gate_passed": energy_ok,
        "independent_numerical_and_work_gates_passed": True,
        "interpretation": "Explicit user acceptance in decision.json supersedes only the original magnitude requirement; this report does not relabel that gate as passed. Five-pair bootstrap intervals describe these retained samples, not an independent replication.",
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, nargs="?", default=Path(__file__).parent
    )
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory), indent=2, allow_nan=False))
