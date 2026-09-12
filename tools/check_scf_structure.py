"""Enforce the dependency direction of shared native SCF modules."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# These are implementation boundaries, independent of #231's scientific CUDA
# ownership inventory. A reference oracle must not acquire a method driver or
# generated backend dependency merely because a future consumer needs it.
ALLOWED = {
    "reference": ("scf/reference/",),
    "initial_guess": ("scf/reference/", "scf/initial_guess/", "core/", "integrals/"),
    "gradient": ("scf/gradient/", "scf/reference/", "integrals/", "core/"),
    "solver": (
        "scf/solver/",
        "scf/gradient/",
        "scf/reference/",
        "scf/initial_guess/",
        "scf/types.hpp",
        "scf/proposals.hpp",
        "scf/fock_prepared.hpp",
        "scf/fock_build.hpp",
        "core/",
        "integrals/",
        "runtime/",
    ),
}
# CUDA planning and linear algebra have separate rebuild ownership. Enumerate
# these extracted owners rather than exempting all historical cuda/ fragments.
CUDA_MODULES = {
    "cuda_planning": (
        "arena",
        "checked_layout",
        "direct_constants",
        "direct_metadata",
        "packed_basis",
        "topology",
        "queue_plan",
        "queue_profile",
    ),
    "cuda_eigensolver": (
        "eigensolver",
        "eigensolver_kernels",
        "eigensolver_types",
        "matrix_index",
        "device_timer",
        "launch_geometry",
    ),
}
CUDA_MODULES["cuda_df_source"] = (
    "df_source",
    "df_source_setup",
    "df_source_internal",
    "df_source_kernels",
    "metadata_upload",
    "df_integral_export",
    "df_integral_export_batch",
)
CUDA_ALLOWED = {
    "cuda_planning": (
        "scf/cuda/arena.",
        "scf/cuda/checked_layout.",
        "scf/cuda/direct_constants.",
        "scf/cuda/direct_metadata.",
        "scf/cuda/packed_basis.",
        "scf/cuda/topology.",
        "scf/cuda/queue_plan.",
        "scf/cuda/queue_profile.",
        "scf/cuda/eigensolver_types.",
        "scf/cuda/launch_geometry.",
        "scf/cuda/rhf_policy.hpp",
        "scf/cuda_batch.hpp",
        "scf/direct_task_layout.hpp",
        "scf/generated_shell_task.hpp",
        "scf/aot_shell_registry.hpp",
        "molecule/",
        "core/",
    ),
    "cuda_eigensolver": (
        "scf/cuda/eigensolver.",
        "scf/cuda/eigensolver_kernels.",
        "scf/cuda/eigensolver_types.",
        "scf/cuda/matrix_index.",
        "scf/cuda/device_timer.",
        "scf/cuda/launch_geometry.",
        "scf/cuda_batch.hpp",
    ),
}
CUDA_ALLOWED["cuda_df_source"] = (
    "scf/cuda/df_source.",
    "scf/cuda/df_source_setup.",
    "scf/cuda/df_source_internal.",
    "scf/cuda/metadata_upload.",
    "scf/cuda/rhf_policy.hpp",
    "scf/cuda/df_source_kernels.",
    "scf/cuda/df_integral_export.",
    "scf/cuda/df_integral_export_batch.",
    "scf/cuda/packed_basis.",
    "scf/cuda/topology.",
    "scf/cuda/checked_layout.",
    "scf/cuda_density_fitting.hpp",
    "scf/cuda_density_fitting_integrals.hpp",
    "molecule/",
    "runtime/",
    "core/",
)
# DF plan/provider ownership is separate from direct HF queues and from the
# generic SCF driver. Kernel owners cannot acquire host plan or solver state.
CUDA_MODULES["cuda_df_runtime"] = (
    "df_plan",
    "df_plan_setup",
    "df_plan_internal",
    "df_setup_internal",
    "df_plan_lifetime",
    "df_runtime",
    "df_jk",
    "df_jk_internal",
    "df_coulomb",
    "df_exchange",
    "df_force_response",
    "df_scf_state",
    "df_scf_library",
    "df_rhf_scf",
    "df_uhf_scf",
)
CUDA_ALLOWED["cuda_df_runtime"] = tuple(
    "scf/cuda/" + stem + "." for stem in CUDA_MODULES["cuda_df_runtime"]
) + (
    "scf/cuda/df_metric_kernels.",
    "scf/cuda/df_jk_kernels.",
    "scf/cuda/df_scf_kernels.",
    "scf/cuda_density_fitting.hpp",
    "scf/cuda_df_gradient.hpp",
    "scf/density_fitting.hpp",
    "molecule/basis.hpp",
    "runtime/",
)
CUDA_MODULES["cuda_df_kernels"] = (
    "df_metric_kernels",
    "df_jk_kernels",
    "df_scf_kernels",
)
CUDA_ALLOWED["cuda_df_kernels"] = tuple(
    "scf/cuda/" + stem + "." for stem in CUDA_MODULES["cuda_df_kernels"]
)
SUFFIXES = {".cpp", ".hpp", ".cu", ".cuh"}
ROOT = Path(__file__).resolve().parents[1]


def audit_scf_structure(root: Path = ROOT) -> dict:
    """Check quoted/angle source includes, including relative-path spellings.

    Standard-library headers are outside this source graph. Comments do not
    introduce edges. Report source sizes without treating a small line count as
    evidence that the full HF decomposition has been completed.
    """
    source = (root / "src").resolve()
    errors, edges, modules = [], [], []
    groups = [
        (owner, allowed, sorted((source / "scf" / owner).rglob("*")))
        for owner, allowed in ALLOWED.items()
    ]
    for owner, stems in CUDA_MODULES.items():
        paths = [
            source / "scf/cuda" / (stem + suffix)
            for stem in stems
            for suffix in sorted(SUFFIXES)
        ]
        groups.append((owner, CUDA_ALLOWED[owner], paths))
    for owner, allowed, paths in groups:
        for path in paths:
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            content = path.read_text()
            relative = path.relative_to(source).as_posix()
            modules.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "lines": len(content.splitlines()),
                }
            )
            # Preserve line numbers when dropping multiline comments.
            text = re.sub(
                r"/\*.*?\*/|//[^\n]*",
                lambda m: "\n" * m[0].count("\n"),
                content,
                flags=re.DOTALL,
            )
            for match in re.finditer(
                r'^\s*#\s*include\s*["<]([^">]+)[">]', text, re.MULTILINE
            ):
                header = match[1]
                candidate = source / header
                if not candidate.is_file():
                    candidate = path.parent / header
                if not candidate.is_file():
                    continue
                try:
                    target = candidate.resolve().relative_to(source).as_posix()
                except ValueError:
                    continue
                edges.append({"from": relative, "to": target})
                if not target.startswith(allowed):
                    line = text.count("\n", 0, match.start()) + 1
                    errors.append(
                        f"{relative}:{line}: forbidden {owner} dependency on {target}"
                    )
    return {"modules": modules, "edges": edges, "errors": errors}


def main():
    """Return failure for a dependency violation; expose an optional JSON inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_scf_structure()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for error in report["errors"]:
            print(error)
        print(
            f"Checked {len(report['modules'])} shared SCF modules; "
            f"{len(report['errors'])} dependency errors"
        )
    return bool(report["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
