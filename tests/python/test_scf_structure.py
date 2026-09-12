"""Keep method/backend ownership out of reusable CPU SCF reference interfaces."""

import pytest

from tools.check_scf_structure import audit_scf_structure


def test_current_shared_scf_dependencies_are_valid():
    report = audit_scf_structure()
    assert not report["errors"]
    assert report["modules"]


@pytest.mark.parametrize("include", ['"scf/rhf.hpp"', '"../rhf.hpp"', "<scf/rhf.hpp>"])
@pytest.mark.parametrize("owner", ["reference", "solver", "gradient"])
def test_method_dependency_cannot_hide_behind_include_spelling(
    tmp_path, include, owner
):
    source = tmp_path / "src/scf"
    (source / owner).mkdir(parents=True)
    (source / "rhf.hpp").write_text("// Method-owned state\n")
    (source / owner / "implementation.cpp").write_text(f"#include {include}\n")
    report = audit_scf_structure(tmp_path)
    assert len(report["errors"]) == 1
    assert f"forbidden {owner} dependency on scf/rhf.hpp" in report["errors"][0]


def test_gradient_assembly_cannot_depend_on_solver_state(tmp_path):
    source = tmp_path / "src/scf"
    (source / "solver").mkdir(parents=True)
    (source / "gradient").mkdir()
    (source / "solver/diis.hpp").write_text("// Trajectory state\n")
    (source / "gradient/hf_gradient.cpp").write_text('#include "scf/solver/diis.hpp"\n')
    report = audit_scf_structure(tmp_path)
    assert len(report["errors"]) == 1
    assert "forbidden gradient dependency on scf/solver/diis.hpp" in report["errors"][0]


def test_initial_guess_consumes_reference_without_reverse_edge(tmp_path):
    source = tmp_path / "src/scf"
    for directory in ["reference", "initial_guess"]:
        (source / directory).mkdir(parents=True)
    reference = source / "reference/linalg.hpp"
    reference.write_text("// Independent reference declarations\n")
    guess = source / "initial_guess/density.hpp"
    guess.write_text('#include "scf/reference/linalg.hpp"\n')
    assert not audit_scf_structure(tmp_path)["errors"]
    reference.write_text('#include "scf/initial_guess/density.hpp"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


def test_documented_forbidden_example_is_not_an_include(tmp_path):
    source = tmp_path / "src/scf"
    (source / "reference").mkdir(parents=True)
    (source / "rhf.hpp").write_text("// Method-owned state\n")
    (source / "reference/linalg.cpp").write_text(
        '/* Forbidden example:\n#include "scf/rhf.hpp"\n*/\n'
        '// #include "scf/rhf.hpp"\n#include <vector>\n'
    )
    assert not audit_scf_structure(tmp_path)["errors"]


@pytest.mark.parametrize(
    "name, owner",
    [
        ("arena.cpp", "cuda_planning"),
        ("eigensolver.cpp", "cuda_eigensolver"),
        ("df_source_setup.cpp", "cuda_df_source"),
        ("df_plan_setup.cpp", "cuda_df_runtime"),
        ("df_rhf_scf.cpp", "cuda_df_runtime"),
        ("resources.cpp", "cuda_resources"),
        ("matrix_library.cpp", "cuda_matrix_library"),
    ],
)
@pytest.mark.parametrize("include", ['"scf/rhf.hpp"', '"../rhf.hpp"', "<scf/rhf.hpp>"])
def test_cuda_runtime_cannot_depend_on_method_driver(tmp_path, name, owner, include):
    source = tmp_path / "src/scf"
    (source / "cuda").mkdir(parents=True)
    (source / "rhf.hpp").write_text("// Method-owned state\n")
    (source / "cuda" / name).write_text(f"#include {include}\n")
    errors = audit_scf_structure(tmp_path)["errors"]
    assert len(errors) == 1
    assert f"forbidden {owner} dependency on scf/rhf.hpp" in errors[0]


def test_eigensolver_cannot_acquire_direct_queue_policy(tmp_path):
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / "direct_constants.hpp").write_text("// Direct queue policy\n")
    (source / "eigensolver.cpp").write_text('#include "direct_constants.hpp"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


@pytest.mark.parametrize(
    "owner", ["df_jk_kernels.cu", "df_scf_kernels.cu", "scf_density_kernels.cu"]
)
def test_df_kernels_cannot_acquire_host_plan_state(tmp_path, owner):
    """Kernel changes must remain independent of resource and graph lifetimes."""
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / "df_plan_internal.hpp").write_text("// Plan-owned allocations\n")
    (source / owner).write_text('#include "df_plan_internal.hpp"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


def test_matrix_library_cannot_acquire_bucket_resource_owner(tmp_path):
    """Matrix consumers borrow handles without depending on allocation lifetime."""
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / "resources.hpp").write_text("// Stream/graph/arena owner\n")
    (source / "matrix_library.cpp").write_text('#include "resources.hpp"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1


@pytest.mark.parametrize(
    "owner", ["direct_bounded_tasks.cu", "direct_queue_scan.cu", "direct_screening.cuh"]
)
@pytest.mark.parametrize(
    "dependency",
    ["resources.hpp", "one_electron_reference.cuh", "df_plan_internal.hpp"],
)
def test_direct_queue_owners_cannot_acquire_plan_or_integral_state(
    tmp_path, owner, dependency
):
    """Queue rebuilds stay independent of host ownership and integral recurrences."""
    source = tmp_path / "src/scf/cuda"
    source.mkdir(parents=True)
    (source / dependency).write_text("// Separately owned plan or scientific code\n")
    (source / owner).write_text(f'#include "{dependency}"\n')
    assert len(audit_scf_structure(tmp_path)["errors"]) == 1
