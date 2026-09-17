"""CPU proofs of the resident ABI contract emitted on top of a verified plan.

These checks need no GPU and no NVCC: they assert the generated translation
unit keeps the ordinary host-staged entry points intact, pins exactly the
planned input/output offsets, and propagates a non-zero ordinary status
instead of swallowing it. Real-device numerical parity is a separate opt-in
suite.
"""

from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    Symmetry,
    TensorSpec,
    add,
    input_tensor,
    multiply,
)
from vibeqc_compiler.tensor.cuda_plan import plan_cuda
from vibeqc_compiler.tensor.cuda_resident_emit import resident_source

TARGET = cuda_target_info("sm_80")


def doubled_pair_program():
    """Two inputs, two outputs: one elementwise and one symmmetric double."""
    occupied = IndexSpace("occupied", "occupied", 2)
    virtual = IndexSpace("virtual", "virtual", 3)
    i, j = Index("i", occupied), Index("j", occupied)
    a, b = Index("a", virtual), Index("b", virtual)
    common = {
        "representation": "restricted_spatial",
        "role": "parameter",
        "differentiable": True,
    }
    singles = TensorSpec((i, a), **common)
    doubles = TensorSpec((i, j, a, b), symmetries=(Symmetry((1, 0, 3, 2)),), **common)
    t1 = input_tensor("t1", singles)
    t2 = input_tensor("t2", doubles)
    return Program({"squared": add(multiply(t1, t1)), "copy": add(t2)})


def test_resident_source_keeps_the_ordinary_abi():
    """A resident artifact must remain a valid host-staged artifact."""
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    source = resident_source(plan)
    for ordinary in (
        "tensor_plan_identity",
        "tensor_create",
        "tensor_destroy",
        "tensor_run",
        "tensor_probe",
    ):
        assert ordinary in source, ordinary


def test_resident_source_declares_the_resident_abi():
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    source = resident_source(plan)
    for entry in (
        "resident_plan_identity",
        "resident_abi",
        "resident_upload",
        "resident_run",
        "resident_download",
    ):
        assert entry in source, entry
    assert "return 1;" in source


def test_resident_spans_match_the_pinned_plan_offsets():
    """Spans are compiled from pinned offsets; the planner must pin them."""
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    assert all(plan.steps[i].last_use == len(plan.steps) for i in plan.inputs)
    source = resident_source(plan)
    for i in plan.inputs:
        step = plan.steps[i]
        assert f"{{{step.offset}ULL,{step.node.spec.size * 8}ULL}}" in source
    for _, i in plan.outputs:
        step = plan.steps[i]
        assert f"{{{step.offset}ULL,{step.node.spec.size * 8}ULL}}" in source


def test_resident_run_propagates_the_ordinary_status():
    """A native non-finite/division error must not be reported as success."""
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    source = resident_source(plan)
    assert "const int status =" in source
    assert "if (status) return status;" in source


def test_resident_validation_covers_every_input_symmetry():
    """Uploads must be checked like the ordinary path, before any kernel."""
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    source = resident_source(plan)
    for slot in range(len(plan.inputs)):
        assert f"resident_validate_{slot}" in source
    # The doubles input declares simultaneous pair exchange; its validation
    # must compare against the permuted partner with the ordinary tolerance.
    assert "1e-11 + 1e-10 * fabs(peer)" in source


def test_resident_requires_pinned_materialized_outputs():
    """Resident spans are only valid while the planner pins outputs.

    This is the invariant the whole design rests on: every output step must
    be materialized and live to the end of the program, so its arena offset
    is stable and can be re-used by a later run. ``resident_source`` rejects
    a plan that violates it instead of emitting a stale span.
    """
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    for _, i in plan.outputs:
        step = plan.steps[i]
        assert not step.virtual
        assert step.last_use == len(plan.steps)
    # The emitter agrees with the planner rather than re-deriving it.
    assert resident_source(plan)


def test_resident_extension_is_part_of_the_generated_source():
    """Any plan-specific post-run action must be visible in the identity."""
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    clean = resident_source(plan)
    extended = resident_source(plan, extension="// plan-specific tables\n")
    assert clean != extended
    assert "// plan-specific tables" in extended


def test_resident_run_without_extension_has_no_undefined_hook():
    """The default source must not reference an undefined post-run symbol."""
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    source = resident_source(plan)
    assert "vibeqc_resident_action" not in source
    assert "VIBEQC_RESIDENT_POST_RUN" not in source


def test_resident_extension_post_run_is_wired_only_when_declared():
    """A declared post-run action is emitted and its status propagated."""
    plan = plan_cuda(doubled_pair_program(), TARGET, max_bytes=1 << 26)
    declared = resident_source(
        plan,
        extension="__VIBEQC_RESIDENT_POST_RUN_DECL__\n"
        "int vibeqc_resident_post_run(void*, int, Metrics*, char*, size_t) "
        "{ return 0; }\n",
    )
    assert "vibeqc_resident_post_run(pointer, profile, result, error, size)" in declared
    assert "if (status) return status;" in declared
