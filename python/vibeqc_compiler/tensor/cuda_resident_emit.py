"""Resident ABI appended to the verified #146 generated translation unit.

No generated-source replacement or second mathematical lowering: this module
wraps the complete TU produced by
:func:`vibeqc_compiler.tensor.cuda_emit.emit_cuda` and appends span-based
upload/run/download entry points around the same ``Context`` owner. The
resident run path re-enters the *ordinary* generated ``tensor_run`` with
input/output pointers pinned into the retained arena, so kernel launches,
gemm handling, the arithmetic error boundary and metrics are identical to
the host-staged path.

Physical spans are the plan's pinned materialized step offsets. Each named
input/output slot maps to exactly one ``(offset, bytes)`` span emitted at
compile time; a caller cannot edit the table, and a mismatched length fails
the checked span lookup before any copy runs.

The ordinary evaluation's status is propagated unchanged: a native
non-finite/division-by-zero boundary or an allocation-accounting mismatch is
reported to the caller exactly as it is on the host-staged path, never
swallowed. ``extension`` may add plan-specific tables and kernels; when it
defines ``vibeqc_resident_post_run`` the action runs after a *successful*
evaluation and its status is propagated too.
"""

from vibeqc_compiler.tensor.cuda_emit import emit_cuda


def _flat_parts(permuted_axes, shape):
    """Row-major flat index of a symmetry partner of element ``z``.

    Each axis coordinate is ``(z / stride[axis]) % extent[axis]``; the
    partner's flat index sums its coordinates weighted by the target-axis
    stride ``stride[permutation.index(axis)]``, matching the C-order
    symmetry verification the ordinary interpreter performs.
    """
    strides = []
    for axis in range(len(shape)):
        tail = 1
        for extent in shape[axis + 1 :]:
            tail *= extent
        strides.append(tail)
    terms = []
    for axis in range(len(shape)):
        terms.append(
            f"((z) / {max(1, strides[axis])}LL % {max(1, shape[axis])}LL)"
            f" * {max(1, strides[permuted_axes[axis]])}LL"
        )
    return " + ".join(terms)


def _validation_body(plan):
    """Emit one finiteness/symmetry validation kernel per input slot."""
    validations, calls = [], []
    for slot, i in enumerate(plan.inputs):
        step = plan.steps[i]
        node = step.node
        comparisons = []
        for symmetry in node.spec.symmetries:
            partner = _flat_parts(symmetry.permutation, node.spec.shape)
            comparisons.append(
                f"double peer = values[{partner}]; if (!isfinite(peer) || "
                f"fabs(value - ({symmetry.sign}) * peer) > 1e-11 + "
                f"1e-10 * fabs(peer)) atomicCAS(error, 0, {i + 1});"
            )
        body = "".join(f"{{{comparison}}}" for comparison in comparisons)
        validations.append(f"""
__global__ void resident_validate_{slot}(unsigned char* p, int* error) {{
    auto* values = reinterpret_cast<const double*>(p + {step.offset});
    for (int z = int(blockIdx.x) * blockDim.x + threadIdx.x; z < {node.spec.size};
         z += int(blockDim.x) * gridDim.x) {{
        double value = values[z];
        if (!isfinite(value)) atomicCAS(error, 0, {i + 1});
        {body}
    }}
}}
""")
        if node.spec.size:
            calls.append(
                f"resident_validate_{slot}<<<vibeqc_tensor::blocks({node.spec.size}, 128),"
                f"128,0,ctx.stream>>>(p,ctx.error); cuda_check(cudaGetLastError());"
            )
    return "".join(validations), calls


def resident_source(plan, *, prefix="", extension=""):
    """Append the resident ABI to the verified ordinary TU.

    ``prefix`` must match the value used to compile the ordinary artifact.
    ``extension`` supplies optional plan-specific tables/kernels. If it
    defines ``vibeqc_resident_post_run``, that action runs after every
    *successful* ordinary evaluation (for example a bounded scalar residual
    reduction), and it must only touch the plan's own arena and reservation.
    A non-zero return from either the ordinary evaluation or the extension
    action is propagated; the error buffer already carries its detail.
    """
    for _, i in plan.outputs:
        if plan.steps[i].virtual:
            raise ValueError("resident outputs require materialized steps")
    base = emit_cuda(plan, symbol_prefix=prefix)
    validations, _validation_calls = _validation_body(plan)

    inputs = [plan.steps[i] for i in plan.inputs]
    outputs = [plan.steps[i] for _, i in plan.outputs]

    def span_rows(steps):
        return ", ".join(f"{{{s.offset}ULL,{s.node.spec.size * 8}ULL}}" for s in steps)

    in_ptrs = (
        ", ".join(
            f"reinterpret_cast<double*>(ctx.arena + {s.offset}ULL)" for s in inputs
        )
        or "nullptr"
    )
    out_ptrs = (
        ", ".join(
            f"reinterpret_cast<double*>(ctx.arena + {s.offset}ULL)" for s in outputs
        )
        or "nullptr"
    )
    if extension and "__VIBEQC_RESIDENT_POST_RUN_DECL__" in extension:
        post_run = (
            "int status = vibeqc_resident_post_run(pointer, profile, result, error, size);"
            "\n    if (status) return status;"
        )
    else:
        post_run = ""
    return f"""{base}

#include "cuda_resident.cuh"
{validations}
namespace vibeqc_resident {{
static const Span resident_inputs[] = {{ {span_rows(inputs)} }};
static const Span resident_outputs[] = {{ {span_rows(outputs)} }};
}}  // namespace vibeqc_resident

{extension}
extern "C" const char* resident_plan_identity() {{
    return "{plan.identity}:resident-abi-v1";
}}
extern "C" int resident_abi() {{ return 1; }}
extern "C" int resident_upload(void* pointer, size_t slot, const void* host, size_t bytes,
                               char* error, size_t size) {{
    return vibeqc_resident::transfer(pointer, vibeqc_resident::resident_inputs,
                                     {len(inputs)}, slot, const_cast<void*>(host), bytes,
                                     true, error, size);
}}
extern "C" int resident_run(void* pointer, int profile, Metrics* result, char* error,
                            size_t size) {{
    if (!pointer || !result) {{
        vibeqc_tensor::error_text(error, size, "null resident argument"); return 1;
    }}
    auto& ctx = *static_cast<Context*>(pointer);
    const double* stored_inputs[{max(1, len(inputs))}] = {{ {in_ptrs} }};
    double* stored_outputs[{max(1, len(outputs))}] = {{ {out_ptrs} }};
    const int status =
        {prefix}tensor_run(pointer, stored_inputs, stored_outputs, profile, result, error, size);
    (void)ctx;
    if (status) return status;
    {post_run}
    return 0;
}}
extern "C" int resident_download(void* pointer, size_t slot, void* host, size_t bytes,
                                 char* error, size_t size) {{
    return vibeqc_resident::transfer(pointer, vibeqc_resident::resident_outputs,
                                     {len(outputs)}, slot, host, bytes,
                                     false, error, size);
}}
"""
