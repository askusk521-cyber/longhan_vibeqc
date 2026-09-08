"""Development TensorIR: typed equations, CPU execution, replay, and AD rules.

Integral operators remain in vibeqc_codegen.  This package supplies primitive
JVP/VJP rules for every TensorIR operation; it supplies no CCSD method, CUDA
lowering, symmetry-aware packing adjoints, or bounded-recomputation planner.
"""

from .autodiff import (
    AD_PRIMITIVES,
    AD_RULE_VERSION,
    AD_RULES,
    DotTestResult,
    JVPResult,
    VJPResult,
    capabilities,
    dot_test,
    jvp,
    vjp,
)
from .interpreter import Execution, execute
from .ir import (
    PRIMITIVES,
    Node,
    add,
    broadcast,
    constant,
    divide,
    einsum,
    gather,
    input_tensor,
    multiply,
    reduce_sum,
    reshape,
    slice_tensor,
    transpose,
)
from .optimize import PASSES, optimize, rewrite
from .packing import PackedLayout
from .program import Program
from .types import Index, IndexSpace, Symmetry, TensorSpec

__all__ = [
    "AD_PRIMITIVES",
    "AD_RULES",
    "AD_RULE_VERSION",
    "PASSES",
    "PRIMITIVES",
    "DotTestResult",
    "Execution",
    "Index",
    "IndexSpace",
    "JVPResult",
    "Node",
    "PackedLayout",
    "Program",
    "Symmetry",
    "TensorSpec",
    "VJPResult",
    "add",
    "broadcast",
    "capabilities",
    "constant",
    "divide",
    "dot_test",
    "einsum",
    "execute",
    "gather",
    "input_tensor",
    "jvp",
    "multiply",
    "optimize",
    "reduce_sum",
    "reshape",
    "rewrite",
    "slice_tensor",
    "transpose",
    "vjp",
]
