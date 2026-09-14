"""Adjust CuMetal graph declarations for Clang CUDA host/device parsing."""

from __future__ import annotations

import os
from pathlib import Path

header = Path(os.environ["CUMETAL_SOURCE"]) / "runtime/api/cuda_runtime.h"
text = header.read_text(encoding="utf-8")
old = (
    "cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);\n"
    "#ifndef cudaStreamGraphTailLaunch\n"
    "#define cudaStreamGraphTailLaunch ((cudaStream_t)0x3)\n"
    "#endif\n"
    "#if defined(__CUDA_ARCH__)\n"
    "#define cudaGetCurrentGraphExec() ((cudaGraphExec_t)nullptr)\n"
    "#define cudaGraphLaunch(graphExec, stream) (cudaErrorNotSupported)\n"
    "#endif\n"
)
new = (
    "__host__ __device__ cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, "
    "cudaStream_t stream);\n"
    "#ifndef cudaStreamGraphTailLaunch\n"
    "#define cudaStreamGraphTailLaunch ((cudaStream_t)0x3)\n"
    "#endif\n"
    "#if defined(__clang__) && defined(__CUDA__)\n"
    "__device__ __forceinline__ cudaGraphExec_t cudaGetCurrentGraphExec(void) { "
    "return nullptr; }\n"
    "#endif\n"
)
if text.count(old) != 1:
    raise SystemExit("CuMetal graph compatibility patch anchor not found exactly once")
text = text.replace(old, new, 1)
# These CUDA ABI error values distinguish an abandoned capture from unrelated
# runtime failures. The pinned shim omits the names; declaring them does not
# claim that CuMetal reproduces NVIDIA's capture rejection behavior.
error_anchor = "    cudaErrorNotSupported = 801,\n"
if text.count(error_anchor) != 1:
    raise SystemExit("CuMetal capture-error enum anchor not found exactly once")
text = text.replace(
    error_anchor,
    error_anchor
    + "    cudaErrorStreamCaptureUnsupported = 900,\n"
    + "    cudaErrorStreamCaptureInvalidated = 901,\n",
    1,
)
header.write_text(text, encoding="utf-8")
