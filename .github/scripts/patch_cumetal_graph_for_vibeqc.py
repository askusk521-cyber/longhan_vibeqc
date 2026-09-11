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
header.write_text(text.replace(old, new, 1), encoding="utf-8")
