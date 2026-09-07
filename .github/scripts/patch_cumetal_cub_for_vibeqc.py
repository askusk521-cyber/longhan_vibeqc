"""Provide the standard CUB BlockScan surface needed by VibeQC CUDA kernels."""

from __future__ import annotations

import os
from pathlib import Path


root = Path(os.environ["CUMETAL_SOURCE"]) / "runtime/api/cub/block"
header = root / "block_scan.h"
if not header.exists():
    raise SystemExit(f"missing pinned CuMetal BlockScan header: {header}")

header.write_text(
    r'''#pragma once
// CuMetal CUB shim: BlockScan.
//
// Match the cooperative device behavior used by VibeQC.  The stock 0.5.0
// shim is host-sequential only and gives every CUDA thread linear_tid == 0.

#include <cuda_runtime.h>

namespace cub {

enum BlockScanAlgorithm {
    BLOCK_SCAN_RAKING,
    BLOCK_SCAN_RAKING_MEMOIZE,
    BLOCK_SCAN_WARP_SCANS
};

template <typename T, int BLOCK_DIM_X, BlockScanAlgorithm ALGORITHM = BLOCK_SCAN_RAKING,
          int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1, int LEGACY_PTX_ARCH = 0>
class BlockScan {
public:
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

    struct TempStorage {
        T data[BLOCK_THREADS];
        T prefix;
    };

    __host__ __device__ explicit BlockScan(TempStorage& temp)
        : temp_(temp), linear_tid_(RowMajorTid()) {}
    __host__ __device__ BlockScan(TempStorage& temp, int linear_tid)
        : temp_(temp), linear_tid_(linear_tid) {}

    __host__ __device__ void ExclusiveSum(T input, T& output) {
        T aggregate{};
        ExclusiveSum(input, output, aggregate);
    }

    __host__ __device__ void ExclusiveSum(T input, T& output, T& block_aggregate) {
        temp_.data[linear_tid_] = input;
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
        if (linear_tid_ == 0) {
            T running{};
            for (int i = 0; i < BLOCK_THREADS; ++i) {
                const T value = temp_.data[i];
                temp_.data[i] = running;
                running = running + value;
            }
            temp_.prefix = running;
        }
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
        output = temp_.data[linear_tid_];
        block_aggregate = temp_.prefix;
    }

    template <typename ScanOp>
    __host__ __device__ void ExclusiveScan(T input, T& output, T initial_value, ScanOp op) {
        temp_.data[linear_tid_] = input;
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
        if (linear_tid_ == 0) {
            T running = initial_value;
            for (int i = 0; i < BLOCK_THREADS; ++i) {
                const T value = temp_.data[i];
                temp_.data[i] = running;
                running = op(running, value);
            }
            temp_.prefix = running;
        }
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
        output = temp_.data[linear_tid_];
    }

    __host__ __device__ void InclusiveSum(T input, T& output) {
        temp_.data[linear_tid_] = input;
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
        if (linear_tid_ == 0) {
            T running{};
            for (int i = 0; i < BLOCK_THREADS; ++i) {
                running = running + temp_.data[i];
                temp_.data[i] = running;
            }
            temp_.prefix = running;
        }
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
        output = temp_.data[linear_tid_];
    }

    template <typename ScanOp>
    __host__ __device__ void InclusiveScan(T input, T& output, ScanOp op) {
        temp_.data[linear_tid_] = input;
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
        if (linear_tid_ == 0) {
            T running = temp_.data[0];
            for (int i = 1; i < BLOCK_THREADS; ++i) {
                running = op(running, temp_.data[i]);
                temp_.data[i] = running;
            }
            temp_.prefix = running;
        }
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
        output = temp_.data[linear_tid_];
    }

private:
    static __host__ __device__ int RowMajorTid() {
#ifdef __CUDA_ARCH__
        return threadIdx.x + BLOCK_DIM_X * (threadIdx.y + BLOCK_DIM_Y * threadIdx.z);
#else
        return 0;
#endif
    }

    TempStorage& temp_;
    int linear_tid_;
};

}  // namespace cub
''',
    encoding="utf-8",
)

# CUDA Toolkit/CUB exposes .cuh includes.  CuMetal 0.5.0 installs only .h.
(root / "block_scan.cuh").write_text(
    '#pragma once\n#include "block_scan.h"\n', encoding="utf-8"
)
