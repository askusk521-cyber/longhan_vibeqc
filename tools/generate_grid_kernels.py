"""Emit the shared AO policy/runtime composition for the native CUDA build."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.dft.ao_cuda import emit_grid_source

from tools.generate_df_kernels import write_if_changed


def main():
    """Use exactly the JIT generator and preserve unchanged generated mtimes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, _, _ = emit_grid_source()
    # Native semilocal XC borrows the same AO traversal/policy. Keep this
    # extension out of the independent Python JIT grid owner's ABI and arena.
    source += '#include "cuda_xc_kernels.cuh"\n'
    write_if_changed(args.output, source)


if __name__ == "__main__":
    main()
