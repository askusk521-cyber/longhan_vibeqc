"""Independent scientific checks of the emitted shell-shared moment consumer."""

import ctypes
import shutil
import subprocess

import numpy as np
import pytest
from vibeqc_compiler.integral.df_derivatives_cuda import emit_df_derivatives_cuda
from vibeqc_compiler.integral.df_shell_derivatives import (
    SHELL_CLASSES,
    emit_df_shell_derivatives_cuda,
)

from tools.vibeqc_validation.df_derivatives import make_df_derivative_fixture


@pytest.fixture(scope="module")
def shell_library(tmp_path_factory):
    """Compile the exact generated arithmetic without requiring a GPU runtime."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("df_shell_emission")
    (directory / "cuda_runtime.h").write_text("")
    (directory / "generated_df_derivatives.cuh").write_text(emit_df_derivatives_cuda())
    (directory / "generated_df_shell_derivatives.cuh").write_text(
        emit_df_shell_derivatives_cuda()
    )
    source = directory / "fixture.cpp"
    source.write_text(r"""
#define __device__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#include "generated_df_shell_derivatives.cuh"
using namespace vibeqc::scf;
extern "C" void shell_derivative(unsigned code,const double* e,const double* r,
                                const unsigned char* powers,double* out) {
  generated_df_shell::for_each_class([&]<unsigned A,unsigned B,unsigned C>() {
    if(code!=A*16+B*4+C) return;
    using Math=generated_df_shell::Shell<A,B,C>;
    generated_df_derivatives::Geometry g;
    generated_df_derivatives::prepare_geometry(e[0],{r[0],r[1],r[2]},e[1],{r[3],r[4],r[5]},
                                               e[2],{r[6],r[7],r[8]},A+B+C,g);
    double cache[3*Math::axis_size];
    for(unsigned lane=0;lane<32;++lane) Math::prepare(g,cache,lane,32);
    const unsigned i=generated_df_shell::cartesian_index(A,powers);
    const unsigned j=generated_df_shell::cartesian_index(B,powers+3);
    const unsigned p=generated_df_shell::cartesian_index(C,powers+6);
    double independent[6]{};
    Math::accumulate((i*Math::nb+j)*Math::nc+p,e[0],e[1],g,cache,1.0,independent);
    auto result=Math::finish(independent);
    for(unsigned center=0;center<3;++center)
      for(unsigned axis=0;axis<3;++axis) out[3*center+axis]=result.gradient[center][axis];
  });
}
""")
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=c++20",
            "-shared",
            "-fPIC",
            str(source),
            "-I",
            str(directory),
            "-o",
            str(directory / "fixture.so"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    library = ctypes.CDLL(str(directory / "fixture.so"))
    double = ctypes.POINTER(ctypes.c_double)
    library.shell_derivative.argtypes = [
        ctypes.c_uint,
        double,
        double,
        ctypes.POINTER(ctypes.c_ubyte),
        double,
    ]
    library.shell_derivative.restype = None
    return library


@pytest.mark.parametrize("angular", SHELL_CLASSES)
@pytest.mark.parametrize("variant", ["asymmetric", "coincident"])
def test_shell_moments_match_independent_contracted_blocks(
    shell_library, angular, variant
):
    """Libcint checks normalization, Gaussian decay and all three center channels."""
    pytest.importorskip("pyscf")
    fixture = make_df_derivative_fixture(angular, variant=variant)
    output = np.empty((len(fixture.records), 3, 3))
    double = ctypes.POINTER(ctypes.c_double)
    for i, record in enumerate(fixture.records):
        exponents = np.ascontiguousarray(record["exponents"])
        centers = np.ascontiguousarray(record["centers"])
        powers = np.ascontiguousarray(record["angular"], dtype=np.uint8)
        shell_library.shell_derivative(
            angular[0] * 16 + angular[1] * 4 + angular[2],
            exponents.ctypes.data_as(double),
            centers.ctypes.data_as(double),
            powers.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            output[i].ctypes.data_as(double),
        )
    output *= fixture.records["weight"][:, None, None]
    actual = fixture.contract(output)
    np.testing.assert_allclose(actual, fixture.reference, atol=8e-11, rtol=8e-11)
    np.testing.assert_allclose(
        fixture.spherical(actual), fixture.spherical_reference, atol=8e-11, rtol=8e-11
    )
