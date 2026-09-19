"""Resident ownership teardown is testable without allocating a CUDA device."""

import ctypes as ct
from types import SimpleNamespace

import pytest

from tools.vibeqc_response.resident_cuda import CudaResidentRHFResponse


def _owner():
    destroyed = []
    owner = CudaResidentRHFResponse.__new__(CudaResidentRHFResponse)
    owner._lib = SimpleNamespace(
        vibeqc_rhf_response_resident_destroy=lambda handle: destroyed.append(
            handle.value
        )
    )
    owner._handle = ct.c_void_p(123)
    owner._closed = False
    owner._free = [0, 1]
    owner._live = set()
    owner.vector_slots = 2
    return owner, destroyed


@pytest.mark.parametrize("error_type", [MemoryError, ValueError, RuntimeError])
def test_exception_teardown_preserves_error_and_destroys_native_owner(error_type):
    owner, destroyed = _owner()
    original = error_type("injected solver failure")
    with pytest.raises(error_type) as captured, owner:
        # A failed solver frame/traceback retains its vector leases while
        # __exit__ runs; garbage collection cannot make them disappear.
        vector = owner._allocate()
        raise original
    assert captured.value is original
    assert owner._closed
    assert not owner._handle
    assert destroyed == [123]
    vector.release()
    assert not owner._live
    owner.close()
    assert destroyed == [123]


def test_normal_close_still_rejects_live_vector_leases():
    owner, destroyed = _owner()
    vector = owner._allocate()
    with pytest.raises(RuntimeError, match="live vectors"):
        owner.close()
    assert not destroyed
    vector.release()
    owner.close()
    assert destroyed == [123]
