"""Timeline evidence must distinguish overlapping API, compute and DMA clocks."""

import sqlite3

import pytest

from benchmarks.df_response_timeline import (
    intersect_duration,
    interval_union,
    summarize,
)


def test_intersections_do_not_double_count_concurrent_activity():
    assert interval_union([(8, 12), (0, 5), (4, 9), (15, 18)]) == [(0, 12), (15, 18)]
    assert intersect_duration([(0, 10), (5, 15)], [(8, 12), (10, 20)]) == 7
    assert intersect_duration([(0, 10)], [(12, 20)]) == 0
    with pytest.raises(ValueError, match="reversed"):
        interval_union([(2, 1)])


@pytest.mark.parametrize("traced", (False, True))
def test_copy_wait_overlap_and_unassigned_host_residual(tmp_path, traced):
    """A copy API spans a prior kernel, DMA and host-only staging/control.

    Nested NVTX and the untraced original route must identify the same raw
    bytes; dense pinned copies need the NVTX origin to exclude density uploads.
    """
    path = tmp_path / "timeline.sqlite"
    with sqlite3.connect(path) as c:
        c.executescript("""
            create table StringIds(id integer, value text);
            create table CUPTI_ACTIVITY_KIND_RUNTIME(
                start integer, end integer, globalTid integer,
                correlationId integer, nameId integer);
            create table CUPTI_ACTIVITY_KIND_KERNEL(
                start integer, end integer, streamId integer,
                correlationId integer, demangledName integer);
            create table CUPTI_ACTIVITY_KIND_MEMCPY(
                start integer, end integer, streamId integer,
                correlationId integer, bytes integer);
            insert into StringIds values(1,'cudaLaunchKernel'),(2,'prior_compute');
            insert into CUPTI_ACTIVITY_KIND_RUNTIME values(0,1,10,1,1),(5,25,10,2,3);
            insert into CUPTI_ACTIVITY_KIND_KERNEL values(2,15,7,1,2);
            insert into CUPTI_ACTIVITY_KIND_MEMCPY values(20,23,7,2,1024);
        """)
        c.execute(
            "insert into StringIds values(3,?)",
            ("cudaMemcpyAsync" if traced else "cudaMemcpy2DAsync",),
        )
        if traced:
            c.executescript("""
                create table NVTX_EVENTS(
                    start integer, end integer, globalTid integer, text text, textId integer);
                insert into NVTX_EVENTS values
                    (0,30,10,'force_response',null),
                    (3,26,10,'raw_value_slice_upload',null);
            """)
    result = summarize(path)
    raw = result["raw_copies"]
    assert raw["bytes"] == 1024
    assert raw["calls"] == 1
    assert raw["host_api_ms"] == 20 / 1e6
    assert raw["device_dma_ms"] == 3 / 1e6
    assert raw["host_overlap_same_stream_kernels_ms"] == 10 / 1e6
    assert raw["host_overlap_same_stream_copies_ms"] == 3 / 1e6
    assert raw["host_residual_not_device_activity_ms"] == 7 / 1e6
    scope = result["raw_copy_scope"]
    if traced:
        assert scope["host_nvtx_ms"] == 23 / 1e6
        assert scope["host_overlap_same_stream_kernels_ms"] == 12 / 1e6
        assert scope["host_overlap_same_stream_copies_ms"] == 3 / 1e6
        assert scope["host_residual_not_device_activity_ms"] == 8 / 1e6
    else:
        assert scope is None
