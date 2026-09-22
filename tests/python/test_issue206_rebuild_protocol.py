"""Changed endpoints include reset work and reject nonconverged oracles."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import issue206_rebuild as runner


def _engine(clock: list[float], events: list[str], converged: bool) -> SimpleNamespace:
    def reset(molecule: object) -> None:
        clock[0] += 7.0
        events.append("reset")

    def kernel(*, dm0: object) -> float:
        clock[0] += 3.0
        events.append("kernel")
        return -1.0

    def gradient() -> np.ndarray:
        clock[0] += 2.0
        events.append("gradient")
        return np.zeros((2, 3))

    return SimpleNamespace(
        mol=SimpleNamespace(set_geom_=lambda *args, **kwargs: object()),
        reset=reset,
        kernel=kernel,
        converged=converged,
        nuc_grad_method=lambda: SimpleNamespace(kernel=gradient),
    )


def test_changed_reference_timer_includes_geometry_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock, events = [0.0], []
    monkeypatch.setattr(runner, "_sync", lambda cp: events.append("sync"))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(runner, "gpu_convergence_payload", lambda *args: [])
    engine = _engine(clock, events, True)
    seed = np.eye(2)
    result = runner._stock_sample(
        [engine],
        [seed],
        SimpleNamespace(asnumpy=np.asarray),
        systems=[[("H", (0, 0, 0)), ("H", (0, 0, 1))]],
        coordinates=[np.zeros((2, 3))],
    )
    assert result["seconds"] == 12.0
    assert events == ["sync", "reset", "kernel", "gradient", "sync"]
    np.testing.assert_array_equal(seed, np.eye(2))


def test_nonconverged_reference_is_rejected_before_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    monkeypatch.setattr(runner, "_sync", lambda cp: None)
    monkeypatch.setattr(runner, "gpu_convergence_payload", lambda *args: [])
    engine = _engine([0.0], events, False)
    with pytest.raises(RuntimeError, match="converg"):
        runner._stock_sample([engine], [np.eye(2)], SimpleNamespace(asnumpy=np.asarray))
    assert "gradient" not in events
