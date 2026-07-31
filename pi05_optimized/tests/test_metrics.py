from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from tk_infer.pi05_optimized.runtime.metrics import LATENCY_STAGES, InferenceMetrics, InferenceTimings


def _timings(value: float) -> InferenceTimings:
    return InferenceTimings(**dict.fromkeys(LATENCY_STAGES, value))


def test_metrics_window_is_bounded_and_reports_deterministic_percentiles() -> None:
    metrics = InferenceMetrics(window_size=3)
    for value in range(1, 6):
        metrics.record_success(_timings(float(value)))
    metrics.record_failure()

    snapshot = metrics.snapshot()
    total = snapshot.stages["total_s"]

    assert snapshot.window_size == 3
    assert snapshot.success_count == 5
    assert snapshot.failure_count == 1
    assert total.count == 3
    assert total.latest == 5.0
    assert total.p50 == 4.0
    assert total.p95 == pytest.approx(4.9)
    assert total.p99 == pytest.approx(4.98)


def test_metrics_percentiles_use_linear_interpolation_for_even_windows() -> None:
    metrics = InferenceMetrics(window_size=2)
    metrics.record_success(_timings(1.0))
    metrics.record_success(_timings(2.0))

    total = metrics.snapshot().stages["total_s"]
    assert total.p50 == pytest.approx(1.5)
    assert total.p95 == pytest.approx(1.95)
    assert total.p99 == pytest.approx(1.99)


def test_empty_metrics_snapshot_is_finite_and_zeroed() -> None:
    snapshot = InferenceMetrics(window_size=2).snapshot().to_dict()

    assert snapshot["success_count"] == 0
    for stage in snapshot["stages"].values():
        assert stage == {"count": 0, "latest": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True, "1.0"])
def test_inference_timings_reject_invalid_values(value: object) -> None:
    values = dict.fromkeys(LATENCY_STAGES, 0.1)
    values["total_s"] = value
    with pytest.raises(ValueError, match="total_s must be"):
        InferenceTimings(**values)  # type: ignore[arg-type]


def test_metrics_record_and_snapshot_are_thread_safe() -> None:
    metrics = InferenceMetrics(window_size=64)

    def record_many(worker: int) -> None:
        for offset in range(100):
            metrics.record_success(_timings(float(worker * 100 + offset)))
            metrics.snapshot()

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(record_many, range(8)))

    snapshot = metrics.snapshot()
    assert snapshot.success_count == 800
    assert snapshot.failure_count == 0
    assert all(stage.count == 64 for stage in snapshot.stages.values())
