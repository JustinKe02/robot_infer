from __future__ import annotations

from pathlib import Path

from tk_infer.pi05_optimized.tools.offline_observability_soak import run_soak


def test_accelerated_observability_soak_has_bounded_metrics_and_no_failures(tmp_path: Path) -> None:
    report = run_soak(
        duration_s=1.0,
        rate_hz=20.0,
        iterations=100,
        metrics_window_size=32,
        trace_path=tmp_path / "trace.jsonl",
        trace_max_bytes=4096,
        trace_backup_count=2,
    )

    assert report["status"] == "PASS"
    assert report["hardware_access"] is False
    assert report["network_access"] is False
    assert report["mode"] == "accelerated"
    assert report["completed_iterations"] == 100
    assert report["service_metrics"]["success_count"] == 100
    assert report["service_metrics"]["failure_count"] == 0
    assert report["service_metrics"]["stages"]["total_s"]["count"] == 32
    assert report["client_metrics"]["request_count"] == 100
    assert report["client_metrics"]["distributions"]["queue_depth"]["count"] == 32
    assert report["trace"]["written_events"] == 100
    assert report["trace"]["dropped_events"] == 0
    assert report["leaked_threads"] == []
