from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tk_infer.pi05_optimized.runtime.metrics import LATENCY_STAGES, InferenceTimings
from tk_infer.pi05_optimized.runtime.trace import MAX_ERROR_MESSAGE_CHARS, JsonlTraceWriter

from .helpers import make_request


def _timings(value: float = 0.1) -> InferenceTimings:
    return InferenceTimings(**dict.fromkeys(LATENCY_STAGES, value))


def _read_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_trace_writes_success_and_failure_without_observation_or_action_payloads(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = JsonlTraceWriter(
        path,
        wall_time_ns=lambda: 1_700_000_000_000_000_000,
        monotonic_clock=lambda: 12.5,
        clock_source="test_monotonic",
    )
    request = make_request(mode="rtc")

    writer.record_inference(request, _timings())
    writer.record_failure(
        request,
        durations_s={"total_s": 0.3, "backend_s": 0.2},
        error=RuntimeError("backend\nfailed"),
    )

    events = _read_events(path)
    assert [event["event"] for event in events] == ["inference", "failure"]
    assert events[0]["status"] == "ok"
    assert events[0]["clock_source"] == "test_monotonic"
    assert events[1]["status"] == "error"
    assert events[1]["error_type"] == "RuntimeError"
    assert events[1]["error_message"] == "backend\nfailed"
    serialized = path.read_text(encoding="utf-8")
    for forbidden in ("observation_frame", "prev_chunk_left_over", "raw_actions", "processed_actions"):
        assert forbidden not in serialized
    assert writer.stats().written_events == 2
    assert writer.stats().dropped_events == 0


def test_concurrent_trace_writes_produce_complete_parseable_lines(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.jsonl"
    writer = JsonlTraceWriter(
        path,
        wall_time_ns=lambda: 1,
        monotonic_clock=lambda: 2.0,
        clock_source="test_monotonic",
    )

    def write(request_id: int) -> None:
        writer.record_inference(make_request(request_id=request_id), _timings(float(request_id)))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(100)))

    events = _read_events(path)
    assert len(events) == 100
    assert {event["request_id"] for event in events} == set(range(100))
    assert writer.stats().written_events == 100


def test_trace_restart_appends_without_rewriting_existing_events(tmp_path: Path) -> None:
    path = tmp_path / "append.jsonl"
    JsonlTraceWriter(path).record_inference(make_request(request_id=1), _timings())
    JsonlTraceWriter(path).record_inference(make_request(request_id=2), _timings())

    assert [event["request_id"] for event in _read_events(path)] == [1, 2]


def test_trace_write_error_is_counted_or_raised_according_to_strict_mode(tmp_path: Path) -> None:
    invalid_parent = tmp_path / "not-a-directory"
    invalid_parent.write_text("file", encoding="utf-8")
    path = invalid_parent / "trace.jsonl"

    non_strict = JsonlTraceWriter(path, strict=False)
    non_strict.record_inference(make_request(), _timings())
    assert non_strict.stats().written_events == 0
    assert non_strict.stats().dropped_events == 1
    assert "FileExistsError" in non_strict.stats().last_error

    strict = JsonlTraceWriter(path, strict=True)
    with pytest.raises(OSError):
        strict.record_inference(make_request(), _timings())


def test_trace_truncates_error_messages(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = JsonlTraceWriter(path)
    writer.record_failure(
        make_request(),
        durations_s={"total_s": 0.1},
        error=RuntimeError("x" * (MAX_ERROR_MESSAGE_CHARS + 100)),
    )

    assert len(_read_events(path)[0]["error_message"]) == MAX_ERROR_MESSAGE_CHARS


def test_trace_rotation_keeps_a_fixed_number_of_bounded_files(tmp_path: Path) -> None:
    path = tmp_path / "rotating.jsonl"
    writer = JsonlTraceWriter(
        path,
        max_bytes=1024,
        backup_count=2,
        wall_time_ns=lambda: 1,
        monotonic_clock=lambda: 1.0,
        clock_source="test_monotonic",
    )
    for request_id in range(20):
        writer.record_inference(make_request(request_id=request_id), _timings())

    paths = [path, Path(f"{path}.1"), Path(f"{path}.2")]
    assert all(candidate.is_file() for candidate in paths)
    assert all(candidate.stat().st_size <= 1024 for candidate in paths)
    assert not Path(f"{path}.3").exists()
    assert writer.stats().rotation_count > 0
    for candidate in paths:
        _read_events(candidate)


def test_injected_trace_clock_requires_an_explicit_domain(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="clock_source is required"):
        JsonlTraceWriter(tmp_path / "trace.jsonl", monotonic_clock=lambda: 1.0)


def test_non_strict_trace_counts_invalid_payload_and_duration_drops(tmp_path: Path) -> None:
    writer = JsonlTraceWriter(tmp_path / "trace.jsonl", strict=False)
    request = make_request()
    request.mode = object()  # type: ignore[assignment]
    writer.record_inference(request, _timings())
    writer.record_failure(make_request(), durations_s={"total_s": float("nan")}, error=ValueError())

    stats = writer.stats()
    assert stats.written_events == 0
    assert stats.dropped_events == 2
