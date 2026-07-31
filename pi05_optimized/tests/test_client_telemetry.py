from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from tk_infer.pi05_optimized.runtime.client_telemetry import ClientTelemetry

from .helpers import make_request, make_response


def test_client_telemetry_tracks_request_queue_frames_and_control_jitter() -> None:
    telemetry = ClientTelemetry(window_size=4)
    request = make_request(mode="rtc")
    response = make_response(request)

    telemetry.record_request(request, response, total_s=0.2)
    telemetry.record_queue(depth=7, dropped_steps=2, stale_chunk=True)
    telemetry.record_frame(observation_sequence_id=1, source_frame_id=10)
    telemetry.record_frame(observation_sequence_id=2, source_frame_id=10)
    telemetry.record_frame(observation_sequence_id=3, source_frame_id=13)
    telemetry.record_sensor_tick(timestamp_s=1.0, target_period_s=0.05)
    telemetry.record_sensor_tick(timestamp_s=1.06, target_period_s=0.05)
    telemetry.record_actor_tick(timestamp_s=2.0, target_period_s=0.05)
    telemetry.record_actor_tick(timestamp_s=2.04, target_period_s=0.05)
    telemetry.record_queue_empty()

    snapshot = telemetry.snapshot()
    assert snapshot.request_count == 1
    assert snapshot.frame_count == 3
    assert snapshot.repeated_source_frames == 1
    assert snapshot.skipped_source_frames == 2
    assert snapshot.stale_chunks == 1
    assert snapshot.queue_empty_events == 1
    assert snapshot.last_observation_sequence_id == 3
    assert snapshot.distributions["request_total_s"].latest == pytest.approx(0.2)
    assert snapshot.distributions["server_reported_s"].latest == pytest.approx(0.01)
    assert snapshot.distributions["queue_depth"].latest == 7.0
    assert snapshot.distributions["sensor_jitter_s"].latest == pytest.approx(0.01)
    assert snapshot.distributions["actor_jitter_s"].latest == pytest.approx(0.01)


def test_client_telemetry_is_bounded_and_thread_safe() -> None:
    telemetry = ClientTelemetry(window_size=16)

    def record(worker: int) -> None:
        for offset in range(50):
            request = make_request(request_id=worker * 100 + offset)
            telemetry.record_request(request, make_response(request), total_s=float(offset))
            telemetry.snapshot()

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(record, range(4)))

    snapshot = telemetry.snapshot()
    assert snapshot.request_count == 200
    assert snapshot.distributions["request_total_s"].count == 16


def test_client_telemetry_rejects_identity_regressions_and_invalid_values() -> None:
    telemetry = ClientTelemetry()
    request = make_request(request_id=1)
    response = make_response(make_request(request_id=2))
    with pytest.raises(ValueError, match="identity must match"):
        telemetry.record_request(request, response, total_s=0.1)

    telemetry.record_frame(observation_sequence_id=1, source_frame_id="a")
    with pytest.raises(ValueError, match="increase strictly"):
        telemetry.record_frame(observation_sequence_id=1, source_frame_id="b")
    with pytest.raises(ValueError, match="depth must"):
        telemetry.record_queue(depth=-1, dropped_steps=0, stale_chunk=False)

    telemetry.record_actor_tick(timestamp_s=1.0, target_period_s=0.05)
    with pytest.raises(ValueError, match="increase strictly"):
        telemetry.record_actor_tick(timestamp_s=1.0, target_period_s=0.05)
