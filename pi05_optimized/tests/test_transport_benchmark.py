from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from tk_infer.pi05_optimized.tools.offline_transport_benchmark import (
    _make_request,
    _make_response,
    _measure,
    _validate_count,
    run_benchmark,
)


def test_transport_fixture_uses_full_request_and_paired_response_contract() -> None:
    request = _make_request()
    response = _make_response(request)

    request.validate()
    response.validate()
    assert request.observation_frame["observation.images.camera_head"].shape == (720, 1280, 3)
    assert request.observation_frame["observation.images.camera_right"].shape == (480, 640, 3)
    assert response.raw_actions.shape == (50, 16)
    assert response.processed_actions.shape == (50, 18)
    assert (response.processed_actions[:, 15] == 80).all()
    assert (response.processed_actions[:, 17] == 80).all()


def test_measurement_reports_requested_sample_count() -> None:
    calls: list[int] = []
    result = _measure(lambda: calls.append(1), warmup=2, repetitions=3)

    assert len(calls) == 5
    assert result["count"] == 3
    assert result["p50"] >= 0
    assert result["p95"] >= result["p50"]


def test_count_validation_is_strict() -> None:
    with pytest.raises(ValueError, match="warmup"):
        _validate_count("warmup", -1, allow_zero=True)
    with pytest.raises(ValueError, match="repetitions"):
        _validate_count("repetitions", 0, allow_zero=False)


def test_one_iteration_authenticated_loopback_closes_cleanly(tmp_path: Path) -> None:
    report = run_benchmark(
        Namespace(
            warmup=0,
            repetitions=1,
            phase2_report=tmp_path / "missing-phase2.json",
        )
    )

    assert report["status"] == "PASS"
    assert report["hardware_access"] is False
    assert report["external_network_access"] is False
    assert report["loopback_socket_access"] is True
    assert report["authenticated"] is True
    assert report["leaked_threads"] == []
