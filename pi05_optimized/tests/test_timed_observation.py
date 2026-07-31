from __future__ import annotations

import numpy as np
import pytest

from tk_infer.pi05_optimized.runtime.timed_observation import SourceTimestamp, TimedObservation


def test_timed_observation_preserves_clock_domains_without_mixing_them() -> None:
    state = SourceTimestamp(timestamp_s=1000.0, clock_domain="device_state", source="raw18_state")
    camera = SourceTimestamp(timestamp_s=1_700_000_000.0, clock_domain="camera_unix", source="head")
    observation = TimedObservation(
        observation_frame={
            "observation.state": np.zeros(18, dtype=np.float32),
            "observation.images.camera_head": np.zeros((2, 2, 3), dtype=np.uint8),
        },
        sequence_id=4,
        receive_monotonic_s=10.0,
        build_started_monotonic_s=10.1,
        build_ready_monotonic_s=10.3,
        state_source_timestamp=state,
        camera_source_timestamps={"observation.images.camera_head": camera},
    )

    assert observation.build_latency_s == pytest.approx(0.2)
    assert observation.receive_to_ready_s == pytest.approx(0.3)
    assert observation.state_source_timestamp == state
    assert observation.camera_source_timestamps["observation.images.camera_head"] == camera
    observation.require_source_timestamps(camera_keys=("observation.images.camera_head",))


def test_timed_observation_trace_metadata_excludes_payloads() -> None:
    observation = TimedObservation(
        observation_frame={
            "observation.state": np.arange(18, dtype=np.float32),
            "observation.images.camera_head": np.ones((720, 1280, 3), dtype=np.uint8),
        },
        sequence_id=1,
        receive_monotonic_s=1.0,
        build_started_monotonic_s=1.0,
        build_ready_monotonic_s=1.1,
    )

    metadata = observation.trace_metadata()

    assert metadata["observation_keys"] == [
        "observation.images.camera_head",
        "observation.state",
    ]
    assert "observation_frame" not in metadata
    assert "images" not in metadata
    with pytest.raises(TypeError):
        observation.observation_frame["new"] = 1  # type: ignore[index]


def test_strict_timestamp_mode_rejects_missing_state_or_camera_sources() -> None:
    observation = TimedObservation(
        observation_frame={"observation.state": np.zeros(18, dtype=np.float32)},
        sequence_id=1,
        receive_monotonic_s=1.0,
        build_started_monotonic_s=1.0,
        build_ready_monotonic_s=1.0,
    )

    with pytest.raises(ValueError, match="state source timestamp"):
        observation.require_source_timestamps(camera_keys=("observation.images.camera_head",))

    with_state = TimedObservation(
        observation_frame=observation.observation_frame,
        sequence_id=1,
        receive_monotonic_s=1.0,
        build_started_monotonic_s=1.0,
        build_ready_monotonic_s=1.0,
        state_source_timestamp=SourceTimestamp(1.0, "device", "state"),
    )
    with pytest.raises(ValueError, match="missing camera source timestamps"):
        with_state.require_source_timestamps(camera_keys=("observation.images.camera_head",))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sequence_id": -1}, "sequence_id"),
        ({"receive_monotonic_s": float("nan")}, "finite and non-negative"),
        ({"receive_monotonic_s": 2.0, "build_started_monotonic_s": 1.0}, "receive <="),
        ({"build_started_monotonic_s": 2.0, "build_ready_monotonic_s": 1.0}, "receive <="),
    ],
)
def test_timed_observation_rejects_invalid_local_timing(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "observation_frame": {"observation.state": np.zeros(18)},
        "sequence_id": 1,
        "receive_monotonic_s": 1.0,
        "build_started_monotonic_s": 1.0,
        "build_ready_monotonic_s": 2.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        TimedObservation(**values)  # type: ignore[arg-type]


def test_source_timestamp_requires_explicit_finite_domain_and_source() -> None:
    with pytest.raises(ValueError, match="real number"):
        SourceTimestamp(True, "device", "state")
    with pytest.raises(ValueError, match="clock_domain"):
        SourceTimestamp(1.0, "", "state")
    with pytest.raises(ValueError, match="source must"):
        SourceTimestamp(1.0, "device", "")


def test_timed_observation_rejects_invalid_source_types_keys_and_orphans() -> None:
    common = {
        "observation_frame": {"observation.state": np.zeros(18)},
        "sequence_id": 1,
        "receive_monotonic_s": 1.0,
        "build_started_monotonic_s": 1.0,
        "build_ready_monotonic_s": 1.0,
    }
    with pytest.raises(TypeError, match="state_source_timestamp"):
        TimedObservation(**common, state_source_timestamp=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no matching observation frame key"):
        TimedObservation(
            **common,
            camera_source_timestamps={
                "observation.images.camera_head": SourceTimestamp(1.0, "device", "head")
            },
        )
    with pytest.raises(ValueError, match="keys must be non-empty strings"):
        TimedObservation(**(common | {"observation_frame": {1: "invalid"}}))  # type: ignore[arg-type]
