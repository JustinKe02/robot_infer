from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from tk_infer.pi05.runtime.protocol import InferenceRequest, InferenceResponse
from tk_infer.pi05_optimized.runtime.optimized_client import OptimizedClient, OptimizedClientConfig
from tk_infer.pi05_optimized.runtime.timed_observation import SourceTimestamp, TimedObservation
from tk_infer.pi05_optimized.runtime.timestamp_alignment import (
    Raw18StateHistory,
    StateHistorySample,
    TimestampAlignmentConfig,
    TimestampAlignmentShadow,
)
from tk_infer.pi05_optimized.tools.offline_phase5_alignment_replay import run_replay

from .helpers import make_response

CAMERA_KEY = "observation.images.camera_head"


def _config(*, max_skew_s: float = 0.6) -> TimestampAlignmentConfig:
    return TimestampAlignmentConfig(
        camera_keys=(CAMERA_KEY,),
        source_clock_domain="sensor_clock",
        state_delay_s=0.0,
        camera_delay_s={CAMERA_KEY: 0.0},
        readout_delay_s={CAMERA_KEY: 0.0},
        max_skew_s=max_skew_s,
        history_window_s=5.0,
        history_max_samples=4,
    )


def _observation(*, sequence: int, state_time: float, camera_time: float, state_value: float) -> TimedObservation:
    local_time = float(sequence + 1)
    return TimedObservation(
        observation_frame={
            "observation.state": np.full(18, state_value, dtype=np.float32),
            CAMERA_KEY: np.zeros((2, 2, 3), dtype=np.uint8),
        },
        sequence_id=sequence,
        receive_monotonic_s=local_time,
        build_started_monotonic_s=local_time,
        build_ready_monotonic_s=local_time,
        state_source_timestamp=SourceTimestamp(state_time, "sensor_clock", "raw18_state"),
        camera_source_timestamps={
            CAMERA_KEY: SourceTimestamp(camera_time, "sensor_clock", "head_camera")
        },
    )


def test_state_history_interpolates_without_extrapolation_and_returns_readonly_state() -> None:
    history = Raw18StateHistory(max_samples=4, window_s=5.0, max_skew_s=0.6)
    history.append(StateHistorySample(1.0, np.zeros(18, dtype=np.float32)))
    history.append(StateHistorySample(2.0, np.full(18, 2.0, dtype=np.float32)))

    aligned = history.interpolate(1.5)

    np.testing.assert_allclose(aligned.raw18, 1.0)
    assert aligned.interpolation_ratio == pytest.approx(0.5)
    assert aligned.raw18.flags.writeable is False
    with pytest.raises(LookupError, match="does not bracket"):
        history.interpolate(0.9)
    with pytest.raises(LookupError, match="does not bracket"):
        history.interpolate(2.1)


def test_state_history_rejects_regression_and_excessive_skew() -> None:
    history = Raw18StateHistory(max_samples=4, window_s=5.0, max_skew_s=0.1)
    history.append(StateHistorySample(1.0, np.zeros(18)))
    with pytest.raises(ValueError, match="advance strictly"):
        history.append(StateHistorySample(1.0, np.zeros(18)))

    history.append(StateHistorySample(2.0, np.ones(18)))
    with pytest.raises(ValueError, match="skew exceeds"):
        history.interpolate(1.5)


def test_alignment_shadow_warms_up_then_reports_interpolation_without_mutation() -> None:
    shadow = TimestampAlignmentShadow(_config())
    first = _observation(sequence=0, state_time=1.0, camera_time=1.0, state_value=0.0)
    second = _observation(sequence=1, state_time=2.0, camera_time=1.5, state_value=2.0)

    assert shadow.observe(first) == ()
    results = shadow.observe(second)

    assert len(results) == 1
    result = results[0]
    np.testing.assert_allclose(result.aligned_raw18, 1.0)
    np.testing.assert_allclose(result.current_raw18, 2.0)
    assert result.max_abs_delta == pytest.approx(1.0)
    assert result.changed_policy_input is False
    assert shadow.snapshot()["changed_policy_input"] is False
    np.testing.assert_allclose(second.observation_frame["observation.state"], 2.0)


def test_alignment_shadow_rejects_mixed_source_clock_domains() -> None:
    shadow = TimestampAlignmentShadow(_config())
    observation = _observation(sequence=0, state_time=1.0, camera_time=1.0, state_value=0.0)
    mixed = TimedObservation(
        observation_frame=observation.observation_frame,
        sequence_id=0,
        receive_monotonic_s=1.0,
        build_started_monotonic_s=1.0,
        build_ready_monotonic_s=1.0,
        state_source_timestamp=observation.state_source_timestamp,
        camera_source_timestamps={
            CAMERA_KEY: SourceTimestamp(1.0, "camera_clock", "head_camera")
        },
    )

    with pytest.raises(ValueError, match="clock domains differ"):
        shadow.observe(mixed)
    assert shadow.snapshot()["failure_count"] == 1


@dataclass
class SequenceSource:
    observations: list[TimedObservation]

    def read(self) -> TimedObservation:
        return self.observations.pop(0)


@dataclass
class RecordingPolicy:
    requests: list[InferenceRequest] = field(default_factory=list)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        return make_response(request)


@dataclass
class RecordingSink:
    actions: list[torch.Tensor] = field(default_factory=list)

    def write(self, action: torch.Tensor) -> None:
        self.actions.append(action)


def test_client_shadow_alignment_never_replaces_policy_observation_state() -> None:
    observations = [
        _observation(sequence=0, state_time=1.0, camera_time=1.0, state_value=0.0),
        _observation(sequence=1, state_time=2.0, camera_time=1.5, state_value=2.0),
    ]
    policy = RecordingPolicy()
    shadow = TimestampAlignmentShadow(_config())
    clock = iter((1.01, 1.02, 1.03, 2.01, 2.02, 2.03))
    client = OptimizedClient(
        config=OptimizedClientConfig(task="task", mode="single_step"),
        observation_source=SequenceSource(observations),
        policy_client=policy,
        action_sink=RecordingSink(),
        alignment_shadow=shadow,
        clock=lambda: next(clock),
    )

    client.run_cycles(2)

    np.testing.assert_allclose(policy.requests[0].observation_frame["observation.state"], 0.0)
    np.testing.assert_allclose(policy.requests[1].observation_frame["observation.state"], 2.0)
    assert shadow.snapshot()["result_count"] == 1
    assert shadow.snapshot()["latest"]["max_abs_delta"] == pytest.approx(1.0)


def test_deterministic_alignment_replay_is_exact_and_bounded() -> None:
    report = run_replay(
        Namespace(
            duration_s=1.0,
            rate_hz=20.0,
            camera_delay_s=0.03,
            readout_delay_s=0.005,
        )
    )

    assert report["status"] == "PASS"
    assert report["changed_policy_input"] is False
    assert report["interpolation_error"]["max"] <= 1e-5
    assert report["max_history_samples"] <= report["history_capacity"]
    assert report["live_motion_delay_identified"] is False
