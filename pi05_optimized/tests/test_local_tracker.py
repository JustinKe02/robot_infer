from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from tk_infer.pi05.runtime.protocol import InferenceRequest, InferenceResponse
from tk_infer.pi05_optimized.runtime.local_tracker import (
    FirstOrderLagEstimator,
    LocalActionTracker,
    LocalTrackerConfig,
    LocalTrackerDeadlineError,
    LocalTrackerError,
    TrackerStateHistory,
    TrackerStateSample,
)
from tk_infer.pi05_optimized.runtime.optimized_client import OptimizedClient, OptimizedClientConfig
from tk_infer.pi05_optimized.runtime.timed_observation import TimedObservation

from .helpers import make_response


def _raw18(value: float = 0.0, *, force: float = 80.0) -> torch.Tensor:
    action = torch.zeros(18, dtype=torch.float32)
    action[:14] = value
    action[15] = force
    action[17] = force
    return action


def test_tracker_rate_limits_joints_but_not_grippers_and_restores_force() -> None:
    tracker = LocalActionTracker()
    requested = _raw18(1.0)
    requested[14] = 90.0
    requested[16] = 10.0

    first = tracker.track(
        requested_action=requested,
        observed_state=np.zeros(18, dtype=np.float32),
        timestamp_s=1.0,
    )
    second = tracker.track(
        requested_action=requested,
        observed_state=np.full(18, 0.01, dtype=np.float32),
        timestamp_s=1.05,
    )

    torch.testing.assert_close(first[:14], torch.full((14,), 0.02))
    torch.testing.assert_close(second[:14], torch.full((14,), 0.04))
    assert float(first[14]) == 90.0
    assert float(first[16]) == 10.0
    assert float(first[15]) == 80.0
    assert float(first[17]) == 80.0
    assert tracker.last_report is not None
    assert tracker.last_report.reference == "previous_command"
    assert tracker.last_report.output_max_joint_delta_rad <= 0.0200001


def test_contact_innovation_only_slows_joint_step_and_is_not_safety() -> None:
    tracker = LocalActionTracker(
        LocalTrackerConfig(contact_slowdown_gain=1.0, min_contact_slowdown_factor=0.25)
    )

    output = tracker.track(
        requested_action=_raw18(1.0),
        observed_state=np.zeros(18, dtype=np.float32),
        timestamp_s=1.0,
        contact_innovation=3.0,
    )

    torch.testing.assert_close(output[:14], torch.full((14,), 0.005))
    assert tracker.last_report is not None
    assert tracker.last_report.contact_slowdown_factor == pytest.approx(0.25)
    assert tracker.last_report.contact_used_as_safety is False
    assert tracker.health()["contact_innovation_role"] == "slowdown_only_not_safety"


def test_state_history_is_bounded_and_rejects_timestamp_regression() -> None:
    history = TrackerStateHistory(max_samples=3, window_s=1.0)
    for index in range(4):
        history.append(TrackerStateSample(float(index), np.zeros(18, dtype=np.float32)))

    assert [sample.timestamp_s for sample in history.snapshot()] == [2.0, 3.0]
    with pytest.raises(LocalTrackerError, match="advance strictly"):
        history.append(TrackerStateSample(3.0, np.zeros(18, dtype=np.float32)))


def test_first_order_lag_estimator_is_deterministic_and_resettable() -> None:
    estimator = FirstOrderLagEstimator(time_constant_s=0.1, innovation_gain=0.5)
    first = estimator.update(command_joints=np.ones(14), observed_joints=np.zeros(14), dt_s=0.05)
    second = estimator.update(command_joints=np.ones(14), observed_joints=np.full(14, 0.2), dt_s=0.05)

    assert first.max_abs_innovation == 0.0
    assert second.max_abs_innovation > 0.0
    assert 0.0 < second.alpha < 1.0
    estimator.reset()
    reset = estimator.update(command_joints=np.ones(14), observed_joints=np.zeros(14), dt_s=0.05)
    np.testing.assert_array_equal(reset.estimated_joints, first.estimated_joints)


def test_mpc_startup_and_runtime_failures_never_fall_back() -> None:
    with pytest.raises(ValueError, match="tracker_replay_passed"):
        LocalTrackerConfig(mpc_enabled=True)

    class FailingMPC:
        @property
        def name(self) -> str:
            return "failing_mpc"

        def solve(self, **_kwargs: object) -> torch.Tensor:
            raise RuntimeError("infeasible")

    tracker = LocalActionTracker(
        LocalTrackerConfig(mpc_enabled=True, tracker_replay_passed=True),
        mpc_solver=FailingMPC(),
        dependency_check=lambda: None,
    )
    with pytest.raises(LocalTrackerError, match="infeasible"):
        tracker.track(
            requested_action=_raw18(1.0),
            observed_state=np.zeros(18),
            timestamp_s=1.0,
        )
    assert tracker.last_report is None

    class UnsafeMPC:
        @property
        def name(self) -> str:
            return "unsafe_mpc"

        def solve(self, *, rate_limited_action: torch.Tensor, **_kwargs: object) -> torch.Tensor:
            output = rate_limited_action.clone()
            output[:14] += 1.0
            return output

    unsafe = LocalActionTracker(
        LocalTrackerConfig(mpc_enabled=True, tracker_replay_passed=True),
        mpc_solver=UnsafeMPC(),
        dependency_check=lambda: None,
    )
    with pytest.raises(LocalTrackerError, match="exceeds allowed"):
        unsafe.track(
            requested_action=_raw18(1.0),
            observed_state=np.zeros(18),
            timestamp_s=1.0,
        )


def test_tracker_and_mpc_deadline_misses_fail_closed() -> None:
    tracker_clock = iter((0.0, 0.02))
    tracker = LocalActionTracker(
        LocalTrackerConfig(tracker_deadline_s=0.01),
        clock=lambda: next(tracker_clock),
    )
    with pytest.raises(LocalTrackerDeadlineError, match="tracker exceeded"):
        tracker.track(
            requested_action=_raw18(0.01),
            observed_state=np.zeros(18),
            timestamp_s=1.0,
        )
    assert tracker.last_report is None

    class IdentityMPC:
        @property
        def name(self) -> str:
            return "identity_mpc"

        def solve(self, *, rate_limited_action: torch.Tensor, **_kwargs: object) -> torch.Tensor:
            return rate_limited_action

    mpc_clock = iter((0.0, 0.001, 0.02))
    mpc_tracker = LocalActionTracker(
        LocalTrackerConfig(
            tracker_deadline_s=0.1,
            mpc_enabled=True,
            tracker_replay_passed=True,
            mpc_deadline_s=0.01,
        ),
        mpc_solver=IdentityMPC(),
        dependency_check=lambda: None,
        clock=lambda: next(mpc_clock),
    )
    with pytest.raises(LocalTrackerDeadlineError, match="MPC exceeded"):
        mpc_tracker.track(
            requested_action=_raw18(0.01),
            observed_state=np.zeros(18),
            timestamp_s=1.0,
        )
    assert mpc_tracker.last_report is None


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


def _observation(sequence: int, timestamp_s: float) -> TimedObservation:
    return TimedObservation(
        observation_frame={"observation.state": np.zeros(18, dtype=np.float32)},
        sequence_id=sequence,
        receive_monotonic_s=timestamp_s,
        build_started_monotonic_s=timestamp_s,
        build_ready_monotonic_s=timestamp_s,
    )


def test_client_tracker_failure_clears_queue_and_blocks_later_writes() -> None:
    tracker = LocalActionTracker()
    sink = RecordingSink()
    clock = iter((1.0, 1.01, 1.02, 1.05, 1.06, 1.07))
    invalid_state = _observation(2, 1.05)
    invalid_state.observation_frame["observation.state"][0] = np.nan
    client = OptimizedClient(
        config=OptimizedClientConfig(
            task="tracker failure test",
            mode="rtc",
            local_tracker_enabled=True,
        ),
        observation_source=SequenceSource([_observation(1, 1.0), invalid_state]),
        policy_client=RecordingPolicy(),
        action_sink=sink,
        local_tracker=tracker,
        clock=lambda: next(clock),
    )

    assert client.run_cycle().tracker_applied is True
    with pytest.raises(LocalTrackerError, match="contains NaN/Inf"):
        client.run_cycle()

    assert len(sink.actions) == 1
    assert client.stop_state.stopped is True
    assert client.action_queue.depth() == 0
    assert client.last_failure_diagnostics is not None
    assert client.last_failure_diagnostics["queue_depth_before_clear"] > 0
    assert client.last_failure_diagnostics["queue_depth_after_clear"] == 0
    assert client.last_failure_diagnostics["no_later_action_write"] is True
    assert tracker.health()["reset_count"] == 1
    with pytest.raises(RuntimeError, match="optimized client is stopped"):
        client.run_cycle()
    assert len(sink.actions) == 1


def test_client_config_requires_explicit_tracker_pairing() -> None:
    source = SequenceSource([_observation(1, 1.0)])
    policy = RecordingPolicy()
    sink = RecordingSink()
    with pytest.raises(ValueError, match="must exactly match"):
        OptimizedClient(
            config=OptimizedClientConfig(task="mismatch", local_tracker_enabled=True),
            observation_source=source,
            policy_client=policy,
            action_sink=sink,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"control_period_s": 0.0}, "positive"),
        ({"max_joint_step_rad": float("nan")}, "finite"),
        ({"lag_innovation_gain": 2.0}, "in 0..1"),
        ({"history_max_samples": 1}, ">= 2"),
        ({"min_contact_slowdown_factor": 1.1}, "<= 1.0"),
    ],
)
def test_tracker_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LocalTrackerConfig(**kwargs)  # type: ignore[arg-type]
