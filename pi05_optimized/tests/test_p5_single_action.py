from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest
import torch

from tk_infer.pi05.runtime.protocol import InferenceRequest, InferenceResponse
from tk_infer.pi05_optimized.runtime.local_tracker import LocalActionTracker
from tk_infer.pi05_optimized.runtime.optimized_client import OptimizedClient, OptimizedClientConfig
from tk_infer.pi05_optimized.runtime.p5_single_action import (
    DEFAULT_CHECKPOINT_FINGERPRINT,
    P5_AUTHORIZATION_SCOPE,
    P5InterlockSnapshot,
    P5SingleActionError,
    P5SingleActionGuardedSink,
    P5StateSnapshot,
)
from tk_infer.pi05_optimized.runtime.timed_observation import TimedObservation

from .helpers import make_response


def _raw18(value: float = 0.0) -> torch.Tensor:
    action = torch.zeros(18, dtype=torch.float32)
    action[:14] = value
    action[14] = 50.0
    action[15] = 80.0
    action[16] = 50.0
    action[17] = 80.0
    return action


def _state(**changes: object) -> P5StateSnapshot:
    values: dict[str, object] = {
        "raw18": _raw18().numpy(),
        "sequence": 10,
        "stamp_ns": 1_000_000_000,
        "received_monotonic_s": 9.95,
        "sender_ip": "192.168.1.81",
        "robot": "robot1",
    }
    values.update(changes)
    return P5StateSnapshot(**values)  # type: ignore[arg-type]


def _interlock(**changes: object) -> P5InterlockSnapshot:
    values: dict[str, object] = {
        "authorization_id": "onsite-authorization-001",
        "scope": P5_AUTHORIZATION_SCOPE,
        "issued_monotonic_s": 9.0,
        "expires_monotonic_s": 11.0,
        "max_actions": 1,
        "checkpoint_step": 15900,
        "checkpoint_fingerprint": DEFAULT_CHECKPOINT_FINGERPRINT,
        "backend_label": "inference_mode",
        "physical_emergency_stop_verified": True,
        "workspace_clear": True,
        "operator_present": True,
        "robot_powered": True,
    }
    values.update(changes)
    return P5InterlockSnapshot(**values)  # type: ignore[arg-type]


@dataclass
class StateSource:
    snapshot: object = field(default_factory=_state)
    reads: int = 0

    def read_state(self) -> object:
        self.reads += 1
        return self.snapshot


@dataclass
class InterlockSource:
    snapshot: object = field(default_factory=_interlock)
    reads: int = 0

    def read_interlock(self) -> object:
        self.reads += 1
        return self.snapshot


@dataclass
class InMemoryInhibitableSink:
    hardware_access: bool = False
    command_transport: str | None = None
    armed_capability: bool = False
    actions: list[torch.Tensor] = field(default_factory=list)
    inhibit_reasons: list[str] = field(default_factory=list)
    write_error: Exception | None = None
    inhibit_error: Exception | None = None

    def write(self, action: torch.Tensor) -> None:
        if self.write_error is not None:
            raise self.write_error
        self.actions.append(action.detach().clone())

    def inhibit(self, reason: str) -> None:
        self.inhibit_reasons.append(reason)
        if self.inhibit_error is not None:
            raise self.inhibit_error


def _guard(
    *,
    state: object | None = None,
    interlock: object | None = None,
    delegate: InMemoryInhibitableSink | None = None,
) -> tuple[P5SingleActionGuardedSink, InMemoryInhibitableSink, StateSource, InterlockSource]:
    sink = delegate or InMemoryInhibitableSink()
    state_source = StateSource(_state() if state is None else state)
    interlock_source = InterlockSource(_interlock() if interlock is None else interlock)
    guard = P5SingleActionGuardedSink(
        delegate=sink,
        state_source=state_source,  # type: ignore[arg-type]
        interlock_source=interlock_source,  # type: ignore[arg-type]
        clock=lambda: 10.0,
    )
    return guard, sink, state_source, interlock_source


def test_exactly_one_safe_action_returns_then_exhausts_budget() -> None:
    guard, sink, state_source, interlock_source = _guard()
    action = _raw18(0.02)

    guard.write(action)

    assert len(sink.actions) == 1
    torch.testing.assert_close(sink.actions[0], action)
    assert state_source.reads == 1
    assert interlock_source.reads == 1
    assert sink.inhibit_reasons == ["single_action_budget_exhausted"]
    health = guard.health()
    assert health["delegate_write_attempts"] == 1
    assert health["delegate_write_successes"] == 1
    assert health["delivery_state"] == "delegate_returned"
    assert health["no_retry"] is True
    assert health["delegate_hardware_access"] is False
    assert health["delegate_command_transport"] is None
    assert health["delegate_armed_capability"] is False
    assert health["armed_launcher"] is False
    assert health["last_validation"]["max_initial_joint_delta_rad"] == pytest.approx(0.02)
    with pytest.raises(P5SingleActionError, match="budget_exhausted"):
        guard.write(action)
    assert len(sink.actions) == 1


@pytest.mark.parametrize(
    ("interlock", "message"),
    [
        (_interlock(scope="wrong"), "scope"),
        (_interlock(expires_monotonic_s=9.5), "not currently valid"),
        (_interlock(max_actions=2), "exactly one"),
        (_interlock(checkpoint_step=10600), "checkpoint step"),
        (_interlock(checkpoint_fingerprint="0" * 64), "fingerprint"),
        (_interlock(backend_label="plain"), "backend"),
        (_interlock(physical_emergency_stop_verified=False), "physical_emergency_stop_verified"),
        (_interlock(workspace_clear=False), "workspace_clear"),
        (_interlock(operator_present=False), "operator_present"),
        (_interlock(robot_powered=False), "robot_powered"),
    ],
)
def test_interlock_failure_latches_and_never_calls_delegate(
    interlock: P5InterlockSnapshot,
    message: str,
) -> None:
    guard, sink, _state_source, _interlock_source = _guard(interlock=interlock)

    with pytest.raises(P5SingleActionError, match=message):
        guard.write(_raw18())

    assert sink.actions == []
    assert len(sink.inhibit_reasons) == 1
    assert guard.health()["delegate_write_attempts"] == 0
    with pytest.raises(P5SingleActionError, match="stopped"):
        guard.write(_raw18())
    assert sink.actions == []


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (_state(received_monotonic_s=9.0), "state age"),
        (_state(received_monotonic_s=10.1), "state age"),
        (_state(sender_ip="192.168.1.99"), "state sender"),
        (_state(robot="robot2"), "state robot"),
        (_state(sequence_advanced=False), "sequence did not advance"),
    ],
)
def test_state_failure_latches_before_delegate(state: P5StateSnapshot, message: str) -> None:
    guard, sink, _state_source, _interlock_source = _guard(state=state)

    with pytest.raises(P5SingleActionError, match=message):
        guard.write(_raw18())

    assert sink.actions == []
    assert guard.health()["delivery_state"] == "not_attempted"


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (torch.zeros(17), "shape"),
        (torch.full((18,), float("nan")), "finite"),
        (_raw18(0.021), "initial joint delta"),
    ],
)
def test_action_contract_failure_latches_before_delegate(action: torch.Tensor, message: str) -> None:
    guard, sink, _state_source, _interlock_source = _guard()

    with pytest.raises(P5SingleActionError, match=message):
        guard.write(action)

    assert sink.actions == []


def test_force_and_gripper_failures_never_call_delegate() -> None:
    actions = []
    bad_force = _raw18()
    bad_force[15] = 79.0
    actions.append((bad_force, "force"))
    bad_left = _raw18()
    bad_left[14] = 101.0
    actions.append((bad_left, "left gripper"))
    bad_right = _raw18()
    bad_right[16] = -1.0
    actions.append((bad_right, "right gripper"))

    for action, message in actions:
        guard, sink, _state_source, _interlock_source = _guard()
        with pytest.raises(P5SingleActionError, match=message):
            guard.write(action)
        assert sink.actions == []


def test_invalid_snapshot_types_fail_before_delegate() -> None:
    for state, interlock in ((object(), _interlock()), (_state(), object())):
        guard, sink, _state_source, _interlock_source = _guard(state=state, interlock=interlock)
        with pytest.raises(P5SingleActionError, match="invalid snapshot"):
            guard.write(_raw18())
        assert sink.actions == []


def test_delegate_failure_is_unknown_delivery_and_cannot_retry() -> None:
    delegate = InMemoryInhibitableSink(write_error=OSError("injected write failure"))
    guard, sink, _state_source, _interlock_source = _guard(delegate=delegate)

    with pytest.raises(P5SingleActionError, match="delegate_failed"):
        guard.write(_raw18())

    health = guard.health()
    assert health["delegate_write_attempts"] == 1
    assert health["delegate_write_successes"] == 0
    assert health["delivery_state"] == "unknown_after_delegate_failure"
    assert health["delegate_inhibited"] is True
    with pytest.raises(P5SingleActionError, match="stopped"):
        guard.write(_raw18())
    assert sink.actions == []


@pytest.mark.parametrize(
    "changes",
    [
        {"hardware_access": True},
        {"command_transport": "udp"},
        {"armed_capability": True},
    ],
)
def test_software_readiness_rejects_hardware_or_armed_delegate(changes: dict[str, object]) -> None:
    delegate = InMemoryInhibitableSink(**changes)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rejects hardware"):
        _guard(delegate=delegate)


def test_preflight_deadline_stops_before_delegate() -> None:
    sink = InMemoryInhibitableSink()
    guard = P5SingleActionGuardedSink(
        delegate=sink,
        state_source=StateSource(_state()),  # type: ignore[arg-type]
        interlock_source=InterlockSource(_interlock()),  # type: ignore[arg-type]
        clock=iter((9.9, 10.0)).__next__,
    )

    with pytest.raises(P5SingleActionError, match="preflight exceeded"):
        guard.write(_raw18())

    assert sink.actions == []
    assert guard.health()["delegate_write_attempts"] == 0


def test_delegate_deadline_after_return_is_unknown_for_retry_purposes() -> None:
    sink = InMemoryInhibitableSink()
    guard = P5SingleActionGuardedSink(
        delegate=sink,
        state_source=StateSource(_state()),  # type: ignore[arg-type]
        interlock_source=InterlockSource(_interlock()),  # type: ignore[arg-type]
        clock=iter((9.99, 10.0, 10.0, 10.03)).__next__,
    )

    with pytest.raises(P5SingleActionError, match="returned after its deadline"):
        guard.write(_raw18())

    assert len(sink.actions) == 1
    health = guard.health()
    assert health["delivery_state"] == "delegate_returned_after_deadline"
    assert health["last_delegate_elapsed_s"] == pytest.approx(0.03)
    with pytest.raises(P5SingleActionError, match="stopped"):
        guard.write(_raw18())
    assert len(sink.actions) == 1


def test_delegate_start_clock_failure_stops_before_delegate() -> None:
    sink = InMemoryInhibitableSink()
    guard = P5SingleActionGuardedSink(
        delegate=sink,
        state_source=StateSource(_state()),  # type: ignore[arg-type]
        interlock_source=InterlockSource(_interlock()),  # type: ignore[arg-type]
        clock=iter((9.99, 10.0, float("nan"))).__next__,
    )

    with pytest.raises(P5SingleActionError, match="delegate_start_clock_failed"):
        guard.write(_raw18())

    assert sink.actions == []
    assert guard.health()["delegate_write_attempts"] == 0


def test_delegate_finish_clock_failure_blocks_retry_after_return() -> None:
    sink = InMemoryInhibitableSink()
    guard = P5SingleActionGuardedSink(
        delegate=sink,
        state_source=StateSource(_state()),  # type: ignore[arg-type]
        interlock_source=InterlockSource(_interlock()),  # type: ignore[arg-type]
        clock=iter((9.99, 10.0, 10.0, float("nan"))).__next__,
    )

    with pytest.raises(P5SingleActionError, match="delivery timing failed"):
        guard.write(_raw18())

    assert len(sink.actions) == 1
    assert guard.health()["delivery_state"] == "delegate_returned_timing_failure"
    with pytest.raises(P5SingleActionError, match="stopped"):
        guard.write(_raw18())
    assert len(sink.actions) == 1


def test_inhibit_failure_after_delegate_return_is_reported_without_retry() -> None:
    delegate = InMemoryInhibitableSink(inhibit_error=OSError("injected inhibit failure"))
    guard, sink, _state_source, _interlock_source = _guard(delegate=delegate)

    with pytest.raises(P5SingleActionError, match="delegate inhibition failed"):
        guard.write(_raw18())

    assert len(sink.actions) == 1
    health = guard.health()
    assert health["delegate_write_successes"] == 1
    assert health["delegate_inhibit_error"] == "OSError: injected inhibit failure"
    with pytest.raises(P5SingleActionError, match="stopped"):
        guard.write(_raw18())
    assert len(sink.actions) == 1


def test_external_inhibit_prevents_first_write() -> None:
    guard, sink, state_source, interlock_source = _guard()

    guard.inhibit("operator cancelled")

    with pytest.raises(P5SingleActionError, match="operator cancelled"):
        guard.write(_raw18())
    assert sink.actions == []
    assert state_source.reads == 0
    assert interlock_source.reads == 0


def test_concurrent_writes_allow_exactly_one_delegate_call() -> None:
    guard, sink, _state_source, _interlock_source = _guard()
    barrier = threading.Barrier(3)
    results: list[str] = []

    def run() -> None:
        barrier.wait()
        try:
            guard.write(_raw18())
        except P5SingleActionError:
            results.append("blocked")
        else:
            results.append("written")

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    assert sorted(results) == ["blocked", "written"]
    assert len(sink.actions) == 1
    assert guard.health()["delegate_write_attempts"] == 1


@dataclass
class ObservationSequence:
    observations: list[TimedObservation]

    def read(self) -> TimedObservation:
        return self.observations.pop(0)


@dataclass
class DeterministicPolicy:
    requests: list[InferenceRequest] = field(default_factory=list)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        return make_response(request)


def _observation(sequence: int, timestamp_s: float) -> TimedObservation:
    return TimedObservation(
        observation_frame={"observation.state": _raw18().numpy()},
        sequence_id=sequence,
        receive_monotonic_s=timestamp_s,
        build_started_monotonic_s=timestamp_s,
        build_ready_monotonic_s=timestamp_s,
    )


def test_optimized_client_tracker_and_guard_allow_one_in_memory_write_only() -> None:
    guard, sink, state_source, interlock_source = _guard()
    clock_values = iter((1.0, 1.01, 1.02, 1.05, 1.06, 1.07))
    client = OptimizedClient(
        config=OptimizedClientConfig(
            task="P5 software readiness integration",
            mode="single_step",
            local_tracker_enabled=True,
        ),
        observation_source=ObservationSequence([_observation(1, 1.0), _observation(2, 1.05)]),
        policy_client=DeterministicPolicy(),
        action_sink=guard,
        local_tracker=LocalActionTracker(),
        clock=lambda: next(clock_values),
    )

    first = client.run_cycle()
    assert first.action_written is True
    assert first.tracker_applied is True
    assert len(sink.actions) == 1
    assert float(torch.max(torch.abs(sink.actions[0][:14])).item()) <= 0.0200001

    with pytest.raises(P5SingleActionError, match="stopped"):
        client.run_cycle()

    assert len(sink.actions) == 1
    assert state_source.reads == 1
    assert interlock_source.reads == 1
    assert client.stop_state.stopped is True
    assert client.action_queue.depth() == 0
    assert client.last_failure_diagnostics is not None
    assert client.last_failure_diagnostics["failed_stage"] == "action"
    assert client.last_failure_diagnostics["no_later_action_write"] is True
