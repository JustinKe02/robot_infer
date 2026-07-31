from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from tk_infer.pi05.runtime.http_server import make_server
from tk_infer.pi05.runtime.protocol import InferenceRequest, InferenceResponse
from tk_infer.pi05.runtime.remote_client import RemotePolicyClient
from tk_infer.pi05_optimized.backends.torch_backend import TorchPolicyBackend
from tk_infer.pi05_optimized.runtime.optimized_client import OptimizedClient, OptimizedClientConfig
from tk_infer.pi05_optimized.runtime.policy_service import OptimizedPolicyService
from tk_infer.pi05_optimized.runtime.timed_observation import SourceTimestamp, TimedObservation

from .helpers import FakeReferenceService, make_response


@dataclass
class FakeObservationSource:
    observations: list[TimedObservation]
    reads: int = 0

    def read(self) -> TimedObservation:
        if self.reads >= len(self.observations):
            raise RuntimeError("fake observation source is exhausted")
        observation = self.observations[self.reads]
        self.reads += 1
        return observation


@dataclass
class FakePolicyClient:
    requests: list[InferenceRequest] = field(default_factory=list)
    response_mutator: object = None

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        response = make_response(request)
        if callable(self.response_mutator):
            self.response_mutator(response)
        return response


@dataclass
class RecordingSink:
    actions: list[torch.Tensor] = field(default_factory=list)

    def write(self, action: torch.Tensor) -> None:
        self.actions.append(action.detach().clone())


def _observation(
    *,
    sequence_id: int,
    ready_s: float,
    source_timestamp_s: float | None = None,
) -> TimedObservation:
    camera_sources = None
    frame = {"observation.state": np.zeros(18, dtype=np.float32)}
    if source_timestamp_s is not None:
        frame["observation.images.camera_head"] = np.zeros((2, 2, 3), dtype=np.uint8)
        camera_sources = {
            "observation.images.camera_head": SourceTimestamp(
                source_timestamp_s,
                "camera_device",
                "head",
            )
        }
    return TimedObservation(
        observation_frame=frame,
        sequence_id=sequence_id,
        receive_monotonic_s=ready_s,
        build_started_monotonic_s=ready_s,
        build_ready_monotonic_s=ready_s,
        state_source_timestamp=(
            SourceTimestamp(float(sequence_id), "state_device", "raw18")
            if source_timestamp_s is not None
            else None
        ),
        camera_source_timestamps=camera_sources,
    )


def _client(
    *,
    mode: str,
    observations: list[TimedObservation],
    clock_values: tuple[float, ...],
    policy: FakePolicyClient | None = None,
    sink: RecordingSink | None = None,
    **config_overrides: object,
) -> tuple[OptimizedClient, FakeObservationSource, FakePolicyClient, RecordingSink]:
    source = FakeObservationSource(observations)
    selected_policy = policy or FakePolicyClient()
    selected_sink = sink or RecordingSink()
    clock = iter(clock_values)
    client = OptimizedClient(
        config=OptimizedClientConfig(
            task="offline optimized client test",
            mode=mode,  # type: ignore[arg-type]
            **config_overrides,
        ),
        observation_source=source,
        policy_client=selected_policy,
        action_sink=selected_sink,
        clock=lambda: next(clock),
    )
    return client, source, selected_policy, selected_sink


@contextmanager
def _remote_policy() -> Iterator[tuple[RemotePolicyClient, FakeReferenceService]]:
    reference = FakeReferenceService()
    service = OptimizedPolicyService(
        backend=TorchPolicyBackend(reference),  # type: ignore[arg-type]
    )
    server = make_server(
        host="127.0.0.1",
        port=0,
        service=service,  # type: ignore[arg-type]
        auth_token="optimized-client-test",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield (
            RemotePolicyClient(
                f"http://{host}:{port}",
                auth_token="optimized-client-test",
                timeout_s=2.0,
            ),
            reference,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def test_client_construction_has_no_source_policy_or_sink_side_effects() -> None:
    client, source, policy, sink = _client(
        mode="single_step",
        observations=[_observation(sequence_id=1, ready_s=1.0)],
        clock_values=(1.0, 1.01, 1.02),
    )

    assert client.stop_state.stopped is False
    assert source.reads == 0
    assert policy.requests == []
    assert sink.actions == []


def test_single_step_selects_first_fresh_action_and_records_telemetry() -> None:
    client, _source, policy, sink = _client(
        mode="single_step",
        observations=[_observation(sequence_id=1, ready_s=1.0)],
        clock_values=(1.0, 1.01, 1.02),
    )

    result = client.run_cycle()

    assert result.action_written is True
    assert result.dropped_steps == 1
    assert result.queue_depth == 0
    assert len(policy.requests) == 1
    assert policy.requests[0].prev_chunk_left_over is None
    assert len(sink.actions) == 1
    expected = make_response(policy.requests[0]).processed_actions[1]
    torch.testing.assert_close(sink.actions[0], torch.as_tensor(expected))
    snapshot = client.telemetry.snapshot()
    assert snapshot.request_count == 1
    assert snapshot.distributions["dropped_steps"].latest == 1.0


def test_client_integrates_authenticated_protocol_v3_loopback_without_hardware() -> None:
    source = FakeObservationSource([_observation(sequence_id=1, ready_s=1.0)])
    sink = RecordingSink()
    clock = iter((1.0, 1.01, 1.02))
    with _remote_policy() as (remote_policy, reference):
        client = OptimizedClient(
            config=OptimizedClientConfig(task="loopback optimized client test", mode="single_step"),
            observation_source=source,
            policy_client=remote_policy,
            action_sink=sink,
            clock=lambda: next(clock),
        )
        result = client.run_cycle()

    assert result.action_written is True
    assert len(reference.infer_requests) == 1
    assert len(sink.actions) == 1
    assert client.telemetry.snapshot().request_count == 1


def test_rtc_cycle_returns_unconsumed_model16_as_next_leftover() -> None:
    client, _source, policy, sink = _client(
        mode="rtc",
        observations=[
            _observation(sequence_id=1, ready_s=1.0),
            _observation(sequence_id=2, ready_s=1.05),
        ],
        clock_values=(1.0, 1.01, 1.02, 1.05, 1.06, 1.07),
    )

    first, second = client.run_cycles(2)

    assert first.action_written is True
    assert second.action_written is True
    assert len(sink.actions) == 2
    assert policy.requests[0].prev_chunk_left_over is None
    leftover = policy.requests[1].prev_chunk_left_over
    assert leftover is not None
    assert leftover.shape == (1, 16)
    expected_leftover = make_response(policy.requests[0]).raw_actions[2:]
    np.testing.assert_array_equal(leftover, expected_leftover)
    assert policy.requests[1].predicted_delay_steps == 1


def test_three_fully_stale_chunks_stop_without_writing_actions() -> None:
    client, source, _policy, sink = _client(
        mode="single_step",
        observations=[
            _observation(sequence_id=1, ready_s=0.0),
            _observation(sequence_id=2, ready_s=1.0),
            _observation(sequence_id=3, ready_s=2.0),
            _observation(sequence_id=4, ready_s=3.0),
        ],
        clock_values=(0.0, 3.0, 3.1, 3.2, 4.0, 4.1, 4.2, 5.0, 5.1),
    )

    assert client.run_cycle().fully_stale is True
    assert client.run_cycle().fully_stale is True
    with pytest.raises(RuntimeError, match="3 consecutive inference chunks were fully stale"):
        client.run_cycle()

    assert client.stop_state.stopped is True
    assert sink.actions == []
    reads_after_stop = source.reads
    with pytest.raises(RuntimeError, match="optimized client is stopped"):
        client.run_cycle()
    assert source.reads == reads_after_stop


def test_invalid_force_fails_closed_before_sink_write() -> None:
    def corrupt_force(response: InferenceResponse) -> None:
        response.processed_actions[0, 15] = 0.0

    policy = FakePolicyClient(response_mutator=corrupt_force)
    sink = RecordingSink()
    client, _source, _policy, _sink = _client(
        mode="single_step",
        observations=[_observation(sequence_id=1, ready_s=1.0)],
        clock_values=(1.0, 1.01),
        policy=policy,
        sink=sink,
    )

    with pytest.raises(ValueError, match="left_gripper.force"):
        client.run_cycle()

    assert sink.actions == []
    assert "queue" in client.stop_state.reason


def test_repeated_camera_source_frame_stops_before_second_policy_request() -> None:
    client, _source, policy, sink = _client(
        mode="single_step",
        observations=[
            _observation(sequence_id=1, ready_s=1.0, source_timestamp_s=100.0),
            _observation(sequence_id=2, ready_s=1.05, source_timestamp_s=100.0),
        ],
        clock_values=(1.0, 1.01, 1.02),
    )

    assert client.run_cycle().action_written is True
    with pytest.raises(RuntimeError, match="source frame was reused"):
        client.run_cycle()

    assert len(policy.requests) == 1
    assert len(sink.actions) == 1
    assert client.telemetry.snapshot().repeated_source_frames == 1


def test_strict_timestamp_mode_rejects_missing_sources_without_policy_call() -> None:
    client, _source, policy, sink = _client(
        mode="single_step",
        observations=[_observation(sequence_id=1, ready_s=1.0)],
        clock_values=(),
        strict_source_timestamps=True,
        required_camera_keys=("observation.images.camera_head",),
    )

    with pytest.raises(ValueError, match="state source timestamp"):
        client.run_cycle()

    assert policy.requests == []
    assert sink.actions == []


def test_external_stop_while_policy_is_in_flight_blocks_sink_write() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingPolicy:
        def infer(self, request: InferenceRequest) -> InferenceResponse:
            entered.set()
            assert release.wait(timeout=2.0)
            return make_response(request)

    source = FakeObservationSource([_observation(sequence_id=1, ready_s=1.0)])
    sink = RecordingSink()
    clock = iter((1.0, 1.01, 1.02))
    client = OptimizedClient(
        config=OptimizedClientConfig(task="concurrent stop test", mode="single_step"),
        observation_source=source,
        policy_client=BlockingPolicy(),
        action_sink=sink,
        clock=lambda: next(clock),
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            client.run_cycle()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(timeout=2.0)
    client.stop_state.request_stop("external emergency stop")
    release.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert sink.actions == []
    assert len(errors) == 1
    assert "external emergency stop" in str(errors[0])


def test_policy_and_sink_failures_stop_without_later_source_reads() -> None:
    class FailingPolicy:
        def infer(self, _request: InferenceRequest) -> InferenceResponse:
            raise TimeoutError("policy deadline exceeded")

    source = FakeObservationSource([_observation(sequence_id=1, ready_s=1.0)])
    sink = RecordingSink()
    client = OptimizedClient(
        config=OptimizedClientConfig(task="policy failure", mode="single_step"),
        observation_source=source,
        policy_client=FailingPolicy(),
        action_sink=sink,
        clock=lambda: 1.0,
    )
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        client.run_cycle()
    with pytest.raises(RuntimeError, match="optimized client is stopped"):
        client.run_cycle()
    assert source.reads == 1
    assert sink.actions == []

    class FailingSink:
        calls = 0

        def write(self, _action: torch.Tensor) -> None:
            self.calls += 1
            raise OSError("sink write failed")

    failing_sink = FailingSink()
    sink_client = OptimizedClient(
        config=OptimizedClientConfig(task="sink failure", mode="single_step"),
        observation_source=FakeObservationSource([_observation(sequence_id=1, ready_s=1.0)]),
        policy_client=FakePolicyClient(),
        action_sink=failing_sink,
        clock=iter((1.0, 1.01, 1.02)).__next__,
    )
    with pytest.raises(OSError, match="sink write failed"):
        sink_client.run_cycle()
    assert failing_sink.calls == 1
    assert "action: OSError" in sink_client.stop_state.reason


def test_response_identity_nonfinite_and_queue_empty_fail_before_sink() -> None:
    def wrong_identity(response: InferenceResponse) -> None:
        response.request_id += 1

    for mutator, message in (
        (wrong_identity, "identity must match"),
        (lambda response: response.raw_actions.__setitem__((0, 0), np.nan), "non-finite"),
    ):
        client, _source, _policy, sink = _client(
            mode="single_step",
            observations=[_observation(sequence_id=1, ready_s=1.0)],
            clock_values=(1.0, 1.01),
            policy=FakePolicyClient(response_mutator=mutator),
        )
        with pytest.raises(ValueError, match=message):
            client.run_cycle()
        assert sink.actions == []

    rtc_client, _source, _policy, rtc_sink = _client(
        mode="rtc",
        observations=[_observation(sequence_id=1, ready_s=1.0)],
        clock_values=(1.0, 1.01, 1.02),
    )
    rtc_client.action_queue.pop_processed_action = lambda: None  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="returned no executable action"):
        rtc_client.run_cycle()
    assert rtc_client.telemetry.snapshot().queue_empty_events == 1
    assert rtc_sink.actions == []


def test_clock_domain_mismatch_and_strict_timestamp_success() -> None:
    observation = _observation(sequence_id=1, ready_s=1.0, source_timestamp_s=10.0)
    mismatch, _source, policy, sink = _client(
        mode="single_step",
        observations=[observation],
        clock_values=(),
        clock_domain="another_monotonic_clock",
    )
    with pytest.raises(ValueError, match="clock domains differ"):
        mismatch.run_cycle()
    assert policy.requests == []
    assert sink.actions == []

    strict, _source, _policy, strict_sink = _client(
        mode="single_step",
        observations=[observation],
        clock_values=(1.0, 1.01, 1.02),
        strict_source_timestamps=True,
        required_camera_keys=("observation.images.camera_head",),
    )
    assert strict.run_cycle().action_written is True
    assert len(strict_sink.actions) == 1
