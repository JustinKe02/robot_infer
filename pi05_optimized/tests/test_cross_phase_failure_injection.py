from __future__ import annotations

import http.client
import json
import pickle
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import torch

from tk_infer.pi05.runtime.action_queue import ActionChunk, ActionChunkQueue
from tk_infer.pi05.runtime.http_server import make_server
from tk_infer.pi05.runtime.protocol import (
    CONTENT_TYPE,
    PROTOCOL_VERSION,
    WIRE_PICKLE_PROTOCOL,
    InferenceRequest,
    InferenceResponse,
    dumps_payload,
)
from tk_infer.pi05.runtime.remote_client import RemotePolicyClient
from tk_infer.pi05_optimized.backends.torch_backend import TorchPolicyBackend
from tk_infer.pi05_optimized.runtime.optimized_client import OptimizedClient, OptimizedClientConfig
from tk_infer.pi05_optimized.runtime.policy_service import OptimizedPolicyService
from tk_infer.pi05_optimized.runtime.timed_observation import SourceTimestamp, TimedObservation

from .helpers import FakeReferenceService, make_request, make_response


@dataclass
class SequenceSource:
    observations: list[TimedObservation]
    reads: int = 0

    def read(self) -> TimedObservation:
        observation = self.observations[self.reads]
        self.reads += 1
        return observation


@dataclass
class RecordingSink:
    actions: list[torch.Tensor] = field(default_factory=list)

    def write(self, action: torch.Tensor) -> None:
        self.actions.append(action.detach().clone())


@dataclass
class MutatingPolicy:
    mutator: object

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        response = make_response(request)
        assert callable(self.mutator)
        self.mutator(response)
        return response


def _observation(
    sequence_id: int,
    timestamp_s: float,
    *,
    state_source_s: float | None = None,
    camera_source_s: float | None = None,
) -> TimedObservation:
    frame: dict[str, object] = {"observation.state": np.zeros(18, dtype=np.float32)}
    camera_sources = None
    if camera_source_s is not None:
        key = "observation.images.camera_head"
        frame[key] = np.zeros((2, 2, 3), dtype=np.uint8)
        camera_sources = {key: SourceTimestamp(camera_source_s, "sensor_clock", "head")}
    return TimedObservation(
        observation_frame=frame,
        sequence_id=sequence_id,
        receive_monotonic_s=timestamp_s,
        build_started_monotonic_s=timestamp_s,
        build_ready_monotonic_s=timestamp_s,
        state_source_timestamp=(
            None
            if state_source_s is None
            else SourceTimestamp(state_source_s, "sensor_clock", "raw18")
        ),
        camera_source_timestamps=camera_sources,
    )


def _client(
    policy: object,
    *,
    observations: list[TimedObservation] | None = None,
    mode: str = "single_step",
) -> tuple[OptimizedClient, SequenceSource, RecordingSink]:
    source = SequenceSource(observations or [_observation(1, 1.0)])
    sink = RecordingSink()
    client = OptimizedClient(
        config=OptimizedClientConfig(task="cross-phase failure injection", mode=mode),  # type: ignore[arg-type]
        observation_source=source,
        policy_client=policy,  # type: ignore[arg-type]
        action_sink=sink,
        clock=lambda: 1.0,
    )
    return client, source, sink


def _assert_client_stopped_without_write(client: OptimizedClient, sink: RecordingSink) -> None:
    assert client.stop_state.stopped is True
    assert sink.actions == []
    assert client.action_queue.depth() == 0
    assert client.last_failure_diagnostics is not None
    assert client.last_failure_diagnostics["stop_reason"] == client.stop_state.reason
    assert client.last_failure_diagnostics["queue_depth_after_clear"] == 0
    assert client.last_failure_diagnostics["no_later_action_write"] is True
    with pytest.raises(RuntimeError, match="optimized client is stopped"):
        client.run_cycle()
    assert sink.actions == []


class _FailureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, mode: str) -> None:
        self.mode = mode
        super().__init__(("127.0.0.1", 0), _FailureHandler)


class _FailureHandler(BaseHTTPRequestHandler):
    server: _FailureServer

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(content_length)
        if self.server.mode == "slow":
            time.sleep(0.1)
            self._send(200, CONTENT_TYPE, b"late")
        elif self.server.mode == "http_500":
            self._send(500, "application/json", json.dumps({"error": "injected 500"}).encode())
        elif self.server.mode == "wrong_content_type":
            self._send(200, "application/json", b"{}")
        elif self.server.mode == "malformed_payload":
            self._send(200, CONTENT_TYPE, b"not-a-valid-protocol-payload")
        else:
            self._send(500, "application/json", b"{}")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def _failure_server(mode: str) -> Iterator[str]:
    server = _FailureServer(mode)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        assert not thread.is_alive()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("http_500", "HTTP 500"),
        ("wrong_content_type", "Unexpected response Content-Type"),
        ("malformed_payload", "pickle|protocol|invalid|unpickling"),
        ("slow", "timed out"),
    ],
)
def test_remote_http_failures_stop_client_and_retain_diagnostics(mode: str, message: str) -> None:
    with _failure_server(mode) as url:
        remote = RemotePolicyClient(url, timeout_s=0.02)
        client, _source, sink = _client(remote)
        with pytest.raises(Exception, match=message):
            client.run_cycle()
        _assert_client_stopped_without_write(client, sink)


def test_unavailable_http_server_stops_client_without_retry_or_write() -> None:
    reserved = socket.socket()
    reserved.bind(("127.0.0.1", 0))
    port = reserved.getsockname()[1]
    reserved.close()
    remote = RemotePolicyClient(f"http://127.0.0.1:{port}", timeout_s=0.05)
    client, source, sink = _client(remote)

    with pytest.raises(OSError):
        client.run_cycle()

    assert source.reads == 1
    _assert_client_stopped_without_write(client, sink)


@contextmanager
def _optimized_server(
    *, max_request_bytes: int = 4096
) -> Iterator[tuple[tuple[str, int], FakeReferenceService]]:
    reference = FakeReferenceService()
    service = OptimizedPolicyService(
        backend=TorchPolicyBackend(reference),  # type: ignore[arg-type]
    )
    server = make_server(
        host="127.0.0.1",
        port=0,
        service=service,  # type: ignore[arg-type]
        auth_token="secret",
        max_request_bytes=max_request_bytes,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield (str(host), int(port)), reference
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def _http_post(
    address: tuple[str, int], body: bytes, headers: dict[str, str]
) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection(*address, timeout=2.0)
    try:
        connection.request("POST", "/infer", body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_optimized_http_rejects_auth_version_content_type_and_oversize_before_inference() -> None:
    wrong_version = pickle.dumps(
        {"version": PROTOCOL_VERSION + 1, "payload": make_request()},
        protocol=WIRE_PICKLE_PROTOCOL,
    )
    with _optimized_server(max_request_bytes=4096) as (address, reference):
        status_auth, _ = _http_post(address, b"not-pickle", {"Content-Type": CONTENT_TYPE})
        status_type, _ = _http_post(
            address,
            dumps_payload(make_request()),
            {"Authorization": "Bearer secret", "Content-Type": "application/octet-stream"},
        )
        status_version, version_body = _http_post(
            address,
            wrong_version,
            {"Authorization": "Bearer secret", "Content-Type": CONTENT_TYPE},
        )
        status_oversize, oversize_body = _http_post(
            address,
            b"x" * 4097,
            {"Authorization": "Bearer secret", "Content-Type": CONTENT_TYPE},
        )

    assert status_auth == 401
    assert status_type == 415
    assert status_version == 400
    assert b"Unsupported protocol version" in version_body
    assert status_oversize == 400
    assert b"too large" in oversize_body
    assert reference.infer_requests == []


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda response: (
                setattr(response, "raw_actions", np.zeros((3, 15), dtype=np.float32)),
                setattr(response, "raw_action_shape", (3, 15)),
            ),
            "raw_actions must have shape",
        ),
        (
            lambda response: (
                setattr(response, "processed_actions", np.zeros((2, 18), dtype=np.float32)),
                setattr(response, "processed_action_shape", (2, 18)),
            ),
            "same temporal length",
        ),
        (lambda response: response.raw_actions.__setitem__((0, 0), np.nan), "non-finite"),
        (lambda response: response.processed_actions.__setitem__((0, 15), 0.0), "force"),
    ],
)
def test_invalid_action_contract_stops_before_sink_and_retains_diagnostics(
    mutator: object, message: str
) -> None:
    client, _source, sink = _client(MutatingPolicy(mutator))

    with pytest.raises(ValueError, match=message):
        client.run_cycle()

    _assert_client_stopped_without_write(client, sink)


def test_sequence_and_state_source_regression_stop_before_second_policy_request() -> None:
    @dataclass
    class RecordingPolicy:
        requests: list[InferenceRequest] = field(default_factory=list)

        def infer(self, request: InferenceRequest) -> InferenceResponse:
            self.requests.append(request)
            return make_response(request)

    for observations, message in (
        (
            [_observation(1, 1.0), _observation(1, 1.05)],
            "observation_sequence_id must increase strictly",
        ),
        (
            [
                _observation(1, 1.0, state_source_s=10.0, camera_source_s=20.0),
                _observation(2, 1.05, state_source_s=10.0, camera_source_s=20.05),
            ],
            "source timestamp did not advance",
        ),
    ):
        policy = RecordingPolicy()
        client, _source, sink = _client(policy, observations=observations)
        assert client.run_cycle().action_written is True
        with pytest.raises((ValueError, RuntimeError), match=message):
            client.run_cycle()
        assert len(policy.requests) == 1
        assert len(sink.actions) == 1
        assert client.stop_state.stopped is True
        assert client.last_failure_diagnostics is not None
        with pytest.raises(RuntimeError, match="optimized client is stopped"):
            client.run_cycle()
        assert len(sink.actions) == 1


def test_full_rtc_queue_is_bounded_and_keeps_model_raw_pairing() -> None:
    raw = torch.arange(50 * 16, dtype=torch.float32).reshape(50, 16)
    processed = torch.zeros(50, 18, dtype=torch.float32)
    processed[:, :14] = raw[:, :14]
    processed[:, 14] = raw[:, 14]
    processed[:, 15] = 80.0
    processed[:, 16] = raw[:, 15]
    processed[:, 17] = 80.0
    queue = ActionChunkQueue(max_queue_size=50, empty_queue_strategy="stop")

    merged = queue.merge_rtc(
        ActionChunk(
            raw_actions=raw,
            processed_actions=processed,
            observation_timestamp_s=1.0,
            ready_timestamp_s=1.0,
            drop_steps=0,
            predicted_delay_steps=0,
            source_observation_seq=1,
        )
    )

    assert merged.queue_depth_after == 50
    assert queue.depth() == 50
    leftover = queue.get_raw_leftover()
    assert leftover is not None
    torch.testing.assert_close(leftover, raw)
    first = queue.pop_processed_action()
    assert first is not None
    torch.testing.assert_close(first, processed[0])
    assert queue.depth() == 49
