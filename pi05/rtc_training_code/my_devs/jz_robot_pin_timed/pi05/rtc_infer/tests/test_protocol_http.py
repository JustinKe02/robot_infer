from __future__ import annotations

import http.client
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pytest

from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime import http_server
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.http_server import make_server
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.protocol import (
    CONTENT_TYPE,
    PROTOCOL_VERSION,
    InferenceRequest,
    InferenceResponse,
    dumps_payload,
    loads_payload,
)
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.remote_client import (
    RemotePolicyClient,
)
from my_devs.jz_robot_pin_timed.pi05.rtc_infer import run_robot_client
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.run_robot_client import (
    validate_server_health,
)


def _request(*, request_id: int = 7, mode: str = "single_step") -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        mode=mode,  # type: ignore[arg-type]
        observation_frame={"observation.state": np.zeros(18, dtype=np.float32)},
        task="jz robot pin timed vr teleoperation",
        robot_type="jz_robot_pin_timed",
        obs_sequence_id=11,
        execution_horizon=10,
    )


def _response(request: InferenceRequest) -> InferenceResponse:
    raw = np.arange(3 * 16, dtype=np.float32).reshape(3, 16)
    processed = np.zeros((3, 18), dtype=np.float32)
    processed[:, :14] = raw[:, :14]
    processed[:, 14] = raw[:, 14]
    processed[:, 15] = 80.0
    processed[:, 16] = raw[:, 15]
    processed[:, 17] = 80.0
    return InferenceResponse(
        request_id=request.request_id,
        mode=request.mode,
        raw_actions=raw,
        processed_actions=processed,
        server_latency_s=0.01,
        model_latency_s=0.005,
        raw_action_shape=raw.shape,
        processed_action_shape=processed.shape,
    )


@dataclass
class FakePolicyService:
    infer_requests: list[InferenceRequest] = field(default_factory=list)

    def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "supported_modes": ["single_step", "rtc"],
            "model_action_dim": 16,
            "wire_action_dim": 18,
        }

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        self.infer_requests.append(request)
        return _response(request)


@contextmanager
def _running_server(
    service: FakePolicyService,
    *,
    auth_token: str | None = "test-secret",
    max_request_bytes: int = 1024 * 1024,
) -> Iterator[tuple[str, int]]:
    server = make_server(
        host="127.0.0.1",
        port=0,
        service=service,  # type: ignore[arg-type]
        auth_token=auth_token,
        max_request_bytes=max_request_bytes,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield str(host), int(port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def _http_request(
    address: tuple[str, int],
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(*address, timeout=2.0)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_protocol_pickle_roundtrip_preserves_numpy_payload() -> None:
    request = _request(mode="rtc")
    request.prev_chunk_left_over = np.arange(32, dtype=np.float32).reshape(2, 16)

    decoded = loads_payload(dumps_payload(request))

    assert isinstance(decoded, InferenceRequest)
    assert decoded.request_id == request.request_id
    assert decoded.mode == "rtc"
    np.testing.assert_array_equal(decoded.prev_chunk_left_over, request.prev_chunk_left_over)
    decoded.validate()


def test_robot_entrypoint_uses_canonical_protocol_package_for_pickle() -> None:
    assert RemotePolicyClient.__module__ == (
        "my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.remote_client"
    )
    assert run_robot_client.RemotePolicyClient is RemotePolicyClient
    assert InferenceRequest.__module__ == (
        "my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.protocol"
    )


def test_protocol_rejects_invalid_request_and_response_modes() -> None:
    with pytest.raises(ValueError, match="Unsupported inference mode"):
        _request(mode="streaming").validate()

    response = _response(_request())
    response.mode = "streaming"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported response mode"):
        response.validate()


def test_protocol_rejects_wrong_envelope_version() -> None:
    import pickle

    data = pickle.dumps({"version": PROTOCOL_VERSION + 1, "payload": _request()})
    with pytest.raises(ValueError, match="Unsupported protocol version"):
        loads_payload(data)


def test_http_authentication_happens_before_unpickle(monkeypatch: pytest.MonkeyPatch) -> None:
    service = FakePolicyService()
    loads_calls: list[bytes] = []

    def fail_if_called(data: bytes) -> object:
        loads_calls.append(data)
        raise AssertionError("unauthenticated request body was unpickled")

    monkeypatch.setattr(http_server, "loads_payload", fail_if_called)
    with _running_server(service) as address:
        status, headers, body = _http_request(
            address,
            "POST",
            "/infer",
            body=b"definitely-not-a-pickle",
            headers={"Content-Type": CONTENT_TYPE},
        )

    assert status == 401
    assert headers["WWW-Authenticate"] == "Bearer"
    assert b"unauthorized" in body
    assert loads_calls == []
    assert service.infer_requests == []


def test_http_rejects_wrong_content_type_without_inference() -> None:
    service = FakePolicyService()
    with _running_server(service) as address:
        status, _headers, body = _http_request(
            address,
            "POST",
            "/infer",
            body=dumps_payload(_request()),
            headers={
                "Authorization": "Bearer test-secret",
                "Content-Type": "application/octet-stream",
            },
        )

    assert status == 415
    assert CONTENT_TYPE.encode() in body
    assert service.infer_requests == []


def test_http_rejects_oversized_body_without_unpickle(monkeypatch: pytest.MonkeyPatch) -> None:
    service = FakePolicyService()
    loads_calls = 0

    def counting_loads(data: bytes) -> object:
        nonlocal loads_calls
        loads_calls += 1
        return loads_payload(data)

    monkeypatch.setattr(http_server, "loads_payload", counting_loads)
    with _running_server(service, max_request_bytes=32) as address:
        status, _headers, body = _http_request(
            address,
            "POST",
            "/infer",
            body=b"x" * 33,
            headers={
                "Authorization": "Bearer test-secret",
                "Content-Type": CONTENT_TYPE,
            },
        )

    assert status == 400
    assert b"too large" in body
    assert loads_calls == 0
    assert service.infer_requests == []


def test_remote_client_health_and_infer_roundtrip() -> None:
    service = FakePolicyService()
    with _running_server(service) as (host, port):
        client = RemotePolicyClient(
            server_url=f"http://{host}:{port}",
            auth_token="test-secret",
            timeout_s=2.0,
        )
        health = client.health()
        response = client.infer(_request(request_id=23, mode="rtc"))

    assert health["ok"] is True
    assert health["protocol_version"] == PROTOCOL_VERSION
    assert health["model_action_dim"] == 16
    assert health["wire_action_dim"] == 18
    assert response.request_id == 23
    assert response.raw_actions.shape == (3, 16)
    assert response.processed_actions.shape == (3, 18)
    assert [request.request_id for request in service.infer_requests] == [23]


def test_remote_client_health_handshake_rejects_bad_token() -> None:
    with _running_server(FakePolicyService()) as (host, port):
        client = RemotePolicyClient(
            server_url=f"http://{host}:{port}",
            auth_token="wrong-secret",
            timeout_s=2.0,
        )
        with pytest.raises(RuntimeError, match=r"HTTP 401: unauthorized"):
            client.health()


def test_remote_client_enforces_response_size_limit() -> None:
    with _running_server(FakePolicyService()) as (host, port):
        client = RemotePolicyClient(
            server_url=f"http://{host}:{port}",
            auth_token="test-secret",
            timeout_s=2.0,
            max_response_bytes=8,
        )
        with pytest.raises(ValueError, match="response is too large"):
            client.health()


def _compatible_health() -> dict[str, object]:
    return {
        "ok": True,
        "protocol_version": PROTOCOL_VERSION,
        "policy_type": "pi05",
        "model_state_dim": 16,
        "model_action_dim": 16,
        "wire_action_dim": 18,
        "schema_id": "jz_pin_opening16_v1",
        "schema_version": 1,
        "complete_step": True,
        "camera_keys": [
            "observation.images.camera_head",
            "observation.images.camera_left",
            "observation.images.camera_right",
        ],
        "camera_shapes": {
            "observation.images.camera_head": [3, 720, 1280],
            "observation.images.camera_left": [3, 480, 640],
            "observation.images.camera_right": [3, 480, 640],
        },
        "supported_modes": ["single_step", "rtc"],
    }


def test_client_health_handshake_accepts_exact_jz_contract() -> None:
    validate_server_health(_compatible_health(), mode="single_step")
    validate_server_health(_compatible_health(), mode="rtc")


def test_client_health_handshake_rejects_dimension_or_camera_mismatch() -> None:
    health = _compatible_health()
    health["wire_action_dim"] = 16
    camera_shapes = dict(health["camera_shapes"])  # type: ignore[arg-type]
    camera_shapes["observation.images.camera_head"] = [3, 480, 640]
    health["camera_shapes"] = camera_shapes

    with pytest.raises(RuntimeError, match="wire_action_dim.*camera_shapes"):
        validate_server_health(health, mode="rtc")
