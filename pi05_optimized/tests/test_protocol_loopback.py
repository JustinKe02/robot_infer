from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

from tk_infer.pi05.runtime.http_server import make_server
from tk_infer.pi05.runtime.remote_client import RemotePolicyClient
from tk_infer.pi05_optimized.backends.torch_backend import TorchPolicyBackend
from tk_infer.pi05_optimized.runtime.policy_service import OptimizedPolicyService

from .helpers import FakeReferenceService, make_request, make_response


@contextmanager
def _running_optimized_server(*, auth_token: str = "optimized-test-secret") -> Iterator[str]:
    backend = TorchPolicyBackend(FakeReferenceService())  # type: ignore[arg-type]
    service = OptimizedPolicyService(backend=backend)
    server = make_server(
        host="127.0.0.1",
        port=0,
        service=service,  # type: ignore[arg-type]
        auth_token=auth_token,
    )
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


def test_protocol_v3_authenticated_loopback_roundtrip_is_exact() -> None:
    request = make_request(mode="rtc")
    expected = make_response(request)
    with _running_optimized_server() as url:
        client = RemotePolicyClient(url, auth_token="optimized-test-secret", timeout_s=2.0)
        health = client.health()
        actual = client.infer(request)

    assert health["optimized_runtime"] is True
    assert health["backend"] == "torch"
    np.testing.assert_array_equal(actual.raw_actions, expected.raw_actions)
    np.testing.assert_array_equal(actual.processed_actions, expected.processed_actions)


def test_protocol_v3_rejects_wrong_bearer_token() -> None:
    with _running_optimized_server() as url:
        client = RemotePolicyClient(url, auth_token="wrong-token", timeout_s=2.0)
        with pytest.raises(RuntimeError, match="HTTP 401"):
            client.health()
