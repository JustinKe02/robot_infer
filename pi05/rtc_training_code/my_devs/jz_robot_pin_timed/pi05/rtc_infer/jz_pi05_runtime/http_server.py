from __future__ import annotations

import hmac
import ipaddress
import json
import pickle
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

from .protocol import CONTENT_TYPE, InferenceRequest, dumps_payload, loads_payload

if TYPE_CHECKING:
    from .policy_service import PolicyService

DEFAULT_MAX_REQUEST_BYTES = 64 * 1024 * 1024


class PolicyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        service: PolicyService,
        *,
        auth_token: str | None,
        max_request_bytes: int,
    ) -> None:
        super().__init__(server_address, PolicyRequestHandler)
        self.service = service
        self.auth_token = auth_token
        self.max_request_bytes = max_request_bytes


class PolicyRequestHandler(BaseHTTPRequestHandler):
    server: PolicyHTTPServer
    server_version = "JZPI05PolicyHTTP/2"

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        if not self._authenticate():
            return
        self._send_json(self.server.service.health())

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return

        # Authentication is deliberately checked before reading or unpickling the body.
        if not self._authenticate():
            return
        media_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if media_type != CONTENT_TYPE:
            self._send_json(
                {"error": f"Content-Type must be {CONTENT_TYPE}"},
                status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
            return

        try:
            body = self._read_request_body()
            request = loads_payload(body)
            if not isinstance(request, InferenceRequest):
                raise ValueError(f"Expected InferenceRequest, got {type(request)}")
            request.validate()
        except (TypeError, ValueError, pickle.UnpicklingError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            response = self.server.service.infer(request)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_pickle(response)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _authenticate(self) -> bool:
        expected = self.server.auth_token
        if expected is None:
            return True
        authorization = self.headers.get("Authorization", "")
        scheme, separator, supplied = authorization.partition(" ")
        valid = (
            bool(separator)
            and scheme.lower() == "bearer"
            and hmac.compare_digest(supplied.strip(), expected)
        )
        if valid:
            return True
        self._send_json(
            {"error": "unauthorized"},
            status=HTTPStatus.UNAUTHORIZED,
            extra_headers={"WWW-Authenticate": "Bearer"},
        )
        return False

    def _read_request_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if length <= 0:
            raise ValueError("Content-Length must be positive")
        if length > self.server.max_request_bytes:
            raise ValueError(
                f"Request body is too large: {length} > {self.server.max_request_bytes}"
            )
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError(f"Request body ended early: expected {length}, got {len(body)}")
        return body

    def _send_json(
        self,
        payload: dict[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_pickle(self, payload: Any) -> None:
        body = dumps_payload(payload)
        self.send_response(int(HTTPStatus.OK))
        self.send_header("Content-Type", CONTENT_TYPE)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def validate_bind_security(host: str, auth_token: str | None) -> str | None:
    token = None if auth_token is None or not auth_token.strip() else auth_token.strip()
    if not is_loopback_host(host) and token is None:
        raise ValueError(
            f"Refusing non-loopback bind host={host!r} without an authentication token"
        )
    return token


def is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def make_server(
    *,
    host: str,
    port: int,
    service: PolicyService,
    auth_token: str | None = None,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> PolicyHTTPServer:
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError(f"port must be in 0..65535, got {port!r}")
    if max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be positive")
    auth_token = validate_bind_security(host, auth_token)
    return PolicyHTTPServer(
        (host, port),
        service,
        auth_token=auth_token,
        max_request_bytes=max_request_bytes,
    )
