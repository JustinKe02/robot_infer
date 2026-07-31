from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .http_server import DEFAULT_MAX_REQUEST_BYTES
from .protocol import CONTENT_TYPE, InferenceRequest, InferenceResponse, dumps_payload, loads_payload

DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024


@dataclass(slots=True)
class RemotePolicyClient:
    server_url: str
    timeout_s: float = 120.0
    auth_token: str | None = None
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES

    def __post_init__(self) -> None:
        self.server_url = self.server_url.rstrip("/")
        parsed = urlsplit(self.server_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid policy server URL: {self.server_url!r}")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Policy server credentials must use the Bearer token, not URL userinfo")
        if parsed.path or parsed.query or parsed.fragment:
            raise ValueError("Policy server URL must be a bare scheme://host:port base URL")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.max_request_bytes <= 0 or self.max_response_bytes <= 0:
            raise ValueError("request/response byte limits must be positive")
        if self.auth_token is not None:
            self.auth_token = self.auth_token.strip() or None

    def health(self) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.server_url}/health",
            method="GET",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                self._require_content_type(response, "application/json")
                payload = json.loads(self._read_limited(response).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Health response must be a JSON object, got {type(payload)}")
        return payload

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        request.validate()
        body = dumps_payload(request)
        if len(body) > self.max_request_bytes:
            raise ValueError(f"Inference request is too large: {len(body)} > {self.max_request_bytes}")
        headers = self._headers()
        headers["Content-Type"] = CONTENT_TYPE
        http_request = urllib.request.Request(
            f"{self.server_url}/infer",
            data=body,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                self._require_content_type(response, CONTENT_TYPE)
                payload = loads_payload(self._read_limited(response))
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc

        if not isinstance(payload, InferenceResponse):
            raise ValueError(f"Expected InferenceResponse, got {type(payload)}")
        if payload.error:
            raise RuntimeError(payload.error)
        if payload.request_id != request.request_id:
            raise RuntimeError(
                f"Response request_id mismatch: {payload.request_id} != {request.request_id}"
            )
        if payload.mode != request.mode:
            raise RuntimeError(f"Response mode mismatch: {payload.mode!r} != {request.mode!r}")
        payload.validate()
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json, application/x-python-pickle"}
        if self.auth_token is not None:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _read_limited(self, response: Any) -> bytes:
        raw_length = response.headers.get("Content-Length")
        if raw_length is not None:
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ValueError("Server returned an invalid Content-Length") from exc
            if length > self.max_response_bytes:
                raise ValueError(
                    f"Server response is too large: {length} > {self.max_response_bytes}"
                )
        body = response.read(self.max_response_bytes + 1)
        if len(body) > self.max_response_bytes:
            raise ValueError(f"Server response exceeds {self.max_response_bytes} bytes")
        if raw_length is not None and len(body) != int(raw_length):
            raise ValueError(
                f"Server response ended early: expected {raw_length}, got {len(body)}"
            )
        return body

    @staticmethod
    def _require_content_type(response: Any, expected: str) -> None:
        media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if media_type != expected:
            raise ValueError(f"Unexpected response Content-Type: {media_type!r}; expected {expected!r}")

    def _http_error(self, exc: urllib.error.HTTPError) -> RuntimeError:
        content_type = exc.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        try:
            body = self._read_limited(exc)
            if content_type == "application/json":
                payload = json.loads(body.decode("utf-8"))
                message = payload.get("error") if isinstance(payload, dict) else payload
            else:
                message = body.decode("utf-8", errors="replace")
        except Exception as parse_error:
            message = f"unreadable error response: {parse_error}"
        return RuntimeError(f"Policy server HTTP {exc.code}: {message}")
