from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from tk_infer.pi05.runtime.protocol import InferenceRequest, InferenceResponse


@runtime_checkable
class PolicyBackend(Protocol):
    """Backend boundary used by the optimized service."""

    @property
    def name(self) -> str: ...

    def health(self) -> dict[str, Any]: ...

    def infer(self, request: InferenceRequest) -> InferenceResponse: ...


__all__ = ["PolicyBackend"]
