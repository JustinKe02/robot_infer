from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from tk_infer.pi05.runtime.protocol import InferenceRequest
from tk_infer.pi05_optimized.constants import DEFAULT_TRACE_BACKUP_COUNT, DEFAULT_TRACE_MAX_BYTES

from .metrics import InferenceTimings

TRACE_SCHEMA_VERSION = 1
MAX_ERROR_MESSAGE_CHARS = 2000


@dataclass(frozen=True, slots=True)
class TraceWriterStats:
    written_events: int
    dropped_events: int
    rotation_count: int
    last_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "written_events": self.written_events,
            "dropped_events": self.dropped_events,
            "rotation_count": self.rotation_count,
            "last_error": self.last_error,
        }


@runtime_checkable
class InferenceTraceRecorder(Protocol):
    def record_inference(self, request: InferenceRequest, timings: InferenceTimings) -> None: ...

    def record_failure(
        self,
        request: InferenceRequest,
        *,
        durations_s: dict[str, float],
        error: BaseException,
    ) -> None: ...

    def stats(self) -> TraceWriterStats: ...


class JsonlTraceWriter:
    """Append-only, payload-free JSONL trace writer with complete-line locking."""

    def __init__(
        self,
        path: str | Path,
        *,
        strict: bool = False,
        fsync: bool = False,
        max_bytes: int = DEFAULT_TRACE_MAX_BYTES,
        backup_count: int = DEFAULT_TRACE_BACKUP_COUNT,
        wall_time_ns: Callable[[], int] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        clock_source: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.strict = bool(strict)
        self.fsync = bool(fsync)
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 256:
            raise ValueError("max_bytes must be an integer >= 256")
        if isinstance(backup_count, bool) or not isinstance(backup_count, int) or backup_count < 0:
            raise ValueError("backup_count must be a non-negative integer")
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._wall_time_ns = wall_time_ns or time.time_ns
        self._monotonic_clock = monotonic_clock or time.perf_counter
        if monotonic_clock is not None and clock_source is None:
            raise ValueError("clock_source is required when monotonic_clock is injected")
        selected_clock_source = clock_source or "process_perf_counter"
        if not isinstance(selected_clock_source, str) or not selected_clock_source.strip():
            raise ValueError("clock_source must be a non-empty string")
        self.clock_source = selected_clock_source.strip()
        self._lock = threading.Lock()
        self._written_events = 0
        self._dropped_events = 0
        self._rotation_count = 0
        self._last_error: str | None = None

    def record_inference(self, request: InferenceRequest, timings: InferenceTimings) -> None:
        try:
            payload = self._base_event(request) | {
                "event": "inference",
                "status": "ok",
                "durations_s": timings.to_dict(),
                "error_type": None,
                "error_message": None,
            }
        except Exception as exc:
            self._record_drop(exc)
            return
        self._write(payload)

    def record_failure(
        self,
        request: InferenceRequest,
        *,
        durations_s: dict[str, float],
        error: BaseException,
    ) -> None:
        try:
            validated_durations = {
                name: _finite_non_negative(f"durations_s[{name!r}]", value)
                for name, value in durations_s.items()
            }
            payload = self._base_event(request) | {
                "event": "failure",
                "status": "error",
                "durations_s": validated_durations,
                "error_type": type(error).__name__,
                "error_message": str(error)[:MAX_ERROR_MESSAGE_CHARS],
            }
        except Exception as exc:
            self._record_drop(exc)
            return
        self._write(payload)

    def stats(self) -> TraceWriterStats:
        with self._lock:
            return TraceWriterStats(
                written_events=self._written_events,
                dropped_events=self._dropped_events,
                rotation_count=self._rotation_count,
                last_error=self._last_error,
            )

    def _base_event(self, request: InferenceRequest) -> dict[str, Any]:
        wall_time_ns = self._wall_time_ns()
        if isinstance(wall_time_ns, bool) or not isinstance(wall_time_ns, int) or wall_time_ns < 0:
            raise ValueError("wall_time_ns must return a non-negative integer")
        monotonic_s = _finite_non_negative("monotonic_s", self._monotonic_clock())
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "event": None,
            "trace_wall_time_ns": wall_time_ns,
            "trace_monotonic_s": monotonic_s,
            "clock_source": self.clock_source,
            "request_id": request.request_id,
            "mode": request.mode,
            "observation_sequence_id": request.obs_sequence_id,
            "predicted_delay_steps": request.predicted_delay_steps,
            "execution_horizon": request.execution_horizon,
        }

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            encoded_bytes = len(encoded.encode("utf-8")) + 1
            if encoded_bytes > self.max_bytes:
                raise ValueError(
                    f"trace event is larger than max_bytes: event={encoded_bytes} max_bytes={self.max_bytes}"
                )
        except (TypeError, ValueError) as exc:
            self._record_drop(exc)
            return
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed(encoded_bytes)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(encoded)
                    stream.write("\n")
                    stream.flush()
                    if self.fsync:
                        os.fsync(stream.fileno())
                self._written_events += 1
                self._last_error = None
            except OSError as exc:
                self._record_drop_locked(exc)

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        current_bytes = self.path.stat().st_size if self.path.exists() else 0
        if current_bytes + incoming_bytes <= self.max_bytes:
            return
        if self.backup_count == 0:
            self.path.unlink(missing_ok=True)
            self._rotation_count += 1
            return
        for index in range(self.backup_count, 0, -1):
            destination = Path(f"{self.path}.{index}")
            source = self.path if index == 1 else Path(f"{self.path}.{index - 1}")
            destination.unlink(missing_ok=True)
            if source.exists():
                source.replace(destination)
        self._rotation_count += 1

    def _record_drop(self, error: BaseException) -> None:
        with self._lock:
            self._record_drop_locked(error)

    def _record_drop_locked(self, error: BaseException) -> None:
        self._dropped_events += 1
        self._last_error = f"{type(error).__name__}: {error}"[:MAX_ERROR_MESSAGE_CHARS]
        if self.strict:
            raise error


def _finite_non_negative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


__all__ = [
    "InferenceTraceRecorder",
    "JsonlTraceWriter",
    "DEFAULT_TRACE_BACKUP_COUNT",
    "DEFAULT_TRACE_MAX_BYTES",
    "MAX_ERROR_MESSAGE_CHARS",
    "TRACE_SCHEMA_VERSION",
    "TraceWriterStats",
]
