from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from numbers import Real
from typing import Any

from tk_infer.pi05.runtime.protocol import MAX_ACTION_CHUNK_STEPS, InferenceRequest, InferenceResponse

CLIENT_DISTRIBUTIONS = (
    "request_total_s",
    "server_reported_s",
    "model_reported_s",
    "queue_depth",
    "dropped_steps",
    "predicted_delay_steps",
    "sensor_period_s",
    "sensor_jitter_s",
    "actor_period_s",
    "actor_jitter_s",
)


@dataclass(frozen=True, slots=True)
class ClientDistributionSnapshot:
    count: int
    latest: float
    p50: float
    p95: float
    p99: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "latest": self.latest,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


@dataclass(frozen=True, slots=True)
class ClientTelemetrySnapshot:
    window_size: int
    request_count: int
    frame_count: int
    repeated_source_frames: int
    skipped_source_frames: int
    stale_chunks: int
    queue_empty_events: int
    last_observation_sequence_id: int | None
    distributions: dict[str, ClientDistributionSnapshot]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "request_count": self.request_count,
            "frame_count": self.frame_count,
            "repeated_source_frames": self.repeated_source_frames,
            "skipped_source_frames": self.skipped_source_frames,
            "stale_chunks": self.stale_chunks,
            "queue_empty_events": self.queue_empty_events,
            "last_observation_sequence_id": self.last_observation_sequence_id,
            "distributions": {
                name: distribution.to_dict() for name, distribution in self.distributions.items()
            },
        }


class ClientTelemetry:
    """Bounded client/control telemetry that is independent of robot I/O."""

    def __init__(self, *, window_size: int = 512) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        self.window_size = window_size
        self._samples = {name: deque(maxlen=window_size) for name in CLIENT_DISTRIBUTIONS}
        self._request_count = 0
        self._frame_count = 0
        self._repeated_source_frames = 0
        self._skipped_source_frames = 0
        self._stale_chunks = 0
        self._queue_empty_events = 0
        self._last_observation_sequence_id: int | None = None
        self._last_source_frame_id: int | str | None = None
        self._last_sensor_tick_s: float | None = None
        self._last_actor_tick_s: float | None = None
        self._lock = threading.RLock()

    def record_request(
        self,
        request: InferenceRequest,
        response: InferenceResponse,
        *,
        total_s: float,
    ) -> None:
        request.validate()
        response.validate()
        if response.request_id != request.request_id or response.mode != request.mode:
            raise ValueError("request and response identity must match before telemetry is recorded")
        values = {
            "request_total_s": _finite_non_negative("total_s", total_s),
            "server_reported_s": _finite_non_negative("server_latency_s", response.server_latency_s),
            "model_reported_s": _finite_non_negative("model_latency_s", response.model_latency_s),
            "predicted_delay_steps": float(request.predicted_delay_steps),
        }
        with self._lock:
            for name, value in values.items():
                self._samples[name].append(value)
            self._request_count += 1

    def record_queue(
        self,
        *,
        depth: int,
        dropped_steps: int,
        stale_chunk: bool,
    ) -> None:
        depth_value = _non_negative_integer("depth", depth)
        dropped_value = _non_negative_integer("dropped_steps", dropped_steps)
        if not isinstance(stale_chunk, bool):
            raise ValueError("stale_chunk must be boolean")
        with self._lock:
            self._samples["queue_depth"].append(float(depth_value))
            self._samples["dropped_steps"].append(float(dropped_value))
            if stale_chunk:
                self._stale_chunks += 1

    def record_frame(self, *, observation_sequence_id: int, source_frame_id: int | str | None) -> None:
        sequence_id = _non_negative_integer("observation_sequence_id", observation_sequence_id)
        if source_frame_id is not None and (
            isinstance(source_frame_id, bool) or not isinstance(source_frame_id, int | str)
        ):
            raise ValueError("source_frame_id must be an integer, string, or None")
        with self._lock:
            if (
                self._last_observation_sequence_id is not None
                and sequence_id <= self._last_observation_sequence_id
            ):
                raise ValueError("observation_sequence_id must increase strictly")
            if source_frame_id is not None and source_frame_id == self._last_source_frame_id:
                self._repeated_source_frames += 1
            if (
                isinstance(source_frame_id, int)
                and isinstance(self._last_source_frame_id, int)
                and source_frame_id > self._last_source_frame_id + 1
            ):
                self._skipped_source_frames += source_frame_id - self._last_source_frame_id - 1
            self._last_observation_sequence_id = sequence_id
            self._last_source_frame_id = source_frame_id
            self._frame_count += 1

    def record_sensor_tick(self, *, timestamp_s: float, target_period_s: float) -> None:
        self._record_tick(
            timestamp_s=timestamp_s,
            target_period_s=target_period_s,
            previous_attribute="_last_sensor_tick_s",
            period_distribution="sensor_period_s",
            jitter_distribution="sensor_jitter_s",
        )

    def record_actor_tick(self, *, timestamp_s: float, target_period_s: float) -> None:
        self._record_tick(
            timestamp_s=timestamp_s,
            target_period_s=target_period_s,
            previous_attribute="_last_actor_tick_s",
            period_distribution="actor_period_s",
            jitter_distribution="actor_jitter_s",
        )

    def record_queue_empty(self) -> None:
        with self._lock:
            self._queue_empty_events += 1

    def predicted_delay_steps(
        self,
        *,
        control_period_s: float,
        max_steps: int = MAX_ACTION_CHUNK_STEPS,
    ) -> int:
        period = _finite_positive("control_period_s", control_period_s)
        maximum = _non_negative_integer("max_steps", max_steps)
        with self._lock:
            values = tuple(self._samples["request_total_s"])
        if not values:
            return 0
        p95_s = _distribution(values).p95
        return min(maximum, int(math.ceil(p95_s / period)))

    def snapshot(self) -> ClientTelemetrySnapshot:
        with self._lock:
            samples = {name: tuple(values) for name, values in self._samples.items()}
            return ClientTelemetrySnapshot(
                window_size=self.window_size,
                request_count=self._request_count,
                frame_count=self._frame_count,
                repeated_source_frames=self._repeated_source_frames,
                skipped_source_frames=self._skipped_source_frames,
                stale_chunks=self._stale_chunks,
                queue_empty_events=self._queue_empty_events,
                last_observation_sequence_id=self._last_observation_sequence_id,
                distributions={name: _distribution(values) for name, values in samples.items()},
            )

    def _record_tick(
        self,
        *,
        timestamp_s: float,
        target_period_s: float,
        previous_attribute: str,
        period_distribution: str,
        jitter_distribution: str,
    ) -> None:
        timestamp = _finite_non_negative("timestamp_s", timestamp_s)
        target = _finite_positive("target_period_s", target_period_s)
        with self._lock:
            previous = getattr(self, previous_attribute)
            if previous is not None:
                period = timestamp - previous
                if period <= 0:
                    raise ValueError("control-loop tick timestamps must increase strictly")
                self._samples[period_distribution].append(period)
                self._samples[jitter_distribution].append(abs(period - target))
            setattr(self, previous_attribute, timestamp)


def _distribution(values: tuple[float, ...]) -> ClientDistributionSnapshot:
    if not values:
        return ClientDistributionSnapshot(0, 0.0, 0.0, 0.0, 0.0)
    ordered = sorted(values)
    return ClientDistributionSnapshot(
        count=len(values),
        latest=values[-1],
        p50=_percentile(ordered, 0.50),
        p95=_percentile(ordered, 0.95),
        p99=_percentile(ordered, 0.99),
    )


def _percentile(ordered: list[float], quantile: float) -> float:
    position = quantile * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(ordered[lower_index])
    weight = position - lower_index
    return float(ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight)


def _non_negative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_non_negative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


def _finite_positive(name: str, value: object) -> float:
    converted = _finite_non_negative(name, value)
    if converted <= 0:
        raise ValueError(f"{name} must be positive")
    return converted


__all__ = [
    "CLIENT_DISTRIBUTIONS",
    "ClientDistributionSnapshot",
    "ClientTelemetry",
    "ClientTelemetrySnapshot",
]
