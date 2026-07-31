from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from numbers import Real
from typing import Any

LATENCY_STAGES = (
    "total_s",
    "lock_wait_s",
    "backend_s",
    "trajectory_s",
    "response_s",
    "backend_reported_server_s",
    "backend_reported_model_s",
)


@dataclass(frozen=True, slots=True)
class InferenceTimings:
    total_s: float
    lock_wait_s: float
    backend_s: float
    trajectory_s: float
    response_s: float
    backend_reported_server_s: float
    backend_reported_model_s: float

    def __post_init__(self) -> None:
        for name in LATENCY_STAGES:
            object.__setattr__(self, name, _finite_non_negative(name, getattr(self, name)))

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in LATENCY_STAGES}


@dataclass(frozen=True, slots=True)
class DistributionSnapshot:
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
class MetricsSnapshot:
    window_size: int
    success_count: int
    failure_count: int
    stages: dict[str, DistributionSnapshot]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "stages": {name: snapshot.to_dict() for name, snapshot in self.stages.items()},
        }


class InferenceMetrics:
    """Thread-safe bounded latency distributions for the optimized service."""

    def __init__(self, *, window_size: int = 512) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        self.window_size = window_size
        self._samples = {name: deque(maxlen=window_size) for name in LATENCY_STAGES}
        self._success_count = 0
        self._failure_count = 0
        self._lock = threading.RLock()

    def record_success(self, timings: InferenceTimings) -> None:
        if not isinstance(timings, InferenceTimings):
            raise TypeError(f"timings must be InferenceTimings, got {type(timings)}")
        with self._lock:
            for name, value in timings.to_dict().items():
                self._samples[name].append(value)
            self._success_count += 1

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            samples = {name: tuple(values) for name, values in self._samples.items()}
            success_count = self._success_count
            failure_count = self._failure_count
        return MetricsSnapshot(
            window_size=self.window_size,
            success_count=success_count,
            failure_count=failure_count,
            stages={name: _distribution(values) for name, values in samples.items()},
        )


def _distribution(values: tuple[float, ...]) -> DistributionSnapshot:
    if not values:
        return DistributionSnapshot(count=0, latest=0.0, p50=0.0, p95=0.0, p99=0.0)
    ordered = sorted(values)
    return DistributionSnapshot(
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


def _finite_non_negative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


__all__ = [
    "DistributionSnapshot",
    "InferenceMetrics",
    "InferenceTimings",
    "LATENCY_STAGES",
    "MetricsSnapshot",
]
