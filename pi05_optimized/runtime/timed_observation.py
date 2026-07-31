from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceTimestamp:
    """One source timestamp whose clock domain is explicit and never inferred."""

    timestamp_s: float
    clock_domain: str
    source: str

    def __post_init__(self) -> None:
        timestamp_s = _finite_non_negative("timestamp_s", self.timestamp_s)
        if not isinstance(self.clock_domain, str) or not self.clock_domain.strip():
            raise ValueError("clock_domain must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(self, "clock_domain", self.clock_domain.strip())
        object.__setattr__(self, "source", self.source.strip())


@dataclass(frozen=True, slots=True)
class TimedObservation:
    """Observation payload plus explicit source and local monotonic timing metadata.

    Source timestamps can originate from device, Unix, or another process clock.
    They are retained for later calibration but are not subtracted from the local
    process-monotonic timestamps in this type.
    """

    observation_frame: Mapping[str, Any]
    sequence_id: int
    receive_monotonic_s: float
    build_started_monotonic_s: float
    build_ready_monotonic_s: float
    local_clock_domain: str = "process_perf_counter"
    state_source_timestamp: SourceTimestamp | None = None
    camera_source_timestamps: Mapping[str, SourceTimestamp] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observation_frame, Mapping) or not self.observation_frame:
            raise ValueError("observation_frame must be a non-empty mapping")
        invalid_keys = [key for key in self.observation_frame if not isinstance(key, str) or not key]
        if invalid_keys:
            raise ValueError(f"observation_frame keys must be non-empty strings: {invalid_keys!r}")
        if (
            isinstance(self.sequence_id, bool)
            or not isinstance(self.sequence_id, int)
            or self.sequence_id < 0
        ):
            raise ValueError("sequence_id must be a non-negative integer")
        receive_s = _finite_non_negative("receive_monotonic_s", self.receive_monotonic_s)
        build_started_s = _finite_non_negative("build_started_monotonic_s", self.build_started_monotonic_s)
        build_ready_s = _finite_non_negative("build_ready_monotonic_s", self.build_ready_monotonic_s)
        if receive_s > build_started_s or build_started_s > build_ready_s:
            raise ValueError(
                "local observation timestamps must satisfy receive <= build_started <= build_ready"
            )
        if not isinstance(self.local_clock_domain, str) or not self.local_clock_domain.strip():
            raise ValueError("local_clock_domain must be a non-empty string")
        if self.state_source_timestamp is not None and not isinstance(
            self.state_source_timestamp, SourceTimestamp
        ):
            raise TypeError("state_source_timestamp must be SourceTimestamp or None")

        camera_sources = dict(self.camera_source_timestamps or {})
        for key, timestamp in camera_sources.items():
            if not isinstance(key, str) or not key.startswith("observation.images."):
                raise ValueError(f"camera source timestamp key is not a camera observation key: {key!r}")
            if not isinstance(timestamp, SourceTimestamp):
                raise TypeError(f"camera source timestamp for {key!r} must be SourceTimestamp")
            if key not in self.observation_frame:
                raise ValueError(f"camera source timestamp has no matching observation frame key: {key!r}")

        object.__setattr__(self, "observation_frame", MappingProxyType(dict(self.observation_frame)))
        object.__setattr__(self, "receive_monotonic_s", receive_s)
        object.__setattr__(self, "build_started_monotonic_s", build_started_s)
        object.__setattr__(self, "build_ready_monotonic_s", build_ready_s)
        object.__setattr__(self, "local_clock_domain", self.local_clock_domain.strip())
        object.__setattr__(self, "camera_source_timestamps", MappingProxyType(camera_sources))

    @property
    def build_latency_s(self) -> float:
        return self.build_ready_monotonic_s - self.build_started_monotonic_s

    @property
    def receive_to_ready_s(self) -> float:
        return self.build_ready_monotonic_s - self.receive_monotonic_s

    def require_source_timestamps(self, *, camera_keys: tuple[str, ...]) -> None:
        if self.state_source_timestamp is None:
            raise ValueError("strict timestamp mode requires a state source timestamp")
        missing = [key for key in camera_keys if key not in self.camera_source_timestamps]
        if missing:
            raise ValueError(f"strict timestamp mode is missing camera source timestamps: {missing}")

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "receive_monotonic_s": self.receive_monotonic_s,
            "build_started_monotonic_s": self.build_started_monotonic_s,
            "build_ready_monotonic_s": self.build_ready_monotonic_s,
            "local_clock_domain": self.local_clock_domain,
            "build_latency_s": self.build_latency_s,
            "receive_to_ready_s": self.receive_to_ready_s,
            "state_source": _source_metadata(self.state_source_timestamp),
            "camera_sources": {
                key: _source_metadata(value) for key, value in self.camera_source_timestamps.items()
            },
            "observation_keys": sorted(self.observation_frame),
        }


def _source_metadata(timestamp: SourceTimestamp | None) -> dict[str, Any] | None:
    if timestamp is None:
        return None
    return {
        "timestamp_s": timestamp.timestamp_s,
        "clock_domain": timestamp.clock_domain,
        "source": timestamp.source,
    }


def _finite_non_negative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


__all__ = ["SourceTimestamp", "TimedObservation"]
