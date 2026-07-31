from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np

from .timed_observation import TimedObservation

RAW_STATE_DIM = 18


@dataclass(frozen=True, slots=True)
class TimestampAlignmentConfig:
    camera_keys: tuple[str, ...]
    source_clock_domain: str
    state_delay_s: float
    camera_delay_s: Mapping[str, float]
    readout_delay_s: Mapping[str, float]
    max_skew_s: float = 0.1
    history_window_s: float = 5.0
    history_max_samples: int = 512

    def __post_init__(self) -> None:
        camera_keys = tuple(self.camera_keys)
        if not camera_keys or len(set(camera_keys)) != len(camera_keys):
            raise ValueError("camera_keys must be non-empty and unique")
        if any(not key.startswith("observation.images.") for key in camera_keys):
            raise ValueError("camera_keys must contain camera observation keys")
        if not isinstance(self.source_clock_domain, str) or not self.source_clock_domain.strip():
            raise ValueError("source_clock_domain must be a non-empty string")
        state_delay_s = _finite_non_negative("state_delay_s", self.state_delay_s)
        max_skew_s = _finite_positive("max_skew_s", self.max_skew_s)
        history_window_s = _finite_positive("history_window_s", self.history_window_s)
        if (
            isinstance(self.history_max_samples, bool)
            or not isinstance(self.history_max_samples, int)
            or self.history_max_samples < 2
        ):
            raise ValueError("history_max_samples must be an integer >= 2")
        camera_delay = _delay_mapping("camera_delay_s", self.camera_delay_s, camera_keys)
        readout_delay = _delay_mapping("readout_delay_s", self.readout_delay_s, camera_keys)
        object.__setattr__(self, "camera_keys", camera_keys)
        object.__setattr__(self, "source_clock_domain", self.source_clock_domain.strip())
        object.__setattr__(self, "state_delay_s", state_delay_s)
        object.__setattr__(self, "camera_delay_s", MappingProxyType(camera_delay))
        object.__setattr__(self, "readout_delay_s", MappingProxyType(readout_delay))
        object.__setattr__(self, "max_skew_s", max_skew_s)
        object.__setattr__(self, "history_window_s", history_window_s)


@dataclass(frozen=True, slots=True)
class StateHistorySample:
    timestamp_s: float
    raw18: np.ndarray

    def __post_init__(self) -> None:
        timestamp_s = _finite_non_negative("timestamp_s", self.timestamp_s)
        raw18 = _raw18(self.raw18, label="raw18")
        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(self, "raw18", raw18)


@dataclass(frozen=True, slots=True)
class InterpolatedState:
    timestamp_s: float
    raw18: np.ndarray
    before_timestamp_s: float
    after_timestamp_s: float
    interpolation_ratio: float

    def __post_init__(self) -> None:
        timestamp_s = _finite_non_negative("timestamp_s", self.timestamp_s)
        before_s = _finite_non_negative("before_timestamp_s", self.before_timestamp_s)
        after_s = _finite_non_negative("after_timestamp_s", self.after_timestamp_s)
        ratio = _finite_non_negative("interpolation_ratio", self.interpolation_ratio)
        if before_s > timestamp_s or timestamp_s > after_s:
            raise ValueError("interpolation timestamps must satisfy before <= target <= after")
        if ratio > 1:
            raise ValueError("interpolation_ratio must be in 0..1")
        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(self, "raw18", _raw18(self.raw18, label="interpolated raw18"))
        object.__setattr__(self, "before_timestamp_s", before_s)
        object.__setattr__(self, "after_timestamp_s", after_s)
        object.__setattr__(self, "interpolation_ratio", ratio)


class Raw18StateHistory:
    def __init__(self, *, max_samples: int, window_s: float, max_skew_s: float) -> None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 2:
            raise ValueError("max_samples must be an integer >= 2")
        self.max_samples = max_samples
        self.window_s = _finite_positive("window_s", window_s)
        self.max_skew_s = _finite_positive("max_skew_s", max_skew_s)
        self._samples: deque[StateHistorySample] = deque(maxlen=max_samples)
        self._lock = threading.RLock()

    def append(self, sample: StateHistorySample) -> None:
        if not isinstance(sample, StateHistorySample):
            raise TypeError("sample must be StateHistorySample")
        with self._lock:
            if self._samples and sample.timestamp_s <= self._samples[-1].timestamp_s:
                raise ValueError(
                    "state source timestamp must advance strictly: "
                    f"{sample.timestamp_s} <= {self._samples[-1].timestamp_s}"
                )
            self._samples.append(sample)
            newest_s = sample.timestamp_s
            while self._samples and newest_s - self._samples[0].timestamp_s > self.window_s:
                self._samples.popleft()

    def interpolate(self, timestamp_s: float) -> InterpolatedState:
        target_s = _finite_non_negative("timestamp_s", timestamp_s)
        with self._lock:
            samples = tuple(self._samples)
        if not samples:
            raise LookupError("state history is empty")
        for sample in samples:
            if sample.timestamp_s == target_s:
                return InterpolatedState(
                    timestamp_s=target_s,
                    raw18=sample.raw18,
                    before_timestamp_s=target_s,
                    after_timestamp_s=target_s,
                    interpolation_ratio=0.0,
                )
        before = None
        after = None
        for sample in samples:
            if sample.timestamp_s < target_s:
                before = sample
                continue
            if sample.timestamp_s > target_s:
                after = sample
                break
        if before is None or after is None:
            raise LookupError(
                f"state history does not bracket timestamp {target_s}; "
                f"range={samples[0].timestamp_s}..{samples[-1].timestamp_s}"
            )
        before_skew = target_s - before.timestamp_s
        after_skew = after.timestamp_s - target_s
        if max(before_skew, after_skew) > self.max_skew_s:
            raise ValueError(
                f"state/image skew exceeds {self.max_skew_s}: "
                f"before={before_skew} after={after_skew}"
            )
        ratio = before_skew / (after.timestamp_s - before.timestamp_s)
        interpolated = before.raw18.astype(np.float64) + ratio * (
            after.raw18.astype(np.float64) - before.raw18.astype(np.float64)
        )
        return InterpolatedState(
            timestamp_s=target_s,
            raw18=interpolated.astype(np.float32),
            before_timestamp_s=before.timestamp_s,
            after_timestamp_s=after.timestamp_s,
            interpolation_ratio=ratio,
        )

    def snapshot(self) -> tuple[StateHistorySample, ...]:
        with self._lock:
            return tuple(self._samples)


@dataclass(frozen=True, slots=True)
class AlignmentShadowResult:
    camera_key: str
    target_timestamp_s: float
    aligned_raw18: np.ndarray
    current_raw18: np.ndarray
    before_timestamp_s: float
    after_timestamp_s: float
    interpolation_ratio: float
    max_abs_delta: float
    mean_abs_delta: float
    changed_policy_input: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.camera_key, str) or not self.camera_key.startswith("observation.images."):
            raise ValueError("camera_key must be a camera observation key")
        target_s = _finite_non_negative("target_timestamp_s", self.target_timestamp_s)
        max_delta = _finite_non_negative("max_abs_delta", self.max_abs_delta)
        mean_delta = _finite_non_negative("mean_abs_delta", self.mean_abs_delta)
        if self.changed_policy_input is not False:
            raise ValueError("Phase 5 alignment must remain shadow-only")
        object.__setattr__(self, "target_timestamp_s", target_s)
        object.__setattr__(self, "aligned_raw18", _raw18(self.aligned_raw18, label="aligned_raw18"))
        object.__setattr__(self, "current_raw18", _raw18(self.current_raw18, label="current_raw18"))
        object.__setattr__(self, "max_abs_delta", max_delta)
        object.__setattr__(self, "mean_abs_delta", mean_delta)

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "camera_key": self.camera_key,
            "target_timestamp_s": self.target_timestamp_s,
            "before_timestamp_s": self.before_timestamp_s,
            "after_timestamp_s": self.after_timestamp_s,
            "interpolation_ratio": self.interpolation_ratio,
            "max_abs_delta": self.max_abs_delta,
            "mean_abs_delta": self.mean_abs_delta,
            "changed_policy_input": False,
        }


class TimestampAlignmentShadow:
    """Bounded source-clock alignment observer that never mutates policy input."""

    def __init__(self, config: TimestampAlignmentConfig) -> None:
        self.config = config
        self.history = Raw18StateHistory(
            max_samples=config.history_max_samples,
            window_s=config.history_window_s,
            max_skew_s=config.max_skew_s,
        )
        self._results: deque[AlignmentShadowResult] = deque(maxlen=config.history_max_samples)
        self._lock = threading.RLock()
        self.observation_count = 0
        self.warmup_count = 0
        self.failure_count = 0

    def observe(self, observation: TimedObservation) -> tuple[AlignmentShadowResult, ...]:
        try:
            results = self._observe(observation)
        except Exception:
            with self._lock:
                self.failure_count += 1
            raise
        with self._lock:
            self.observation_count += 1
            if not results:
                self.warmup_count += 1
            self._results.extend(results)
        return results

    def _observe(self, observation: TimedObservation) -> tuple[AlignmentShadowResult, ...]:
        if not isinstance(observation, TimedObservation):
            raise TypeError("alignment shadow requires TimedObservation")
        observation.require_source_timestamps(camera_keys=self.config.camera_keys)
        state_source = observation.state_source_timestamp
        assert state_source is not None
        if state_source.clock_domain != self.config.source_clock_domain:
            raise ValueError(
                f"state clock domain differs: {state_source.clock_domain!r} != "
                f"{self.config.source_clock_domain!r}"
            )
        state_timestamp_s = state_source.timestamp_s - self.config.state_delay_s
        if state_timestamp_s < 0:
            raise ValueError("state delay correction produced a negative timestamp")
        current_raw18 = _raw18(observation.observation_frame.get("observation.state"), label="state")
        camera_targets: list[tuple[str, float]] = []
        for camera_key in self.config.camera_keys:
            camera_source = observation.camera_source_timestamps[camera_key]
            if camera_source.clock_domain != self.config.source_clock_domain:
                raise ValueError(
                    f"camera/state clock domains differ for {camera_key}: "
                    f"{camera_source.clock_domain!r} != {self.config.source_clock_domain!r}"
                )
            target_s = (
                camera_source.timestamp_s
                - self.config.camera_delay_s[camera_key]
                - self.config.readout_delay_s[camera_key]
            )
            if target_s < 0:
                raise ValueError(f"camera delay correction produced a negative timestamp for {camera_key}")
            camera_targets.append((camera_key, target_s))
        self.history.append(StateHistorySample(state_timestamp_s, current_raw18))
        if len(self.history.snapshot()) < 2:
            return ()
        results = []
        for camera_key, target_s in camera_targets:
            aligned = self.history.interpolate(target_s)
            delta = np.abs(aligned.raw18.astype(np.float64) - current_raw18.astype(np.float64))
            results.append(
                AlignmentShadowResult(
                    camera_key=camera_key,
                    target_timestamp_s=target_s,
                    aligned_raw18=aligned.raw18,
                    current_raw18=current_raw18,
                    before_timestamp_s=aligned.before_timestamp_s,
                    after_timestamp_s=aligned.after_timestamp_s,
                    interpolation_ratio=aligned.interpolation_ratio,
                    max_abs_delta=float(delta.max(initial=0.0)),
                    mean_abs_delta=float(delta.mean()),
                )
            )
        return tuple(results)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            results = tuple(self._results)
            observation_count = self.observation_count
            warmup_count = self.warmup_count
            failure_count = self.failure_count
        return {
            "mode": "shadow",
            "changed_policy_input": False,
            "observation_count": observation_count,
            "warmup_count": warmup_count,
            "failure_count": failure_count,
            "history_samples": len(self.history.snapshot()),
            "result_count": len(results),
            "latest": None if not results else results[-1].trace_metadata(),
        }


def _raw18(value: object, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (RAW_STATE_DIM,):
        raise ValueError(f"{label} must have shape (18,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN/Inf")
    copied = np.ascontiguousarray(array).copy()
    copied.setflags(write=False)
    return copied


def _delay_mapping(name: str, value: Mapping[str, float], camera_keys: tuple[str, ...]) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(camera_keys):
        raise ValueError(f"{name} must explicitly contain exactly {list(camera_keys)}")
    return {key: _finite_non_negative(f"{name}[{key}]", value[key]) for key in camera_keys}


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
    "AlignmentShadowResult",
    "InterpolatedState",
    "Raw18StateHistory",
    "StateHistorySample",
    "TimestampAlignmentConfig",
    "TimestampAlignmentShadow",
]
