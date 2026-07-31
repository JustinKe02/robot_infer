from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .paired_trajectory import COMMAND_FORCE, LEFT_FORCE_INDEX, RIGHT_FORCE_INDEX
from .temporal_optimizer import JOINT_DIM, optional_qp_dependency_status, require_qp_dependencies

RAW_DIM = 18
_BOUND_TOLERANCE = 1e-7


class LocalTrackerError(RuntimeError):
    """Base fail-closed tracker error."""


class LocalTrackerDeadlineError(LocalTrackerError):
    """Raised when tracker or MPC work exceeds its deadline."""


@dataclass(frozen=True, slots=True)
class LocalTrackerConfig:
    control_period_s: float = 0.05
    max_joint_step_rad: float = 0.02
    lag_time_constant_s: float = 0.15
    lag_innovation_gain: float = 0.5
    history_window_s: float = 5.0
    history_max_samples: int = 512
    contact_slowdown_gain: float = 1.0
    min_contact_slowdown_factor: float = 0.25
    tracker_deadline_s: float = 0.01
    mpc_enabled: bool = False
    tracker_replay_passed: bool = False
    mpc_deadline_s: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "control_period_s",
            "max_joint_step_rad",
            "lag_time_constant_s",
            "history_window_s",
            "tracker_deadline_s",
            "mpc_deadline_s",
        ):
            object.__setattr__(self, name, _finite_positive(name, getattr(self, name)))
        for name in ("lag_innovation_gain", "contact_slowdown_gain"):
            value = _finite_non_negative(name, getattr(self, name))
            object.__setattr__(self, name, value)
        slowdown = _finite_positive("min_contact_slowdown_factor", self.min_contact_slowdown_factor)
        if slowdown > 1.0:
            raise ValueError("min_contact_slowdown_factor must be <= 1.0")
        object.__setattr__(self, "min_contact_slowdown_factor", slowdown)
        if not 0.0 <= self.lag_innovation_gain <= 1.0:
            raise ValueError("lag_innovation_gain must be in 0..1")
        if (
            isinstance(self.history_max_samples, bool)
            or not isinstance(self.history_max_samples, int)
            or self.history_max_samples < 2
        ):
            raise ValueError("history_max_samples must be an integer >= 2")
        if not isinstance(self.mpc_enabled, bool) or not isinstance(self.tracker_replay_passed, bool):
            raise ValueError("mpc_enabled and tracker_replay_passed must be boolean")
        if self.mpc_enabled and not self.tracker_replay_passed:
            raise ValueError("MPC requires tracker_replay_passed=true")


@dataclass(frozen=True, slots=True)
class TrackerStateSample:
    timestamp_s: float
    raw18: NDArray[np.float32]

    def __post_init__(self) -> None:
        timestamp_s = _finite_non_negative("timestamp_s", self.timestamp_s)
        raw18 = _raw18(self.raw18, label="tracker state")
        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(self, "raw18", raw18)


class TrackerStateHistory:
    def __init__(self, *, max_samples: int, window_s: float) -> None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 2:
            raise ValueError("max_samples must be an integer >= 2")
        self.max_samples = max_samples
        self.window_s = _finite_positive("window_s", window_s)
        self._samples: deque[TrackerStateSample] = deque(maxlen=max_samples)
        self._lock = threading.RLock()

    def append(self, sample: TrackerStateSample) -> None:
        if not isinstance(sample, TrackerStateSample):
            raise TypeError("sample must be TrackerStateSample")
        with self._lock:
            if self._samples and sample.timestamp_s <= self._samples[-1].timestamp_s:
                raise LocalTrackerError(
                    "tracker state timestamp must advance strictly: "
                    f"{sample.timestamp_s} <= {self._samples[-1].timestamp_s}"
                )
            self._samples.append(sample)
            cutoff = sample.timestamp_s - self.window_s
            while len(self._samples) > 1 and self._samples[0].timestamp_s < cutoff:
                self._samples.popleft()

    def snapshot(self) -> tuple[TrackerStateSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()


@dataclass(frozen=True, slots=True)
class LagEstimate:
    estimated_joints: NDArray[np.float32]
    innovation: NDArray[np.float32]
    max_abs_innovation: float
    alpha: float


class FirstOrderLagEstimator:
    def __init__(self, *, time_constant_s: float, innovation_gain: float) -> None:
        self.time_constant_s = _finite_positive("time_constant_s", time_constant_s)
        self.innovation_gain = _finite_non_negative("innovation_gain", innovation_gain)
        if self.innovation_gain > 1.0:
            raise ValueError("innovation_gain must be <= 1")
        self._estimate: NDArray[np.float64] | None = None
        self._previous_command: NDArray[np.float64] | None = None

    def update(
        self,
        *,
        command_joints: object,
        observed_joints: object,
        dt_s: float,
    ) -> LagEstimate:
        command = _joints(command_joints, label="lag command")
        observed = _joints(observed_joints, label="lag observation")
        dt_s = _finite_positive("dt_s", dt_s)
        alpha = 1.0 - math.exp(-dt_s / self.time_constant_s)
        if self._estimate is None or self._previous_command is None:
            predicted = observed.astype(np.float64)
        else:
            predicted = self._estimate + alpha * (self._previous_command - self._estimate)
        innovation = observed.astype(np.float64) - predicted
        estimate = predicted + self.innovation_gain * innovation
        if not np.isfinite(estimate).all() or not np.isfinite(innovation).all():
            raise LocalTrackerError("lag estimator produced NaN/Inf")
        self._estimate = estimate
        self._previous_command = command.astype(np.float64)
        estimated_output = np.ascontiguousarray(estimate.astype(np.float32))
        innovation_output = np.ascontiguousarray(innovation.astype(np.float32))
        estimated_output.setflags(write=False)
        innovation_output.setflags(write=False)
        return LagEstimate(
            estimated_joints=estimated_output,
            innovation=innovation_output,
            max_abs_innovation=float(np.max(np.abs(innovation), initial=0.0)),
            alpha=alpha,
        )

    def reset(self) -> None:
        self._estimate = None
        self._previous_command = None


@runtime_checkable
class MPCSolver(Protocol):
    @property
    def name(self) -> str: ...

    def solve(
        self,
        *,
        rate_limited_action: Tensor,
        observed_state: Tensor,
        control_period_s: float,
        deadline_s: float,
    ) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class LocalTrackerReport:
    timestamp_s: float
    dt_s: float
    reference: str
    requested_max_joint_delta_rad: float
    output_max_joint_delta_rad: float
    rate_limit_scale: float
    contact_innovation: float
    contact_slowdown_factor: float
    contact_used_as_safety: bool
    lag_alpha: float
    lag_max_abs_innovation: float
    history_samples: int
    mpc_enabled: bool
    mpc_solver: str | None
    tracker_elapsed_s: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LocalActionTracker:
    def __init__(
        self,
        config: LocalTrackerConfig | None = None,
        *,
        mpc_solver: MPCSolver | None = None,
        dependency_check: Callable[[], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or LocalTrackerConfig()
        self._clock = clock or time.perf_counter
        self.history = TrackerStateHistory(
            max_samples=self.config.history_max_samples,
            window_s=self.config.history_window_s,
        )
        self.lag_estimator = FirstOrderLagEstimator(
            time_constant_s=self.config.lag_time_constant_s,
            innovation_gain=self.config.lag_innovation_gain,
        )
        if self.config.mpc_enabled:
            (dependency_check or require_qp_dependencies)()
            if mpc_solver is None or not isinstance(mpc_solver, MPCSolver):
                raise LocalTrackerError("MPC enabled but no valid MPCSolver was supplied")
        elif mpc_solver is not None:
            raise ValueError("mpc_solver requires mpc_enabled=true")
        self.mpc_solver = mpc_solver
        self._last_command: Tensor | None = None
        self._last_timestamp_s: float | None = None
        self._last_report: LocalTrackerReport | None = None
        self._reset_count = 0
        self._last_reset_reason: str | None = None
        self._lock = threading.RLock()

    @property
    def last_report(self) -> LocalTrackerReport | None:
        with self._lock:
            return self._last_report

    def health(self) -> dict[str, object]:
        report = self.last_report
        with self._lock:
            reset_count = self._reset_count
            last_reset_reason = self._last_reset_reason
        return {
            "enabled": True,
            "phase": 7,
            "history_samples": len(self.history.snapshot()),
            "history_capacity": self.config.history_max_samples,
            "max_joint_step_rad": self.config.max_joint_step_rad,
            "lag_time_constant_s": self.config.lag_time_constant_s,
            "contact_innovation_role": "slowdown_only_not_safety",
            "mpc_enabled": self.config.mpc_enabled,
            "mpc_solver": None if self.mpc_solver is None else self.mpc_solver.name,
            "optional_qp_dependencies": optional_qp_dependency_status(),
            "reset_count": reset_count,
            "last_reset_reason": last_reset_reason,
            "last_report": None if report is None else report.to_dict(),
        }

    def track(
        self,
        *,
        requested_action: object,
        observed_state: object,
        timestamp_s: float,
        contact_innovation: float = 0.0,
    ) -> Tensor:
        started_s = self._clock()
        timestamp_s = _finite_non_negative("timestamp_s", timestamp_s)
        requested = _action_tensor(requested_action, label="requested_action")
        observed = _action_tensor(observed_state, label="observed_state", require_force=False)
        contact_innovation = _finite_non_negative("contact_innovation", contact_innovation)
        self.history.append(TrackerStateSample(timestamp_s, observed.numpy()))
        if self._last_timestamp_s is None:
            dt_s = self.config.control_period_s
            reference = observed
            reference_name = "observed_state"
        else:
            dt_s = timestamp_s - self._last_timestamp_s
            if not math.isfinite(dt_s) or dt_s <= 0:
                raise LocalTrackerError("tracker timestamp must advance strictly")
            reference = self._last_command
            if reference is None:
                raise LocalTrackerError("tracker lost its previous command state")
            reference_name = "previous_command"
        delta = requested[:JOINT_DIM] - reference[:JOINT_DIM]
        requested_max_delta = float(torch.max(torch.abs(delta)).item())
        slowdown = max(
            self.config.min_contact_slowdown_factor,
            1.0 / (1.0 + self.config.contact_slowdown_gain * contact_innovation),
        )
        allowed_step = self.config.max_joint_step_rad * slowdown
        rate_scale = 1.0 if requested_max_delta <= allowed_step else allowed_step / requested_max_delta
        tracked = requested.detach().clone()
        tracked[:JOINT_DIM] = reference[:JOINT_DIM] + delta * rate_scale
        tracked[LEFT_FORCE_INDEX] = COMMAND_FORCE
        tracked[RIGHT_FORCE_INDEX] = COMMAND_FORCE
        lag = self.lag_estimator.update(
            command_joints=tracked[:JOINT_DIM].numpy(),
            observed_joints=observed[:JOINT_DIM].numpy(),
            dt_s=dt_s,
        )
        if self.config.mpc_enabled:
            tracked = self._run_mpc(tracked, observed)
        _validate_tracked_action(tracked, reference, allowed_step)
        elapsed_s = self._clock() - started_s
        _check_deadline("tracker", elapsed_s, self.config.tracker_deadline_s)
        output_max_delta = float(
            torch.max(torch.abs(tracked[:JOINT_DIM] - reference[:JOINT_DIM])).item()
        )
        report = LocalTrackerReport(
            timestamp_s=timestamp_s,
            dt_s=dt_s,
            reference=reference_name,
            requested_max_joint_delta_rad=requested_max_delta,
            output_max_joint_delta_rad=output_max_delta,
            rate_limit_scale=rate_scale,
            contact_innovation=contact_innovation,
            contact_slowdown_factor=slowdown,
            contact_used_as_safety=False,
            lag_alpha=lag.alpha,
            lag_max_abs_innovation=lag.max_abs_innovation,
            history_samples=len(self.history.snapshot()),
            mpc_enabled=self.config.mpc_enabled,
            mpc_solver=None if self.mpc_solver is None else self.mpc_solver.name,
            tracker_elapsed_s=elapsed_s,
        )
        with self._lock:
            self._last_command = tracked.detach().clone()
            self._last_timestamp_s = timestamp_s
            self._last_report = report
        return tracked.detach().clone()

    def reset(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("tracker reset reason must be a non-empty string")
        self.history.clear()
        self.lag_estimator.reset()
        with self._lock:
            self._last_command = None
            self._last_timestamp_s = None
            self._last_report = None
            self._reset_count += 1
            self._last_reset_reason = reason.strip()

    def _run_mpc(self, rate_limited: Tensor, observed: Tensor) -> Tensor:
        if self.mpc_solver is None:
            raise LocalTrackerError("MPC enabled without a solver")
        started_s = self._clock()
        try:
            output = self.mpc_solver.solve(
                rate_limited_action=rate_limited.detach().clone(),
                observed_state=observed.detach().clone(),
                control_period_s=self.config.control_period_s,
                deadline_s=self.config.mpc_deadline_s,
            )
        except Exception as exc:
            raise LocalTrackerError(
                f"MPC solver {self.mpc_solver.name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        elapsed_s = self._clock() - started_s
        _check_deadline("MPC", elapsed_s, self.config.mpc_deadline_s)
        return _action_tensor(output, label="MPC output")


def _validate_tracked_action(output: Tensor, reference: Tensor, allowed_step: float) -> None:
    checked = _action_tensor(output, label="tracked action")
    measured = float(torch.max(torch.abs(checked[:JOINT_DIM] - reference[:JOINT_DIM])).item())
    if measured > allowed_step + _BOUND_TOLERANCE:
        raise LocalTrackerError(
            f"tracked joint step {measured:.9f} exceeds allowed {allowed_step:.9f} rad"
        )


def _action_tensor(value: object, *, label: str, require_force: bool = True) -> Tensor:
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
    if tensor.shape != (RAW_DIM,):
        raise LocalTrackerError(f"{label} must have shape (18,), got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise LocalTrackerError(f"{label} contains NaN/Inf")
    if require_force and (float(tensor[LEFT_FORCE_INDEX]) != 80.0 or float(tensor[RIGHT_FORCE_INDEX]) != 80.0):
        raise LocalTrackerError(f"{label} force slots must be exactly 80")
    return tensor.detach().clone()


def _raw18(value: object, *, label: str) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (RAW_DIM,):
        raise ValueError(f"{label} must have shape (18,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN/Inf")
    output = np.ascontiguousarray(array).copy()
    output.setflags(write=False)
    return output


def _joints(value: object, *, label: str) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (JOINT_DIM,):
        raise LocalTrackerError(f"{label} must have shape ({JOINT_DIM},), got {array.shape}")
    if not np.isfinite(array).all():
        raise LocalTrackerError(f"{label} contains NaN/Inf")
    return np.ascontiguousarray(array)


def _check_deadline(label: str, elapsed_s: float, deadline_s: float) -> None:
    if not math.isfinite(elapsed_s) or elapsed_s < 0:
        raise LocalTrackerError(f"{label} clock moved backwards or returned NaN/Inf")
    if elapsed_s > deadline_s:
        raise LocalTrackerDeadlineError(
            f"{label} exceeded {deadline_s:.6f}s deadline: {elapsed_s:.6f}s"
        )


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
    "FirstOrderLagEstimator",
    "LagEstimate",
    "LocalActionTracker",
    "LocalTrackerConfig",
    "LocalTrackerDeadlineError",
    "LocalTrackerError",
    "LocalTrackerReport",
    "MPCSolver",
    "TrackerStateHistory",
    "TrackerStateSample",
]
