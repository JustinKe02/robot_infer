from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from tk_infer.pi05.runtime.safety import ActionSafety

from .paired_trajectory import LEFT_FORCE_INDEX, RIGHT_FORCE_INDEX
from .temporal_optimizer import JOINT_DIM

RAW_DIM = 18
LEFT_GRIPPER_INDEX = 14
RIGHT_GRIPPER_INDEX = 16
P5_AUTHORIZATION_SCOPE = "p5_single_action"
DEFAULT_CHECKPOINT_STEP = 15900
DEFAULT_CHECKPOINT_FINGERPRINT = "9d6d37f6111a034209c9bdc2899423a3258cc35070cb8294194c9c594197b58a"
_BOUND_TOLERANCE = 1e-7


class P5SingleActionError(RuntimeError):
    """Raised when the software-only P5 single-action boundary fails closed."""


@dataclass(frozen=True, slots=True)
class P5SingleActionConfig:
    required_sender_ip: str = "192.168.1.81"
    required_robot: str = "robot1"
    state_max_age_s: float = 0.1
    max_initial_joint_delta_rad: float = 0.02
    gripper_width_min: float = 0.0
    gripper_width_max: float = 100.0
    preflight_deadline_s: float = 0.02
    delegate_deadline_s: float = 0.02
    checkpoint_step: int = DEFAULT_CHECKPOINT_STEP
    checkpoint_fingerprint: str = DEFAULT_CHECKPOINT_FINGERPRINT
    backend_label: str = "inference_mode"

    def __post_init__(self) -> None:
        for name in ("required_sender_ip", "required_robot", "backend_label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        for name in (
            "state_max_age_s",
            "max_initial_joint_delta_rad",
            "gripper_width_min",
            "gripper_width_max",
            "preflight_deadline_s",
            "delegate_deadline_s",
        ):
            object.__setattr__(self, name, _finite_number(name, getattr(self, name)))
        if self.state_max_age_s <= 0:
            raise ValueError("state_max_age_s must be positive")
        if self.max_initial_joint_delta_rad <= 0:
            raise ValueError("max_initial_joint_delta_rad must be positive")
        if self.preflight_deadline_s <= 0 or self.delegate_deadline_s <= 0:
            raise ValueError("P5 preflight and delegate deadlines must be positive")
        if self.gripper_width_min > self.gripper_width_max:
            raise ValueError("gripper_width_min must be <= gripper_width_max")
        if isinstance(self.checkpoint_step, bool) or not isinstance(self.checkpoint_step, int):
            raise ValueError("checkpoint_step must be an integer")
        if self.checkpoint_step <= 0:
            raise ValueError("checkpoint_step must be positive")
        if not _is_sha256(self.checkpoint_fingerprint):
            raise ValueError("checkpoint_fingerprint must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class P5StateSnapshot:
    raw18: NDArray[np.float32]
    sequence: int
    stamp_ns: int
    received_monotonic_s: float
    sender_ip: str
    robot: str
    sequence_advanced: bool = True

    def __post_init__(self) -> None:
        raw18 = _raw18_array(self.raw18, label="P5 state", require_force=False)
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("P5 state sequence must be a non-negative integer")
        if isinstance(self.stamp_ns, bool) or not isinstance(self.stamp_ns, int) or self.stamp_ns <= 0:
            raise ValueError("P5 state stamp_ns must be a positive integer")
        received = _finite_number("P5 state received_monotonic_s", self.received_monotonic_s)
        if received < 0:
            raise ValueError("P5 state received_monotonic_s must be non-negative")
        for name in ("sender_ip", "robot"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"P5 state {name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if not isinstance(self.sequence_advanced, bool):
            raise ValueError("P5 state sequence_advanced must be boolean")
        object.__setattr__(self, "raw18", raw18)
        object.__setattr__(self, "received_monotonic_s", received)


@dataclass(frozen=True, slots=True)
class P5InterlockSnapshot:
    authorization_id: str
    scope: str
    issued_monotonic_s: float
    expires_monotonic_s: float
    max_actions: int
    checkpoint_step: int
    checkpoint_fingerprint: str
    backend_label: str
    physical_emergency_stop_verified: bool
    workspace_clear: bool
    operator_present: bool
    robot_powered: bool

    def __post_init__(self) -> None:
        for name in ("authorization_id", "scope", "backend_label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"P5 interlock {name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        issued = _finite_number("issued_monotonic_s", self.issued_monotonic_s)
        expires = _finite_number("expires_monotonic_s", self.expires_monotonic_s)
        if issued < 0 or expires <= issued:
            raise ValueError("P5 interlock validity window must advance from a non-negative issue time")
        if isinstance(self.max_actions, bool) or not isinstance(self.max_actions, int):
            raise ValueError("P5 interlock max_actions must be an integer")
        if isinstance(self.checkpoint_step, bool) or not isinstance(self.checkpoint_step, int):
            raise ValueError("P5 interlock checkpoint_step must be an integer")
        if not _is_sha256(self.checkpoint_fingerprint):
            raise ValueError("P5 interlock checkpoint_fingerprint must be a lowercase SHA-256 hex digest")
        for name in (
            "physical_emergency_stop_verified",
            "workspace_clear",
            "operator_present",
            "robot_powered",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"P5 interlock {name} must be boolean")
        object.__setattr__(self, "issued_monotonic_s", issued)
        object.__setattr__(self, "expires_monotonic_s", expires)


@runtime_checkable
class P5StateSource(Protocol):
    def read_state(self) -> P5StateSnapshot: ...


@runtime_checkable
class P5InterlockSource(Protocol):
    def read_interlock(self) -> P5InterlockSnapshot: ...


@runtime_checkable
class InhibitableActionSink(Protocol):
    @property
    def hardware_access(self) -> bool: ...

    @property
    def command_transport(self) -> str | None: ...

    @property
    def armed_capability(self) -> bool: ...

    def write(self, action: Tensor) -> None: ...

    def inhibit(self, reason: str) -> None: ...


@dataclass(frozen=True, slots=True)
class P5ValidationReport:
    authorization_id: str
    state_sequence: int
    state_stamp_ns: int
    state_age_s: float
    max_initial_joint_delta_rad: float
    left_gripper_width: float
    right_gripper_width: float
    preflight_elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class P5SingleActionGuardedSink:
    """One-shot fail-closed wrapper with no concrete robot or transport implementation."""

    def __init__(
        self,
        *,
        delegate: InhibitableActionSink,
        state_source: P5StateSource,
        interlock_source: P5InterlockSource,
        config: P5SingleActionConfig | None = None,
        safety: ActionSafety | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(delegate, InhibitableActionSink):
            raise TypeError("P5 delegate must implement isolation metadata, write(), and inhibit()")
        if not isinstance(delegate.hardware_access, bool) or not isinstance(delegate.armed_capability, bool):
            raise TypeError("P5 delegate isolation metadata must be boolean")
        if delegate.hardware_access or delegate.command_transport is not None or delegate.armed_capability:
            raise ValueError("P5 software-readiness guard rejects hardware, command, or armed delegates")
        if not isinstance(state_source, P5StateSource):
            raise TypeError("P5 state_source must implement read_state()")
        if not isinstance(interlock_source, P5InterlockSource):
            raise TypeError("P5 interlock_source must implement read_interlock()")
        if not callable(clock):
            raise TypeError("P5 clock must be callable")
        self.delegate = delegate
        self.state_source = state_source
        self.interlock_source = interlock_source
        self.config = config or P5SingleActionConfig()
        self.safety = safety or ActionSafety()
        self._clock = clock
        self._lock = threading.RLock()
        self._latched_reason: str | None = None
        self._validation_attempts = 0
        self._delegate_write_attempts = 0
        self._delegate_write_successes = 0
        self._delegate_inhibit_attempts = 0
        self._delegate_inhibited = False
        self._delegate_inhibit_error: str | None = None
        self._delivery_state = "not_attempted"
        self._last_validation: P5ValidationReport | None = None
        self._last_delegate_elapsed_s: float | None = None

    @property
    def latched_reason(self) -> str | None:
        with self._lock:
            return self._latched_reason

    @property
    def stopped(self) -> bool:
        return self.latched_reason is not None

    def write(self, action: Tensor) -> None:
        with self._lock:
            if self._latched_reason is not None:
                raise P5SingleActionError(f"P5 single-action guard is stopped: {self._latched_reason}")
            self._validation_attempts += 1
            try:
                checked = self.safety.check_tensor(action).detach().to(device="cpu", dtype=torch.float32)
                preflight_started_s = _finite_number("P5 preflight start time", self._clock())
                state = self.state_source.read_state()
                interlock = self.interlock_source.read_interlock()
                now_s = _finite_number("P5 current monotonic time", self._clock())
                preflight_elapsed_s = _elapsed("P5 preflight", preflight_started_s, now_s)
                if preflight_elapsed_s > self.config.preflight_deadline_s:
                    raise TimeoutError(
                        f"P5 preflight exceeded {self.config.preflight_deadline_s:.6f}s deadline: "
                        f"{preflight_elapsed_s:.6f}s"
                    )
                validation = self._validate(
                    checked,
                    state,
                    interlock,
                    now_s=now_s,
                    preflight_elapsed_s=preflight_elapsed_s,
                )
            except Exception as exc:
                self._latch_and_inhibit(f"validation_failed: {type(exc).__name__}: {exc}")
                raise P5SingleActionError(str(self._latched_reason)) from exc

            self._last_validation = validation
            try:
                delegate_started_s = _finite_number("P5 delegate start time", self._clock())
            except Exception as exc:
                self._latch_and_inhibit(f"delegate_start_clock_failed: {type(exc).__name__}: {exc}")
                raise P5SingleActionError(str(self._latched_reason)) from exc
            self._delegate_write_attempts += 1
            try:
                self.delegate.write(checked.detach().clone())
            except Exception as exc:
                self._delivery_state = "unknown_after_delegate_failure"
                self._latch_and_inhibit(f"delegate_failed: {type(exc).__name__}: {exc}")
                raise P5SingleActionError(str(self._latched_reason)) from exc

            self._delegate_write_successes += 1
            try:
                delegate_finished_s = _finite_number("P5 delegate finish time", self._clock())
                delegate_elapsed_s = _elapsed("P5 delegate", delegate_started_s, delegate_finished_s)
            except Exception as exc:
                self._delivery_state = "delegate_returned_timing_failure"
                self._latch_and_inhibit(f"delegate_finish_clock_failed: {type(exc).__name__}: {exc}")
                raise P5SingleActionError(
                    "single action delegate returned but delivery timing failed; delivery may have occurred and "
                    f"no retry is permitted: {type(exc).__name__}: {exc}"
                ) from exc
            self._last_delegate_elapsed_s = delegate_elapsed_s
            deadline_missed = delegate_elapsed_s > self.config.delegate_deadline_s
            self._delivery_state = (
                "delegate_returned_after_deadline" if deadline_missed else "delegate_returned"
            )
            inhibit_error = self._latch_and_inhibit("single_action_budget_exhausted")
            if inhibit_error is not None:
                raise P5SingleActionError(
                    "single action delegate returned, but delegate inhibition failed; no retry is permitted: "
                    f"{inhibit_error}"
                )
            if deadline_missed:
                raise P5SingleActionError(
                    "single action delegate returned after its deadline; delivery may have occurred and no retry "
                    f"is permitted: {delegate_elapsed_s:.6f}s > {self.config.delegate_deadline_s:.6f}s"
                )

    def inhibit(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("P5 inhibit reason must be a non-empty string")
        with self._lock:
            inhibit_error = self._latch_and_inhibit(f"external_inhibit: {reason.strip()}")
            if inhibit_error is not None:
                raise P5SingleActionError(f"delegate inhibition failed: {inhibit_error}")

    def health(self) -> dict[str, Any]:
        with self._lock:
            validation = self._last_validation
            return {
                "phase": "P5_software_readiness",
                "delegate_hardware_access": self.delegate.hardware_access,
                "delegate_command_transport": self.delegate.command_transport,
                "delegate_armed_capability": self.delegate.armed_capability,
                "armed_launcher": False,
                "max_actions": 1,
                "stopped": self._latched_reason is not None,
                "latched_reason": self._latched_reason,
                "validation_attempts": self._validation_attempts,
                "delegate_write_attempts": self._delegate_write_attempts,
                "delegate_write_successes": self._delegate_write_successes,
                "delegate_inhibit_attempts": self._delegate_inhibit_attempts,
                "delegate_inhibited": self._delegate_inhibited,
                "delegate_inhibit_error": self._delegate_inhibit_error,
                "delivery_state": self._delivery_state,
                "last_delegate_elapsed_s": self._last_delegate_elapsed_s,
                "no_retry": self._latched_reason is not None,
                "config": asdict(self.config),
                "last_validation": None if validation is None else validation.to_dict(),
            }

    def _validate(
        self,
        action: Tensor,
        state: object,
        interlock: object,
        *,
        now_s: float,
        preflight_elapsed_s: float,
    ) -> P5ValidationReport:
        if not isinstance(state, P5StateSnapshot):
            raise TypeError("P5 state source returned an invalid snapshot")
        if not isinstance(interlock, P5InterlockSnapshot):
            raise TypeError("P5 interlock source returned an invalid snapshot")
        if now_s < 0:
            raise ValueError("P5 current monotonic time must be non-negative")
        state_age_s = now_s - state.received_monotonic_s
        if state_age_s < 0 or state_age_s > self.config.state_max_age_s:
            raise TimeoutError(f"P5 state age {state_age_s:.6f}s exceeds {self.config.state_max_age_s:.6f}s")
        if state.sender_ip != self.config.required_sender_ip:
            raise RuntimeError(
                f"P5 state sender {state.sender_ip!r} does not match {self.config.required_sender_ip!r}"
            )
        if state.robot != self.config.required_robot:
            raise RuntimeError(
                f"P5 state robot {state.robot!r} does not match {self.config.required_robot!r}"
            )
        if not state.sequence_advanced:
            raise TimeoutError("P5 state sequence did not advance")
        self._validate_interlock(interlock, now_s=now_s)
        state_tensor = torch.from_numpy(np.array(state.raw18, dtype=np.float32, copy=True))
        initial_delta = float(torch.max(torch.abs(action[:JOINT_DIM] - state_tensor[:JOINT_DIM])).item())
        if initial_delta > self.config.max_initial_joint_delta_rad + _BOUND_TOLERANCE:
            raise ValueError(
                f"P5 initial joint delta {initial_delta:.9f} exceeds "
                f"{self.config.max_initial_joint_delta_rad:.9f} rad"
            )
        left_gripper = float(action[LEFT_GRIPPER_INDEX])
        right_gripper = float(action[RIGHT_GRIPPER_INDEX])
        for label, value in (("left", left_gripper), ("right", right_gripper)):
            if not self.config.gripper_width_min <= value <= self.config.gripper_width_max:
                raise ValueError(
                    f"P5 {label} gripper width {value} is outside "
                    f"{self.config.gripper_width_min}..{self.config.gripper_width_max}"
                )
        if float(action[LEFT_FORCE_INDEX]) != 80.0 or float(action[RIGHT_FORCE_INDEX]) != 80.0:
            raise ValueError("P5 action force slots must be exactly 80")
        return P5ValidationReport(
            authorization_id=interlock.authorization_id,
            state_sequence=state.sequence,
            state_stamp_ns=state.stamp_ns,
            state_age_s=state_age_s,
            max_initial_joint_delta_rad=initial_delta,
            left_gripper_width=left_gripper,
            right_gripper_width=right_gripper,
            preflight_elapsed_s=preflight_elapsed_s,
        )

    def _validate_interlock(self, interlock: P5InterlockSnapshot, *, now_s: float) -> None:
        if interlock.scope != P5_AUTHORIZATION_SCOPE:
            raise RuntimeError(
                f"P5 interlock scope must be {P5_AUTHORIZATION_SCOPE!r}, got {interlock.scope!r}"
            )
        if not interlock.issued_monotonic_s <= now_s <= interlock.expires_monotonic_s:
            raise TimeoutError("P5 interlock authorization is not currently valid")
        if interlock.max_actions != 1:
            raise RuntimeError("P5 interlock must authorize exactly one action")
        if interlock.checkpoint_step != self.config.checkpoint_step:
            raise RuntimeError("P5 interlock checkpoint step does not match the locked profile")
        if interlock.checkpoint_fingerprint != self.config.checkpoint_fingerprint:
            raise RuntimeError("P5 interlock checkpoint fingerprint does not match the locked profile")
        if interlock.backend_label != self.config.backend_label:
            raise RuntimeError("P5 interlock backend does not match the evaluated backend")
        required = {
            "physical_emergency_stop_verified": interlock.physical_emergency_stop_verified,
            "workspace_clear": interlock.workspace_clear,
            "operator_present": interlock.operator_present,
            "robot_powered": interlock.robot_powered,
        }
        missing = [name for name, enabled in required.items() if not enabled]
        if missing:
            raise RuntimeError(f"P5 interlock confirmations are incomplete: {missing}")

    def _latch_and_inhibit(self, reason: str) -> str | None:
        if self._latched_reason is None:
            self._latched_reason = reason
        if self._delegate_inhibited or self._delegate_inhibit_error is not None:
            return self._delegate_inhibit_error
        self._delegate_inhibit_attempts += 1
        try:
            self.delegate.inhibit(self._latched_reason)
        except Exception as exc:
            self._delegate_inhibit_error = f"{type(exc).__name__}: {exc}"
            return self._delegate_inhibit_error
        self._delegate_inhibited = True
        return None


def _raw18_array(value: object, *, label: str, require_force: bool) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (RAW_DIM,):
        raise ValueError(f"{label} must have shape (18,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN/Inf")
    if require_force and (array[LEFT_FORCE_INDEX] != 80.0 or array[RIGHT_FORCE_INDEX] != 80.0):
        raise ValueError(f"{label} force slots must be exactly 80")
    output = np.ascontiguousarray(array).copy()
    output.setflags(write=False)
    return output


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _elapsed(label: str, started_s: float, finished_s: float) -> float:
    elapsed_s = finished_s - started_s
    if not math.isfinite(elapsed_s) or elapsed_s < 0:
        raise ValueError(f"{label} clock moved backwards or returned NaN/Inf")
    return elapsed_s


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "DEFAULT_CHECKPOINT_FINGERPRINT",
    "DEFAULT_CHECKPOINT_STEP",
    "InhibitableActionSink",
    "P5_AUTHORIZATION_SCOPE",
    "P5InterlockSnapshot",
    "P5InterlockSource",
    "P5SingleActionConfig",
    "P5SingleActionError",
    "P5SingleActionGuardedSink",
    "P5StateSnapshot",
    "P5StateSource",
    "P5ValidationReport",
]
