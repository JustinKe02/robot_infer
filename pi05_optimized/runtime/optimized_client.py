from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import torch
from torch import Tensor

from tk_infer.pi05.runtime.action_queue import ActionChunk, ActionChunkQueue, select_single_step
from tk_infer.pi05.runtime.protocol import MAX_ACTION_CHUNK_STEPS, InferenceRequest, InferenceResponse
from tk_infer.pi05.runtime.safety import ActionSafety

from .client_telemetry import ClientTelemetry
from .local_tracker import LocalActionTracker
from .timed_observation import TimedObservation
from .timestamp_alignment import TimestampAlignmentShadow

ClientMode: TypeAlias = Literal["single_step", "rtc"]


@dataclass(frozen=True, slots=True)
class OptimizedClientConfig:
    task: str
    mode: ClientMode = "rtc"
    robot_type: str = "jz_robot_pin_timed"
    control_hz: float = 20.0
    execution_horizon: int = 10
    max_consecutive_stale_chunks: int = 3
    max_consecutive_repeated_source_frames: int = 1
    strict_source_timestamps: bool = False
    required_camera_keys: tuple[str, ...] = ()
    clock_domain: str = "process_perf_counter"
    local_tracker_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be a non-empty string")
        if self.mode not in {"single_step", "rtc"}:
            raise ValueError(f"mode must be single_step or rtc, got {self.mode!r}")
        if self.robot_type != "jz_robot_pin_timed":
            raise ValueError("robot_type must be jz_robot_pin_timed")
        control_hz = _finite_positive("control_hz", self.control_hz)
        if (
            isinstance(self.execution_horizon, bool)
            or not isinstance(self.execution_horizon, int)
            or not 1 <= self.execution_horizon <= MAX_ACTION_CHUNK_STEPS
        ):
            raise ValueError(f"execution_horizon must be in 1..{MAX_ACTION_CHUNK_STEPS}")
        if (
            isinstance(self.max_consecutive_stale_chunks, bool)
            or not isinstance(self.max_consecutive_stale_chunks, int)
            or self.max_consecutive_stale_chunks <= 0
        ):
            raise ValueError("max_consecutive_stale_chunks must be a positive integer")
        if (
            isinstance(self.max_consecutive_repeated_source_frames, bool)
            or not isinstance(self.max_consecutive_repeated_source_frames, int)
            or self.max_consecutive_repeated_source_frames <= 0
        ):
            raise ValueError("max_consecutive_repeated_source_frames must be a positive integer")
        if not isinstance(self.strict_source_timestamps, bool):
            raise ValueError("strict_source_timestamps must be boolean")
        if any(
            not isinstance(key, str) or not key.startswith("observation.images.")
            for key in self.required_camera_keys
        ):
            raise ValueError("required_camera_keys must contain camera observation keys")
        if not isinstance(self.clock_domain, str) or not self.clock_domain.strip():
            raise ValueError("clock_domain must be a non-empty string")
        if not isinstance(self.local_tracker_enabled, bool):
            raise ValueError("local_tracker_enabled must be boolean")
        object.__setattr__(self, "task", self.task.strip())
        object.__setattr__(self, "control_hz", control_hz)
        object.__setattr__(self, "required_camera_keys", tuple(self.required_camera_keys))
        object.__setattr__(self, "clock_domain", self.clock_domain.strip())

    @property
    def control_period_s(self) -> float:
        return 1.0 / self.control_hz


@runtime_checkable
class ObservationSource(Protocol):
    def read(self) -> TimedObservation: ...


@runtime_checkable
class PolicyClient(Protocol):
    def infer(self, request: InferenceRequest) -> InferenceResponse: ...


@runtime_checkable
class ActionSink(Protocol):
    def write(self, action: Tensor) -> None: ...


@dataclass(frozen=True, slots=True)
class ClientCycleResult:
    request_id: int
    observation_sequence_id: int
    predicted_delay_steps: int
    dropped_steps: int
    queue_depth: int
    action_written: bool
    fully_stale: bool
    tracker_applied: bool


class ClientStopState:
    def __init__(self) -> None:
        self._reason: str | None = None
        self._lock = threading.RLock()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    @property
    def stopped(self) -> bool:
        return self.reason is not None

    def request_stop(self, reason: str) -> str:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("stop reason must be a non-empty string")
        with self._lock:
            if self._reason is None:
                self._reason = reason.strip()
            return self._reason

    def raise_if_stopped(self) -> None:
        reason = self.reason
        if reason is not None:
            raise RuntimeError(f"optimized client is stopped: {reason}")

    def run_if_active(self, operation: Callable[[], None]) -> None:
        """Serialize the final stop check with one externally visible operation."""

        with self._lock:
            if self._reason is not None:
                raise RuntimeError(f"optimized client is stopped: {self._reason}")
            operation()


class OptimizedClient:
    """Synchronous, dependency-injected Phase 1 client with no hardware adapter."""

    def __init__(
        self,
        *,
        config: OptimizedClientConfig,
        observation_source: ObservationSource,
        policy_client: PolicyClient,
        action_sink: ActionSink,
        telemetry: ClientTelemetry | None = None,
        alignment_shadow: TimestampAlignmentShadow | None = None,
        local_tracker: LocalActionTracker | None = None,
        safety: ActionSafety | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(observation_source, ObservationSource):
            raise TypeError("observation_source does not implement ObservationSource")
        if not isinstance(policy_client, PolicyClient):
            raise TypeError("policy_client does not implement PolicyClient")
        if not isinstance(action_sink, ActionSink):
            raise TypeError("action_sink does not implement ActionSink")
        if alignment_shadow is not None and not isinstance(alignment_shadow, TimestampAlignmentShadow):
            raise TypeError("alignment_shadow must be TimestampAlignmentShadow or None")
        if config.local_tracker_enabled != (local_tracker is not None):
            raise ValueError(
                "local_tracker_enabled must exactly match whether a LocalActionTracker is supplied"
            )
        self.config = config
        self.observation_source = observation_source
        self.policy_client = policy_client
        self.action_sink = action_sink
        self.telemetry = telemetry or ClientTelemetry()
        self.alignment_shadow = alignment_shadow
        self.local_tracker = local_tracker
        self.safety = safety or ActionSafety()
        self._clock = clock or time.perf_counter
        self.action_queue = ActionChunkQueue(
            max_queue_size=MAX_ACTION_CHUNK_STEPS, empty_queue_strategy="stop"
        )
        self.stop_state = ClientStopState()
        self._next_request_id = 0
        self._consecutive_stale_chunks = 0
        self._last_source_frame_id: int | str | None = None
        self._consecutive_repeated_source_frames = 0
        self._last_source_timestamps: dict[str, float] = {}
        self.last_failure_diagnostics: dict[str, object] | None = None

    def run_cycle(self) -> ClientCycleResult:
        self.stop_state.raise_if_stopped()
        stage = "observation"
        try:
            observation = self.observation_source.read()
            if not isinstance(observation, TimedObservation):
                raise TypeError(f"observation source must return TimedObservation, got {type(observation)}")
            if self.config.strict_source_timestamps:
                observation.require_source_timestamps(camera_keys=self.config.required_camera_keys)
            if observation.local_clock_domain != self.config.clock_domain:
                raise ValueError(
                    "observation/client local clock domains differ: "
                    f"{observation.local_clock_domain!r} != {self.config.clock_domain!r}"
                )
            source_frame_id = _source_frame_id(observation)
            if source_frame_id == self._last_source_frame_id:
                self._consecutive_repeated_source_frames += 1
            else:
                self._consecutive_repeated_source_frames = 0
            self._last_source_frame_id = source_frame_id
            self.telemetry.record_frame(
                observation_sequence_id=observation.sequence_id,
                source_frame_id=source_frame_id,
            )
            if self._consecutive_repeated_source_frames >= self.config.max_consecutive_repeated_source_frames:
                raise RuntimeError(
                    f"source frame was reused {self._consecutive_repeated_source_frames} consecutive times"
                )
            self._validate_source_timestamp_progress(observation)
            if self.alignment_shadow is not None:
                stage = "alignment_shadow"
                self.alignment_shadow.observe(observation)
                stage = "observation"
            self.telemetry.record_sensor_tick(
                timestamp_s=observation.build_ready_monotonic_s,
                target_period_s=self.config.control_period_s,
            )

            predicted_delay_steps = (
                self.telemetry.predicted_delay_steps(control_period_s=self.config.control_period_s)
                if self.config.mode == "rtc"
                else 0
            )
            leftover = self.action_queue.get_raw_leftover() if self.config.mode == "rtc" else None
            request_id = self._next_request_id
            request = InferenceRequest(
                request_id=request_id,
                mode=self.config.mode,
                observation_frame=dict(observation.observation_frame),
                task=self.config.task,
                robot_type=self.config.robot_type,
                obs_sequence_id=observation.sequence_id,
                predicted_delay_steps=predicted_delay_steps,
                prev_chunk_left_over=(
                    None
                    if leftover is None
                    else leftover.detach().to(device="cpu", dtype=torch.float32).numpy()
                ),
                execution_horizon=self.config.execution_horizon,
            )
            request.validate()
            self._next_request_id += 1

            stage = "policy"
            request_started_s = self._clock()
            response = self.policy_client.infer(request)
            ready_s = self._clock()
            request_total_s = _duration("request_total_s", request_started_s, ready_s)
            self.telemetry.record_request(request, response, total_s=request_total_s)

            stage = "queue"
            chunk = _response_to_chunk(
                response=response,
                request=request,
                observation=observation,
                ready_s=ready_s,
                control_period_s=self.config.control_period_s,
                safety=self.safety,
            )
            if self.config.mode == "rtc":
                merge = self.action_queue.merge_rtc(chunk)
                dropped_steps = merge.dropped_steps
                queue_depth = merge.queue_depth_after
                fully_stale = merge.dropped_all
            else:
                selection = select_single_step(chunk)
                dropped_steps = selection.dropped_steps
                queue_depth = 0
                fully_stale = selection.dropped_all

            self.telemetry.record_queue(
                depth=queue_depth,
                dropped_steps=dropped_steps,
                stale_chunk=fully_stale,
            )
            actor_tick_s = self._clock()
            self.telemetry.record_actor_tick(
                timestamp_s=actor_tick_s,
                target_period_s=self.config.control_period_s,
            )

            if fully_stale:
                self._consecutive_stale_chunks += 1
                if self._consecutive_stale_chunks >= self.config.max_consecutive_stale_chunks:
                    raise RuntimeError(
                        f"{self._consecutive_stale_chunks} consecutive inference chunks were fully stale"
                    )
                return ClientCycleResult(
                    request_id=request_id,
                    observation_sequence_id=observation.sequence_id,
                    predicted_delay_steps=predicted_delay_steps,
                    dropped_steps=dropped_steps,
                    queue_depth=queue_depth,
                    action_written=False,
                    fully_stale=True,
                    tracker_applied=False,
                )
            self._consecutive_stale_chunks = 0

            stage = "action"
            if self.config.mode == "rtc":
                executable = self.action_queue.pop_processed_action()
                if executable is None:
                    self.telemetry.record_queue_empty()
                    raise RuntimeError("RTC action queue returned no executable action")
                queue_depth = self.action_queue.depth()
            else:
                executable = selection.processed_action
                if executable is None:
                    raise RuntimeError("single-step selection returned no executable action")
            tracker_applied = False
            if self.local_tracker is not None:
                stage = "tracker"
                executable = self.local_tracker.track(
                    requested_action=executable,
                    observed_state=observation.observation_frame.get("observation.state"),
                    timestamp_s=observation.build_ready_monotonic_s,
                )
                tracker_applied = True
            stage = "action"
            checked_action = self.safety.check_tensor(executable)
            self.stop_state.run_if_active(lambda: self.action_sink.write(checked_action.detach().clone()))
            return ClientCycleResult(
                request_id=request_id,
                observation_sequence_id=observation.sequence_id,
                predicted_delay_steps=predicted_delay_steps,
                dropped_steps=dropped_steps,
                queue_depth=queue_depth,
                action_written=True,
                fully_stale=False,
                tracker_applied=tracker_applied,
            )
        except Exception as error:
            stop_reason = self.stop_state.request_stop(f"{stage}: {type(error).__name__}: {error}")
            queue_snapshot = self.action_queue.snapshot()
            self.action_queue = ActionChunkQueue(
                max_queue_size=MAX_ACTION_CHUNK_STEPS,
                empty_queue_strategy="stop",
            )
            tracker_reset_error = None
            if self.local_tracker is not None:
                try:
                    self.local_tracker.reset(stop_reason)
                except Exception as reset_error:
                    tracker_reset_error = f"{type(reset_error).__name__}: {reset_error}"
            self.last_failure_diagnostics = {
                "stop_reason": stop_reason,
                "failed_stage": stage,
                "queue_depth_before_clear": queue_snapshot.depth,
                "queue_depth_after_clear": self.action_queue.depth(),
                "tracker_reset_error": tracker_reset_error,
                "no_later_action_write": True,
            }
            raise

    def run_cycles(self, count: int) -> tuple[ClientCycleResult, ...]:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        return tuple(self.run_cycle() for _ in range(count))

    def _validate_source_timestamp_progress(self, observation: TimedObservation) -> None:
        timestamps: dict[str, float] = {}
        if observation.state_source_timestamp is not None:
            state = observation.state_source_timestamp
            timestamps[f"state:{state.source}:{state.clock_domain}"] = state.timestamp_s
        for key, camera in observation.camera_source_timestamps.items():
            timestamps[f"{key}:{camera.source}:{camera.clock_domain}"] = camera.timestamp_s
        for source, timestamp_s in timestamps.items():
            previous = self._last_source_timestamps.get(source)
            if previous is not None and timestamp_s <= previous:
                raise RuntimeError(
                    f"source timestamp did not advance for {source}: "
                    f"current={timestamp_s} previous={previous}"
                )
        self._last_source_timestamps.update(timestamps)


def _response_to_chunk(
    *,
    response: InferenceResponse,
    request: InferenceRequest,
    observation: TimedObservation,
    ready_s: float,
    control_period_s: float,
    safety: ActionSafety,
) -> ActionChunk:
    if not isinstance(response, InferenceResponse):
        raise TypeError(f"policy client must return InferenceResponse, got {type(response)}")
    response.validate()
    if response.request_id != request.request_id or response.mode != request.mode:
        raise ValueError("policy response identity does not match the request")
    model_actions = _chunk_tensor(response.raw_actions, expected_dim=16, label="model16")
    robot_actions = _chunk_tensor(response.processed_actions, expected_dim=18, label="raw18")
    if model_actions.shape[0] != robot_actions.shape[0]:
        raise ValueError("policy response model16/raw18 time dimensions differ")
    for action in robot_actions:
        safety.check_tensor(action)
    age_s = _duration("observation_age_s", observation.build_ready_monotonic_s, ready_s)
    drop_steps = min(MAX_ACTION_CHUNK_STEPS, int(math.ceil(age_s / control_period_s))) if age_s else 0
    return ActionChunk(
        raw_actions=model_actions,
        processed_actions=robot_actions,
        observation_timestamp_s=observation.build_ready_monotonic_s,
        ready_timestamp_s=ready_s,
        drop_steps=drop_steps,
        predicted_delay_steps=request.predicted_delay_steps,
        source_observation_seq=observation.sequence_id,
    )


def _chunk_tensor(value: object, *, expected_dim: int, label: str) -> Tensor:
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != expected_dim:
        raise ValueError(f"{label} actions must have shape (T,{expected_dim}), got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{label} actions contain non-finite values")
    return tensor.clone()


def _source_frame_id(observation: TimedObservation) -> int | str:
    if observation.camera_source_timestamps:
        parts = []
        for key in sorted(observation.camera_source_timestamps):
            timestamp = observation.camera_source_timestamps[key]
            parts.append(f"{key}:{timestamp.clock_domain}:{timestamp.timestamp_s:.9f}")
        return "|".join(parts)
    if observation.state_source_timestamp is not None:
        timestamp = observation.state_source_timestamp
        return f"state:{timestamp.clock_domain}:{timestamp.timestamp_s:.9f}"
    return observation.sequence_id


def _duration(name: str, started_s: object, finished_s: object) -> float:
    start = _finite_non_negative(f"{name} start", started_s)
    finish = _finite_non_negative(f"{name} finish", finished_s)
    if finish < start:
        raise ValueError(f"monotonic clock moved backwards while measuring {name}")
    return finish - start


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
    "ActionSink",
    "ClientCycleResult",
    "ClientMode",
    "ClientStopState",
    "ObservationSource",
    "OptimizedClient",
    "OptimizedClientConfig",
    "PolicyClient",
]
