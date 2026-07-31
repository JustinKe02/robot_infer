from __future__ import annotations

import copy
import math
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import OBS_STR
from lerobot.utils.robot_utils import precise_sleep

from .action_queue import ActionChunk, ActionChunkQueue, QueueSnapshot, select_single_step
from .protocol import InferenceMode, InferenceRequest
from .remote_client import RemotePolicyClient
from .robot_io import SerializedRobotIO
from .safety import ActionSafety

RuntimeMode = Literal["single_step", "async_single_step", "rtc"]


@dataclass(slots=True)
class ClientRuntimeConfig:
    task: str
    server_url: str = "http://127.0.0.1:8088"
    mode: RuntimeMode = "single_step"
    sensor_fps: int = 20
    control_fps: int = 20
    run_time_s: float = 0.0
    queue_low_watermark: int = 30
    max_queue_size: int = 50
    first_chunk_timeout_s: float = 120.0
    rtc_execution_horizon: int = 10
    empty_queue_strategy: str = "stop"
    metrics_log_interval_s: float = 2.0
    producer_idle_sleep_s: float = 0.005
    fully_stale_chunk_limit: int = 3
    max_sent_actions: int = 0

    def __post_init__(self) -> None:
        self.server_url = self.server_url.rstrip("/")
        if not self.task.strip():
            raise ValueError("task must be non-empty")
        if self.mode not in {"single_step", "async_single_step", "rtc"}:
            raise ValueError("mode must be single_step, async_single_step, or rtc")
        if self.sensor_fps <= 0 or self.control_fps <= 0:
            raise ValueError("sensor_fps and control_fps must be positive")
        if self.run_time_s < 0:
            raise ValueError("run_time_s must be non-negative")
        if self.queue_low_watermark < 0:
            raise ValueError("queue_low_watermark must be non-negative")
        if self.max_queue_size <= 0 or self.queue_low_watermark >= self.max_queue_size:
            raise ValueError("queue_low_watermark must be smaller than positive max_queue_size")
        if self.first_chunk_timeout_s <= 0:
            raise ValueError("first_chunk_timeout_s must be positive")
        if self.rtc_execution_horizon <= 0:
            raise ValueError("rtc_execution_horizon must be positive")
        if self.empty_queue_strategy not in {"stop", "skip_send", "hold_last_action"}:
            raise ValueError("unsupported empty_queue_strategy")
        if self.metrics_log_interval_s <= 0:
            raise ValueError("metrics_log_interval_s must be positive")
        if self.producer_idle_sleep_s < 0:
            raise ValueError("producer_idle_sleep_s must be non-negative")
        if self.fully_stale_chunk_limit <= 0:
            raise ValueError("fully_stale_chunk_limit must be positive")
        if isinstance(self.max_sent_actions, bool) or self.max_sent_actions < 0:
            raise ValueError("max_sent_actions must be a non-negative integer")

    @property
    def sensor_dt_s(self) -> float:
        return 1.0 / self.sensor_fps

    @property
    def control_dt_s(self) -> float:
        return 1.0 / self.control_fps


@dataclass(frozen=True, slots=True)
class FrameSnapshot:
    raw_observation: dict[str, Any]
    observation_frame: dict[str, Any]
    timestamp_s: float
    sequence_id: int


class FrameBuffer:
    """Latest-only observation buffer shared by the RTC sensor and producer."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._snapshot: FrameSnapshot | None = None
        self._sequence_id = 0

    def update(
        self,
        *,
        raw_observation: Mapping[str, Any],
        observation_frame: Mapping[str, Any],
        timestamp_s: float,
    ) -> FrameSnapshot:
        with self._condition:
            self._sequence_id += 1
            snapshot = FrameSnapshot(
                raw_observation=copy.deepcopy(dict(raw_observation)),
                observation_frame=copy.deepcopy(dict(observation_frame)),
                timestamp_s=float(timestamp_s),
                sequence_id=self._sequence_id,
            )
            self._snapshot = snapshot
            self._condition.notify_all()
            return _copy_frame_snapshot(snapshot)

    def latest(
        self,
        *,
        timeout_s: float | None = None,
        min_sequence_id: int | None = None,
    ) -> FrameSnapshot | None:
        with self._condition:
            if not self._has_snapshot(min_sequence_id):
                self._condition.wait_for(lambda: self._has_snapshot(min_sequence_id), timeout=timeout_s)
            return None if not self._has_snapshot(min_sequence_id) else _copy_frame_snapshot(self._snapshot)

    def _has_snapshot(self, min_sequence_id: int | None) -> bool:
        if self._snapshot is None:
            return False
        return min_sequence_id is None or self._snapshot.sequence_id >= min_sequence_id


@dataclass(frozen=True, slots=True)
class RuntimeErrorRecord:
    source: str
    message: str
    traceback_text: str


class ClientRuntimeState:
    def __init__(self) -> None:
        self.start_time_s = time.perf_counter()
        self.stop_event = threading.Event()
        self.first_chunk_ready = threading.Event()
        self.stop_reason: str | None = None
        self.last_error: RuntimeErrorRecord | None = None
        self.sensor_ticks = 0
        self.actor_ticks = 0
        self.sent_actions = 0
        self.inference_requests = 0
        self.fully_stale_chunks_in_a_row = 0
        self._lock = threading.RLock()
        self._actuation_lock = threading.Lock()

    @property
    def running(self) -> bool:
        return not self.stop_event.is_set()

    def call_if_running(self, callback: Callable[[], Any]) -> tuple[bool, Any | None]:
        """Gate a new actuator call without delaying a concurrent stop request.

        An already-running transport call cannot be cancelled here, but once the
        stop event is set no subsequent callback may begin.
        """

        with self._actuation_lock:
            if self.stop_event.is_set():
                return False, None
            return True, callback()

    def request_stop(self, reason: str) -> None:
        with self._lock:
            if self.stop_event.is_set():
                return
            self.stop_reason = reason
            self.stop_event.set()
            self.first_chunk_ready.set()

    def record_error(self, source: str, exc: BaseException) -> None:
        record = RuntimeErrorRecord(
            source=source,
            message=str(exc),
            traceback_text="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        with self._lock:
            self.last_error = record
        self.request_stop(f"{source}: {exc}")

    def note_fully_stale_chunk(self, *, dropped_all: bool, limit: int) -> None:
        with self._lock:
            if dropped_all:
                self.fully_stale_chunks_in_a_row += 1
            else:
                self.fully_stale_chunks_in_a_row = 0
            if self.fully_stale_chunks_in_a_row >= limit:
                self.request_stop(
                    f"{self.fully_stale_chunks_in_a_row} consecutive inference chunks were fully stale"
                )


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    latest_request_s: float
    p95_request_s: float
    latest_server_s: float
    latest_drop_steps: int
    latest_predicted_delay_steps: int
    request_count: int


class ClientMetrics:
    def __init__(self, *, maxlen: int = 100) -> None:
        self._request_latencies: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.RLock()
        self.latest_server_s = 0.0
        self.latest_drop_steps = 0
        self.latest_predicted_delay_steps = 0

    def record_request(
        self,
        *,
        total_s: float,
        server_s: float,
        drop_steps: int,
        predicted_delay_steps: int,
    ) -> None:
        with self._lock:
            self._request_latencies.append(max(0.0, float(total_s)))
            self.latest_server_s = max(0.0, float(server_s))
            self.latest_drop_steps = int(drop_steps)
            self.latest_predicted_delay_steps = int(predicted_delay_steps)

    def predicted_delay_steps(self, *, control_dt_s: float) -> int:
        with self._lock:
            p95 = _percentile_95(self._request_latencies)
        return latency_to_steps(p95, control_dt_s)

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                latest_request_s=self._request_latencies[-1] if self._request_latencies else 0.0,
                p95_request_s=_percentile_95(self._request_latencies),
                latest_server_s=self.latest_server_s,
                latest_drop_steps=self.latest_drop_steps,
                latest_predicted_delay_steps=self.latest_predicted_delay_steps,
                request_count=len(self._request_latencies),
            )


@dataclass(frozen=True, slots=True)
class ClientRuntimeResult:
    mode: RuntimeMode
    stop_reason: str | None
    sensor_ticks: int
    actor_ticks: int
    sent_actions: int
    inference_requests: int
    metrics: MetricsSnapshot
    queue: QueueSnapshot | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True, slots=True)
class InferenceSmokeResult:
    request_id: int
    observation_sequence_id: int
    model_steps: int
    raw_action_shape: tuple[int, ...]
    processed_action_shape: tuple[int, ...]
    dropped_steps: int
    selected_action_index: int
    server_latency_s: float
    model_latency_s: float
    robot_action_keys: tuple[str, ...]
    processed_action_names: tuple[str, ...]
    selected_robot_action: dict[str, float]
    observation_frame: dict[str, np.ndarray]
    raw_actions: np.ndarray
    processed_actions: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "observation_sequence_id": self.observation_sequence_id,
            "model_steps": self.model_steps,
            "raw_action_shape": self.raw_action_shape,
            "processed_action_shape": self.processed_action_shape,
            "dropped_steps": self.dropped_steps,
            "selected_action_index": self.selected_action_index,
            "server_latency_s": self.server_latency_s,
            "model_latency_s": self.model_latency_s,
            "robot_action_keys": self.robot_action_keys,
            "processed_action_names": self.processed_action_names,
        }


def build_live_observation_frame(
    *,
    raw_observation: Mapping[str, Any],
    dataset_features: dict[str, dict[str, Any]],
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    processed = (
        dict(robot_observation_processor(raw_observation))
        if robot_observation_processor is not None
        else dict(raw_observation)
    )
    return build_dataset_frame(dataset_features, processed, prefix=OBS_STR)


def build_robot_action(
    *,
    action: Tensor,
    dataset_features: dict[str, dict[str, Any]],
    robot_action_processor: Callable[[tuple[dict[str, float], Any]], Any] | None,
    observation: Mapping[str, Any],
    safety: ActionSafety,
) -> dict[str, Any]:
    checked_action = safety.check_tensor(action)
    robot_action = make_robot_action(checked_action.unsqueeze(0), dataset_features)
    if robot_action_processor is not None:
        robot_action = robot_action_processor((robot_action, observation))
    safety.check_robot_action(robot_action)
    return robot_action


def make_request(
    *,
    request_id: int,
    mode: InferenceMode,
    snapshot: FrameSnapshot,
    task: str,
    robot_type: str,
    predicted_delay_steps: int,
    prev_chunk_left_over: Tensor | None,
    execution_horizon: int,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        mode=mode,
        observation_frame=snapshot.observation_frame,
        task=task,
        robot_type=robot_type,
        obs_sequence_id=snapshot.sequence_id,
        predicted_delay_steps=int(predicted_delay_steps),
        prev_chunk_left_over=(
            None
            if prev_chunk_left_over is None
            else prev_chunk_left_over.detach().to(device="cpu", dtype=torch.float32).numpy()
        ),
        execution_horizon=int(execution_horizon),
    )


def response_to_chunk(
    *,
    response: Any,
    request_id: int,
    snapshot: FrameSnapshot,
    ready_timestamp_s: float,
    control_dt_s: float,
    predicted_delay_steps: int,
    safety: ActionSafety,
) -> ActionChunk:
    response_request_id = getattr(response, "request_id", request_id)
    if int(response_request_id) != int(request_id):
        raise RuntimeError(
            f"Policy response request_id={response_request_id} does not match request_id={request_id}"
        )
    raw_actions = _as_chunk_tensor(response.raw_actions, expected_dim=16, label="model16")
    processed_actions = _as_chunk_tensor(
        response.processed_actions,
        expected_dim=18,
        label="raw18",
    )
    if raw_actions.shape[0] != processed_actions.shape[0]:
        raise ValueError(
            "Server response model16/raw18 time dimensions differ: "
            f"{raw_actions.shape[0]} vs {processed_actions.shape[0]}"
        )
    for action in processed_actions:
        safety.check_tensor(action)
    return ActionChunk(
        raw_actions=raw_actions,
        processed_actions=processed_actions,
        observation_timestamp_s=snapshot.timestamp_s,
        ready_timestamp_s=ready_timestamp_s,
        drop_steps=latency_to_steps(ready_timestamp_s - snapshot.timestamp_s, control_dt_s),
        predicted_delay_steps=int(predicted_delay_steps),
        source_observation_seq=snapshot.sequence_id,
    )


def run_inference_smoke(
    *,
    config: ClientRuntimeConfig,
    remote_policy: RemotePolicyClient,
    robot_io: SerializedRobotIO,
    dataset_features: dict[str, dict[str, Any]],
    robot_action_processor: Callable[[tuple[dict[str, float], Any]], Any] | None,
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    safety: ActionSafety,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> InferenceSmokeResult:
    """Validate one live observation and policy response without sending an action."""

    if config.mode != "single_step":
        raise ValueError("inference smoke requires mode=single_step")

    raw_observation = robot_io.get_observation()
    observation_timestamp_s = perf_counter()
    snapshot = FrameSnapshot(
        raw_observation=copy.deepcopy(raw_observation),
        observation_frame=build_live_observation_frame(
            raw_observation=raw_observation,
            dataset_features=dataset_features,
            robot_observation_processor=robot_observation_processor,
        ),
        timestamp_s=observation_timestamp_s,
        sequence_id=1,
    )
    request_id = 1
    request = make_request(
        request_id=request_id,
        mode="single_step",
        snapshot=snapshot,
        task=config.task,
        robot_type=robot_io.robot_type,
        predicted_delay_steps=0,
        prev_chunk_left_over=None,
        execution_horizon=config.rtc_execution_horizon,
    )
    response = remote_policy.infer(request)
    ready_timestamp_s = perf_counter()
    chunk = response_to_chunk(
        response=response,
        request_id=request_id,
        snapshot=snapshot,
        ready_timestamp_s=ready_timestamp_s,
        control_dt_s=config.control_dt_s,
        predicted_delay_steps=0,
        safety=safety,
    )
    selection = select_single_step(chunk)
    if selection.processed_action is None:
        raise RuntimeError(
            "Inference smoke response was fully stale before validation; "
            f"dropped={selection.dropped_steps} steps={selection.input_steps}"
        )

    # Build and validate the exact robot-facing dictionary, but deliberately do
    # not call robot_io.send_action(). This is the smoke mode's hard no-motion boundary.
    robot_action = build_robot_action(
        action=selection.processed_action,
        dataset_features=dataset_features,
        robot_action_processor=robot_action_processor,
        observation=snapshot.raw_observation,
        safety=safety,
    )
    action_feature = dataset_features.get("action", {})
    action_names = tuple(str(name) for name in action_feature.get("names", ()))

    def numpy_copy(value: Any) -> np.ndarray:
        if isinstance(value, Tensor):
            return value.detach().to(device="cpu").numpy().copy()
        return np.asarray(value).copy()

    return InferenceSmokeResult(
        request_id=request_id,
        observation_sequence_id=snapshot.sequence_id,
        model_steps=int(chunk.raw_actions.shape[0]),
        raw_action_shape=tuple(chunk.raw_actions.shape),
        processed_action_shape=tuple(chunk.processed_actions.shape),
        dropped_steps=selection.dropped_steps,
        selected_action_index=selection.dropped_steps,
        server_latency_s=float(getattr(response, "server_latency_s", 0.0)),
        model_latency_s=float(getattr(response, "model_latency_s", 0.0)),
        robot_action_keys=tuple(sorted(robot_action)),
        processed_action_names=action_names,
        selected_robot_action={key: float(np.asarray(value).item()) for key, value in robot_action.items()},
        observation_frame={key: numpy_copy(value) for key, value in snapshot.observation_frame.items()},
        raw_actions=chunk.raw_actions.detach().to(device="cpu").numpy().copy(),
        processed_actions=chunk.processed_actions.detach().to(device="cpu").numpy().copy(),
    )


def run_single_step_runtime(
    *,
    config: ClientRuntimeConfig,
    remote_policy: RemotePolicyClient,
    robot_io: SerializedRobotIO,
    dataset_features: dict[str, dict[str, Any]],
    robot_action_processor: Callable[[tuple[dict[str, float], Any]], Any] | None,
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    safety: ActionSafety,
    state: ClientRuntimeState | None = None,
    metrics: ClientMetrics | None = None,
    perf_counter: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = precise_sleep,
) -> ClientRuntimeResult:
    """Run the conservative sequential mode: one request, then at most one send."""

    if config.mode != "single_step":
        raise ValueError("run_single_step_runtime requires mode=single_step")
    state = ClientRuntimeState() if state is None else state
    metrics = ClientMetrics() if metrics is None else metrics
    request_id = 0
    observation_sequence_id = 0
    next_tick_s = perf_counter()

    while state.running:
        now_s = perf_counter()
        if _duration_elapsed(config, state, now_s):
            break
        try:
            raw_observation = robot_io.get_observation()
            observation_timestamp_s = perf_counter()
            observation_sequence_id += 1
            observation_frame = build_live_observation_frame(
                raw_observation=raw_observation,
                dataset_features=dataset_features,
                robot_observation_processor=robot_observation_processor,
            )
            snapshot = FrameSnapshot(
                raw_observation=copy.deepcopy(raw_observation),
                observation_frame=observation_frame,
                timestamp_s=observation_timestamp_s,
                sequence_id=observation_sequence_id,
            )
            with state._lock:
                state.sensor_ticks += 1

            request_id += 1
            request = make_request(
                request_id=request_id,
                mode="single_step",
                snapshot=snapshot,
                task=config.task,
                robot_type=robot_io.robot_type,
                predicted_delay_steps=0,
                prev_chunk_left_over=None,
                execution_horizon=config.rtc_execution_horizon,
            )
            request_started_s = perf_counter()
            response = remote_policy.infer(request)
            ready_timestamp_s = perf_counter()
            chunk = response_to_chunk(
                response=response,
                request_id=request_id,
                snapshot=snapshot,
                ready_timestamp_s=ready_timestamp_s,
                control_dt_s=config.control_dt_s,
                predicted_delay_steps=0,
                safety=safety,
            )
            selection = select_single_step(chunk)
            state.note_fully_stale_chunk(
                dropped_all=selection.dropped_all,
                limit=config.fully_stale_chunk_limit,
            )
            metrics.record_request(
                total_s=ready_timestamp_s - request_started_s,
                server_s=float(getattr(response, "server_latency_s", 0.0)),
                drop_steps=selection.dropped_steps,
                predicted_delay_steps=0,
            )
            with state._lock:
                state.inference_requests += 1

            if selection.processed_action is not None and state.running:
                robot_action = build_robot_action(
                    action=selection.processed_action,
                    dataset_features=dataset_features,
                    robot_action_processor=robot_action_processor,
                    observation=snapshot.raw_observation,
                    safety=safety,
                )
                did_send, sent_action = state.call_if_running(
                    lambda action=robot_action: robot_io.send_action(action)
                )
                if not did_send:
                    break
                if isinstance(sent_action, Mapping):
                    safety.check_robot_action(sent_action)
                with state._lock:
                    state.actor_ticks += 1
                    state.sent_actions += 1
                    sent_actions = state.sent_actions
                if config.max_sent_actions > 0 and sent_actions >= config.max_sent_actions:
                    state.request_stop(f"max_sent_actions reached: {config.max_sent_actions}")
                    break
            next_tick_s += config.control_dt_s
            remaining_s = next_tick_s - perf_counter()
            if remaining_s > 0:
                sleep_fn(remaining_s)
            else:
                next_tick_s = perf_counter()
        except KeyboardInterrupt:
            state.request_stop("KeyboardInterrupt")
            break
        except Exception as exc:
            state.record_error("single_step_runtime", exc)
            break

    return _runtime_result(mode="single_step", state=state, metrics=metrics, queue=None)


def run_sensor_loop(
    *,
    config: ClientRuntimeConfig,
    state: ClientRuntimeState,
    robot_io: SerializedRobotIO,
    frame_buffer: FrameBuffer,
    dataset_features: dict[str, dict[str, Any]],
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    perf_counter: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = precise_sleep,
) -> int:
    ticks = 0
    next_tick_s = perf_counter()
    while state.running:
        try:
            raw_observation = robot_io.get_observation()
            observation_timestamp_s = perf_counter()
            observation_frame = build_live_observation_frame(
                raw_observation=raw_observation,
                dataset_features=dataset_features,
                robot_observation_processor=robot_observation_processor,
            )
            frame_buffer.update(
                raw_observation=raw_observation,
                observation_frame=observation_frame,
                timestamp_s=observation_timestamp_s,
            )
            ticks += 1
            with state._lock:
                state.sensor_ticks += 1
        except Exception as exc:
            state.record_error("sensor_loop", exc)
            break

        next_tick_s += config.sensor_dt_s
        remaining_s = next_tick_s - perf_counter()
        if remaining_s > 0:
            sleep_fn(remaining_s)
        else:
            next_tick_s = perf_counter()
    return ticks


def run_producer_loop(
    *,
    config: ClientRuntimeConfig,
    state: ClientRuntimeState,
    metrics: ClientMetrics,
    remote_policy: RemotePolicyClient,
    frame_buffer: FrameBuffer,
    action_queue: ActionChunkQueue,
    robot_type: str,
    safety: ActionSafety,
    perf_counter: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    if config.mode not in {"async_single_step", "rtc"}:
        raise ValueError("run_producer_loop requires mode=async_single_step or rtc")
    requests = 0
    request_id = 0
    last_observation_sequence_id = 0
    while state.running:
        if action_queue.depth() > config.queue_low_watermark:
            sleep_fn(config.producer_idle_sleep_s)
            continue
        snapshot = frame_buffer.latest(
            timeout_s=config.first_chunk_timeout_s if requests == 0 else 0.0,
            min_sequence_id=last_observation_sequence_id + 1,
        )
        if snapshot is None:
            sleep_fn(config.producer_idle_sleep_s)
            continue
        last_observation_sequence_id = snapshot.sequence_id

        request_id += 1
        rtc_enabled = config.mode == "rtc"
        predicted_delay_steps = (
            metrics.predicted_delay_steps(control_dt_s=config.control_dt_s) if rtc_enabled else 0
        )
        request = make_request(
            request_id=request_id,
            mode="rtc" if rtc_enabled else "single_step",
            snapshot=snapshot,
            task=config.task,
            robot_type=robot_type,
            predicted_delay_steps=predicted_delay_steps,
            prev_chunk_left_over=action_queue.get_raw_leftover() if rtc_enabled else None,
            execution_horizon=config.rtc_execution_horizon,
        )
        request_started_s = perf_counter()
        try:
            response = remote_policy.infer(request)
            ready_timestamp_s = perf_counter()
            chunk = response_to_chunk(
                response=response,
                request_id=request_id,
                snapshot=snapshot,
                ready_timestamp_s=ready_timestamp_s,
                control_dt_s=config.control_dt_s,
                predicted_delay_steps=predicted_delay_steps,
                safety=safety,
            )
            merge_result = action_queue.merge_rtc(chunk)
            state.note_fully_stale_chunk(
                dropped_all=merge_result.dropped_all,
                limit=config.fully_stale_chunk_limit,
            )
            if merge_result.enqueued_steps > 0:
                state.first_chunk_ready.set()
            metrics.record_request(
                total_s=ready_timestamp_s - request_started_s,
                server_s=float(getattr(response, "server_latency_s", 0.0)),
                drop_steps=merge_result.dropped_steps,
                predicted_delay_steps=predicted_delay_steps,
            )
            requests += 1
            with state._lock:
                state.inference_requests += 1
        except Exception as exc:
            state.record_error("producer_loop", exc)
            break
    return requests


def run_actor_loop(
    *,
    config: ClientRuntimeConfig,
    state: ClientRuntimeState,
    robot_io: SerializedRobotIO,
    frame_buffer: FrameBuffer,
    action_queue: ActionChunkQueue,
    dataset_features: dict[str, dict[str, Any]],
    robot_action_processor: Callable[[tuple[dict[str, float], Any]], Any] | None,
    safety: ActionSafety,
    perf_counter: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = precise_sleep,
) -> int:
    ticks = 0
    next_tick_s = perf_counter()
    first_chunk_deadline_s = state.start_time_s + config.first_chunk_timeout_s
    while state.running:
        now_s = perf_counter()
        if _duration_elapsed(config, state, now_s):
            break
        if not state.first_chunk_ready.is_set() and now_s >= first_chunk_deadline_s:
            state.request_stop(f"first action chunk timeout after {config.first_chunk_timeout_s:.3f}s")
            break

        try:
            if state.first_chunk_ready.is_set():
                action = action_queue.pop_processed_action()
                if action is not None:
                    snapshot = frame_buffer.latest(timeout_s=0.0)
                    observation = {} if snapshot is None else snapshot.raw_observation
                    robot_action = build_robot_action(
                        action=action,
                        dataset_features=dataset_features,
                        robot_action_processor=robot_action_processor,
                        observation=observation,
                        safety=safety,
                    )
                    did_send, sent_action = state.call_if_running(
                        lambda action=robot_action: robot_io.send_action(action)
                    )
                    if not did_send:
                        break
                    if isinstance(sent_action, Mapping):
                        safety.check_robot_action(sent_action)
                    with state._lock:
                        state.sent_actions += 1
                        sent_actions = state.sent_actions
                    if config.max_sent_actions > 0 and sent_actions >= config.max_sent_actions:
                        state.request_stop(f"max_sent_actions reached: {config.max_sent_actions}")
                        break
            ticks += 1
            with state._lock:
                state.actor_ticks += 1
        except Exception as exc:
            state.record_error("actor_loop", exc)
            break

        next_tick_s += config.control_dt_s
        remaining_s = next_tick_s - perf_counter()
        if remaining_s > 0:
            sleep_fn(remaining_s)
        else:
            next_tick_s = perf_counter()
    return ticks


def run_rtc_runtime(
    *,
    config: ClientRuntimeConfig,
    remote_policy: RemotePolicyClient,
    robot_io: SerializedRobotIO,
    dataset_features: dict[str, dict[str, Any]],
    robot_action_processor: Callable[[tuple[dict[str, float], Any]], Any] | None,
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    safety: ActionSafety,
    log_fn: Callable[[str], None] | None = print,
) -> ClientRuntimeResult:
    if config.mode not in {"async_single_step", "rtc"}:
        raise ValueError("run_rtc_runtime requires mode=async_single_step or rtc")
    state = ClientRuntimeState()
    metrics = ClientMetrics()
    frame_buffer = FrameBuffer()
    action_queue = ActionChunkQueue(
        max_queue_size=config.max_queue_size,
        empty_queue_strategy=config.empty_queue_strategy,
    )
    threads = [
        threading.Thread(
            target=run_sensor_loop,
            kwargs={
                "config": config,
                "state": state,
                "robot_io": robot_io,
                "frame_buffer": frame_buffer,
                "dataset_features": dataset_features,
                "robot_observation_processor": robot_observation_processor,
            },
            name="JZPI05Sensor",
            daemon=True,
        ),
        threading.Thread(
            target=run_producer_loop,
            kwargs={
                "config": config,
                "state": state,
                "metrics": metrics,
                "remote_policy": remote_policy,
                "frame_buffer": frame_buffer,
                "action_queue": action_queue,
                "robot_type": robot_io.robot_type,
                "safety": safety,
            },
            name="JZPI05Producer",
            daemon=True,
        ),
        threading.Thread(
            target=run_actor_loop,
            kwargs={
                "config": config,
                "state": state,
                "robot_io": robot_io,
                "frame_buffer": frame_buffer,
                "action_queue": action_queue,
                "dataset_features": dataset_features,
                "robot_action_processor": robot_action_processor,
                "safety": safety,
            },
            name="JZPI05Actor",
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    last_log_s = time.perf_counter()
    try:
        while state.running:
            time.sleep(0.1)
            now_s = time.perf_counter()
            if log_fn is not None and now_s - last_log_s >= config.metrics_log_interval_s:
                log_fn(format_runtime_metrics(state, metrics, action_queue))
                last_log_s = now_s
    except KeyboardInterrupt:
        state.request_stop("KeyboardInterrupt")
    finally:
        state.request_stop(state.stop_reason or f"{config.mode} client runtime exiting")
        for thread in threads:
            thread.join(timeout=5.0)
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive and state.last_error is None:
            state.record_error("run_rtc_runtime", RuntimeError(f"threads did not stop: {alive}"))

    return _runtime_result(mode=config.mode, state=state, metrics=metrics, queue=action_queue)


def run_client_runtime(
    *,
    config: ClientRuntimeConfig,
    remote_policy: RemotePolicyClient,
    robot_io: SerializedRobotIO,
    dataset_features: dict[str, dict[str, Any]],
    robot_action_processor: Callable[[tuple[dict[str, float], Any]], Any] | None,
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    safety: ActionSafety,
    log_fn: Callable[[str], None] | None = print,
) -> ClientRuntimeResult:
    if config.mode == "single_step":
        result = run_single_step_runtime(
            config=config,
            remote_policy=remote_policy,
            robot_io=robot_io,
            dataset_features=dataset_features,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            safety=safety,
        )
    else:
        result = run_rtc_runtime(
            config=config,
            remote_policy=remote_policy,
            robot_io=robot_io,
            dataset_features=dataset_features,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            safety=safety,
            log_fn=log_fn,
        )
    raise_for_runtime_error(result)
    return result


def raise_for_runtime_error(result: ClientRuntimeResult) -> None:
    """Turn fail-closed stop reasons into a non-zero client exit."""

    if result.stop_reason and any(
        marker in result.stop_reason
        for marker in ("_loop:", "runtime:", "fully stale", "queue is empty", "timeout")
    ):
        raise RuntimeError(result.stop_reason)


def format_runtime_metrics(
    state: ClientRuntimeState,
    metrics: ClientMetrics,
    action_queue: ActionChunkQueue,
) -> str:
    metric_snapshot = metrics.snapshot()
    queue_snapshot = action_queue.snapshot()
    return (
        "[jz/pi05/rtc-client] "
        f"sensor={state.sensor_ticks} actor={state.actor_ticks} sent={state.sent_actions} "
        f"requests={state.inference_requests} queue={queue_snapshot.depth} "
        f"request_ms={metric_snapshot.latest_request_s * 1000:.1f} "
        f"p95_ms={metric_snapshot.p95_request_s * 1000:.1f} "
        f"server_ms={metric_snapshot.latest_server_s * 1000:.1f} "
        f"drop={metric_snapshot.latest_drop_steps} "
        f"pred_delay={metric_snapshot.latest_predicted_delay_steps}"
    )


def latency_to_steps(latency_s: float, control_dt_s: float) -> int:
    if latency_s <= 0:
        return 0
    if control_dt_s <= 0:
        raise ValueError("control_dt_s must be positive")
    return int(math.ceil(float(latency_s) / float(control_dt_s)))


def _as_chunk_tensor(value: np.ndarray | Tensor, *, expected_dim: int, label: str) -> Tensor:
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != expected_dim:
        raise ValueError(
            f"Server {label} actions must have shape (T,{expected_dim}), got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Server {label} actions contain non-finite values")
    return tensor.clone()


def _copy_frame_snapshot(snapshot: FrameSnapshot) -> FrameSnapshot:
    return FrameSnapshot(
        raw_observation=copy.deepcopy(snapshot.raw_observation),
        observation_frame=copy.deepcopy(snapshot.observation_frame),
        timestamp_s=snapshot.timestamp_s,
        sequence_id=snapshot.sequence_id,
    )


def _duration_elapsed(
    config: ClientRuntimeConfig,
    state: ClientRuntimeState,
    now_s: float,
) -> bool:
    if config.run_time_s <= 0 or now_s - state.start_time_s < config.run_time_s:
        return False
    state.request_stop(f"run_time_s elapsed: {config.run_time_s:.3f}s")
    return True


def _percentile_95(values: deque[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return float(ordered[index])


def _runtime_result(
    *,
    mode: RuntimeMode,
    state: ClientRuntimeState,
    metrics: ClientMetrics,
    queue: ActionChunkQueue | None,
) -> ClientRuntimeResult:
    if state.last_error is not None:
        raise RuntimeError(state.last_error.traceback_text)
    if state.stop_reason is not None and not (
        state.stop_reason.startswith("run_time_s elapsed")
        or state.stop_reason.startswith("max_sent_actions reached")
        or state.stop_reason
        in {
            "KeyboardInterrupt",
            "RTC client runtime exiting",
            "async_single_step client runtime exiting",
            "rtc client runtime exiting",
        }
    ):
        raise RuntimeError(state.stop_reason)
    return ClientRuntimeResult(
        mode=mode,
        stop_reason=state.stop_reason,
        sensor_ticks=state.sensor_ticks,
        actor_ticks=state.actor_ticks,
        sent_actions=state.sent_actions,
        inference_requests=state.inference_requests,
        metrics=metrics.snapshot(),
        queue=None if queue is None else queue.snapshot(),
    )


__all__ = [
    "ClientMetrics",
    "ClientRuntimeConfig",
    "ClientRuntimeResult",
    "ClientRuntimeState",
    "FrameBuffer",
    "FrameSnapshot",
    "InferenceSmokeResult",
    "build_live_observation_frame",
    "build_robot_action",
    "format_runtime_metrics",
    "latency_to_steps",
    "make_request",
    "raise_for_runtime_error",
    "response_to_chunk",
    "run_actor_loop",
    "run_client_runtime",
    "run_inference_smoke",
    "run_producer_loop",
    "run_rtc_runtime",
    "run_sensor_loop",
    "run_single_step_runtime",
]
