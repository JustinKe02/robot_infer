from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np

PROTOCOL_VERSION = 2
CONTENT_TYPE = "application/x-python-pickle"
SINGLE_STEP_MODE = "single_step"
RTC_MODE = "rtc"
SUPPORTED_MODES = (SINGLE_STEP_MODE, RTC_MODE)
MODEL_ACTION_DIM = 16
WIRE_ACTION_DIM = 18

InferenceMode: TypeAlias = Literal["single_step", "rtc"]


@dataclass(slots=True)
class InferenceRequest:
    request_id: int
    mode: InferenceMode
    observation_frame: dict[str, Any]
    task: str
    robot_type: str
    obs_sequence_id: int
    predicted_delay_steps: int = 0
    prev_chunk_left_over: np.ndarray | None = None
    execution_horizon: int = 10

    def validate(self) -> None:
        if isinstance(self.request_id, bool) or not isinstance(self.request_id, int) or self.request_id < 0:
            raise ValueError("request_id must be a non-negative integer")
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported inference mode: {self.mode!r}")
        if not isinstance(self.observation_frame, dict) or not self.observation_frame:
            raise ValueError("observation_frame must be a non-empty dictionary")
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be a non-empty string")
        if self.robot_type != "jz_robot_pin_timed":
            raise ValueError(f"robot_type must be 'jz_robot_pin_timed', got {self.robot_type!r}")
        if isinstance(self.obs_sequence_id, bool) or not isinstance(self.obs_sequence_id, int):
            raise ValueError("obs_sequence_id must be an integer")
        if isinstance(self.predicted_delay_steps, bool) or not isinstance(self.predicted_delay_steps, int):
            raise ValueError("predicted_delay_steps must be an integer")
        if self.predicted_delay_steps < 0:
            raise ValueError("predicted_delay_steps must be non-negative")
        if isinstance(self.execution_horizon, bool) or not isinstance(self.execution_horizon, int):
            raise ValueError("execution_horizon must be an integer")
        if self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")

        if self.mode == SINGLE_STEP_MODE and self.prev_chunk_left_over is not None:
            raise ValueError("single_step requests must not include prev_chunk_left_over")
        if self.prev_chunk_left_over is not None:
            leftover = np.asarray(self.prev_chunk_left_over)
            if leftover.ndim != 2 or leftover.shape[1] != MODEL_ACTION_DIM:
                raise ValueError(
                    "prev_chunk_left_over must have shape (T,16), "
                    f"got {tuple(leftover.shape)}"
                )
            if not np.isfinite(leftover).all():
                raise ValueError("prev_chunk_left_over contains non-finite values")


@dataclass(slots=True)
class InferenceResponse:
    request_id: int
    mode: InferenceMode
    raw_actions: np.ndarray
    processed_actions: np.ndarray
    server_latency_s: float
    model_latency_s: float
    raw_action_shape: tuple[int, ...]
    processed_action_shape: tuple[int, ...]
    error: str | None = None

    @property
    def action_shape(self) -> tuple[int, ...]:
        """Compatibility alias for the robot-ready processed action shape."""

        return self.processed_action_shape

    def validate(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported response mode: {self.mode!r}")
        raw = np.asarray(self.raw_actions)
        processed = np.asarray(self.processed_actions)
        if raw.ndim != 2 or raw.shape[1] != MODEL_ACTION_DIM:
            raise ValueError(f"raw_actions must have shape (T,16), got {tuple(raw.shape)}")
        if processed.ndim != 2 or processed.shape[1] != WIRE_ACTION_DIM:
            raise ValueError(f"processed_actions must have shape (T,18), got {tuple(processed.shape)}")
        if raw.shape[0] != processed.shape[0]:
            raise ValueError(
                "raw_actions and processed_actions must have the same temporal length, "
                f"got {raw.shape[0]} and {processed.shape[0]}"
            )
        if tuple(raw.shape) != tuple(self.raw_action_shape):
            raise ValueError("raw_action_shape does not match raw_actions")
        if tuple(processed.shape) != tuple(self.processed_action_shape):
            raise ValueError("processed_action_shape does not match processed_actions")
        if not np.isfinite(raw).all() or not np.isfinite(processed).all():
            raise ValueError("Inference response contains non-finite actions")


def dumps_payload(payload: Any) -> bytes:
    envelope = {"version": PROTOCOL_VERSION, "payload": payload}
    return pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)


def loads_payload(data: bytes) -> Any:
    if not isinstance(data, bytes | bytearray | memoryview):
        raise TypeError("Protocol payload must be bytes-like")
    envelope = pickle.loads(data)
    if not isinstance(envelope, dict):
        raise ValueError(f"Invalid protocol envelope type: {type(envelope)}")
    version = envelope.get("version")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {version}; expected {PROTOCOL_VERSION}")
    if "payload" not in envelope:
        raise ValueError("Protocol envelope does not contain payload")
    return envelope["payload"]
