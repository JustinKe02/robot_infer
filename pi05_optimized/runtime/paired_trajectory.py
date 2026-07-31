from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tk_infer.pi05.runtime.protocol import (
    MAX_ACTION_CHUNK_STEPS,
    MODEL_ACTION_DIM,
    WIRE_ACTION_DIM,
    InferenceResponse,
)

LEFT_FORCE_INDEX = 15
RIGHT_FORCE_INDEX = 17
COMMAND_FORCE = 80.0


@dataclass(frozen=True, slots=True)
class PairedTrajectory:
    """Time-aligned model16 and raw18 trajectories with immutable array storage."""

    model_actions: NDArray[np.float32]
    robot_actions: NDArray[np.float32]
    request_id: int
    mode: str
    source_observation_seq: int
    predicted_delay_steps: int

    def __post_init__(self) -> None:
        model_actions = _readonly_float32_copy(self.model_actions, name="model_actions")
        robot_actions = _readonly_float32_copy(self.robot_actions, name="robot_actions")
        if model_actions.ndim != 2 or model_actions.shape[1] != MODEL_ACTION_DIM:
            raise ValueError(f"model_actions must have shape (T,16), got {tuple(model_actions.shape)}")
        if robot_actions.ndim != 2 or robot_actions.shape[1] != WIRE_ACTION_DIM:
            raise ValueError(f"robot_actions must have shape (T,18), got {tuple(robot_actions.shape)}")
        if model_actions.shape[0] != robot_actions.shape[0]:
            raise ValueError(
                "model16 and raw18 trajectories must have the same temporal length, "
                f"got {model_actions.shape[0]} and {robot_actions.shape[0]}"
            )
        if not 1 <= model_actions.shape[0] <= MAX_ACTION_CHUNK_STEPS:
            raise ValueError(
                f"trajectory length must be in 1..{MAX_ACTION_CHUNK_STEPS}, got {model_actions.shape[0]}"
            )
        if isinstance(self.request_id, bool) or not isinstance(self.request_id, int) or self.request_id < 0:
            raise ValueError("request_id must be a non-negative integer")
        if isinstance(self.source_observation_seq, bool) or not isinstance(self.source_observation_seq, int):
            raise ValueError("source_observation_seq must be an integer")
        if (
            isinstance(self.predicted_delay_steps, bool)
            or not isinstance(self.predicted_delay_steps, int)
            or self.predicted_delay_steps < 0
        ):
            raise ValueError("predicted_delay_steps must be a non-negative integer")
        _validate_force_slots(robot_actions)
        object.__setattr__(self, "model_actions", model_actions)
        object.__setattr__(self, "robot_actions", robot_actions)

    @property
    def steps(self) -> int:
        return int(self.model_actions.shape[0])

    @classmethod
    def from_response(
        cls,
        response: InferenceResponse,
        *,
        source_observation_seq: int,
        predicted_delay_steps: int,
    ) -> PairedTrajectory:
        response.validate()
        return cls(
            model_actions=response.raw_actions,
            robot_actions=response.processed_actions,
            request_id=response.request_id,
            mode=response.mode,
            source_observation_seq=source_observation_seq,
            predicted_delay_steps=predicted_delay_steps,
        )

    def copy(self) -> PairedTrajectory:
        return PairedTrajectory(
            model_actions=self.model_actions,
            robot_actions=self.robot_actions,
            request_id=self.request_id,
            mode=self.mode,
            source_observation_seq=self.source_observation_seq,
            predicted_delay_steps=self.predicted_delay_steps,
        )


def _readonly_float32_copy(value: object, *, name: str) -> NDArray[np.float32]:
    try:
        array = np.array(value, dtype=np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _validate_force_slots(robot_actions: NDArray[np.float32]) -> None:
    left_force = robot_actions[:, LEFT_FORCE_INDEX]
    right_force = robot_actions[:, RIGHT_FORCE_INDEX]
    if not np.equal(left_force, COMMAND_FORCE).all() or not np.equal(right_force, COMMAND_FORCE).all():
        raise ValueError(
            f"raw18 force slots [{LEFT_FORCE_INDEX}, {RIGHT_FORCE_INDEX}] must be exactly {COMMAND_FORCE:g}"
        )


__all__ = [
    "COMMAND_FORCE",
    "LEFT_FORCE_INDEX",
    "PairedTrajectory",
    "RIGHT_FORCE_INDEX",
]
