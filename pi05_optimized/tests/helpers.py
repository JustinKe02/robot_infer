from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tk_infer.pi05.runtime.protocol import PROTOCOL_VERSION, InferenceRequest, InferenceResponse


def make_request(*, request_id: int = 7, mode: str = "single_step") -> InferenceRequest:
    request = InferenceRequest(
        request_id=request_id,
        mode=mode,  # type: ignore[arg-type]
        observation_frame={"observation.state": np.zeros(18, dtype=np.float32)},
        task="jz robot pin timed vr teleoperation",
        robot_type="jz_robot_pin_timed",
        obs_sequence_id=11,
        predicted_delay_steps=1 if mode == "rtc" else 0,
        execution_horizon=10,
    )
    if mode == "rtc":
        request.prev_chunk_left_over = np.zeros((2, 16), dtype=np.float32)
    return request


def make_response(request: InferenceRequest, *, steps: int = 3) -> InferenceResponse:
    model_actions = np.arange(steps * 16, dtype=np.float32).reshape(steps, 16) / 10.0
    robot_actions = np.zeros((steps, 18), dtype=np.float32)
    robot_actions[:, :14] = model_actions[:, :14]
    robot_actions[:, 14] = model_actions[:, 14]
    robot_actions[:, 15] = 80.0
    robot_actions[:, 16] = model_actions[:, 15]
    robot_actions[:, 17] = 80.0
    return InferenceResponse(
        request_id=request.request_id,
        mode=request.mode,
        raw_actions=model_actions,
        processed_actions=robot_actions,
        server_latency_s=0.01,
        model_latency_s=0.005,
        raw_action_shape=model_actions.shape,
        processed_action_shape=robot_actions.shape,
    )


@dataclass
class FakeReferenceService:
    infer_requests: list[InferenceRequest] = field(default_factory=list)
    response_mutator: Any = None

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "supported_modes": ["single_step", "rtc"],
            "checkpoint_fingerprint": "fake-checkpoint",
        }

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        self.infer_requests.append(request)
        response = make_response(request)
        if self.response_mutator is not None:
            self.response_mutator(response)
        return response


__all__ = ["FakeReferenceService", "make_request", "make_response"]
