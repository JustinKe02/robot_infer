from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from lerobot.robots.jz_robot_pin_timed.training_schema import (
    RAW_FEATURE_NAMES,
    JZPinTrainingSchema,
    build_training_schema_manifest,
)
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime import checkpoint
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.policy_service import PolicyService
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.protocol import (
    InferenceRequest,
    InferenceResponse,
)
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.ybd.policy_service import (
    HEAD_RIGHT_CAMERA_KEYS,
    HEAD_RIGHT_CAMERA_SHAPES,
    YBDPolicyService,
    hold_raw18_left_side,
    install_head_right_camera_contract,
)


def _audited_schema() -> JZPinTrainingSchema:
    return JZPinTrainingSchema(
        build_training_schema_manifest(
            left_observation_source="measured_opening",
            right_observation_source="commanded_opening",
            left_observation_raw_closed=0.0,
            left_observation_raw_open=100.0,
            right_observation_raw_closed=100.0,
            right_observation_raw_open=0.0,
            left_action_raw_closed=100.0,
            left_action_raw_open=0.0,
            right_action_raw_closed=100.0,
            right_action_raw_open=0.0,
            left_command_force=80.0,
            right_command_force=80.0,
        )
    )


def _raw_state() -> torch.Tensor:
    state = torch.arange(18, dtype=torch.float32)
    state[14] = 25.0
    state[15] = 17.0
    state[16] = 40.0
    state[17] = 23.0
    return state


def _fake_checkpoint_postprocess(
    normalized_actions: torch.Tensor,
    *,
    schema: JZPinTrainingSchema,
) -> torch.Tensor:
    physical_model16 = normalized_actions * 2.0 + 10.0
    return torch.as_tensor(schema.expand_action(physical_model16), dtype=torch.float32)


def test_hold_raw18_left_side_preserves_right_predictions_and_input() -> None:
    schema = _audited_schema()
    processed = torch.arange(2 * 18, dtype=torch.float32).reshape(2, 18)
    original = processed.clone()

    held = torch.from_numpy(
        hold_raw18_left_side(
            processed,
            raw_observation_state=_raw_state(),
            schema=schema,
        )
    )

    torch.testing.assert_close(held[:, 0:7], _raw_state()[0:7].expand(2, -1))
    torch.testing.assert_close(held[:, 7:14], original[:, 7:14])
    torch.testing.assert_close(held[:, 14], torch.full((2,), 75.0))
    torch.testing.assert_close(held[:, 15], torch.full((2,), 80.0))
    torch.testing.assert_close(held[:, 16], original[:, 16])
    torch.testing.assert_close(held[:, 17], torch.full((2,), 80.0))
    torch.testing.assert_close(processed, original)
    assert schema.raw_feature_names["action"] == RAW_FEATURE_NAMES


def test_hold_is_applied_after_checkpoint_unnormalization() -> None:
    schema = _audited_schema()
    normalized = torch.linspace(-1.0, 1.0, steps=2 * 16).reshape(2, 16)
    model_state = torch.as_tensor(schema.project_observation(_raw_state()), dtype=torch.float32)
    wrongly_clamped_normalized = normalized.clone()
    wrongly_clamped_normalized[:, 0:7] = model_state[0:7]
    wrongly_clamped_normalized[:, 14] = model_state[14]

    wrong_raw18 = _fake_checkpoint_postprocess(wrongly_clamped_normalized, schema=schema)
    assert not torch.allclose(wrong_raw18[:, 0:7], _raw_state()[0:7].expand(2, -1))

    checkpoint_raw18 = _fake_checkpoint_postprocess(normalized, schema=schema)
    held = torch.from_numpy(
        hold_raw18_left_side(
            checkpoint_raw18,
            raw_observation_state=_raw_state(),
            schema=schema,
        )
    )

    torch.testing.assert_close(held[:, 0:7], _raw_state()[0:7].expand(2, -1))
    torch.testing.assert_close(held[:, 7:14], checkpoint_raw18[:, 7:14])
    torch.testing.assert_close(held[:, 14], torch.full((2,), 75.0))
    torch.testing.assert_close(held[:, 15], torch.full((2,), 80.0))
    torch.testing.assert_close(held[:, 16], checkpoint_raw18[:, 16])
    torch.testing.assert_close(held[:, 17], torch.full((2,), 80.0))


@pytest.mark.parametrize(
    "raw_state",
    [
        None,
        np.zeros(17, dtype=np.float32),
        np.array([*([0.0] * 17), np.nan], dtype=np.float32),
    ],
)
def test_hold_raw18_left_side_rejects_invalid_state(raw_state: Any) -> None:
    with pytest.raises(ValueError, match="raw18 observation.state"):
        hold_raw18_left_side(
            torch.zeros((2, 18), dtype=torch.float32),
            raw_observation_state=raw_state,
            schema=_audited_schema(),
        )


@pytest.mark.parametrize(
    "processed",
    [
        np.zeros((2, 16), dtype=np.float32),
        np.array([[*([0.0] * 17), np.inf]], dtype=np.float32),
    ],
)
def test_hold_raw18_left_side_rejects_invalid_processed_actions(processed: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"processed actions \(T,18\)|processed raw18 actions must be finite"):
        hold_raw18_left_side(
            processed,
            raw_observation_state=_raw_state(),
            schema=_audited_schema(),
        )


def test_install_head_right_camera_contract_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoint, "EXPECTED_CAMERA_KEYS", ("sentinel",))
    monkeypatch.setattr(checkpoint, "EXPECTED_CAMERA_SHAPES", {"sentinel": (1,)})

    install_head_right_camera_contract()

    assert checkpoint.EXPECTED_CAMERA_KEYS == HEAD_RIGHT_CAMERA_KEYS
    assert checkpoint.EXPECTED_CAMERA_SHAPES == HEAD_RIGHT_CAMERA_SHAPES
    assert "observation.images.camera_left" not in checkpoint.EXPECTED_CAMERA_KEYS


def test_ybd_service_clamps_postprocessed_raw18_without_mutating_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _audited_schema()
    raw_state = _raw_state().numpy()
    left_image = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = {
        "observation.state": raw_state,
        "observation.images.camera_head": np.zeros((720, 1280, 3), dtype=np.uint8),
        "observation.images.camera_left": left_image,
        "observation.images.camera_right": np.zeros((480, 640, 3), dtype=np.uint8),
    }
    request = InferenceRequest(
        request_id=1,
        mode="single_step",
        observation_frame=frame,
        task="Put the bottle on the right into the basket on the left.",
        robot_type="jz_robot_pin_timed",
        obs_sequence_id=9,
    )
    normalized = torch.linspace(-1.0, 1.0, steps=2 * 16).reshape(2, 16)
    checkpoint_raw18 = _fake_checkpoint_postprocess(normalized, schema=schema).numpy()
    original_checkpoint_raw18 = checkpoint_raw18.copy()
    captured: dict[str, Any] = {}

    def fake_infer(_self: PolicyService, adapted_request: InferenceRequest) -> InferenceResponse:
        captured["request"] = adapted_request
        return InferenceResponse(
            request_id=adapted_request.request_id,
            mode=adapted_request.mode,
            raw_actions=normalized.numpy().copy(),
            processed_actions=checkpoint_raw18,
            server_latency_s=0.1,
            model_latency_s=0.08,
            raw_action_shape=(2, 16),
            processed_action_shape=(2, 18),
        )

    monkeypatch.setattr(PolicyService, "infer", fake_infer)
    service = object.__new__(YBDPolicyService)
    service.bundle = SimpleNamespace(schema=schema)

    response = service.infer(request)

    adapted_request = captured["request"]
    assert adapted_request is not request
    assert "observation.images.camera_left" not in adapted_request.observation_frame
    assert request.observation_frame is frame
    assert request.observation_frame["observation.images.camera_left"] is left_image
    np.testing.assert_array_equal(response.raw_actions, normalized.numpy())
    np.testing.assert_array_equal(checkpoint_raw18, original_checkpoint_raw18)
    np.testing.assert_allclose(response.processed_actions[:, 0:7], np.broadcast_to(raw_state[0:7], (2, 7)))
    np.testing.assert_allclose(response.processed_actions[:, 7:14], original_checkpoint_raw18[:, 7:14])
    np.testing.assert_allclose(response.processed_actions[:, 14], 75.0)
    np.testing.assert_allclose(response.processed_actions[:, 15], 80.0)
    np.testing.assert_allclose(response.processed_actions[:, 16], original_checkpoint_raw18[:, 16])
    np.testing.assert_allclose(response.processed_actions[:, 17], 80.0)
    assert response.processed_action_shape == (2, 18)
