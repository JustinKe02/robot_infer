from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tk_infer.pi05_optimized.backends.torch_rtc_conditioned_backend import (
    REQUIRED_RTC_PARAMETER_SUFFIXES,
    RTCConditionedCheckpointContract,
    TorchRTCConditionedBackend,
    inspect_rtc_conditioned_checkpoint,
)
from tk_infer.pi05_optimized.runtime.optimized_client import OptimizedClient, OptimizedClientConfig
from tk_infer.pi05_optimized.runtime.policy_service import OptimizedPolicyService
from tk_infer.pi05_optimized.runtime.timed_observation import TimedObservation

from .helpers import make_request


class FakeMetadata:
    def health_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_fingerprint": "fake-rtc-conditioned",
            "checkpoint_step": 10,
            "configured_steps": 10,
            "complete_step": True,
            "model_action_dim": 16,
            "wire_action_dim": 18,
            "camera_profile": "three_camera",
            "camera_shapes": {},
        }


class FakePolicy:
    def __init__(self, *, include_rtc_parameters: bool = True) -> None:
        self.config = SimpleNamespace(
            rtc_training=SimpleNamespace(enabled=True),
            rtc_config=None,
            use_amp=False,
        )
        self.calls: list[dict[str, Any]] = []
        self.include_rtc_parameters = include_rtc_parameters

    def named_parameters(self):
        if not self.include_rtc_parameters:
            return iter(())
        return iter(
            (f"model.{suffix}", torch.nn.Parameter(torch.zeros(1)))
            for suffix in REQUIRED_RTC_PARAMETER_SUFFIXES
        )

    def predict_action_chunk(self, _batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        self.calls.append({"kwargs": kwargs, "inference_mode": torch.is_inference_mode_enabled()})
        actions = torch.arange(3 * 16, dtype=torch.float32).reshape(1, 3, 16)
        prefix = kwargs.get("action_prefix")
        prefix_length = int(kwargs.get("prefix_length", 0))
        if prefix is not None:
            actions[:, :prefix_length] = prefix[:, :prefix_length]
        return actions


class FakePipeline:
    def __init__(self, function: Any) -> None:
        self.function = function

    def __call__(self, value: Any) -> Any:
        return self.function(value)


def _expand_raw18(actions: torch.Tensor) -> torch.Tensor:
    output = torch.zeros((*actions.shape[:-1], 18), dtype=torch.float32)
    output[..., :14] = actions[..., :14]
    output[..., 14] = actions[..., 14]
    output[..., 15] = 80
    output[..., 16] = actions[..., 15]
    output[..., 17] = 80
    return output


def _backend(*, include_rtc_parameters: bool = True) -> tuple[TorchRTCConditionedBackend, FakePolicy]:
    policy = FakePolicy(include_rtc_parameters=include_rtc_parameters)
    bundle = SimpleNamespace(
        policy=policy,
        preprocessor=FakePipeline(lambda batch: batch),
        postprocessor=FakePipeline(_expand_raw18),
        metadata=FakeMetadata(),
    )
    contract = RTCConditionedCheckpointContract(
        chunk_size=3,
        max_delay=2,
        min_postfix_steps=1,
        observed_delay_histogram=(),
        observed_histogram_weight=0.9,
    )
    return (
        TorchRTCConditionedBackend(  # type: ignore[arg-type]
            bundle,
            contract=contract,
            device=torch.device("cpu"),
            expected_task="jz robot pin timed vr teleoperation",
        ),
        policy,
    )


def test_rtc_request_uses_delay_sized_clean_prefix_without_vjp() -> None:
    backend, policy = _backend()
    request = make_request(mode="rtc")

    response = backend.infer(request)

    call = policy.calls[-1]
    assert call["inference_mode"] is True
    assert call["kwargs"]["prefix_length"] == 1
    np.testing.assert_array_equal(
        call["kwargs"]["action_prefix"].cpu().numpy()[0],
        request.prev_chunk_left_over[:1],
    )
    assert "inference_delay" not in call["kwargs"]
    assert "prev_chunk_left_over" not in call["kwargs"]
    np.testing.assert_array_equal(response.raw_actions[:1], request.prev_chunk_left_over[:1])
    health = backend.health()
    assert health["backend"] == "torch_rtc_conditioned"
    assert health["inference_time_vjp_rtc_enabled"] is False
    assert health["backend_prefix_clamp_checks"] == 1


def test_single_step_and_zero_delay_rtc_use_full_chunk_path() -> None:
    backend, policy = _backend()

    backend.infer(make_request(mode="single_step"))
    zero_delay = make_request(mode="rtc")
    zero_delay.predicted_delay_steps = 0
    zero_delay.prev_chunk_left_over = None
    backend.infer(zero_delay)

    assert policy.calls[-2]["kwargs"] == {}
    assert policy.calls[-1]["kwargs"] == {}


def test_conditioned_backend_integrates_with_two_rtc_client_cycles() -> None:
    backend, policy = _backend()
    service = OptimizedPolicyService(backend=backend)

    class ObservationSource:
        def __init__(self) -> None:
            self._observations = iter(
                (
                    TimedObservation(
                        observation_frame={"observation.state": np.zeros(18, dtype=np.float32)},
                        sequence_id=1,
                        receive_monotonic_s=1.0,
                        build_started_monotonic_s=1.0,
                        build_ready_monotonic_s=1.0,
                    ),
                    TimedObservation(
                        observation_frame={"observation.state": np.zeros(18, dtype=np.float32)},
                        sequence_id=2,
                        receive_monotonic_s=1.05,
                        build_started_monotonic_s=1.05,
                        build_ready_monotonic_s=1.05,
                    ),
                )
            )

        def read(self) -> TimedObservation:
            return next(self._observations)

    class CapturingPolicyClient:
        def __init__(self) -> None:
            self.requests = []
            self.responses = []

        def infer(self, request):
            self.requests.append(request)
            response = service.infer(request)
            self.responses.append(response)
            return response

    class RecordingSink:
        def __init__(self) -> None:
            self.actions = []

        def write(self, action):
            self.actions.append(action.detach().clone())

    policy_client = CapturingPolicyClient()
    sink = RecordingSink()
    clock = iter((1.0, 1.01, 1.02, 1.05, 1.06, 1.07))
    client = OptimizedClient(
        config=OptimizedClientConfig(
            task="jz robot pin timed vr teleoperation",
            mode="rtc",
        ),
        observation_source=ObservationSource(),
        policy_client=policy_client,
        action_sink=sink,
        clock=lambda: next(clock),
    )

    first, second = client.run_cycles(2)

    assert first.predicted_delay_steps == 0
    assert policy_client.requests[0].prev_chunk_left_over is None
    assert policy.calls[0]["kwargs"] == {}

    assert second.predicted_delay_steps == 1
    leftover = policy_client.requests[1].prev_chunk_left_over
    assert leftover is not None
    np.testing.assert_array_equal(leftover, policy_client.responses[0].raw_actions[2:])
    assert policy.calls[1]["kwargs"]["prefix_length"] == second.predicted_delay_steps
    np.testing.assert_array_equal(
        policy.calls[1]["kwargs"]["action_prefix"].cpu().numpy()[0],
        leftover[: second.predicted_delay_steps],
    )
    np.testing.assert_array_equal(
        policy_client.responses[1].raw_actions[: second.predicted_delay_steps],
        leftover[: second.predicted_delay_steps],
    )

    snapshot = client.action_queue.snapshot()
    assert snapshot.merge_calls == 2
    assert snapshot.popped_actions == 2
    assert snapshot.raw_leftover_steps == 1
    assert len(sink.actions) == 2
    torch.testing.assert_close(
        sink.actions[1],
        torch.as_tensor(policy_client.responses[1].processed_actions[1]),
    )


def test_rtc_prefix_overflow_and_missing_leftover_fail_closed() -> None:
    backend, _policy = _backend()
    overflow = make_request(mode="rtc")
    overflow.predicted_delay_steps = 3
    with pytest.raises(ValueError, match="exceeds conditioned max prefix"):
        backend.infer(overflow)

    missing = make_request(mode="rtc")
    missing.prev_chunk_left_over = None
    with pytest.raises(ValueError, match="requires model16 leftover"):
        backend.infer(missing)

    short = make_request(mode="rtc")
    short.predicted_delay_steps = 2
    short.prev_chunk_left_over = np.zeros((1, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="requires 2 leftover steps"):
        backend.infer(short)


def test_backend_rejects_policy_without_learned_rtc_parameters() -> None:
    with pytest.raises(ValueError, match="lacks required learned parameters"):
        _backend(include_rtc_parameters=False)


def test_backend_rejects_task_different_from_locked_training_task() -> None:
    backend, policy = _backend()
    request = make_request()
    request.task = "Put the bottle on the right into the basket on the left."

    with pytest.raises(ValueError, match="differs from the locked training task"):
        backend.infer(request)

    assert policy.calls == []


def test_checkpoint_contract_rejects_standard_or_invalid_rtc_config(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    config_path.write_text(json.dumps({"type": "pi05", "chunk_size": 50}), encoding="utf-8")
    with pytest.raises(ValueError, match="rtc_training.enabled=true"):
        inspect_rtc_conditioned_checkpoint(checkpoint)

    config_path.write_text(
        json.dumps(
            {
                "type": "pi05",
                "chunk_size": 50,
                "rtc_config": {"enabled": True},
                "rtc_training": {"enabled": True, "max_delay": 10, "min_postfix_steps": 1},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not enable inference-time VJP"):
        inspect_rtc_conditioned_checkpoint(checkpoint)


def test_real_rtc_checkpoint_declares_expected_contract() -> None:
    checkpoint = (
        Path(__file__).resolve().parents[2]
        / "pi05/checkpoints/pi05_jz100_model16_head_left_right_expert_b_rtc_e10_seed1000_010600/"
        "pretrained_model"
    )

    contract = inspect_rtc_conditioned_checkpoint(checkpoint)

    assert contract.chunk_size == 50
    assert contract.max_delay == 10
    assert contract.min_postfix_steps == 1
    assert contract.maximum_prefix_length == 10
