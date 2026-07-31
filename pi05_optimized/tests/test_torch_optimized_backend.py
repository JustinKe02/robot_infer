from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tk_infer.pi05_optimized.backends.torch_optimized_backend import (
    TorchBackendOptions,
    TorchOptimizedBackend,
)
from tk_infer.pi05_optimized.runtime.backend_manifest import TorchFeatureFlags

from .helpers import make_request


class FakePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(rtc_config=SimpleNamespace(enabled=True))
        self.calls: list[dict[str, Any]] = []
        self.reset_count = 0

    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        self.calls.append(
            {
                "batch": batch,
                "kwargs": kwargs,
                "rtc_enabled": self.config.rtc_config.enabled,
                "inference_mode": torch.is_inference_mode_enabled(),
            }
        )
        return torch.arange(3 * 16, dtype=torch.float32).reshape(1, 3, 16) / 10

    def reset(self) -> None:
        self.reset_count += 1


class FakePipeline:
    def __init__(self, function: Any) -> None:
        self.function = function
        self.reset_count = 0

    def __call__(self, value: Any) -> Any:
        return self.function(value)

    def reset(self) -> None:
        self.reset_count += 1


class FakePolicyService:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.policy = FakePolicy()
        self.preprocessor = FakePipeline(lambda batch: batch)
        self.postprocessor = FakePipeline(_expand_raw18)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "device": "cpu",
            "checkpoint_fingerprint": "fake-phase2-checkpoint",
            "checkpoint_step": 1,
            "model_action_dim": 16,
            "wire_action_dim": 18,
            "camera_profile": "head_right",
            "camera_shapes": {"observation.images.camera_right": [3, 2, 2]},
        }


def _expand_raw18(actions: torch.Tensor) -> torch.Tensor:
    output = torch.zeros((*actions.shape[:-1], 18), dtype=torch.float32)
    output[..., :14] = actions[..., :14]
    output[..., 14] = actions[..., 14]
    output[..., 15] = 80
    output[..., 16] = actions[..., 15]
    output[..., 17] = 80
    return output


@pytest.mark.parametrize("mode", ["single_step", "rtc"])
def test_optimized_backend_preserves_paired_contract_and_rtc_mode(mode: str) -> None:
    service = FakePolicyService()
    backend = TorchOptimizedBackend(service)  # type: ignore[arg-type]

    response = backend.infer(make_request(mode=mode))

    assert response.raw_actions.shape == (3, 16)
    assert response.processed_actions.shape == (3, 18)
    np.testing.assert_array_equal(response.processed_actions[:, 15], 80)
    np.testing.assert_array_equal(response.processed_actions[:, 17], 80)
    call = service.policy.calls[-1]
    assert call["rtc_enabled"] is (mode == "rtc")
    assert service.policy.config.rtc_config.enabled is True
    if mode == "rtc":
        assert call["kwargs"]["prev_chunk_left_over"].shape == (2, 16)
        assert call["kwargs"]["inference_delay"] == 1
    else:
        assert call["kwargs"] == {}


def test_inference_mode_flag_is_observed_only_when_enabled() -> None:
    reference_service = FakePolicyService()
    reference = TorchOptimizedBackend(reference_service)  # type: ignore[arg-type]
    reference.infer(make_request())
    assert reference_service.policy.calls[-1]["inference_mode"] is False

    optimized_service = FakePolicyService()
    optimized = TorchOptimizedBackend(
        optimized_service,  # type: ignore[arg-type]
        options=TorchBackendOptions(features=TorchFeatureFlags(inference_mode=True)),
    )
    optimized.infer(make_request())
    assert optimized_service.policy.calls[-1]["inference_mode"] is True
    assert optimized.health()["supported_modes"] == ["single_step"]
    with pytest.raises(ValueError, match="supports single_step only"):
        optimized.infer(make_request(mode="rtc"))


def test_health_exposes_bounded_stage_metrics_and_manifest() -> None:
    service = FakePolicyService()
    backend = TorchOptimizedBackend(  # type: ignore[arg-type]
        service,
        options=TorchBackendOptions(metrics_window_size=2, synchronize_stages=True),
    )

    for request_id in range(3):
        backend.infer(make_request(request_id=request_id))

    health = backend.health()
    assert health["backend"] == "torch_optimized"
    assert health["backend_phase"] == 2
    assert health["backend_inference_count"] == 3
    assert health["backend_failure_count"] == 0
    assert health["backend_stage_timing_mode"] == "synchronized_boundary_diagnostic"
    assert health["backend_stage_metrics"]["total_s"]["count"] == 2
    assert health["backend_manifest"]["schema_version"] == 1
    assert health["backend_manifest"]["checkpoint_fingerprint"] == "fake-phase2-checkpoint"


def test_deterministic_warmup_is_counted_separately_and_resets_components() -> None:
    service = FakePolicyService()
    backend = TorchOptimizedBackend(service)  # type: ignore[arg-type]

    backend.warmup((make_request(),), iterations=2, seed=99)

    health = backend.health()
    assert health["backend_warmup_count"] == 2
    assert health["backend_inference_count"] == 0
    assert service.policy.reset_count == 3
    assert service.preprocessor.reset_count == 3
    assert service.postprocessor.reset_count == 3


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (TorchFeatureFlags(bf16_autocast=True), "require a CUDA"),
        (TorchFeatureFlags(pinned_memory=True), "require a CUDA"),
        (TorchFeatureFlags(static_buffers=True), "static_buffers is not supported"),
        (TorchFeatureFlags(cuda_graph=True), "cuda_graph is not supported"),
        (TorchFeatureFlags(non_blocking_copies=True), "require a CUDA"),
    ],
)
def test_unsupported_feature_combinations_fail_during_backend_startup(
    features: TorchFeatureFlags,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TorchOptimizedBackend(  # type: ignore[arg-type]
            FakePolicyService(),
            options=TorchBackendOptions(features=features),
        )
