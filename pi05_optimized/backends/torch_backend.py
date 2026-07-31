from __future__ import annotations

from typing import Any

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from tk_infer.pi05.runtime.policy_service import PolicyService, PolicyServiceConfig
from tk_infer.pi05.runtime.protocol import InferenceRequest, InferenceResponse
from tk_infer.pi05_optimized.config import OptimizedRuntimeConfig


class TorchPolicyBackend:
    """Exact adapter around the trusted PI0.5 PyTorch service."""

    def __init__(self, service: PolicyService) -> None:
        self._service = service

    @property
    def name(self) -> str:
        return "torch"

    @property
    def service(self) -> PolicyService:
        return self._service

    @classmethod
    def from_runtime_config(cls, config: OptimizedRuntimeConfig) -> TorchPolicyBackend:
        if config.backend != "torch":
            raise ValueError(f"TorchPolicyBackend cannot load backend={config.backend!r}")
        policy_path, tokenizer_path = config.require_model_paths()
        service_config = PolicyServiceConfig(
            policy_path=policy_path,
            tokenizer_path=tokenizer_path,
            device=config.device,
            require_complete_step=config.require_complete_step,
        )
        rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=RTCAttentionSchedule(config.rtc_prefix_attention_schedule),
            max_guidance_weight=config.rtc_max_guidance_weight,
            execution_horizon=config.rtc_execution_horizon,
            debug=config.rtc_debug,
        )
        return cls(PolicyService.from_config(service_config, rtc_config=rtc_config))

    def health(self) -> dict[str, Any]:
        health = dict(self._service.health())
        health["backend"] = self.name
        health["backend_implementation"] = "tk_infer.pi05.runtime.policy_service.PolicyService"
        return health

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        return self._service.infer(request)


__all__ = ["TorchPolicyBackend"]
