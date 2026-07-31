from __future__ import annotations

import json
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.utils import get_safe_torch_device
from tk_infer.pi05.runtime.checkpoint import PolicyBundle, load_policy_bundle
from tk_infer.pi05.runtime.policy_service import ensure_chunk_batch, postprocess_action_chunk
from tk_infer.pi05.runtime.protocol import (
    MODEL_ACTION_DIM,
    RTC_MODE,
    SINGLE_STEP_MODE,
    InferenceRequest,
    InferenceResponse,
)
from tk_infer.pi05_optimized.config import OptimizedRuntimeConfig
from tk_infer.pi05_optimized.runtime.backend_manifest import (
    TorchFeatureFlags,
    build_torch_backend_manifest,
)
from tk_infer.pi05_optimized.runtime.paired_trajectory import PairedTrajectory

RTC_CONDITIONED_INFERENCE_CONTRACT = "training_time_action_conditioning_v1"
REQUIRED_RTC_PARAMETER_SUFFIXES = (
    "rtc_prefix_embedding.weight",
    "rtc_token_time_mlp_in.weight",
    "rtc_token_time_mlp_in.bias",
    "rtc_token_time_mlp_out.weight",
    "rtc_token_time_mlp_out.bias",
)


@dataclass(frozen=True, slots=True)
class RTCConditionedCheckpointContract:
    chunk_size: int
    max_delay: int
    min_postfix_steps: int
    observed_delay_histogram: tuple[float, ...]
    observed_histogram_weight: float

    @property
    def maximum_prefix_length(self) -> int:
        return min(self.max_delay, self.chunk_size - self.min_postfix_steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "inference_contract": RTC_CONDITIONED_INFERENCE_CONTRACT,
            "chunk_size": self.chunk_size,
            "max_delay": self.max_delay,
            "min_postfix_steps": self.min_postfix_steps,
            "maximum_prefix_length": self.maximum_prefix_length,
            "observed_delay_histogram": list(self.observed_delay_histogram),
            "observed_histogram_weight": self.observed_histogram_weight,
        }


def inspect_rtc_conditioned_checkpoint(policy_path: str | Path) -> RTCConditionedCheckpointContract:
    path = Path(policy_path).expanduser().resolve(strict=True)
    config_path = path / "config.json"
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or config.get("type") != "pi05":
        raise ValueError("RTC-conditioned checkpoint must contain a PI0.5 config.json object")
    if config.get("rtc_config") is not None:
        raise ValueError("RTC-conditioned checkpoint must not enable inference-time VJP rtc_config")
    rtc_training = config.get("rtc_training")
    if not isinstance(rtc_training, dict) or rtc_training.get("enabled") is not True:
        raise ValueError("RTC-conditioned checkpoint requires rtc_training.enabled=true")

    chunk_size = _positive_int("chunk_size", config.get("chunk_size"))
    max_delay = _non_negative_int("rtc_training.max_delay", rtc_training.get("max_delay"))
    min_postfix_steps = _positive_int("rtc_training.min_postfix_steps", rtc_training.get("min_postfix_steps"))
    if min_postfix_steps > chunk_size:
        raise ValueError("rtc_training.min_postfix_steps exceeds chunk_size")
    if max_delay > chunk_size - min_postfix_steps:
        raise ValueError("rtc_training.max_delay does not leave the required postfix")

    raw_histogram = rtc_training.get("observed_delay_histogram", [])
    if not isinstance(raw_histogram, list):
        raise ValueError("rtc_training.observed_delay_histogram must be a list")
    histogram = tuple(
        _finite_non_negative("observed delay histogram entry", value) for value in raw_histogram
    )
    histogram_weight = _finite_non_negative(
        "rtc_training.observed_histogram_weight",
        rtc_training.get("observed_histogram_weight", 0.9),
    )
    if histogram_weight > 1.0:
        raise ValueError("rtc_training.observed_histogram_weight must be <= 1")
    return RTCConditionedCheckpointContract(
        chunk_size=chunk_size,
        max_delay=max_delay,
        min_postfix_steps=min_postfix_steps,
        observed_delay_histogram=histogram,
        observed_histogram_weight=histogram_weight,
    )


class TorchRTCConditionedBackend:
    """PI0.5 clean-prefix RTC backend without inference-time VJP guidance."""

    def __init__(
        self,
        bundle: PolicyBundle,
        *,
        contract: RTCConditionedCheckpointContract,
        device: torch.device,
        expected_task: str,
    ) -> None:
        self.bundle = bundle
        self.policy = bundle.policy
        self.preprocessor = bundle.preprocessor
        self.postprocessor = bundle.postprocessor
        self.contract = contract
        self.device = device
        self.expected_task = expected_task
        self._validate_loaded_policy()
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.inference_count = 0
        self.failure_count = 0
        self.prefix_clamp_checks = 0
        self.last_prefix_length = 0
        self.last_latency_s = 0.0
        checkpoint_health = bundle.metadata.health_dict()
        self.manifest = build_torch_backend_manifest(
            backend=self.name,
            device=device,
            features=TorchFeatureFlags(inference_mode=True),
            checkpoint_health=checkpoint_health,
        )
        self.manifest["rtc_conditioning"] = contract.to_dict()
        self.manifest["inference_time_vjp_rtc_enabled"] = False

    @property
    def name(self) -> str:
        return "torch_rtc_conditioned"

    @classmethod
    def from_runtime_config(cls, config: OptimizedRuntimeConfig) -> TorchRTCConditionedBackend:
        if config.backend != "torch_rtc_conditioned":
            raise ValueError(f"TorchRTCConditionedBackend cannot load backend={config.backend!r}")
        policy_path, tokenizer_path = config.require_model_paths()
        contract = inspect_rtc_conditioned_checkpoint(policy_path)
        bundle = load_policy_bundle(
            policy_path,
            tokenizer_path=tokenizer_path,
            device=config.device,
            require_complete_step=config.require_complete_step,
        )
        return cls(
            bundle,
            contract=contract,
            device=get_safe_torch_device(config.device),
            expected_task=config.rtc_conditioned_task or "",
        )

    def health(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = {
                "backend_inference_count": self.inference_count,
                "backend_failure_count": self.failure_count,
                "backend_prefix_clamp_checks": self.prefix_clamp_checks,
                "backend_last_prefix_length": self.last_prefix_length,
                "backend_last_latency_s": self.last_latency_s,
            }
        return {
            "ok": True,
            **self.bundle.metadata.health_dict(),
            "backend": self.name,
            "backend_implementation": (
                "tk_infer.pi05_optimized.backends.torch_rtc_conditioned_backend.TorchRTCConditionedBackend"
            ),
            "backend_phase": 10,
            "supported_modes": [SINGLE_STEP_MODE, RTC_MODE],
            "rtc_method": "training_time_action_conditioning",
            "expected_task": self.expected_task,
            "rtc_conditioning": self.contract.to_dict(),
            "inference_time_vjp_rtc_enabled": False,
            "backend_features": TorchFeatureFlags(inference_mode=True).to_dict(),
            "backend_manifest": dict(self.manifest),
            **stats,
        }

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        request.validate()
        started_s = time.perf_counter()
        try:
            if request.task != self.expected_task:
                raise ValueError(
                    "RTC-conditioned request task differs from the locked training task: "
                    f"expected={self.expected_task!r} got={request.task!r}"
                )
            prefix, prefix_length = self._action_prefix(request)
            with self._lock:
                batch = prepare_observation_for_inference(
                    dict(request.observation_frame),
                    device=self.device,
                    task=request.task,
                    robot_type=request.robot_type,
                )
                preprocessed = self.preprocessor(batch)
                predict_kwargs: dict[str, Any] = {}
                if prefix is not None:
                    predict_kwargs = {
                        "action_prefix": prefix,
                        "prefix_length": prefix_length,
                    }
                use_amp = bool(getattr(getattr(self.policy, "config", None), "use_amp", False))
                autocast_context = (
                    torch.autocast(device_type=self.device.type)
                    if self.device.type == "cuda" and use_amp
                    else nullcontext()
                )
                with torch.inference_mode(), autocast_context:
                    raw_batch = ensure_chunk_batch(
                        self.policy.predict_action_chunk(preprocessed, **predict_kwargs)
                    )
                if raw_batch.shape[1] != self.contract.chunk_size:
                    raise ValueError(
                        f"RTC-conditioned policy returned {raw_batch.shape[1]} steps; "
                        f"expected {self.contract.chunk_size}"
                    )
                if prefix is not None and not torch.equal(
                    raw_batch[:, :prefix_length, :MODEL_ACTION_DIM],
                    prefix[:, :prefix_length, :MODEL_ACTION_DIM],
                ):
                    raise RuntimeError("RTC-conditioned policy failed the exact clean-prefix clamp contract")
                raw_cpu = raw_batch.detach().to(device="cpu", dtype=torch.float32)
                processed_batch = postprocess_action_chunk(self.postprocessor, raw_cpu)

            raw_actions = np.ascontiguousarray(raw_cpu.squeeze(0).numpy())
            processed_actions = np.ascontiguousarray(
                processed_batch.squeeze(0).to(dtype=torch.float32).numpy()
            )
            elapsed_s = time.perf_counter() - started_s
            response = InferenceResponse(
                request_id=request.request_id,
                mode=request.mode,
                raw_actions=raw_actions,
                processed_actions=processed_actions,
                server_latency_s=elapsed_s,
                model_latency_s=elapsed_s,
                raw_action_shape=tuple(raw_actions.shape),
                processed_action_shape=tuple(processed_actions.shape),
            )
            response.validate()
            PairedTrajectory.from_response(
                response,
                source_observation_seq=request.obs_sequence_id,
                predicted_delay_steps=request.predicted_delay_steps,
            )
        except Exception:
            with self._stats_lock:
                self.failure_count += 1
            raise
        with self._stats_lock:
            self.inference_count += 1
            self.last_prefix_length = prefix_length
            self.last_latency_s = response.server_latency_s
            if prefix_length:
                self.prefix_clamp_checks += 1
        return response

    def _action_prefix(self, request: InferenceRequest) -> tuple[torch.Tensor | None, int]:
        if request.mode == SINGLE_STEP_MODE:
            return None, 0
        if request.mode != RTC_MODE:
            raise ValueError(f"Unsupported inference mode: {request.mode!r}")
        prefix_length = request.predicted_delay_steps
        if prefix_length > self.contract.maximum_prefix_length:
            raise ValueError(
                f"predicted_delay_steps={prefix_length} exceeds conditioned max prefix "
                f"{self.contract.maximum_prefix_length}; refusing untrained overflow"
            )
        if prefix_length == 0:
            return None, 0
        if request.prev_chunk_left_over is None:
            raise ValueError("RTC-conditioned request with nonzero delay requires model16 leftover")
        leftover = np.asarray(request.prev_chunk_left_over, dtype=np.float32)
        if len(leftover) < prefix_length:
            raise ValueError(
                f"RTC-conditioned prefix requires {prefix_length} leftover steps, got {len(leftover)}"
            )
        prefix = torch.as_tensor(
            np.ascontiguousarray(leftover[:prefix_length]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return prefix, prefix_length

    def _validate_loaded_policy(self) -> None:
        if not self.expected_task:
            raise ValueError("RTC-conditioned backend requires a non-empty expected task")
        config = getattr(self.policy, "config", None)
        rtc_training = getattr(config, "rtc_training", None)
        if rtc_training is None or getattr(rtc_training, "enabled", None) is not True:
            raise ValueError("loaded policy does not expose rtc_training.enabled=true")
        rtc_config = getattr(config, "rtc_config", None)
        if rtc_config is not None and getattr(rtc_config, "enabled", False):
            raise ValueError("loaded policy unexpectedly enables inference-time VJP RTC")
        names = {name for name, _parameter in self.policy.named_parameters()}
        missing = [
            suffix
            for suffix in REQUIRED_RTC_PARAMETER_SUFFIXES
            if not any(name.endswith(suffix) for name in names)
        ]
        if missing:
            raise ValueError(f"RTC-conditioned checkpoint lacks required learned parameters: {missing}")


def _positive_int(name: str, value: object) -> int:
    parsed = _non_negative_int(name, value)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_non_negative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite non-negative number")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return parsed


__all__ = [
    "REQUIRED_RTC_PARAMETER_SUFFIXES",
    "RTC_CONDITIONED_INFERENCE_CONTRACT",
    "RTCConditionedCheckpointContract",
    "TorchRTCConditionedBackend",
    "inspect_rtc_conditioned_checkpoint",
]
