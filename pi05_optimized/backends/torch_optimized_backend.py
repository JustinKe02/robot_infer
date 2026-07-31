from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from tk_infer.pi05.runtime.checkpoint import restore_policy_rtc_enabled, set_policy_rtc_enabled
from tk_infer.pi05.runtime.policy_service import (
    PolicyService,
    PolicyServiceConfig,
    ensure_chunk_batch,
    postprocess_action_chunk,
)
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

BACKEND_STAGE_NAMES = (
    "observation_prepare_s",
    "preprocess_s",
    "rtc_prepare_s",
    "model_s",
    "device_to_host_s",
    "postprocess_s",
    "response_s",
    "total_s",
)


@dataclass(frozen=True, slots=True)
class TorchBackendOptions:
    features: TorchFeatureFlags = TorchFeatureFlags()
    warmup_iterations: int = 0
    warmup_seed: int = 12345
    metrics_window_size: int = 512
    synchronize_stages: bool = False

    @classmethod
    def from_runtime_config(cls, config: OptimizedRuntimeConfig) -> TorchBackendOptions:
        return cls(
            features=TorchFeatureFlags(
                inference_mode=config.torch_inference_mode,
                bf16_autocast=config.torch_bf16_autocast,
                pinned_memory=config.torch_pinned_memory,
                non_blocking_copies=config.torch_non_blocking_copies,
                static_buffers=config.torch_static_buffers,
                cuda_graph=config.torch_cuda_graph,
            ),
            warmup_iterations=config.torch_warmup_iterations,
            warmup_seed=config.torch_warmup_seed,
            metrics_window_size=config.metrics_window_size,
        )

    def __post_init__(self) -> None:
        if (
            isinstance(self.warmup_iterations, bool)
            or not isinstance(self.warmup_iterations, int)
            or self.warmup_iterations < 0
        ):
            raise ValueError("warmup_iterations must be a non-negative integer")
        if isinstance(self.warmup_seed, bool) or not isinstance(self.warmup_seed, int):
            raise ValueError("warmup_seed must be an integer")
        if (
            isinstance(self.metrics_window_size, bool)
            or not isinstance(self.metrics_window_size, int)
            or self.metrics_window_size <= 0
        ):
            raise ValueError("metrics_window_size must be a positive integer")
        if not isinstance(self.synchronize_stages, bool):
            raise ValueError("synchronize_stages must be boolean")


class TorchOptimizedBackend:
    """Independent PI0.5 PyTorch inference path with explicit, audited feature flags."""

    def __init__(self, service: PolicyService, *, options: TorchBackendOptions | None = None) -> None:
        self._service = service
        self.options = options or TorchBackendOptions()
        self.device = torch.device(service.device)
        self._validate_startup()
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stage_samples = {
            name: deque(maxlen=self.options.metrics_window_size) for name in BACKEND_STAGE_NAMES
        }
        self.inference_count = 0
        self.failure_count = 0
        self.warmup_count = 0
        self.last_stage_timings: dict[str, float] = {}
        checkpoint_health = dict(service.health())
        self.manifest = build_torch_backend_manifest(
            backend=self.name,
            device=self.device,
            features=self.options.features,
            checkpoint_health=checkpoint_health,
        )

    @property
    def name(self) -> str:
        return "torch_optimized"

    @property
    def service(self) -> PolicyService:
        return self._service

    @classmethod
    def from_runtime_config(cls, config: OptimizedRuntimeConfig) -> TorchOptimizedBackend:
        if config.backend != "torch_optimized":
            raise ValueError(f"TorchOptimizedBackend cannot load backend={config.backend!r}")
        policy_path, tokenizer_path = config.require_model_paths()
        service = PolicyService.from_config(
            PolicyServiceConfig(
                policy_path=policy_path,
                tokenizer_path=tokenizer_path,
                device=config.device,
                require_complete_step=config.require_complete_step,
            ),
            rtc_config=RTCConfig(
                enabled=True,
                prefix_attention_schedule=RTCAttentionSchedule(config.rtc_prefix_attention_schedule),
                max_guidance_weight=config.rtc_max_guidance_weight,
                execution_horizon=config.rtc_execution_horizon,
                debug=config.rtc_debug,
            ),
        )
        backend = cls(service, options=TorchBackendOptions.from_runtime_config(config))
        if backend.options.warmup_iterations:
            backend.warmup(
                backend.deterministic_warmup_requests(),
                iterations=backend.options.warmup_iterations,
                seed=backend.options.warmup_seed,
            )
        return backend

    def health(self) -> dict[str, Any]:
        health = dict(self._service.health())
        supported_modes = [SINGLE_STEP_MODE]
        if not self.options.features.inference_mode:
            supported_modes.append(RTC_MODE)
        with self._stats_lock:
            stage_samples = {name: tuple(samples) for name, samples in self._stage_samples.items()}
            inference_count = self.inference_count
            failure_count = self.failure_count
            warmup_count = self.warmup_count
            last_stage_timings = dict(self.last_stage_timings)
        health.update(
            {
                "backend": self.name,
                "backend_implementation": (
                    "tk_infer.pi05_optimized.backends.torch_optimized_backend.TorchOptimizedBackend"
                ),
                "backend_phase": 2,
                "backend_inference_count": inference_count,
                "backend_failure_count": failure_count,
                "backend_warmup_count": warmup_count,
                "backend_features": self.options.features.to_dict(),
                "supported_modes": supported_modes,
                "inference_mode_rtc_compatible": not self.options.features.inference_mode,
                "backend_manifest": dict(self.manifest),
                "backend_stage_timing_mode": (
                    "synchronized_boundary_diagnostic"
                    if self.options.synchronize_stages
                    else "host_elapsed_unsynchronized"
                ),
                "backend_last_stage_timings": last_stage_timings,
                "backend_stage_metrics": {
                    name: _distribution(samples) for name, samples in stage_samples.items()
                },
            }
        )
        return health

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        self._validate_request_features(request)
        try:
            response, timings = self._infer(request)
        except Exception:
            with self._stats_lock:
                self.failure_count += 1
            raise
        with self._stats_lock:
            self.inference_count += 1
            self.last_stage_timings = dict(timings)
            for name, value in timings.items():
                self._stage_samples[name].append(value)
        return response

    def warmup(
        self,
        requests: tuple[InferenceRequest, ...],
        *,
        iterations: int,
        seed: int,
    ) -> None:
        if not requests:
            raise ValueError("warmup requires at least one request")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
            raise ValueError("warmup iterations must be a non-negative integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("warmup seed must be an integer")
        numpy_state = np.random.get_state()
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        try:
            with torch.random.fork_rng(devices=devices):
                for iteration in range(iterations):
                    for request_index, request in enumerate(requests):
                        self._validate_request_features(request)
                        current_seed = seed + iteration * len(requests) + request_index
                        np.random.seed(current_seed)
                        torch.manual_seed(current_seed)
                        if self.device.type == "cuda":
                            torch.cuda.manual_seed_all(current_seed)
                        self._reset_components()
                        self._infer(request)
                        with self._stats_lock:
                            self.warmup_count += 1
                self._synchronize_device()
        finally:
            np.random.set_state(numpy_state)
            self._reset_components()

    def deterministic_warmup_requests(self) -> tuple[InferenceRequest, ...]:
        health = self._service.health()
        camera_shapes = health.get("camera_shapes")
        if not isinstance(camera_shapes, dict) or not camera_shapes:
            raise ValueError("checkpoint health lacks fixed camera_shapes required for deterministic warmup")
        observation: dict[str, np.ndarray] = {
            "observation.state": np.zeros(18, dtype=np.float32)
        }
        for key, shape_value in camera_shapes.items():
            shape = tuple(shape_value)
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"warmup camera shape must be CHW with C=3, got {key}={shape}")
            channels, height, width = shape
            observation[key] = np.zeros((height, width, channels), dtype=np.uint8)
        common = {
            "observation_frame": observation,
            "task": "jz robot pin timed vr teleoperation",
            "robot_type": "jz_robot_pin_timed",
            "obs_sequence_id": -1,
            "execution_horizon": 10,
        }
        single_step = InferenceRequest(request_id=0, mode=SINGLE_STEP_MODE, **common)
        if self.options.features.inference_mode:
            return (single_step,)
        return (
            single_step,
            InferenceRequest(
                request_id=1,
                mode=RTC_MODE,
                predicted_delay_steps=1,
                prev_chunk_left_over=np.zeros((40, MODEL_ACTION_DIM), dtype=np.float32),
                **common,
            ),
        )

    def _infer(self, request: InferenceRequest) -> tuple[InferenceResponse, dict[str, float]]:
        request.validate()
        total_started_s = time.perf_counter()
        timings: dict[str, float] = {}
        with self._lock:
            batch = self._timed(
                timings,
                "observation_prepare_s",
                lambda: self._prepare_observation(request.observation_frame, request.task, request.robot_type),
            )
            preprocessed = self._timed(timings, "preprocess_s", lambda: self._service.preprocessor(batch))
            predict_kwargs = self._timed(
                timings,
                "rtc_prepare_s",
                lambda: self._predict_kwargs(request),
            )
            raw_chunk = self._timed(
                timings,
                "model_s",
                lambda: self._predict(preprocessed, request.mode, predict_kwargs),
            )
            raw_cpu = self._timed(
                timings,
                "device_to_host_s",
                lambda: raw_chunk.detach().to(dtype=torch.float32, device="cpu"),
            )
            processed_cpu = self._timed(
                timings,
                "postprocess_s",
                lambda: postprocess_action_chunk(self._service.postprocessor, raw_cpu.unsqueeze(0)).squeeze(0),
            )

        def build_response() -> InferenceResponse:
            raw_np = np.ascontiguousarray(raw_cpu.numpy())
            processed_np = np.ascontiguousarray(processed_cpu.to(dtype=torch.float32).numpy())
            elapsed_s = time.perf_counter() - total_started_s
            response = InferenceResponse(
                request_id=request.request_id,
                mode=request.mode,
                raw_actions=raw_np,
                processed_actions=processed_np,
                server_latency_s=elapsed_s,
                model_latency_s=elapsed_s,
                raw_action_shape=tuple(raw_np.shape),
                processed_action_shape=tuple(processed_np.shape),
            )
            response.validate()
            PairedTrajectory.from_response(
                response,
                source_observation_seq=request.obs_sequence_id,
                predicted_delay_steps=request.predicted_delay_steps,
            )
            return response

        response = self._timed(timings, "response_s", build_response)
        timings["total_s"] = _finite_duration(time.perf_counter() - total_started_s)
        response.server_latency_s = timings["total_s"]
        response.model_latency_s = timings["total_s"]
        response.validate()
        return response, timings

    def _prepare_observation(
        self,
        observation_frame: dict[str, Any],
        task: str,
        robot_type: str,
    ) -> dict[str, Any]:
        features = self.options.features
        if not features.pinned_memory:
            return prepare_observation_for_inference(
                dict(observation_frame),
                device=self.device,
                task=task,
                robot_type=robot_type,
            )
        observation: dict[str, Any] = {}
        for name, value in observation_frame.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f"observation {name!r} must be a NumPy array, got {type(value)}")
            tensor = torch.from_numpy(value)
            if "image" in name:
                if tensor.ndim != 3 or tensor.shape[-1] != 3:
                    raise ValueError(f"observation image {name!r} must be HWC with C=3")
                tensor = tensor.to(torch.float32).div(255).permute(2, 0, 1).contiguous()
            tensor = tensor.unsqueeze(0).pin_memory()
            observation[name] = tensor.to(
                self.device,
                non_blocking=features.non_blocking_copies,
            )
        observation["task"] = task
        observation["robot_type"] = robot_type
        return observation

    def _predict_kwargs(self, request: InferenceRequest) -> dict[str, Any]:
        if request.mode == SINGLE_STEP_MODE:
            return {}
        if request.mode != RTC_MODE:
            raise ValueError(f"Unsupported inference mode: {request.mode!r}")
        left_over_tensor = None
        if request.prev_chunk_left_over is not None:
            left_over = np.ascontiguousarray(request.prev_chunk_left_over, dtype=np.float32)
            left_over_tensor = torch.from_numpy(left_over)
            if self.options.features.pinned_memory:
                left_over_tensor = left_over_tensor.pin_memory()
            left_over_tensor = left_over_tensor.to(
                self.device,
                non_blocking=self.options.features.non_blocking_copies,
            )
        return {
            "inference_delay": request.predicted_delay_steps,
            "prev_chunk_left_over": left_over_tensor,
            "execution_horizon": request.execution_horizon,
        }

    def _predict(
        self,
        preprocessed: dict[str, Any],
        mode: str,
        predict_kwargs: dict[str, Any],
    ) -> Tensor:
        inference_context = torch.inference_mode() if self.options.features.inference_mode else nullcontext()
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.options.features.bf16_autocast
            else nullcontext()
        )
        with (
            _policy_mode(self._service.policy, rtc_enabled=mode == RTC_MODE),
            inference_context,
            autocast_context,
        ):
            raw_chunk = ensure_chunk_batch(
                self._service.policy.predict_action_chunk(preprocessed, **predict_kwargs)
            )
        return raw_chunk.squeeze(0)

    def _timed(
        self,
        timings: dict[str, float],
        name: str,
        operation: Callable[[], Any],
    ) -> Any:
        if self.options.synchronize_stages:
            self._synchronize_device()
        started_s = time.perf_counter()
        result = operation()
        if self.options.synchronize_stages:
            self._synchronize_device()
        timings[name] = _finite_duration(time.perf_counter() - started_s)
        return result

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _reset_components(self) -> None:
        for component in (self._service.policy, self._service.preprocessor, self._service.postprocessor):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def _validate_startup(self) -> None:
        features = self.options.features
        if features.static_buffers:
            raise ValueError(
                "torch_static_buffers is not supported by the current dynamic observation/token ownership; "
                "enable only after a fixed-shape buffer implementation passes parity"
            )
        if features.cuda_graph:
            raise ValueError(
                "torch_cuda_graph is not supported by the dynamic KV cache, RTC branch, and denoising loop"
            )
        cuda_features = (
            features.bf16_autocast or features.pinned_memory or features.non_blocking_copies
        )
        if cuda_features and self.device.type != "cuda":
            raise ValueError("BF16, pinned memory, and non-blocking copies require a CUDA backend device")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA backend requested but torch.cuda.is_available() is false")
        if features.bf16_autocast and not torch.cuda.is_bf16_supported():
            raise ValueError("torch_bf16_autocast requested but the CUDA device does not support BF16")
        if features.non_blocking_copies and not features.pinned_memory:
            raise ValueError("non_blocking_copies requires pinned_memory")

    def _validate_request_features(self, request: InferenceRequest) -> None:
        if self.options.features.inference_mode and request.mode == RTC_MODE:
            raise ValueError(
                "torch_inference_mode supports single_step only: inference-time RTC guidance requires "
                "torch.enable_grad and torch.autograd.grad"
            )


@contextmanager
def _policy_mode(policy: Any, *, rtc_enabled: bool) -> Iterator[None]:
    changed = set_policy_rtc_enabled(policy, rtc_enabled)
    try:
        yield
    finally:
        restore_policy_rtc_enabled(changed)


def _finite_duration(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("backend duration must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError("backend duration must be finite and non-negative")
    return converted


def _distribution(values: tuple[float, ...]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "latest": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "latest": values[-1],
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


__all__ = [
    "BACKEND_STAGE_NAMES",
    "TorchBackendOptions",
    "TorchOptimizedBackend",
]
