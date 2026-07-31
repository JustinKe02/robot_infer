from __future__ import annotations

import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.utils import get_safe_torch_device

from .checkpoint import (
    PolicyBundle,
    install_policy_rtc,
    load_policy_bundle,
    restore_policy_rtc_enabled,
    set_policy_rtc_enabled,
)
from .protocol import (
    MODEL_ACTION_DIM,
    PROTOCOL_VERSION,
    RTC_MODE,
    SINGLE_STEP_MODE,
    SUPPORTED_MODES,
    WIRE_ACTION_DIM,
    InferenceRequest,
    InferenceResponse,
)


@dataclass(slots=True)
class PolicyServiceConfig:
    policy_path: Path
    tokenizer_path: Path
    device: str = "cuda"
    require_complete_step: bool = True

    def __post_init__(self) -> None:
        self.policy_path = Path(self.policy_path).expanduser()
        self.tokenizer_path = Path(self.tokenizer_path).expanduser()
        if not self.device:
            raise ValueError("device must be non-empty")


class PolicyService:
    """Serialized PI0.5 service shared by single-step and RTC clients."""

    def __init__(
        self,
        *,
        config: PolicyServiceConfig,
        bundle: PolicyBundle,
        rtc_config: RTCConfig,
    ) -> None:
        self.config = config
        self.bundle = bundle
        self.policy = bundle.policy
        self.preprocessor = bundle.preprocessor
        self.postprocessor = bundle.postprocessor
        self.device = get_safe_torch_device(config.device)
        self.rtc_config = rtc_config
        install_policy_rtc(self.policy, rtc_config)
        self._lock = threading.Lock()
        self.inference_count = 0
        self.mode_counts = {mode: 0 for mode in SUPPORTED_MODES}
        self.last_latency_s = 0.0
        self.last_model_latency_s = 0.0
        self.last_mode: str | None = None

    @classmethod
    def from_config(
        cls,
        config: PolicyServiceConfig,
        *,
        rtc_config: RTCConfig,
    ) -> PolicyService:
        bundle = load_policy_bundle(
            config.policy_path,
            tokenizer_path=config.tokenizer_path,
            device=config.device,
            require_complete_step=config.require_complete_step,
        )
        return cls(config=config, bundle=bundle, rtc_config=rtc_config)

    def health(self) -> dict[str, Any]:
        health = {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "supported_modes": list(SUPPORTED_MODES),
            "device": str(self.device),
            "inference_count": self.inference_count,
            "mode_counts": dict(self.mode_counts),
            "last_latency_s": self.last_latency_s,
            "last_model_latency_s": self.last_model_latency_s,
            "last_mode": self.last_mode,
            "rtc_execution_horizon": self.rtc_config.execution_horizon,
            "rtc_max_guidance_weight": self.rtc_config.max_guidance_weight,
        }
        health.update(self.bundle.metadata.health_dict())
        return health

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        request.validate()
        started_s = time.perf_counter()
        with self._lock:
            model_started_s = time.perf_counter()
            raw_chunk, processed_chunk = run_policy_chunk_inference(
                policy=self.policy,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                observation_frame=request.observation_frame,
                device=self.device,
                task=request.task,
                robot_type=request.robot_type,
                mode=request.mode,
                predicted_delay_steps=request.predicted_delay_steps,
                prev_chunk_left_over=request.prev_chunk_left_over,
                execution_horizon=request.execution_horizon,
            )
            model_latency_s = time.perf_counter() - model_started_s
            self.inference_count += 1
            self.mode_counts[request.mode] += 1
            self.last_mode = request.mode
            self.last_model_latency_s = model_latency_s
            self.last_latency_s = time.perf_counter() - started_s

        raw_np = raw_chunk.detach().to(dtype=torch.float32, device="cpu").numpy()
        processed_np = processed_chunk.detach().to(dtype=torch.float32, device="cpu").numpy()
        response = InferenceResponse(
            request_id=request.request_id,
            mode=request.mode,
            raw_actions=raw_np,
            processed_actions=processed_np,
            server_latency_s=time.perf_counter() - started_s,
            model_latency_s=model_latency_s,
            raw_action_shape=tuple(raw_np.shape),
            processed_action_shape=tuple(processed_np.shape),
        )
        response.validate()
        return response


def run_policy_chunk_inference(
    *,
    policy: Any,
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    postprocessor: Callable[[Tensor], Tensor],
    observation_frame: dict[str, Any],
    device: torch.device,
    task: str,
    robot_type: str,
    mode: str,
    predicted_delay_steps: int,
    prev_chunk_left_over: np.ndarray | Tensor | None,
    execution_horizon: int,
) -> tuple[Tensor, Tensor]:
    batch = prepare_observation_for_inference(
        dict(observation_frame),
        device=device,
        task=task,
        robot_type=robot_type,
    )
    preprocessed_batch = preprocessor(batch)

    predict_kwargs: dict[str, Any] = {}
    if mode == RTC_MODE:
        left_over_tensor = None
        if prev_chunk_left_over is not None:
            left_over_tensor = torch.as_tensor(
                prev_chunk_left_over,
                dtype=torch.float32,
                device=device,
            )
        predict_kwargs = {
            "inference_delay": int(predicted_delay_steps),
            "prev_chunk_left_over": left_over_tensor,
            "execution_horizon": int(execution_horizon),
        }
    elif mode != SINGLE_STEP_MODE:
        raise ValueError(f"Unsupported inference mode: {mode!r}")

    use_amp = bool(getattr(getattr(policy, "config", None), "use_amp", False))
    autocast_context = (
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and use_amp
        else nullcontext()
    )
    with _policy_mode(policy, rtc_enabled=mode == RTC_MODE), autocast_context:
        raw_chunk = ensure_chunk_batch(policy.predict_action_chunk(preprocessed_batch, **predict_kwargs))
        postprocess_input = raw_chunk.detach().to(device="cpu", dtype=torch.float32)
        processed_chunk = postprocess_action_chunk(postprocessor, postprocess_input)

    raw_chunk = raw_chunk.squeeze(0)
    processed_chunk = processed_chunk.squeeze(0)
    _validate_action_boundaries(raw_chunk, processed_chunk)
    return raw_chunk, processed_chunk


@contextmanager
def _policy_mode(policy: Any, *, rtc_enabled: bool) -> Iterator[None]:
    changed = set_policy_rtc_enabled(policy, rtc_enabled)
    try:
        yield
    finally:
        restore_policy_rtc_enabled(changed)


def postprocess_action_chunk(postprocessor: Callable[[Tensor], Tensor], raw_chunk: Tensor) -> Tensor:
    raw_chunk = ensure_chunk_batch(raw_chunk)
    try:
        return ensure_chunk_batch(torch.as_tensor(postprocessor(raw_chunk)))
    except Exception as direct_error:
        batch_size, chunk_len, action_dim = raw_chunk.shape
        flattened = raw_chunk.reshape(batch_size * chunk_len, action_dim)
        try:
            processed = torch.as_tensor(postprocessor(flattened))
        except Exception:
            raise direct_error
        processed = processed.reshape(batch_size, chunk_len, -1)
        return ensure_chunk_batch(processed)


def ensure_chunk_batch(actions: Tensor) -> Tensor:
    if actions.ndim == 2:
        return actions.unsqueeze(0)
    if actions.ndim == 3:
        if actions.shape[0] != 1:
            raise ValueError(f"Only batch_size=1 is supported, got {tuple(actions.shape)}")
        return actions
    raise ValueError(f"Expected action chunk with shape (T,D) or (1,T,D), got {tuple(actions.shape)}")


def _validate_action_boundaries(raw_actions: Tensor, processed_actions: Tensor) -> None:
    if raw_actions.ndim != 2 or raw_actions.shape[1] != MODEL_ACTION_DIM:
        raise ValueError(f"PI0.5 raw chunk must be (T,16), got {tuple(raw_actions.shape)}")
    if processed_actions.ndim != 2 or processed_actions.shape[1] != WIRE_ACTION_DIM:
        raise ValueError(f"JZ processed chunk must be (T,18), got {tuple(processed_actions.shape)}")
    if raw_actions.shape[0] != processed_actions.shape[0]:
        raise ValueError("Raw model16 and processed raw18 chunks have different temporal lengths")
    if not torch.isfinite(raw_actions).all() or not torch.isfinite(processed_actions).all():
        raise ValueError("Policy produced non-finite actions")
