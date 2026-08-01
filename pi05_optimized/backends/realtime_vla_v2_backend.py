from __future__ import annotations

import gc
import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
from lerobot.policies.utils import prepare_observation_for_inference
from tk_infer.pi05.runtime.checkpoint import CheckpointMetadata, load_policy_bundle
from tk_infer.pi05.runtime.policy_service import postprocess_action_chunk
from tk_infer.pi05.runtime.protocol import (
    MODEL_ACTION_DIM,
    RTC_MODE,
    SINGLE_STEP_MODE,
    WIRE_ACTION_DIM,
    InferenceRequest,
    InferenceResponse,
)
from tk_infer.pi05_optimized.config import OptimizedRuntimeConfig
from tk_infer.pi05_optimized.runtime.paired_trajectory import PairedTrajectory
from tk_infer.pi05_optimized.runtime.realtime_vla_v2_contract import (
    ARTIFACT_FORMAT,
    CAMERA_KEYS,
    CAMERA_PROFILE,
    CHUNK_SIZE,
    CONVERTER_VERSION,
    EXPECTED_OUTPUT_SHAPES,
    FORCE_SLOT_INDICES,
    FORCE_SLOT_VALUES,
    INTERNAL_ACTION_DIM,
    KERNEL_CONTRACT,
    RTC_INFERENCE_CONTRACT,
    RTC_SOURCE_TENSOR_SHAPES,
    SUPPORTED_MODES,
    expected_manifest_values,
)
from tk_infer.pi05_optimized.third_party.realtime_vla_v2 import (
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
)
from tk_infer.pi05_optimized.third_party.realtime_vla_v2.pi05rtc_infer import Pi05RTCInference

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class RealtimeVLAV2Artifact:
    directory: Path
    model_path: Path
    manifest_path: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls, directory: str | Path) -> RealtimeVLAV2Artifact:
        resolved = Path(directory).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise FileNotFoundError(f"Realtime-VLA v2 artifact is not a directory: {resolved}")
        model_path = resolved / "model.safetensors"
        manifest_path = resolved / "manifest.json"
        if not model_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("Realtime-VLA v2 artifact requires model.safetensors and manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Realtime-VLA v2 manifest must be a JSON object")
        _validate_manifest(manifest, model_path=model_path)

        actual_sha256 = _sha256_file(model_path)
        if actual_sha256 != manifest["output_sha256"]:
            raise ValueError(
                f"Realtime-VLA v2 artifact SHA-256 mismatch: {actual_sha256} != {manifest['output_sha256']}"
            )
        with safe_open(model_path, framework="pt", device="cpu") as source:
            actual_keys = set(source.keys())
            expected_keys = set(EXPECTED_OUTPUT_SHAPES)
            if actual_keys != expected_keys:
                raise ValueError(
                    "Realtime-VLA v2 tensor key mismatch: "
                    f"missing={sorted(expected_keys - actual_keys)}, "
                    f"extra={sorted(actual_keys - expected_keys)}"
                )
            tensor_errors = {}
            for name, expected_shape in EXPECTED_OUTPUT_SHAPES.items():
                tensor_slice = source.get_slice(name)
                actual_shape = tuple(tensor_slice.get_shape())
                dtype = tensor_slice.get_dtype()
                if actual_shape != expected_shape or dtype != "BF16":
                    tensor_errors[name] = {
                        "expected_shape": expected_shape,
                        "actual_shape": actual_shape,
                        "dtype": dtype,
                    }
            if tensor_errors:
                raise ValueError(f"Realtime-VLA v2 tensor contract failed: {tensor_errors}")
            metadata = source.metadata() or {}
        expected_header = {
            "format": ARTIFACT_FORMAT,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "kernel_contract": KERNEL_CONTRACT,
            "converter_version": str(CONVERTER_VERSION),
            "source_sha256": manifest["source_sha256"],
            "checkpoint_fingerprint": manifest["checkpoint_fingerprint"],
        }
        header_mismatches = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in expected_header.items()
            if metadata.get(key) != expected
        }
        if header_mismatches:
            raise ValueError(f"Realtime-VLA v2 safetensors header is incompatible: {header_mismatches}")
        return cls(
            directory=resolved,
            model_path=model_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    @property
    def maximum_prefix_length(self) -> int:
        return int(self.manifest["rtc_training"]["maximum_prefix_length"])


@dataclass(slots=True)
class _ProcessorOwner:
    metadata: CheckpointMetadata
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]]
    postprocessor: Callable[[Any], Any]
    device: torch.device


class RealtimeVLAV2PolicyBackend:
    """Local RTC-conditioned Realtime-VLA v2 backend with no legacy fallback."""

    def __init__(
        self,
        *,
        processors: _ProcessorOwner,
        artifact: RealtimeVLAV2Artifact,
        tokenizer_path: str | Path,
        expected_task: str,
        model_factory: Any = Pi05RTCInference,
    ) -> None:
        self.processors = processors
        self.artifact = artifact
        self.tokenizer_path = Path(tokenizer_path).expanduser().resolve(strict=True)
        self.expected_task = expected_task.strip()
        self.device = processors.device
        if not self.expected_task:
            raise ValueError("Realtime-VLA v2 backend requires a non-empty expected task")
        if self.expected_task != artifact.manifest["rtc_conditioned_task"]:
            raise ValueError("Realtime-VLA v2 runtime task differs from the converted artifact task")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Realtime-VLA v2 backend requires an available CUDA device")
        metadata = processors.metadata
        if metadata.checkpoint_fingerprint != artifact.manifest["checkpoint_fingerprint"]:
            raise ValueError("Realtime-VLA v2 artifact and processor checkpoint fingerprints differ")
        if metadata.camera_profile != CAMERA_PROFILE or tuple(metadata.camera_keys) != CAMERA_KEYS:
            raise ValueError("Realtime-VLA v2 processors require the head_right camera profile")
        if metadata.model_action_dim != MODEL_ACTION_DIM or metadata.wire_action_dim != WIRE_ACTION_DIM:
            raise ValueError("Realtime-VLA v2 processors must preserve model16/raw18")

        checkpoint = load_file(artifact.model_path, device="cpu")
        try:
            self._model = model_factory(
                checkpoint=checkpoint,
                num_views=len(CAMERA_KEYS),
                chunk_size=CHUNK_SIZE,
                tokenizer_path=str(self.tokenizer_path),
                max_tokenize_len=200,
                discrete_state_input=True,
                max_prompt_text=self.expected_task,
                state_dim_for_max_prompt=INTERNAL_ACTION_DIM,
            )
        finally:
            checkpoint.clear()
            gc.collect()
        if hasattr(self._model, "checkpoint"):
            self._model.checkpoint = None
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.inference_count = 0
        self.failure_count = 0
        self.prefix_clamp_checks = 0
        self.last_prefix_length = 0
        self.last_latency_s = 0.0

    @classmethod
    def from_runtime_config(cls, config: OptimizedRuntimeConfig) -> RealtimeVLAV2PolicyBackend:
        if config.backend != "realtime_vla_v2":
            raise ValueError(f"RealtimeVLAV2PolicyBackend cannot load backend={config.backend!r}")
        policy_path, tokenizer_path = config.require_model_paths()
        artifact = RealtimeVLAV2Artifact.load(config.require_realtime_vla_v2_artifact_path())
        bundle = load_policy_bundle(
            policy_path,
            tokenizer_path=tokenizer_path,
            device=config.device,
            require_complete_step=config.require_complete_step,
        )
        processors = _ProcessorOwner(
            metadata=bundle.metadata,
            preprocessor=bundle.preprocessor,
            postprocessor=bundle.postprocessor,
            device=torch.device(config.device),
        )
        bundle.policy = None
        bundle.policy_config = None
        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cls(
            processors=processors,
            artifact=artifact,
            tokenizer_path=tokenizer_path,
            expected_task=config.rtc_conditioned_task or "",
        )

    @property
    def name(self) -> str:
        return "realtime_vla_v2"

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
            **self.processors.metadata.health_dict(),
            "backend": self.name,
            "backend_implementation": (
                "tk_infer.pi05_optimized.backends.realtime_vla_v2_backend.RealtimeVLAV2PolicyBackend"
            ),
            "backend_phase": 11,
            "supported_modes": list(SUPPORTED_MODES),
            "rtc_supported": True,
            "rtc_method": "training_time_action_conditioning",
            "rtc_inference_contract": RTC_INFERENCE_CONTRACT,
            "inference_time_vjp_rtc_enabled": False,
            "expected_task": self.expected_task,
            "maximum_prefix_length": self.artifact.maximum_prefix_length,
            "realtime_vla_v2_artifact": {
                "directory": str(self.artifact.directory),
                "output_sha256": self.artifact.manifest["output_sha256"],
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
                "kernel_contract": KERNEL_CONTRACT,
                "converter_version": CONVERTER_VERSION,
                "dtype": "bfloat16",
                "num_views": len(CAMERA_KEYS),
                "chunk_size": CHUNK_SIZE,
                "internal_action_dim": INTERNAL_ACTION_DIM,
                "exposed_model_action_dim": MODEL_ACTION_DIM,
                "wire_action_dim": WIRE_ACTION_DIM,
            },
            "reference_policy_owner_resident": False,
            **stats,
        }

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        request.validate()
        started_s = time.perf_counter()
        try:
            if request.task != self.expected_task:
                raise ValueError(
                    "Realtime-VLA v2 request task differs from the locked training task: "
                    f"expected={self.expected_task!r} got={request.task!r}"
                )
            clean_prefix, internal_prefix, prefix_length = self._action_prefix(request)
            with self._lock, torch.inference_mode():
                images, state_tokens = self._prepare_inputs(request)
                noise = torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=(CHUNK_SIZE, INTERNAL_ACTION_DIM),
                    dtype=torch.float32,
                    device=self.device,
                ).to(torch.bfloat16)
                torch.cuda.synchronize(self.device)
                model_started_s = time.perf_counter()
                internal_actions = self._model.forward(
                    observation_images_normalized=images,
                    diffusion_noise=noise,
                    task_prompt=request.task,
                    state_tokens=state_tokens,
                    action_prefill_len=prefix_length,
                    prefill_actions=internal_prefix,
                )
                torch.cuda.synchronize(self.device)
                model_latency_s = time.perf_counter() - model_started_s
                self._validate_internal_actions(
                    internal_actions,
                    internal_prefix=internal_prefix,
                    prefix_length=prefix_length,
                )
                model_actions = internal_actions[:, :MODEL_ACTION_DIM].to(
                    dtype=torch.float32,
                    device="cpu",
                )
                if clean_prefix is not None:
                    model_actions[:prefix_length].copy_(torch.from_numpy(clean_prefix))
                processed = postprocess_action_chunk(
                    self.processors.postprocessor,
                    model_actions.unsqueeze(0),
                ).squeeze(0)
                raw_np = np.ascontiguousarray(model_actions.numpy())
                processed_np = np.ascontiguousarray(processed.to(torch.float32).numpy())
                _validate_wire_actions(processed_np)
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
                PairedTrajectory.from_response(
                    response,
                    source_observation_seq=request.obs_sequence_id,
                    predicted_delay_steps=request.predicted_delay_steps,
                )
        except Exception:
            with self._stats_lock:
                self.failure_count += 1
            raise
        response.server_latency_s = time.perf_counter() - started_s
        response.validate()
        with self._stats_lock:
            self.inference_count += 1
            self.last_prefix_length = prefix_length
            self.last_latency_s = response.server_latency_s
            if prefix_length:
                self.prefix_clamp_checks += 1
        return response

    def _action_prefix(
        self,
        request: InferenceRequest,
    ) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        if request.mode == SINGLE_STEP_MODE:
            return None, None, 0
        if request.mode != RTC_MODE:
            raise ValueError(f"Unsupported Realtime-VLA v2 mode: {request.mode!r}")
        prefix_length = request.predicted_delay_steps
        if prefix_length > self.artifact.maximum_prefix_length:
            raise ValueError(
                f"predicted_delay_steps={prefix_length} exceeds conditioned max prefix "
                f"{self.artifact.maximum_prefix_length}; refusing untrained overflow"
            )
        if prefix_length == 0:
            return None, None, 0
        if request.prev_chunk_left_over is None:
            raise ValueError("Realtime-VLA v2 RTC request with nonzero delay requires model16 leftover")
        leftover = np.asarray(request.prev_chunk_left_over, dtype=np.float32)
        if len(leftover) < prefix_length:
            raise ValueError(
                f"Realtime-VLA v2 prefix requires {prefix_length} leftover steps, got {len(leftover)}"
            )
        clean_prefix = np.ascontiguousarray(leftover[:prefix_length])
        if clean_prefix.shape != (prefix_length, MODEL_ACTION_DIM):
            raise ValueError(f"Realtime-VLA v2 clean prefix must be model16, got {clean_prefix.shape}")
        if not np.isfinite(clean_prefix).all():
            raise ValueError("Realtime-VLA v2 clean prefix contains NaN or Inf")
        internal_prefix = np.zeros((prefix_length, INTERNAL_ACTION_DIM), dtype=np.float32)
        internal_prefix[:, :MODEL_ACTION_DIM] = clean_prefix
        return clean_prefix, internal_prefix, prefix_length

    def _validate_internal_actions(
        self,
        actions: torch.Tensor,
        *,
        internal_prefix: np.ndarray | None,
        prefix_length: int,
    ) -> None:
        if tuple(actions.shape) != (CHUNK_SIZE, INTERNAL_ACTION_DIM):
            raise ValueError(f"Realtime-VLA v2 kernel output must be (50,32), got {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            raise ValueError("Realtime-VLA v2 kernel output contains NaN or Inf")
        if prefix_length:
            expected = torch.as_tensor(
                internal_prefix,
                dtype=torch.bfloat16,
                device=actions.device,
            )
            if not torch.equal(actions[:prefix_length], expected):
                raise RuntimeError("Realtime-VLA v2 kernel failed the exact BF16 clean-prefix clamp")

    def _prepare_inputs(self, request: InferenceRequest) -> tuple[torch.Tensor, np.ndarray]:
        batch = prepare_observation_for_inference(
            dict(request.observation_frame),
            device=self.device,
            task=request.task,
            robot_type=request.robot_type,
        )
        preprocessed = self.processors.preprocessor(batch)
        images = []
        for key in CAMERA_KEYS:
            image = preprocessed.get(key)
            if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1:
                raise ValueError(f"Realtime-VLA v2 camera {key!r} must be one BCHW tensor")
            if image.shape[1] != 3:
                raise ValueError(f"Realtime-VLA v2 camera {key!r} must have three channels")
            image_hwc = image.to(torch.float32).permute(0, 2, 3, 1)
            image_hwc = resize_with_pad_torch(image_hwc, 224, 224)
            image_hwc = image_hwc.mul(2.0).sub(1.0)
            images.append(image_hwc.squeeze(0))
        image_tensor = torch.stack(images).to(device=self.device, dtype=torch.bfloat16)
        state = preprocessed.get("observation.state")
        if not isinstance(state, torch.Tensor) or state.shape != (1, MODEL_ACTION_DIM):
            raise ValueError(
                f"normalized Realtime-VLA v2 state must be (1,16), got {getattr(state, 'shape', None)}"
            )
        padded_state = torch.zeros(1, INTERNAL_ACTION_DIM, dtype=torch.float32)
        padded_state[:, :MODEL_ACTION_DIM] = state.detach().to(dtype=torch.float32, device="cpu")
        state_tokens = (
            np.digitize(
                padded_state.numpy(),
                bins=np.linspace(-1, 1, 256 + 1)[:-1],
            )
            - 1
        )
        return image_tensor, np.ascontiguousarray(state_tokens.squeeze(0))


def _validate_manifest(manifest: dict[str, Any], *, model_path: Path) -> None:
    expected = expected_manifest_values()
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Realtime-VLA v2 artifact manifest is incompatible: {mismatches}")
    for field in ("source_sha256", "output_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"Realtime-VLA v2 manifest {field} must be a lowercase SHA-256")
    for field in ("checkpoint_fingerprint", "rtc_conditioned_task", "action_mapping_proof"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Realtime-VLA v2 manifest {field} must be non-empty")
    if manifest.get("status") != "PASS":
        raise ValueError("Realtime-VLA v2 artifact must originate from a passing source validation")
    if manifest.get("validate_only") is not False or manifest.get("output_written") is not True:
        raise ValueError("Realtime-VLA v2 manifest does not describe a published artifact")
    output_bytes = manifest.get("output_bytes")
    if isinstance(output_bytes, bool) or not isinstance(output_bytes, int) or output_bytes <= 0:
        raise ValueError("Realtime-VLA v2 manifest output_bytes must be a positive integer")
    if model_path.stat().st_size != output_bytes:
        raise ValueError("Realtime-VLA v2 model size differs from manifest output_bytes")
    action_names = manifest.get("exposed_action_names")
    if (
        not isinstance(action_names, list)
        or len(action_names) != MODEL_ACTION_DIM
        or not all(isinstance(name, str) and name for name in action_names)
    ):
        raise ValueError("Realtime-VLA v2 manifest must prove all 16 exposed action names")
    if manifest.get("required_rtc_source_tensors") != sorted(RTC_SOURCE_TENSOR_SHAPES):
        raise ValueError("Realtime-VLA v2 manifest does not prove all learned RTC source tensors")
    rtc_training = manifest.get("rtc_training")
    if not isinstance(rtc_training, dict):
        raise ValueError("Realtime-VLA v2 manifest rtc_training must be an object")
    required_rtc_values = {
        "enabled": True,
        "inference_contract": RTC_INFERENCE_CONTRACT,
        "chunk_size": CHUNK_SIZE,
    }
    rtc_mismatches = {
        key: {"expected": value, "actual": rtc_training.get(key)}
        for key, value in required_rtc_values.items()
        if rtc_training.get(key) != value
    }
    if rtc_mismatches:
        raise ValueError(f"Realtime-VLA v2 RTC training contract is incompatible: {rtc_mismatches}")
    max_delay = _non_negative_int("rtc_training.max_delay", rtc_training.get("max_delay"))
    min_postfix = _positive_int(
        "rtc_training.min_postfix_steps",
        rtc_training.get("min_postfix_steps"),
    )
    if max_delay > CHUNK_SIZE - min_postfix:
        raise ValueError("Realtime-VLA v2 rtc_training.max_delay does not leave the required postfix")
    maximum_prefix = min(max_delay, CHUNK_SIZE - min_postfix)
    if rtc_training.get("maximum_prefix_length") != maximum_prefix:
        raise ValueError("Realtime-VLA v2 rtc_training.maximum_prefix_length is inconsistent")


def _validate_wire_actions(actions: np.ndarray) -> None:
    if actions.shape != (CHUNK_SIZE, WIRE_ACTION_DIM):
        raise ValueError(f"Realtime-VLA v2 processed actions must be (50,18), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Realtime-VLA v2 processed actions contain NaN or Inf")
    for index, expected in zip(FORCE_SLOT_INDICES, FORCE_SLOT_VALUES, strict=True):
        if not np.all(actions[:, index] == expected):
            raise ValueError(f"Realtime-VLA v2 force slot {index} must remain exactly {expected}")


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    parsed = _non_negative_int(name, value)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["RealtimeVLAV2Artifact", "RealtimeVLAV2PolicyBackend"]
