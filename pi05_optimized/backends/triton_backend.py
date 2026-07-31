from __future__ import annotations

import gc
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from tk_infer.pi05.runtime.policy_service import (
    PolicyService,
    PolicyServiceConfig,
    postprocess_action_chunk,
)
from tk_infer.pi05.runtime.protocol import (
    MODEL_ACTION_DIM,
    RTC_MODE,
    SINGLE_STEP_MODE,
    InferenceRequest,
    InferenceResponse,
)
from tk_infer.pi05_optimized.runtime.paired_trajectory import PairedTrajectory
from tk_infer.pi05_optimized.third_party.realtime_vla import UPSTREAM_COMMIT
from tk_infer.pi05_optimized.third_party.realtime_vla.pi05_infer import Pi05Inference

TRITON_ARTIFACT_SCHEMA_VERSION = 1
TRITON_INTERNAL_ACTION_DIM = 32
TRITON_NUM_VIEWS = 2
TRITON_CHUNK_SIZE = 50


@dataclass(frozen=True, slots=True)
class TritonArtifact:
    directory: Path
    model_path: Path
    manifest_path: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls, directory: str | Path, *, verify_sha256: bool = True) -> TritonArtifact:
        resolved = Path(directory).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise FileNotFoundError(f"Triton artifact is not a directory: {resolved}")
        model_path = resolved / "model.safetensors"
        manifest_path = resolved / "manifest.json"
        if not model_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("Triton artifact requires model.safetensors and manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Triton manifest must be a JSON object")
        expected = {
            "manifest_schema_version": TRITON_ARTIFACT_SCHEMA_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "dtype": "bfloat16",
            "num_views": TRITON_NUM_VIEWS,
            "camera_profile": "head_right",
            "chunk_size": TRITON_CHUNK_SIZE,
            "internal_action_dim": TRITON_INTERNAL_ACTION_DIM,
            "exposed_model_action_dim": MODEL_ACTION_DIM,
            "supported_modes": [SINGLE_STEP_MODE],
            "rtc_supported": False,
            "output_tensor_count": 46,
        }
        mismatches = {
            key: {"expected": value, "actual": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if manifest.get("exposed_action_indices") != list(range(MODEL_ACTION_DIM)):
            mismatches["exposed_action_indices"] = {
                "expected": list(range(MODEL_ACTION_DIM)),
                "actual": manifest.get("exposed_action_indices"),
            }
        if mismatches:
            raise ValueError(f"Triton artifact manifest is incompatible: {mismatches}")
        if verify_sha256:
            actual_sha256 = _sha256_file(model_path)
            if actual_sha256 != manifest.get("output_sha256"):
                raise ValueError(
                    f"Triton artifact SHA-256 mismatch: {actual_sha256} != {manifest.get('output_sha256')}"
                )
        with safe_open(model_path, framework="pt", device="cpu") as source:
            if len(source.keys()) != 46:
                raise ValueError(f"Triton safetensors must contain 46 tensors, got {len(source.keys())}")
            metadata = source.metadata() or {}
        if metadata.get("upstream_commit") != UPSTREAM_COMMIT:
            raise ValueError("Triton safetensors header upstream_commit differs from the pinned runtime")
        if metadata.get("checkpoint_fingerprint") != manifest.get("checkpoint_fingerprint"):
            raise ValueError("Triton safetensors header checkpoint fingerprint differs from manifest")
        return cls(
            directory=resolved,
            model_path=model_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )


class TritonPolicyBackend:
    """Single-step Realtime-VLA v1 backend behind a strict, non-fallback boundary."""

    def __init__(
        self,
        *,
        processor_service: PolicyService,
        artifact: TritonArtifact,
        tokenizer_path: str | Path,
        model_factory: Any = Pi05Inference,
    ) -> None:
        self._processor_service = processor_service
        self.artifact = artifact
        self.tokenizer_path = Path(tokenizer_path).expanduser().resolve(strict=True)
        self.device = torch.device(processor_service.device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Triton PI0.5 backend requires an available CUDA device")
        service_health = processor_service.health()
        if service_health.get("checkpoint_fingerprint") != artifact.manifest.get(
            "checkpoint_fingerprint"
        ):
            raise ValueError("Triton artifact and processor checkpoint fingerprints differ")
        if service_health.get("camera_profile") != "head_right":
            raise ValueError("Triton backend requires the head_right camera profile")
        if service_health.get("model_action_dim") != MODEL_ACTION_DIM:
            raise ValueError("Triton processor service must expose model16")
        checkpoint = load_file(artifact.model_path, device="cpu")
        try:
            self._model = model_factory(
                checkpoint=checkpoint,
                num_views=TRITON_NUM_VIEWS,
                chunk_size=TRITON_CHUNK_SIZE,
                tokenizer_path=str(self.tokenizer_path),
                max_tokenize_len=200,
                discrete_state_input=True,
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
        self.last_latency_s = 0.0

    @classmethod
    def from_runtime_config(cls, config: Any) -> TritonPolicyBackend:
        if config.backend != "triton":
            raise ValueError(f"TritonPolicyBackend cannot load backend={config.backend!r}")
        policy_path, tokenizer_path = config.require_model_paths()
        processor_service = PolicyService.from_config(
            PolicyServiceConfig(
                policy_path=policy_path,
                tokenizer_path=tokenizer_path,
                device=config.device,
                require_complete_step=config.require_complete_step,
            ),
            rtc_config=RTCConfig(
                enabled=True,
                prefix_attention_schedule=RTCAttentionSchedule(
                    config.rtc_prefix_attention_schedule
                ),
                max_guidance_weight=config.rtc_max_guidance_weight,
                execution_horizon=config.rtc_execution_horizon,
                debug=config.rtc_debug,
            ),
        )
        artifact = TritonArtifact.load(config.require_triton_artifact_path())
        return cls(
            processor_service=processor_service,
            artifact=artifact,
            tokenizer_path=tokenizer_path,
        )

    @property
    def name(self) -> str:
        return "triton"

    def health(self) -> dict[str, Any]:
        health = dict(self._processor_service.health())
        with self._stats_lock:
            inference_count = self.inference_count
            failure_count = self.failure_count
            last_latency_s = self.last_latency_s
        health.update(
            {
                "backend": self.name,
                "backend_implementation": (
                    "tk_infer.pi05_optimized.backends.triton_backend.TritonPolicyBackend"
                ),
                "backend_phase": 3,
                "supported_modes": [SINGLE_STEP_MODE],
                "rtc_supported": False,
                "triton_inference_count": inference_count,
                "triton_failure_count": failure_count,
                "triton_last_latency_s": last_latency_s,
                "triton_artifact": {
                    "directory": str(self.artifact.directory),
                    "output_sha256": self.artifact.manifest["output_sha256"],
                    "upstream_commit": self.artifact.manifest["upstream_commit"],
                    "converter_version": self.artifact.manifest["converter_version"],
                    "dtype": self.artifact.manifest["dtype"],
                    "num_views": self.artifact.manifest["num_views"],
                    "chunk_size": self.artifact.manifest["chunk_size"],
                    "internal_action_dim": self.artifact.manifest["internal_action_dim"],
                    "exposed_model_action_dim": self.artifact.manifest["exposed_model_action_dim"],
                },
                "reference_processor_owner_resident": True,
                "reference_processor_owner_note": (
                    "evaluation backend currently retains the trusted PyTorch policy service solely to reuse "
                    "its audited pre/post processors; no performance promotion is implied"
                ),
            }
        )
        return health

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        request.validate()
        if request.mode == RTC_MODE:
            raise ValueError("Triton backend supports single_step only; RTC remains on the PyTorch backend")
        if request.mode != SINGLE_STEP_MODE:
            raise ValueError(f"unsupported Triton mode: {request.mode!r}")
        started_s = time.perf_counter()
        try:
            with self._lock, torch.inference_mode():
                images, state_tokens = self._prepare_inputs(request)
                noise = torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=(TRITON_CHUNK_SIZE, TRITON_INTERNAL_ACTION_DIM),
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
                )
                torch.cuda.synchronize(self.device)
                model_latency_s = time.perf_counter() - model_started_s
                model_actions = internal_actions[:, :MODEL_ACTION_DIM].to(
                    dtype=torch.float32,
                    device="cpu",
                )
                processed = postprocess_action_chunk(
                    self._processor_service.postprocessor,
                    model_actions.unsqueeze(0),
                ).squeeze(0)
                raw_np = np.ascontiguousarray(model_actions.numpy())
                processed_np = np.ascontiguousarray(processed.to(torch.float32).numpy())
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
            self.last_latency_s = response.server_latency_s
        return response

    def _prepare_inputs(self, request: InferenceRequest) -> tuple[torch.Tensor, np.ndarray]:
        batch = prepare_observation_for_inference(
            dict(request.observation_frame),
            device=self.device,
            task=request.task,
            robot_type=request.robot_type,
        )
        preprocessed = self._processor_service.preprocessor(batch)
        camera_keys = self.artifact.manifest["camera_keys"]
        images = []
        for key in camera_keys:
            image = preprocessed.get(key)
            if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1:
                raise ValueError(f"Triton camera {key!r} must be one BCHW tensor")
            if image.shape[1] != 3:
                raise ValueError(f"Triton camera {key!r} must have three channels")
            image_hwc = image.to(torch.float32).permute(0, 2, 3, 1)
            image_hwc = resize_with_pad_torch(image_hwc, 224, 224)
            image_hwc = image_hwc.mul(2.0).sub(1.0)
            images.append(image_hwc.squeeze(0))
        image_tensor = torch.stack(images).to(device=self.device, dtype=torch.bfloat16)
        state = preprocessed.get("observation.state")
        if not isinstance(state, torch.Tensor) or state.shape != (1, MODEL_ACTION_DIM):
            raise ValueError(f"normalized Triton state must be (1,16), got {getattr(state, 'shape', None)}")
        padded_state = torch.zeros(1, TRITON_INTERNAL_ACTION_DIM, dtype=torch.float32)
        padded_state[:, :MODEL_ACTION_DIM] = state.detach().to(dtype=torch.float32, device="cpu")
        state_tokens = np.digitize(
            padded_state.numpy(),
            bins=np.linspace(-1, 1, 256 + 1)[:-1],
        ) - 1
        return image_tensor, np.ascontiguousarray(state_tokens.squeeze(0))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["TritonArtifact", "TritonPolicyBackend"]
