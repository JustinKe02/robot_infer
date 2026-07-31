from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any

import torch

BACKEND_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TorchFeatureFlags:
    inference_mode: bool = False
    bf16_autocast: bool = False
    pinned_memory: bool = False
    non_blocking_copies: bool = False
    static_buffers: bool = False
    cuda_graph: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "inference_mode": self.inference_mode,
            "bf16_autocast": self.bf16_autocast,
            "pinned_memory": self.pinned_memory,
            "non_blocking_copies": self.non_blocking_copies,
            "static_buffers": self.static_buffers,
            "cuda_graph": self.cuda_graph,
        }


def build_torch_backend_manifest(
    *,
    backend: str,
    device: torch.device,
    features: TorchFeatureFlags,
    checkpoint_health: dict[str, Any],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": BACKEND_MANIFEST_SCHEMA_VERSION,
        "backend": backend,
        "device": str(device),
        "features": features.to_dict(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "checkpoint_fingerprint": checkpoint_health.get("checkpoint_fingerprint"),
        "checkpoint_step": checkpoint_health.get("checkpoint_step"),
        "model_action_dim": checkpoint_health.get("model_action_dim"),
        "wire_action_dim": checkpoint_health.get("wire_action_dim"),
        "camera_profile": checkpoint_health.get("camera_profile"),
        "camera_shapes": checkpoint_health.get("camera_shapes"),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        manifest["cuda_device"] = {
            "index": index,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    else:
        manifest["cuda_device"] = None
    try:
        import triton

        manifest["triton_version"] = triton.__version__
    except ImportError:
        manifest["triton_version"] = None
    return manifest


__all__ = [
    "BACKEND_MANIFEST_SCHEMA_VERSION",
    "TorchFeatureFlags",
    "build_torch_backend_manifest",
]
