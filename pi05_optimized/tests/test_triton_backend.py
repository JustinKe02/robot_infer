from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tk_infer.pi05_optimized.backends.triton_backend import (
    TritonArtifact,
    TritonPolicyBackend,
)
from tk_infer.pi05_optimized.third_party.realtime_vla import UPSTREAM_COMMIT

from .helpers import make_request


def _write_minimal_artifact(path: Path) -> Path:
    path.mkdir()
    tensors = {f"tensor_{index}": torch.zeros(1) for index in range(46)}
    save_file(
        tensors,
        path / "model.safetensors",
        metadata={
            "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint_fingerprint": "test-fingerprint",
        },
    )
    manifest = {
        "manifest_schema_version": 1,
        "upstream_commit": UPSTREAM_COMMIT,
        "dtype": "bfloat16",
        "num_views": 2,
        "camera_profile": "head_right",
        "chunk_size": 50,
        "internal_action_dim": 32,
        "exposed_model_action_dim": 16,
        "exposed_action_indices": list(range(16)),
        "supported_modes": ["single_step"],
        "rtc_supported": False,
        "output_tensor_count": 46,
        "output_sha256": "not-verified-in-unit-test",
        "checkpoint_fingerprint": "test-fingerprint",
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_artifact_loader_requires_pinned_manifest_and_safetensors_header(tmp_path: Path) -> None:
    artifact = TritonArtifact.load(_write_minimal_artifact(tmp_path / "artifact"), verify_sha256=False)

    assert artifact.manifest["upstream_commit"] == UPSTREAM_COMMIT
    assert artifact.manifest["supported_modes"] == ["single_step"]


def test_artifact_loader_rejects_unproven_action_mapping(tmp_path: Path) -> None:
    directory = _write_minimal_artifact(tmp_path / "artifact")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["exposed_action_indices"] = list(range(1, 17))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="exposed_action_indices"):
        TritonArtifact.load(directory, verify_sha256=False)


def test_triton_backend_rejects_rtc_without_fallback() -> None:
    backend = object.__new__(TritonPolicyBackend)

    with pytest.raises(ValueError, match="single_step only"):
        backend.infer(make_request(mode="rtc"))


def test_triton_input_preparation_preserves_view_order_and_discretizes_state() -> None:
    class FakePreprocessor:
        def __call__(self, batch: dict[str, object]) -> dict[str, object]:
            batch["observation.state"] = batch["observation.state"][:, :16]  # type: ignore[index]
            return batch

    backend = object.__new__(TritonPolicyBackend)
    backend.device = torch.device("cpu")
    backend.artifact = SimpleNamespace(
        manifest={
            "camera_keys": [
                "observation.images.camera_head",
                "observation.images.camera_right",
            ]
        }
    )
    backend._processor_service = SimpleNamespace(preprocessor=FakePreprocessor())
    request = make_request()
    request.observation_frame = {
        "observation.state": np.zeros(18, dtype=np.float32),
        "observation.images.camera_head": np.zeros((4, 6, 3), dtype=np.uint8),
        "observation.images.camera_right": np.full((2, 3, 3), 255, dtype=np.uint8),
    }

    images, state_tokens = backend._prepare_inputs(request)

    assert images.shape == (2, 224, 224, 3)
    assert images.dtype == torch.bfloat16
    assert torch.all(images[0, 112, 112] == -1)
    assert torch.all(images[1, 112, 112] == 1)
    assert torch.all(images[:, 0, 0] == -3)
    assert state_tokens.shape == (32,)
    assert state_tokens.min() >= 0
    assert state_tokens.max() <= 255
