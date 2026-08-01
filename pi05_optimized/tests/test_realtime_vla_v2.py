from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tk_infer.pi05_optimized.backends import realtime_vla_v2_backend as backend_module
from tk_infer.pi05_optimized.backends.realtime_vla_v2_backend import (
    RealtimeVLAV2Artifact,
    RealtimeVLAV2PolicyBackend,
    _validate_wire_actions,
)
from tk_infer.pi05_optimized.runtime.realtime_vla_v2_contract import (
    ARTIFACT_FORMAT,
    EXPECTED_OUTPUT_SHAPES,
    KERNEL_CONTRACT,
    RTC_INFERENCE_CONTRACT,
    RTC_SOURCE_TENSOR_SHAPES,
    expected_manifest_values,
)
from tk_infer.pi05_optimized.third_party.realtime_vla_v2 import (
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
)
from tk_infer.pi05_optimized.third_party.realtime_vla_v2.pi05rtc_infer import (
    _posemb_sincos_torch,
    build_rtc_token_condition_bases,
)
from tk_infer.pi05_optimized.tools import (
    convert_pi05_safetensors_to_realtime_vla_v2 as converter_module,
)
from tk_infer.pi05_optimized.tools.convert_pi05_safetensors_to_realtime_vla_v2 import (
    _convert_rtc_tensors,
)

from .helpers import make_request

VENDORED_ROOT = Path(__file__).resolve().parents[1] / "third_party/realtime_vla_v2"


class _FakeSlice:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = shape

    def get_shape(self) -> list[int]:
        return list(self._shape)

    def get_dtype(self) -> str:
        return "BF16"


class _FakeSafeOpen:
    def __init__(self, metadata: dict[str, str]) -> None:
        self._metadata = metadata

    def __enter__(self) -> _FakeSafeOpen:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def keys(self) -> list[str]:
        return list(EXPECTED_OUTPUT_SHAPES)

    def get_slice(self, name: str) -> _FakeSlice:
        return _FakeSlice(EXPECTED_OUTPUT_SHAPES[name])

    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_header(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        "format": ARTIFACT_FORMAT,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "kernel_contract": KERNEL_CONTRACT,
        "converter_version": "1",
        "source_sha256": manifest["source_sha256"],
        "checkpoint_fingerprint": manifest["checkpoint_fingerprint"],
    }


def _write_manifest_artifact(path: Path) -> tuple[Path, dict[str, Any]]:
    path.mkdir()
    model_path = path / "model.safetensors"
    model_path.write_bytes(b"unit-test-v2-artifact")
    manifest = {
        **expected_manifest_values(),
        "status": "PASS",
        "validate_only": False,
        "output_written": True,
        "source_sha256": "1" * 64,
        "output_sha256": _sha256(model_path),
        "output_bytes": model_path.stat().st_size,
        "checkpoint_fingerprint": "test-v2-checkpoint-fingerprint",
        "rtc_conditioned_task": "jz robot pin timed vr teleoperation",
        "action_mapping_proof": "audited model16 to raw18 training schema",
        "exposed_action_names": [f"action_{index}" for index in range(16)],
        "required_rtc_source_tensors": sorted(RTC_SOURCE_TENSOR_SHAPES),
        "rtc_training": {
            "enabled": True,
            "inference_contract": RTC_INFERENCE_CONTRACT,
            "chunk_size": 50,
            "max_delay": 10,
            "min_postfix_steps": 1,
            "maximum_prefix_length": 10,
            "observed_delay_histogram": [],
            "observed_histogram_weight": 0.9,
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_vendored_v2_sources_match_recorded_hashes_and_pinned_commit() -> None:
    manifest = json.loads((VENDORED_ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))

    assert manifest["upstream_commit"] == UPSTREAM_COMMIT
    assert manifest["upstream_repository"] == UPSTREAM_REPOSITORY
    for name, provenance in manifest["files"].items():
        assert _sha256(VENDORED_ROOT / name) == provenance["vendored_sha256"]
        assert len(provenance["git_blob"]) == 40
        assert len(provenance["upstream_sha256"]) == 64
    assert "from .pi0_infer import" in (VENDORED_ROOT / "pi05_infer.py").read_text()
    rtc_source = (VENDORED_ROOT / "pi05rtc_infer.py").read_text()
    assert "from .pi05_infer import" in rtc_source
    assert "build_rtc_token_condition_bases" in rtc_source


def test_v2_artifact_loader_checks_manifest_hash_tensor_contract_and_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, manifest = _write_manifest_artifact(tmp_path / "artifact")
    monkeypatch.setattr(
        backend_module,
        "safe_open",
        lambda *_args, **_kwargs: _FakeSafeOpen(_artifact_header(manifest)),
    )

    artifact = RealtimeVLAV2Artifact.load(directory)

    assert artifact.maximum_prefix_length == 10
    assert artifact.manifest["output_tensor_count"] == 51


def test_v2_artifact_loader_rejects_manifest_and_hash_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, manifest = _write_manifest_artifact(tmp_path / "artifact")
    monkeypatch.setattr(
        backend_module,
        "safe_open",
        lambda *_args, **_kwargs: _FakeSafeOpen(_artifact_header(manifest)),
    )
    manifest_path = directory / "manifest.json"

    tampered = dict(manifest)
    tampered["kernel_contract"] = "upstream_unconditioned_rtc"
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="kernel_contract"):
        RealtimeVLAV2Artifact.load(directory)

    tampered = dict(manifest)
    tampered["output_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        RealtimeVLAV2Artifact.load(directory)


def test_v2_artifact_loader_rejects_safetensors_header_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, manifest = _write_manifest_artifact(tmp_path / "artifact")
    header = _artifact_header(manifest)
    header["format"] = "pi05_realtime_vla_v1_fused"
    monkeypatch.setattr(
        backend_module,
        "safe_open",
        lambda *_args, **_kwargs: _FakeSafeOpen(header),
    )

    with pytest.raises(ValueError, match="safetensors header"):
        RealtimeVLAV2Artifact.load(directory)


def test_v2_converter_maps_all_five_learned_rtc_tensors() -> None:
    values = {
        "model.rtc_prefix_embedding.weight": torch.arange(8).reshape(2, 4),
        "model.rtc_token_time_mlp_in.weight": torch.arange(16).reshape(4, 4),
        "model.rtc_token_time_mlp_in.bias": torch.arange(4),
        "model.rtc_token_time_mlp_out.weight": torch.arange(16, 32).reshape(4, 4),
        "model.rtc_token_time_mlp_out.bias": torch.arange(4, 8),
    }
    source = SimpleNamespace(get_tensor=values.__getitem__)

    converted = _convert_rtc_tensors(source)

    assert set(converted) == {
        "rtc_prefix_embedding",
        "rtc_token_time_mlp_in_w",
        "rtc_token_time_mlp_in_b",
        "rtc_token_time_mlp_out_w",
        "rtc_token_time_mlp_out_b",
    }
    torch.testing.assert_close(
        converted["rtc_token_time_mlp_in_w"],
        values["model.rtc_token_time_mlp_in.weight"].T.to(torch.bfloat16),
    )
    assert all(value.dtype == torch.bfloat16 and value.is_contiguous() for value in converted.values())


def test_v2_converter_cli_records_fail_closed_validation_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fail_validation(_args: object) -> dict[str, Any]:
        raise ValueError("RTC-conditioned checkpoint requires rtc_training.enabled=true")

    def capture_report(_path: Path, report: dict[str, Any]) -> Path:
        captured.update(report)
        return tmp_path / "report.json"

    monkeypatch.setattr(converter_module, "convert", fail_validation)
    monkeypatch.setattr(converter_module, "_write_report", capture_report)

    result = converter_module.main(
        [
            "--policy-path=/tmp/non-rtc-checkpoint",
            "--rtc-conditioned-task=jz robot pin timed vr teleoperation",
            "--validate-only",
        ]
    )

    assert result == 2
    assert captured["status"] == "FAIL"
    assert captured["error_type"] == "ValueError"
    assert captured["validate_only"] is True
    assert captured["output_written"] is False


def test_v2_kernel_condition_matches_local_training_formula() -> None:
    time_embeds = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
        dtype=torch.bfloat16,
    )
    weights = {
        "decoder_time_embeds": time_embeds,
        "rtc_token_time_mlp_in_w": torch.eye(4, dtype=torch.bfloat16),
        "rtc_token_time_mlp_in_b": torch.zeros(4, dtype=torch.bfloat16),
        "rtc_token_time_mlp_out_w": torch.eye(4, dtype=torch.bfloat16),
        "rtc_token_time_mlp_out_b": torch.zeros(4, dtype=torch.bfloat16),
        "rtc_prefix_embedding": torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
            dtype=torch.bfloat16,
        ),
    }

    suffix, prefix = build_rtc_token_condition_bases(weights)

    expected_suffix = (time_embeds.float() * torch.sigmoid(time_embeds.float()) + 1.0).to(torch.bfloat16)
    zero = _posemb_sincos_torch(torch.zeros(1), embedding_dim=4)
    expected_prefix = (zero * torch.sigmoid(zero) + 2.0).to(torch.bfloat16)[0]
    torch.testing.assert_close(suffix, expected_suffix)
    torch.testing.assert_close(prefix, expected_prefix)


def test_v2_prefix_contract_pads_model16_and_rejects_overflow_or_missing_leftover() -> None:
    backend = object.__new__(RealtimeVLAV2PolicyBackend)
    backend.artifact = SimpleNamespace(maximum_prefix_length=10)
    request = make_request(mode="rtc")

    clean, internal, length = backend._action_prefix(request)

    assert length == request.predicted_delay_steps == 1
    assert clean is not None and internal is not None
    assert clean.shape == (1, 16)
    assert internal.shape == (1, 32)
    np.testing.assert_array_equal(internal[:, :16], clean)
    np.testing.assert_array_equal(internal[:, 16:], 0)

    request.predicted_delay_steps = 11
    with pytest.raises(ValueError, match="exceeds conditioned max prefix"):
        backend._action_prefix(request)
    request.predicted_delay_steps = 1
    request.prev_chunk_left_over = None
    with pytest.raises(ValueError, match="requires model16 leftover"):
        backend._action_prefix(request)


def test_v2_wire_contract_requires_raw18_finite_actions_and_exact_force_slots() -> None:
    actions = np.zeros((50, 18), dtype=np.float32)
    actions[:, 15] = 80.0
    actions[:, 17] = 80.0

    _validate_wire_actions(actions)

    actions[0, 17] = 79.0
    with pytest.raises(ValueError, match="force slot 17"):
        _validate_wire_actions(actions)
