from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tk_infer.pi05_optimized.config import DEFAULT_SERVER_PORT, OPTIMIZED_ROOT, OptimizedRuntimeConfig


def test_phase0_defaults_are_reference_and_pass_through() -> None:
    config = OptimizedRuntimeConfig()

    assert config.backend == "torch"
    assert config.trajectory_processor == "pass_through"
    assert config.server_host == "127.0.0.1"
    assert config.server_port == DEFAULT_SERVER_PORT == 18088
    assert config.policy_path is None
    with pytest.raises(ValueError, match="policy_path is required"):
        config.require_model_paths()


def test_config_is_frozen_and_normalizes_paths() -> None:
    config = OptimizedRuntimeConfig(
        policy_path="~/checkpoint",
        tokenizer_path="~/tokenizer",
    )

    assert config.policy_path == Path("~/checkpoint").expanduser()
    assert config.tokenizer_path == Path("~/tokenizer").expanduser()
    with pytest.raises(FrozenInstanceError):
        config.server_port = 8088  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"backend": "jax"}, "backend must be 'torch', 'torch_optimized', 'torch_rtc_conditioned'"),
        ({"trajectory_processor": "temporal"}, "must be 'pass_through' or 'paired_temporal'"),
        ({"server_host": " "}, "server_host must be"),
        ({"server_port": 65536}, "server_port must be in"),
        ({"max_request_bytes": 0}, "max_request_bytes must be positive"),
        ({"device": ""}, "device must be"),
        ({"rtc_execution_horizon": 51}, "rtc_execution_horizon must be in"),
        ({"rtc_max_guidance_weight": True}, "must be a real number"),
        ({"rtc_max_guidance_weight": "1.0"}, "must be a real number"),
        ({"rtc_max_guidance_weight": float("nan")}, "must be finite"),
        ({"rtc_prefix_attention_schedule": "UNKNOWN"}, "must be one of"),
        ({"metrics_window_size": 0}, "metrics_window_size must be"),
        ({"trace_strict": "true"}, "trace_strict must be"),
        ({"trace_max_bytes": 255}, "trace_max_bytes must be"),
        ({"trace_backup_count": -1}, "trace_backup_count must be"),
    ],
)
def test_config_rejects_unsupported_or_unsafe_values(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OptimizedRuntimeConfig(**overrides)  # type: ignore[arg-type]


def test_public_config_does_not_contain_authentication_secrets() -> None:
    payload = OptimizedRuntimeConfig().public_dict()

    assert "auth_token" not in payload
    assert payload["backend"] == "torch"


def test_phase2_flags_are_explicit_and_default_disabled() -> None:
    payload = OptimizedRuntimeConfig().public_dict()

    assert payload["torch_inference_mode"] is False
    assert payload["torch_bf16_autocast"] is False
    assert payload["torch_pinned_memory"] is False
    assert payload["torch_non_blocking_copies"] is False
    assert payload["torch_static_buffers"] is False
    assert payload["torch_cuda_graph"] is False
    assert payload["torch_warmup_iterations"] == 0


def test_phase2_flags_require_optimized_backend_and_valid_dependencies() -> None:
    with pytest.raises(ValueError, match="require backend='torch_optimized'"):
        OptimizedRuntimeConfig(torch_inference_mode=True)
    with pytest.raises(ValueError, match="requires torch_pinned_memory"):
        OptimizedRuntimeConfig(backend="torch_optimized", torch_non_blocking_copies=True)
    with pytest.raises(ValueError, match="requires torch_static_buffers"):
        OptimizedRuntimeConfig(backend="torch_optimized", torch_cuda_graph=True)

    config = OptimizedRuntimeConfig(
        backend="torch_optimized",
        torch_inference_mode=True,
        torch_bf16_autocast=True,
        torch_pinned_memory=True,
        torch_non_blocking_copies=True,
        torch_warmup_iterations=3,
    )
    assert config.backend == "torch_optimized"
    assert config.torch_warmup_iterations == 3


def test_phase3_triton_backend_requires_confined_artifact_at_load_boundary() -> None:
    artifact = OPTIMIZED_ROOT / "artifacts/triton/test-artifact"
    config = OptimizedRuntimeConfig(backend="triton", triton_artifact_path=artifact)

    assert config.require_triton_artifact_path() == artifact
    assert config.public_dict()["triton_artifact_path"] == str(artifact)
    with pytest.raises(ValueError, match="triton_artifact_path is required"):
        OptimizedRuntimeConfig(backend="triton").require_triton_artifact_path()
    with pytest.raises(ValueError, match="must stay inside"):
        OptimizedRuntimeConfig(backend="triton", triton_artifact_path="/tmp/triton-artifact")
    with pytest.raises(ValueError, match="requires a CUDA"):
        OptimizedRuntimeConfig(backend="triton", device="cpu")


def test_rtc_conditioned_backend_is_independent_from_torch_optimization_flags() -> None:
    config = OptimizedRuntimeConfig(
        backend="torch_rtc_conditioned",
        rtc_conditioned_task="jz robot pin timed vr teleoperation",
    )

    assert config.backend == "torch_rtc_conditioned"
    assert config.rtc_conditioned_task == "jz robot pin timed vr teleoperation"
    with pytest.raises(ValueError, match="requires rtc_conditioned_task"):
        OptimizedRuntimeConfig(backend="torch_rtc_conditioned")
    with pytest.raises(ValueError, match="requires backend='torch_rtc_conditioned'"):
        OptimizedRuntimeConfig(rtc_conditioned_task="task")
    with pytest.raises(ValueError, match="require backend='torch_optimized'"):
        OptimizedRuntimeConfig(
            backend="torch_rtc_conditioned",
            rtc_conditioned_task="task",
            torch_inference_mode=True,
        )


def test_phase6_temporal_processor_is_explicit_and_defaults_to_speed_one() -> None:
    config = OptimizedRuntimeConfig(trajectory_processor="paired_temporal")

    assert config.temporal_speed_factor == 1.0
    assert config.temporal_max_joint_step_rad == pytest.approx(0.02)
    assert config.temporal_solver_timeout_s == pytest.approx(0.05)
    assert config.public_dict()["trajectory_processor"] == "paired_temporal"

    with pytest.raises(ValueError, match="require trajectory_processor='paired_temporal'"):
        OptimizedRuntimeConfig(temporal_speed_factor=1.25)

    configured = OptimizedRuntimeConfig(
        trajectory_processor="paired_temporal",
        temporal_speed_factor=1.25,
        temporal_max_joint_step_rad=0.01,
        temporal_solver_timeout_s=0.1,
    )
    assert configured.temporal_speed_factor == 1.25


def test_trace_path_must_stay_inside_optimized_runtime() -> None:
    local_trace = OPTIMIZED_ROOT / "logs/test-trace.jsonl"
    assert OptimizedRuntimeConfig(trace_path=local_trace).trace_path == local_trace

    with pytest.raises(ValueError, match="trace_path must stay inside"):
        OptimizedRuntimeConfig(trace_path="/tmp/outside-optimized-trace.jsonl")
