from __future__ import annotations

import json
from argparse import Namespace

import pytest

from tk_infer.pi05_optimized.runtime.training_conditioning_gate import (
    evaluate_training_conditioning_gate,
    validate_training_conditioning_checkpoint,
)
from tk_infer.pi05_optimized.tools.offline_phase8_training_gate import run_gate


def _manifest(*, p95: float = 2.0, fraction: float = 0.2) -> dict[str, object]:
    return {
        "schema_version": 1,
        "delay_evidence": {
            "evidence_kind": "measured_end_to_end_delay_steps",
            "source": "recorded_optimized_runtime",
            "per_request_trace": True,
            "backend_optimized": True,
            "backend": "triton",
            "checkpoint_fingerprint": "checkpoint",
            "request_count": 500,
            "control_period_s": 0.05,
            "p95_delay_steps": p95,
            "fraction_delay_steps_ge_2": fraction,
            "histogram": {"0": 100, "1": 300, "2": 100},
            "overflow_rate": 0.0,
            "queue_empty_rate": 0.0,
            "stale_chunk_rate": 0.0,
        },
        "training_contract": {
            "randomized_delay_histogram_configured": True,
            "hard_action_prefix_implemented": True,
            "token_wise_flow_timestep_implemented": True,
            "postfix_only_loss_verified": True,
            "checkpoint_metadata_schema_ready": True,
            "old_checkpoint_rejection_verified": True,
            "unified_prefix_source_trace_passed": True,
            "disabled_path_large_model_parity_passed": True,
            "per_parameter_gradient_assertions_passed": True,
            "runtime_carries_action_prefix": True,
            "runtime_carries_prefix_length": True,
            "inference_time_vjp_rtc_disabled": True,
        },
    }


def test_gate_has_three_states_and_never_authorizes_or_invokes_training() -> None:
    ready = evaluate_training_conditioning_gate(_manifest())
    assert ready.status == "READY_FOR_SEPARATE_TRAINING_AUTHORIZATION"
    assert ready.trigger_satisfied is True
    assert ready.contract_ready is True
    assert ready.training_allowed is False
    assert ready.training_command_invoked is False

    not_triggered = evaluate_training_conditioning_gate(_manifest(p95=1.0, fraction=0.01))
    assert not_triggered.status == "NOT_TRIGGERED"
    assert not_triggered.evidence_valid is True
    assert not_triggered.trigger_satisfied is False

    blocked_manifest = _manifest()
    blocked_manifest["delay_evidence"]["evidence_kind"] = "predicted"  # type: ignore[index]
    blocked = evaluate_training_conditioning_gate(blocked_manifest)
    assert blocked.status == "BLOCKED"
    assert blocked.evidence_valid is False


def test_current_evidence_gate_is_blocked_without_starting_training() -> None:
    report = run_gate(
        Namespace(input_json=None, common_delay_fraction=0.1, min_requests=200)
    )

    assert report["status"] == "BLOCKED"
    assert report["hardware_access"] is False
    assert report["network_access"] is False
    assert report["training_process_created"] is False
    assert report["training_command_invoked"] is False
    assert report["decision"]["evidence_valid"] is False
    assert report["source_evidence"]["phase1_predicted_delay_steps"]["p99"] == 1.0
    assert report["source_evidence"]["phase3_rtc_supported"] is False


def test_training_checkpoint_validator_rejects_old_and_accepts_explicit_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"type": "pi05"}), encoding="utf-8")
    with pytest.raises(ValueError, match="old/unconditioned checkpoint rejected"):
        validate_training_conditioning_checkpoint(checkpoint)

    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "pi05",
                "rtc_training": {
                    "enabled": True,
                    "max_delay": 10,
                    "min_postfix_steps": 1,
                    "inference_contract": "training_time_action_conditioning_v1",
                },
            }
        ),
        encoding="utf-8",
    )
    metadata = validate_training_conditioning_checkpoint(checkpoint)
    assert metadata.rtc_training_enabled is True
    assert metadata.max_delay == 10
    assert metadata.inference_contract == "training_time_action_conditioning_v1"
