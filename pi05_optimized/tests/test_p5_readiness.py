from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from tk_infer.pi05_optimized.runtime.p5_readiness import evaluate_p5_readiness
from tk_infer.pi05_optimized.runtime.p5_single_action import (
    DEFAULT_CHECKPOINT_FINGERPRINT,
    P5_AUTHORIZATION_SCOPE,
)
from tk_infer.pi05_optimized.tools.offline_p5_readiness_gate import run_gate


def _backend_ab(*, performance_passed: bool) -> dict[str, object]:
    selected = "inference_mode" if performance_passed else None
    return {
        "status": "PASS",
        "action_sent": False,
        "integrity_failures": [],
        "thresholds": {"request_p95_s": 0.05, "request_p99_s": 0.1},
        "selection": {
            "performance_gate_status": "PASS" if performance_passed else "FAIL",
            "selected_production_candidate": selected,
        },
        "reports": {
            "inference_mode": {
                "checkpoint_step": 15900,
                "checkpoint_fingerprint": DEFAULT_CHECKPOINT_FINGERPRINT,
                "integrity_gate": {"status": "PASS", "action_sent": False},
                "request_total_s": {"p95": 0.04, "p99": 0.08},
            }
        },
    }


def _tracker_replay() -> dict[str, object]:
    action_hash = "a" * 64
    return {
        "status": "PASS",
        "tracker_replay_passed": True,
        "hardware_access": False,
        "network_access": False,
        "action_transport_created": False,
        "deterministic": True,
        "second_action_sha256": action_hash,
        "first_run": {
            "action_sha256": action_hash,
            "finite": True,
            "force_slots_exact_80": True,
            "contact_used_as_safety": False,
            "max_output_joint_step_rad": 0.02,
        },
    }


def _authorization(**changes: object) -> dict[str, object]:
    evidence: dict[str, object] = {
        "authorization_id": "onsite-authorization-001",
        "approved": True,
        "scope": P5_AUTHORIZATION_SCOPE,
        "max_actions": 1,
        "checkpoint_step": 15900,
        "checkpoint_fingerprint": DEFAULT_CHECKPOINT_FINGERPRINT,
        "backend_label": "inference_mode",
        "operator_present": True,
        "physical_emergency_stop_verified": True,
        "workspace_clear": True,
        "robot_powered": True,
    }
    evidence.update(changes)
    return evidence


def test_performance_and_authorization_are_independent_blockers() -> None:
    decision = evaluate_p5_readiness(
        backend_ab=_backend_ab(performance_passed=False),
        tracker_replay=_tracker_replay(),
    )

    assert decision.status == "BLOCKED"
    assert decision.selected_candidate is None
    assert decision.checks["performance_gate"] is False
    assert decision.checks["explicit_p5_authorization"] is False
    assert any("production performance" in reason for reason in decision.blocking_reasons)
    assert any("authorization" in reason for reason in decision.blocking_reasons)


def test_technical_pass_without_authorization_remains_blocked() -> None:
    decision = evaluate_p5_readiness(
        backend_ab=_backend_ab(performance_passed=True),
        tracker_replay=_tracker_replay(),
    )

    assert decision.status == "BLOCKED"
    assert decision.selected_candidate == "inference_mode"
    assert decision.blocking_reasons == ("explicit P5 on-site authorization evidence is absent",)


def test_exact_evidence_can_only_reach_separate_trial_readiness() -> None:
    decision = evaluate_p5_readiness(
        backend_ab=_backend_ab(performance_passed=True),
        tracker_replay=_tracker_replay(),
        authorization=_authorization(),
    )

    assert decision.status == "READY_FOR_SEPARATE_P5_SINGLE_ACTION_TRIAL"
    assert decision.blocking_reasons == ()
    assert all(decision.checks.values())


def test_authorization_mismatch_blocks_readiness() -> None:
    decision = evaluate_p5_readiness(
        backend_ab=_backend_ab(performance_passed=True),
        tracker_replay=_tracker_replay(),
        authorization=_authorization(max_actions=2, workspace_clear=False),
    )

    assert decision.status == "BLOCKED"
    assert decision.checks["authorization_single_action"] is False
    assert decision.checks["authorization_workspace_clear"] is False


def test_actual_default_reports_remain_blocked_without_hardware_access() -> None:
    optimized_root = Path(__file__).resolve().parents[1]
    report = run_gate(
        Namespace(
            backend_ab_json=optimized_root / "outputs/live/live_backend_ab_step015900_20260730.json",
            tracker_replay_json=optimized_root / "outputs/phase7_tracker_replay.json",
            authorization_json=None,
        )
    )

    assert report["status"] == "BLOCKED"
    assert report["hardware_access"] is False
    assert report["network_access"] is False
    assert report["robot_created"] is False
    assert report["action_transport_created"] is False
    assert report["action_sent"] is False
    assert report["armed_launcher_created"] is False
    assert report["decision"]["checks"]["performance_gate"] is False
    assert report["decision"]["checks"]["explicit_p5_authorization"] is False
