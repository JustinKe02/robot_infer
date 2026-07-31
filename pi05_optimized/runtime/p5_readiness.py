from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .p5_single_action import (
    DEFAULT_CHECKPOINT_FINGERPRINT,
    DEFAULT_CHECKPOINT_STEP,
    P5_AUTHORIZATION_SCOPE,
)

P5ReadinessStatus = Literal["BLOCKED", "READY_FOR_SEPARATE_P5_SINGLE_ACTION_TRIAL"]


@dataclass(frozen=True, slots=True)
class P5ReadinessDecision:
    status: P5ReadinessStatus
    selected_candidate: str | None
    checks: dict[str, bool]
    blocking_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_p5_readiness(
    *,
    backend_ab: Mapping[str, Any],
    tracker_replay: Mapping[str, Any],
    authorization: Mapping[str, Any] | None = None,
) -> P5ReadinessDecision:
    if not isinstance(backend_ab, Mapping):
        raise TypeError("backend_ab must be a mapping")
    if not isinstance(tracker_replay, Mapping):
        raise TypeError("tracker_replay must be a mapping")
    if authorization is not None and not isinstance(authorization, Mapping):
        raise TypeError("authorization must be a mapping or None")

    checks: dict[str, bool] = {}
    blocking_reasons: list[str] = []

    def check(name: str, condition: bool, reason: str, *, blocking: bool = True) -> None:
        checks[name] = bool(condition)
        if not condition and blocking:
            blocking_reasons.append(reason)

    selection = _mapping(backend_ab.get("selection"))
    selected = selection.get("selected_production_candidate")
    selected_candidate = selected if isinstance(selected, str) and selected.strip() else None
    reports = _mapping(backend_ab.get("reports"))
    selected_report = _mapping(reports.get(selected_candidate)) if selected_candidate is not None else {}
    selected_integrity = _mapping(selected_report.get("integrity_gate"))
    selected_latency = _mapping(selected_report.get("request_total_s"))
    thresholds = _mapping(backend_ab.get("thresholds"))

    check("backend_ab_status", backend_ab.get("status") == "PASS", "backend A/B report status is not PASS")
    check(
        "backend_ab_no_action",
        backend_ab.get("action_sent") is False,
        "backend A/B evidence does not prove action_sent=false",
    )
    check(
        "backend_ab_integrity",
        backend_ab.get("integrity_failures") == [],
        "backend A/B integrity failures are present",
    )
    check(
        "performance_gate",
        selection.get("performance_gate_status") == "PASS",
        "no backend satisfies the production performance gate",
    )
    check(
        "production_candidate_selected",
        selected_candidate is not None,
        "selected_production_candidate is absent",
    )
    check(
        "candidate_integrity",
        selected_integrity.get("status") == "PASS" and selected_integrity.get("action_sent") is False,
        "selected candidate lacks a passing no-action integrity gate",
        blocking=selected_candidate is not None,
    )
    check(
        "candidate_checkpoint_step",
        selected_report.get("checkpoint_step") == DEFAULT_CHECKPOINT_STEP,
        f"selected candidate is not checkpoint step {DEFAULT_CHECKPOINT_STEP}",
        blocking=selected_candidate is not None,
    )
    check(
        "candidate_checkpoint_fingerprint",
        selected_report.get("checkpoint_fingerprint") == DEFAULT_CHECKPOINT_FINGERPRINT,
        "selected candidate checkpoint fingerprint does not match step-015900",
        blocking=selected_candidate is not None,
    )
    check(
        "candidate_request_p95",
        _at_most(selected_latency.get("p95"), thresholds.get("request_p95_s")),
        "selected candidate request p95 exceeds the configured gate",
        blocking=selected_candidate is not None,
    )
    check(
        "candidate_request_p99",
        _at_most(selected_latency.get("p99"), thresholds.get("request_p99_s")),
        "selected candidate request p99 exceeds the configured gate",
        blocking=selected_candidate is not None,
    )

    first_run = _mapping(tracker_replay.get("first_run"))
    check(
        "tracker_replay_status",
        tracker_replay.get("status") == "PASS" and tracker_replay.get("tracker_replay_passed") is True,
        "deterministic tracker replay has not passed",
    )
    check(
        "tracker_replay_isolation",
        tracker_replay.get("hardware_access") is False
        and tracker_replay.get("network_access") is False
        and tracker_replay.get("action_transport_created") is False,
        "tracker replay evidence is not isolated from hardware/network/action transport",
    )
    check(
        "tracker_replay_deterministic",
        tracker_replay.get("deterministic") is True
        and first_run.get("action_sha256") == tracker_replay.get("second_action_sha256"),
        "tracker replay is not deterministic",
    )
    check(
        "tracker_replay_action_contract",
        first_run.get("finite") is True
        and first_run.get("force_slots_exact_80") is True
        and first_run.get("contact_used_as_safety") is False
        and _at_most(first_run.get("max_output_joint_step_rad"), 0.0200001),
        "tracker replay action contract is not safe for the single-action boundary",
    )

    if authorization is None:
        check("explicit_p5_authorization", False, "explicit P5 on-site authorization evidence is absent")
    else:
        authorization_candidate = authorization.get("backend_label")
        authorization_checks = {
            "authorization_id": isinstance(authorization.get("authorization_id"), str)
            and bool(str(authorization.get("authorization_id")).strip()),
            "authorization_approved": authorization.get("approved") is True,
            "authorization_scope": authorization.get("scope") == P5_AUTHORIZATION_SCOPE,
            "authorization_single_action": authorization.get("max_actions") == 1,
            "authorization_checkpoint_step": authorization.get("checkpoint_step") == DEFAULT_CHECKPOINT_STEP,
            "authorization_checkpoint_fingerprint": authorization.get("checkpoint_fingerprint")
            == DEFAULT_CHECKPOINT_FINGERPRINT,
            "authorization_backend": selected_candidate is not None
            and authorization_candidate == selected_candidate,
            "authorization_operator_present": authorization.get("operator_present") is True,
            "authorization_physical_estop": authorization.get("physical_emergency_stop_verified") is True,
            "authorization_workspace_clear": authorization.get("workspace_clear") is True,
            "authorization_robot_powered": authorization.get("robot_powered") is True,
        }
        for name, passed in authorization_checks.items():
            check(name, passed, f"P5 authorization check failed: {name}")

    return P5ReadinessDecision(
        status=("READY_FOR_SEPARATE_P5_SINGLE_ACTION_TRIAL" if not blocking_reasons else "BLOCKED"),
        selected_candidate=selected_candidate,
        checks=checks,
        blocking_reasons=tuple(blocking_reasons),
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _at_most(value: object, limit: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(limit, bool) or not isinstance(limit, (int, float)):
        return False
    return float(value) <= float(limit)


__all__ = ["P5ReadinessDecision", "P5ReadinessStatus", "evaluate_p5_readiness"]
