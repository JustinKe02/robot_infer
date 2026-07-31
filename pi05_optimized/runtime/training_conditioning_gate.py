from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Literal, TypeAlias

GATE_SCHEMA_VERSION = 1
DEFAULT_COMMON_DELAY_FRACTION = 0.10
DEFAULT_MIN_REQUESTS = 200
EXPECTED_CONTROL_PERIOD_S = 0.05

GateStatus: TypeAlias = Literal[
    "BLOCKED",
    "NOT_TRIGGERED",
    "READY_FOR_SEPARATE_TRAINING_AUTHORIZATION",
]


@dataclass(frozen=True, slots=True)
class TrainingConditioningCheckpointMetadata:
    checkpoint_path: str
    policy_type: str
    rtc_training_enabled: bool
    max_delay: int
    min_postfix_steps: int
    inference_contract: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingConditioningGateDecision:
    status: GateStatus
    trigger_satisfied: bool
    evidence_valid: bool
    contract_ready: bool
    training_allowed: bool
    training_command_invoked: bool
    blocking_reasons: tuple[str, ...]
    trigger_reasons: tuple[str, ...]
    evidence_summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        output = asdict(self)
        output["blocking_reasons"] = list(self.blocking_reasons)
        output["trigger_reasons"] = list(self.trigger_reasons)
        return output


def evaluate_training_conditioning_gate(
    manifest: dict[str, Any],
    *,
    common_delay_fraction: float = DEFAULT_COMMON_DELAY_FRACTION,
    min_requests: int = DEFAULT_MIN_REQUESTS,
) -> TrainingConditioningGateDecision:
    if not isinstance(manifest, dict):
        raise TypeError("training gate manifest must be a JSON object")
    common_delay_fraction = _fraction("common_delay_fraction", common_delay_fraction)
    if isinstance(min_requests, bool) or not isinstance(min_requests, int) or min_requests <= 0:
        raise ValueError("min_requests must be a positive integer")
    blocking_reasons: list[str] = []
    trigger_reasons: list[str] = []
    if manifest.get("schema_version") != GATE_SCHEMA_VERSION:
        blocking_reasons.append("gate manifest schema_version must be 1")

    evidence = manifest.get("delay_evidence")
    evidence_valid = isinstance(evidence, dict)
    if not evidence_valid:
        blocking_reasons.append("missing delay_evidence object")
        evidence = {}
    assert isinstance(evidence, dict)
    if evidence.get("evidence_kind") != "measured_end_to_end_delay_steps":
        evidence_valid = False
        blocking_reasons.append("delay evidence must be measured end-to-end, not predicted or synthetic")
    if evidence.get("source") not in {"recorded_optimized_runtime", "live_read_only_optimized_runtime"}:
        evidence_valid = False
        blocking_reasons.append("delay evidence must come from an optimized recorded/live-read-only runtime")
    if evidence.get("per_request_trace") is not True:
        evidence_valid = False
        blocking_reasons.append("per-request delay trace is required")
    if evidence.get("backend_optimized") is not True:
        evidence_valid = False
        blocking_reasons.append("delay evidence must be collected after backend optimization")
    if not _non_empty_string(evidence.get("backend")):
        evidence_valid = False
        blocking_reasons.append("delay evidence must identify the backend")
    if not _non_empty_string(evidence.get("checkpoint_fingerprint")):
        evidence_valid = False
        blocking_reasons.append("delay evidence must identify the checkpoint fingerprint")
    request_count = _optional_non_negative_int(evidence.get("request_count"))
    if request_count is None or request_count < min_requests:
        evidence_valid = False
        blocking_reasons.append(f"delay evidence requires at least {min_requests} measured requests")
    control_period_s = _optional_finite(evidence.get("control_period_s"))
    if control_period_s is None or not math.isclose(
        control_period_s, EXPECTED_CONTROL_PERIOD_S, rel_tol=0.0, abs_tol=1e-9
    ):
        evidence_valid = False
        blocking_reasons.append("delay evidence control_period_s must be exactly 0.05 for 20 Hz")
    p95_delay_steps = _optional_finite(evidence.get("p95_delay_steps"))
    fraction_ge_2 = _optional_fraction(evidence.get("fraction_delay_steps_ge_2"))
    if p95_delay_steps is None:
        evidence_valid = False
        blocking_reasons.append("delay evidence must report p95_delay_steps")
    if fraction_ge_2 is None:
        evidence_valid = False
        blocking_reasons.append("delay evidence must report fraction_delay_steps_ge_2")
    if not isinstance(evidence.get("histogram"), dict) or not evidence.get("histogram"):
        evidence_valid = False
        blocking_reasons.append("delay evidence must report a non-empty measured histogram")
    for field in ("overflow_rate", "queue_empty_rate", "stale_chunk_rate"):
        if _optional_fraction(evidence.get(field)) is None:
            evidence_valid = False
            blocking_reasons.append(f"delay evidence must report {field}")

    trigger_satisfied = bool(
        evidence_valid
        and p95_delay_steps is not None
        and p95_delay_steps >= 2.0
        and fraction_ge_2 is not None
        and fraction_ge_2 >= common_delay_fraction
    )
    if evidence_valid and not trigger_satisfied:
        if p95_delay_steps is not None and p95_delay_steps < 2.0:
            trigger_reasons.append("measured p95_delay_steps is below 2")
        if fraction_ge_2 is not None and fraction_ge_2 < common_delay_fraction:
            trigger_reasons.append(
                f"fraction_delay_steps_ge_2 is below common threshold {common_delay_fraction:.3f}"
            )

    contract = manifest.get("training_contract")
    contract_ready = isinstance(contract, dict)
    if not contract_ready:
        blocking_reasons.append("missing training_contract object")
        contract = {}
    assert isinstance(contract, dict)
    required_contract_flags = {
        "randomized_delay_histogram_configured": "randomized empirical delay distribution is not configured",
        "hard_action_prefix_implemented": "hard clean action prefix is not implemented",
        "token_wise_flow_timestep_implemented": "token-wise flow timestep is not implemented",
        "postfix_only_loss_verified": "postfix-only loss is not verified",
        "checkpoint_metadata_schema_ready": "training-time checkpoint metadata schema is not ready",
        "old_checkpoint_rejection_verified": "old checkpoint rejection is not verified",
        "unified_prefix_source_trace_passed": "unified runtime prefix source/overflow trace is missing",
        "disabled_path_large_model_parity_passed": "large-model disabled-path parity is missing",
        "per_parameter_gradient_assertions_passed": "per-parameter gradient assertions are missing",
        "runtime_carries_action_prefix": "runtime does not carry action_prefix",
        "runtime_carries_prefix_length": "runtime does not carry prefix_length",
    }
    for field, reason in required_contract_flags.items():
        if contract.get(field) is not True:
            contract_ready = False
            blocking_reasons.append(reason)
    if contract.get("inference_time_vjp_rtc_disabled") is not True:
        contract_ready = False
        blocking_reasons.append("training-time conditioning must be mutually exclusive with VJP RTC")

    if not evidence_valid or (trigger_satisfied and not contract_ready):
        status: GateStatus = "BLOCKED"
    elif not trigger_satisfied:
        status = "NOT_TRIGGERED"
    else:
        status = "READY_FOR_SEPARATE_TRAINING_AUTHORIZATION"
    return TrainingConditioningGateDecision(
        status=status,
        trigger_satisfied=trigger_satisfied,
        evidence_valid=evidence_valid,
        contract_ready=contract_ready,
        training_allowed=False,
        training_command_invoked=False,
        blocking_reasons=tuple(dict.fromkeys(blocking_reasons)),
        trigger_reasons=tuple(trigger_reasons),
        evidence_summary={
            "request_count": request_count,
            "control_period_s": control_period_s,
            "p95_delay_steps": p95_delay_steps,
            "fraction_delay_steps_ge_2": fraction_ge_2,
            "common_delay_fraction_threshold": common_delay_fraction,
            "minimum_request_count": min_requests,
        },
    )


def validate_training_conditioning_checkpoint(
    checkpoint_path: str | Path,
) -> TrainingConditioningCheckpointMetadata:
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint is missing config.json: {path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"checkpoint config.json is invalid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("checkpoint config.json must contain an object")
    if config.get("type") != "pi05":
        raise ValueError("training-time conditioning checkpoint must have type='pi05'")
    rtc_training = config.get("rtc_training")
    if not isinstance(rtc_training, dict) or rtc_training.get("enabled") is not True:
        raise ValueError(
            "old/unconditioned checkpoint rejected: rtc_training.enabled=true is required"
        )
    max_delay = _positive_int("rtc_training.max_delay", rtc_training.get("max_delay"))
    min_postfix_steps = _positive_int(
        "rtc_training.min_postfix_steps", rtc_training.get("min_postfix_steps")
    )
    if max_delay >= 50:
        raise ValueError("rtc_training.max_delay must leave at least one postfix step")
    if min_postfix_steps > 50 - max_delay:
        raise ValueError("rtc_training.min_postfix_steps exceeds the available postfix horizon")
    inference_contract = rtc_training.get("inference_contract")
    if inference_contract != "training_time_action_conditioning_v1":
        raise ValueError(
            "rtc_training.inference_contract must be 'training_time_action_conditioning_v1'"
        )
    return TrainingConditioningCheckpointMetadata(
        checkpoint_path=str(path),
        policy_type="pi05",
        rtc_training_enabled=True,
        max_delay=max_delay,
        min_postfix_steps=min_postfix_steps,
        inference_contract=inference_contract,
    )


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _optional_finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _optional_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_fraction(value: object) -> float | None:
    converted = _optional_finite(value)
    if converted is None or not 0.0 <= converted <= 1.0:
        return None
    return converted


def _fraction(name: str, value: object) -> float:
    converted = _optional_fraction(value)
    if converted is None:
        raise ValueError(f"{name} must be finite and in 0..1")
    return converted


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


__all__ = [
    "DEFAULT_COMMON_DELAY_FRACTION",
    "DEFAULT_MIN_REQUESTS",
    "EXPECTED_CONTROL_PERIOD_S",
    "GATE_SCHEMA_VERSION",
    "TrainingConditioningCheckpointMetadata",
    "TrainingConditioningGateDecision",
    "evaluate_training_conditioning_gate",
    "validate_training_conditioning_checkpoint",
]
