#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def resolve_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
OPTIMIZED_ROOT = REPO_ROOT / "tk_infer/pi05_optimized"
DEFAULT_INPUT_PATH = OPTIMIZED_ROOT / "outputs/phase8_training_gate_input.json"
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/phase8_training_gate.json"
PHASE1_REPORT = OPTIMIZED_ROOT / "outputs/phase1_observability_soak_30m.json"
PHASE3_REPORT = OPTIMIZED_ROOT / "outputs/phase3_triton_benchmark.json"
CURRENT_CHECKPOINT = REPO_ROOT / "tk_infer/pi05/checkpoints/010600/pretrained_model"

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from tk_infer.pi05_optimized.runtime.training_conditioning_gate import (  # noqa: E402
    DEFAULT_COMMON_DELAY_FRACTION,
    DEFAULT_MIN_REQUESTS,
    evaluate_training_conditioning_gate,
    validate_training_conditioning_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Phase 8 training-time action-conditioning decision gate."
    )
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--common-delay-fraction", type=float, default=DEFAULT_COMMON_DELAY_FRACTION)
    parser.add_argument("--min-requests", type=int, default=DEFAULT_MIN_REQUESTS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_json is None:
        manifest, source_evidence = _current_manifest()
        input_source = "current_optimized_reports"
    else:
        input_path = args.input_json.expanduser().resolve(strict=True)
        manifest = _read_json_object(input_path)
        source_evidence = {"explicit_input_json": str(input_path)}
        input_source = "explicit_gate_manifest"
    decision = evaluate_training_conditioning_gate(
        manifest,
        common_delay_fraction=args.common_delay_fraction,
        min_requests=args.min_requests,
    )
    return {
        "phase": 8,
        "status": decision.status,
        "hardware_access": False,
        "network_access": False,
        "training_process_created": False,
        "training_command_invoked": False,
        "input_source": input_source,
        "source_evidence": source_evidence,
        "decision": decision.to_dict(),
        "required_next_evidence": [
            "optimized unified-runtime per-request measured delay histogram",
            "measured overflow/queue-empty/stale rates and checkpoint/backend identity",
            "unified action_prefix/prefix_length source and overflow trace",
            "large-model disabled-path parity",
            "per-parameter gradient assertions",
            "training-time checkpoint metadata contract",
        ],
        "authorization_note": (
            "READY means eligible for a separately authorized training run; this gate never starts training."
        ),
    }


def _current_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    phase1 = _read_json_object(PHASE1_REPORT)
    phase3 = _read_json_object(PHASE3_REPORT)
    predicted_delay = (
        phase1.get("client_metrics", {})
        .get("distributions", {})
        .get("predicted_delay_steps", {})
    )
    backend_health = phase3.get("backend_health", {})
    checkpoint_rejection = None
    try:
        validate_training_conditioning_checkpoint(CURRENT_CHECKPOINT)
    except (FileNotFoundError, ValueError) as exc:
        checkpoint_rejection = str(exc)
    manifest = {
        "schema_version": 1,
        "delay_evidence": {
            "evidence_kind": "synthetic_predicted_delay_steps",
            "source": "offline_fake_soak",
            "per_request_trace": False,
            "backend_optimized": False,
            "backend": "deterministic_fake",
            "checkpoint_fingerprint": phase3.get("checkpoint_fingerprint"),
            "request_count": predicted_delay.get("count"),
            "control_period_s": 0.05,
            "p95_delay_steps": predicted_delay.get("p95"),
            "fraction_delay_steps_ge_2": None,
            "histogram": None,
            "overflow_rate": None,
            "queue_empty_rate": None,
            "stale_chunk_rate": None,
        },
        "training_contract": {
            "randomized_delay_histogram_configured": False,
            "hard_action_prefix_implemented": True,
            "token_wise_flow_timestep_implemented": True,
            "postfix_only_loss_verified": True,
            "checkpoint_metadata_schema_ready": True,
            "old_checkpoint_rejection_verified": checkpoint_rejection is not None,
            "unified_prefix_source_trace_passed": False,
            "disabled_path_large_model_parity_passed": False,
            "per_parameter_gradient_assertions_passed": False,
            "runtime_carries_action_prefix": False,
            "runtime_carries_prefix_length": False,
            "inference_time_vjp_rtc_disabled": False,
        },
    }
    return manifest, {
        "phase1_report": str(PHASE1_REPORT),
        "phase1_hardware_access": phase1.get("hardware_access"),
        "phase1_network_access": phase1.get("network_access"),
        "phase1_predicted_delay_steps": predicted_delay,
        "phase3_report": str(PHASE3_REPORT),
        "phase3_mode": phase3.get("mode"),
        "phase3_checkpoint_step": phase3.get("checkpoint_step"),
        "phase3_complete_step": phase3.get("complete_step"),
        "phase3_supported_modes": backend_health.get("supported_modes"),
        "phase3_rtc_supported": backend_health.get("rtc_supported"),
        "current_checkpoint_rejection": checkpoint_rejection,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def _write_report(path: Path, report: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(OPTIMIZED_ROOT.resolve()):
        raise ValueError(f"output-json must stay inside {OPTIMIZED_ROOT}, got {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gate(args)
    output_path = _write_report(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
