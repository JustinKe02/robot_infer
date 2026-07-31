#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
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
LIVE_OUTPUT_ROOT = OPTIMIZED_ROOT / "outputs/live"
DEFAULT_OUTPUT_PATH = LIVE_OUTPUT_ROOT / "live_backend_ab_step015900_20260730.json"
DEFAULT_REPORTS = {
    "reference": LIVE_OUTPUT_ROOT / "live_reference_step015900_20260730.json",
    "torch_optimized_plain": LIVE_OUTPUT_ROOT / "live_torch_optimized_plain_step015900_20260730.json",
    "inference_mode": LIVE_OUTPUT_ROOT / "live_inference_mode_step015900_20260730.json",
    "inference_mode_bf16": LIVE_OUTPUT_ROOT / "live_inference_mode_bf16_step015900_20260730.json",
    "inference_mode_pinned": LIVE_OUTPUT_ROOT / "live_inference_mode_pinned_step015900_20260730.json",
    "inference_mode_pinned_nonblocking": (
        LIVE_OUTPUT_ROOT / "live_inference_mode_pinned_nonblocking_step015900_20260730.json"
    ),
}
_LABEL_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class PerformanceThresholds:
    request_p95_s: float = 0.050
    request_p99_s: float = 0.100
    reference_p95_improvement_fraction: float = 0.20
    predicted_delay_p99_steps: float = 2.0
    stale_rate: float = 0.001
    repeated_frame_rate: float = 0.001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize completed PI05 live read-only backend A/B reports without hardware access."
    )
    parser.add_argument(
        "--report",
        action="append",
        metavar="LABEL=PATH",
        help="Report input; repeat to replace the fixed step-015900 default set.",
    )
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--request-p95-s", type=float, default=0.050)
    parser.add_argument("--request-p99-s", type=float, default=0.100)
    parser.add_argument("--reference-p95-improvement-fraction", type=float, default=0.20)
    parser.add_argument("--predicted-delay-p99-steps", type=float, default=2.0)
    parser.add_argument("--stale-rate", type=float, default=0.001)
    parser.add_argument("--repeated-frame-rate", type=float, default=0.001)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def _parse_report_specs(values: list[str] | None) -> dict[str, Path]:
    if values is None:
        return dict(DEFAULT_REPORTS)
    reports: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not _LABEL_PATTERN.fullmatch(label) or not raw_path.strip():
            raise ValueError(f"report must use LABEL=PATH with a snake_case label, got {value!r}")
        if label in reports:
            raise ValueError(f"duplicate report label: {label}")
        reports[label] = Path(raw_path).expanduser()
    if len(reports) < 2:
        raise ValueError("at least two reports are required")
    return reports


def _thresholds_from_args(args: argparse.Namespace) -> PerformanceThresholds:
    thresholds = PerformanceThresholds(
        request_p95_s=args.request_p95_s,
        request_p99_s=args.request_p99_s,
        reference_p95_improvement_fraction=args.reference_p95_improvement_fraction,
        predicted_delay_p99_steps=args.predicted_delay_p99_steps,
        stale_rate=args.stale_rate,
        repeated_frame_rate=args.repeated_frame_rate,
    )
    for name, value in asdict(thresholds).items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    return thresholds


def _read_json_object(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {resolved}")
    return value


def _at(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"missing report field: {path}")
        current = current[key]
    return current


def _number(value: Mapping[str, Any], path: str) -> float:
    result = _at(value, path)
    if isinstance(result, bool) or not isinstance(result, (int, float)) or not math.isfinite(float(result)):
        raise ValueError(f"report field must be a finite number: {path}")
    return float(result)


def _integer(value: Mapping[str, Any], path: str) -> int:
    result = _at(value, path)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"report field must be an integer: {path}")
    return result


def _distribution(value: Mapping[str, Any], path: str) -> dict[str, float | int]:
    return {
        "count": _integer(value, f"{path}.count"),
        "p50": _number(value, f"{path}.p50"),
        "p95": _number(value, f"{path}.p95"),
        "p99": _number(value, f"{path}.p99"),
    }


def _summarize_report(label: str, path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    warmup_requests = _integer(report, "warmup_requests")
    measure_requests = _integer(report, "measure_requests")
    expected_total = warmup_requests + measure_requests
    telemetry_path = "measured.client_telemetry"
    output_path = "measured.local_policy_output_records"
    failures: list[str] = []

    def require_equal(field: str, expected: Any) -> None:
        actual = _at(report, field)
        if actual != expected:
            failures.append(f"{field} expected {expected!r}, got {actual!r}")

    require_equal("status", "PASS")
    require_equal("errors", [])
    require_equal("mode", "single_step")
    require_equal("server_count.delta", expected_total)
    require_equal(f"{telemetry_path}.request_count", measure_requests)
    require_equal(f"{telemetry_path}.frame_count", measure_requests)
    require_equal(f"{output_path}.count", measure_requests)
    require_equal(f"{output_path}.finite", True)
    require_equal(f"{output_path}.force_slots_exact", True)
    require_equal(f"{output_path}.shapes", [18])
    require_equal("source_diagnostics.read_count", expected_total)
    require_equal("source_diagnostics.receiver_stopped", True)
    require_equal("safety.action_sent", False)
    require_equal("safety.armed_capability", False)
    require_equal("safety.command_transport_created", False)
    require_equal("safety.live_sensor_access", True)
    require_equal("safety.robot_created", False)
    require_equal("safety.state_receiver_stopped", True)
    require_equal("checkpoint.optimized_failure_count", 0)
    require_equal("checkpoint.trace.dropped_events", 0)
    for field in (
        "queue_empty_events",
        "repeated_source_frames",
        "skipped_source_frames",
        "stale_chunks",
    ):
        require_equal(f"{telemetry_path}.{field}", 0)
    for camera in ("camera_head", "camera_right"):
        for field in (
            "invalid_messages",
            "duplicate_or_out_of_order_messages",
            "raw_queue_drops",
            "sequence_gaps",
        ):
            require_equal(f"source_diagnostics.cameras.{camera}.{field}", 0)

    distributions = _at(report, f"{telemetry_path}.distributions")
    if not isinstance(distributions, Mapping):
        raise ValueError(f"{label}: telemetry distributions must be an object")
    request_latency = _distribution(distributions, "request_total_s")
    model_latency = _distribution(distributions, "model_reported_s")
    dropped_steps = _distribution(distributions, "dropped_steps")
    predicted_delay_steps = _distribution(distributions, "predicted_delay_steps")
    cycle_active = _distribution(report, "measured.cycle_active_s")
    for metric_name, distribution in (
        ("request_total_s", request_latency),
        ("model_reported_s", model_latency),
        ("dropped_steps", dropped_steps),
        ("predicted_delay_steps", predicted_delay_steps),
        ("cycle_active_s", cycle_active),
    ):
        if distribution["count"] != measure_requests:
            failures.append(f"{metric_name}.count expected {measure_requests}, got {distribution['count']}")

    skew = {
        camera: {
            **_distribution(report, f"source_diagnostics.camera_state_receive_skew_ms.{camera}"),
            "max": _number(report, f"source_diagnostics.camera_state_receive_skew_ms.{camera}.max"),
        }
        for camera in ("camera_head", "camera_right")
    }
    stale_chunks = _integer(report, f"{telemetry_path}.stale_chunks")
    repeated_frames = _integer(report, f"{telemetry_path}.repeated_source_frames")
    return {
        "label": label,
        "source_report": str(path.resolve()),
        "backend": _at(report, "checkpoint.backend"),
        "trajectory_processor": _at(report, "checkpoint.trajectory_processor"),
        "checkpoint_fingerprint": _at(report, "checkpoint.checkpoint_fingerprint"),
        "checkpoint_step": _integer(report, "checkpoint.checkpoint_step"),
        "camera_profile": _at(report, "camera_profile"),
        "control_hz": _number(report, "control_hz"),
        "warmup_requests": warmup_requests,
        "measure_requests": measure_requests,
        "request_total_s": request_latency,
        "model_reported_s": model_latency,
        "cycle_active_s": cycle_active,
        "dropped_steps": dropped_steps,
        "predicted_delay_steps": predicted_delay_steps,
        "camera_state_receive_skew_ms": skew,
        "stale_rate": stale_chunks / measure_requests,
        "repeated_frame_rate": repeated_frames / measure_requests,
        "integrity_gate": {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
            "action_sent": _at(report, "safety.action_sent"),
            "raw18_finite": _at(report, f"{output_path}.finite"),
            "force_slots_exact": _at(report, f"{output_path}.force_slots_exact"),
            "server_count_delta": _integer(report, "server_count.delta"),
            "trace_dropped_events": _integer(report, "checkpoint.trace.dropped_events"),
            "optimized_failure_count": _integer(report, "checkpoint.optimized_failure_count"),
            "queue_empty_events": _integer(report, f"{telemetry_path}.queue_empty_events"),
            "repeated_source_frames": repeated_frames,
            "stale_chunks": stale_chunks,
        },
    }


def _comparison_failures(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    failures = []
    for field in (
        "checkpoint_fingerprint",
        "checkpoint_step",
        "camera_profile",
        "control_hz",
        "warmup_requests",
        "measure_requests",
    ):
        if candidate[field] != reference[field]:
            failures.append(f"{field} differs from reference")
    return failures


def _improvement(reference: float, candidate: float) -> float:
    if reference <= 0:
        raise ValueError("reference latency must be positive")
    return (reference - candidate) / reference


def build_summary(
    report_paths: Mapping[str, Path],
    *,
    reference_label: str = "reference",
    thresholds: PerformanceThresholds = PerformanceThresholds(),
) -> dict[str, Any]:
    if reference_label not in report_paths:
        raise ValueError(f"reference label {reference_label!r} is not present")
    reports = {
        label: _summarize_report(label, path, _read_json_object(path)) for label, path in report_paths.items()
    }
    reference = reports[reference_label]
    comparisons: dict[str, Any] = {}
    integrity_failures: list[str] = []
    qualified: list[str] = []
    integrity_candidates: list[str] = []

    for label, summary in reports.items():
        if summary["integrity_gate"]["status"] != "PASS":
            integrity_failures.extend(
                f"{label}: {failure}" for failure in summary["integrity_gate"]["failures"]
            )
        if label == reference_label:
            continue
        comparability_failures = _comparison_failures(reference, summary)
        integrity_failures.extend(f"{label}: {failure}" for failure in comparability_failures)
        request_improvement = {
            percentile: _improvement(
                float(reference["request_total_s"][percentile]),
                float(summary["request_total_s"][percentile]),
            )
            for percentile in ("p50", "p95", "p99")
        }
        model_improvement = {
            percentile: _improvement(
                float(reference["model_reported_s"][percentile]),
                float(summary["model_reported_s"][percentile]),
            )
            for percentile in ("p50", "p95", "p99")
        }
        gate_checks = {
            "integrity_and_comparability": (
                summary["integrity_gate"]["status"] == "PASS" and not comparability_failures
            ),
            "request_p95": summary["request_total_s"]["p95"] <= thresholds.request_p95_s,
            "request_p99": summary["request_total_s"]["p99"] <= thresholds.request_p99_s,
            "reference_p95_improvement": (
                request_improvement["p95"] >= thresholds.reference_p95_improvement_fraction
            ),
            "predicted_delay_p99": (
                summary["predicted_delay_steps"]["p99"] <= thresholds.predicted_delay_p99_steps
            ),
            "stale_rate": summary["stale_rate"] < thresholds.stale_rate,
            "repeated_frame_rate": summary["repeated_frame_rate"] < thresholds.repeated_frame_rate,
        }
        gate_failures = [name for name, passed in gate_checks.items() if not passed]
        comparisons[label] = {
            "request_improvement_fraction": request_improvement,
            "model_improvement_fraction": model_improvement,
            "performance_gate": {
                "status": "PASS" if not gate_failures else "FAIL",
                "checks": gate_checks,
                "failures": gate_failures,
            },
        }
        if gate_checks["integrity_and_comparability"]:
            integrity_candidates.append(label)
        if not gate_failures:
            qualified.append(label)

    def fastest(labels: list[str]) -> str | None:
        if not labels:
            return None
        return min(labels, key=lambda label: reports[label]["request_total_s"]["p95"])

    best_observed = fastest(integrity_candidates)
    selected = fastest(qualified)
    best_observed_improves_reference = bool(
        best_observed
        and reports[best_observed]["request_total_s"]["p95"] < reference["request_total_s"]["p95"]
    )
    readonly_shadow_candidate = best_observed if best_observed_improves_reference else None
    return {
        "schema_version": 1,
        "phase": "P4.5_live_read_only_backend_ab",
        "status": "PASS" if not integrity_failures else "FAIL",
        "hardware_access": False,
        "network_access": False,
        "source_reports_used_live_sensor_access": True,
        "action_sent": any(bool(report["integrity_gate"]["action_sent"]) for report in reports.values()),
        "reference_label": reference_label,
        "thresholds": asdict(thresholds),
        "reports": reports,
        "comparisons": comparisons,
        "integrity_failures": integrity_failures,
        "selection": {
            "best_observed_candidate": best_observed,
            "best_observed_improves_reference": best_observed_improves_reference,
            "best_observed_is_production_qualified": best_observed in qualified,
            "production_qualified_candidates": qualified,
            "selected_production_candidate": selected,
            "selected_readonly_shadow_candidate": readonly_shadow_candidate,
            "performance_gate_status": "PASS" if qualified else "FAIL",
            "note": (
                "The read-only shadow candidate must be the fastest comparable safe observation and improve "
                "reference p95; it is not a production selection unless it also appears in "
                "production_qualified_candidates."
            ),
        },
    }


def _write_report(path: Path, report: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(OPTIMIZED_ROOT.resolve()):
        raise ValueError(f"output-json must stay inside {OPTIMIZED_ROOT}, got {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = _parse_report_specs(args.report)
    thresholds = _thresholds_from_args(args)
    report = build_summary(reports, reference_label=args.reference_label, thresholds=thresholds)
    output_path = _write_report(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
