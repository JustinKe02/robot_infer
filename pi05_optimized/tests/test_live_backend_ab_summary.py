from __future__ import annotations

import json
from pathlib import Path

import pytest

from tk_infer.pi05_optimized.tools.live_backend_ab_summary import (
    PerformanceThresholds,
    _parse_report_specs,
    build_summary,
)


def _distribution(*, p50: float, p95: float, p99: float, count: int = 30) -> dict[str, float | int]:
    return {"count": count, "p50": p50, "p95": p95, "p99": p99}


def _live_report(*, request_p95: float, request_p99: float, unsafe: bool = False) -> dict[str, object]:
    return {
        "status": "PASS",
        "errors": [],
        "mode": "single_step",
        "camera_profile": "head_right",
        "control_hz": 5.0,
        "warmup_requests": 3,
        "measure_requests": 30,
        "server_count": {"before": 0, "after": 33, "delta": 33},
        "checkpoint": {
            "backend": "torch_optimized",
            "trajectory_processor": "pass_through",
            "checkpoint_fingerprint": "fingerprint",
            "checkpoint_step": 15900,
            "optimized_failure_count": 0,
            "trace": {"dropped_events": 0},
        },
        "safety": {
            "action_sent": unsafe,
            "armed_capability": False,
            "command_transport_created": False,
            "live_sensor_access": True,
            "robot_created": False,
            "state_receiver_stopped": True,
        },
        "measured": {
            "client_telemetry": {
                "frame_count": 30,
                "request_count": 30,
                "queue_empty_events": 0,
                "repeated_source_frames": 0,
                "skipped_source_frames": 0,
                "stale_chunks": 0,
                "distributions": {
                    "request_total_s": _distribution(p50=request_p95 * 0.9, p95=request_p95, p99=request_p99),
                    "model_reported_s": _distribution(
                        p50=request_p95 * 0.8,
                        p95=request_p95 * 0.9,
                        p99=request_p99 * 0.9,
                    ),
                    "dropped_steps": _distribution(p50=1.0, p95=1.0, p99=1.0),
                    "predicted_delay_steps": _distribution(p50=0.0, p95=0.0, p99=0.0),
                },
            },
            "cycle_active_s": _distribution(
                p50=request_p95 * 0.95,
                p95=request_p95 * 1.01,
                p99=request_p99 * 1.01,
            ),
            "local_policy_output_records": {
                "count": 30,
                "finite": True,
                "force_slots_exact": True,
                "shapes": [18],
            },
        },
        "source_diagnostics": {
            "read_count": 33,
            "receiver_stopped": True,
            "camera_state_receive_skew_ms": {
                camera: {**_distribution(p50=2.0, p95=4.0, p99=5.0, count=33), "max": 6.0}
                for camera in ("camera_head", "camera_right")
            },
            "cameras": {
                camera: {
                    "invalid_messages": 0,
                    "duplicate_or_out_of_order_messages": 0,
                    "raw_queue_drops": 0,
                    "sequence_gaps": 0,
                }
                for camera in ("camera_head", "camera_right")
            },
        },
    }


def _write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_summary_selects_fastest_qualified_candidate(tmp_path: Path) -> None:
    reports = {
        "reference": _write(tmp_path / "reference.json", _live_report(request_p95=0.100, request_p99=0.120)),
        "fast": _write(tmp_path / "fast.json", _live_report(request_p95=0.040, request_p99=0.080)),
        "faster": _write(tmp_path / "faster.json", _live_report(request_p95=0.035, request_p99=0.070)),
    }

    summary = build_summary(reports)

    assert summary["status"] == "PASS"
    assert summary["selection"]["best_observed_candidate"] == "faster"
    assert summary["selection"]["best_observed_improves_reference"] is True
    assert summary["selection"]["selected_production_candidate"] == "faster"
    assert summary["selection"]["best_observed_is_production_qualified"] is True
    assert summary["comparisons"]["faster"]["request_improvement_fraction"]["p95"] == pytest.approx(0.65)


def test_summary_keeps_safe_shadow_selection_separate_from_performance_gate(tmp_path: Path) -> None:
    reports = {
        "reference": _write(tmp_path / "reference.json", _live_report(request_p95=0.100, request_p99=0.120)),
        "candidate": _write(tmp_path / "candidate.json", _live_report(request_p95=0.080, request_p99=0.095)),
    }

    summary = build_summary(reports)

    assert summary["status"] == "PASS"
    assert summary["selection"]["selected_readonly_shadow_candidate"] == "candidate"
    assert summary["selection"]["selected_production_candidate"] is None
    assert summary["selection"]["performance_gate_status"] == "FAIL"
    assert summary["comparisons"]["candidate"]["performance_gate"]["failures"] == ["request_p95"]


def test_summary_fails_integrity_when_action_was_sent(tmp_path: Path) -> None:
    reports = {
        "reference": _write(tmp_path / "reference.json", _live_report(request_p95=0.100, request_p99=0.120)),
        "unsafe": _write(
            tmp_path / "unsafe.json",
            _live_report(request_p95=0.040, request_p99=0.080, unsafe=True),
        ),
    }

    summary = build_summary(reports)

    assert summary["status"] == "FAIL"
    assert summary["action_sent"] is True
    assert summary["selection"]["best_observed_candidate"] is None
    assert any("safety.action_sent" in failure for failure in summary["integrity_failures"])


def test_report_spec_parser_rejects_duplicates_and_invalid_labels() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _parse_report_specs(["reference=a.json", "reference=b.json"])
    with pytest.raises(ValueError, match="LABEL=PATH"):
        _parse_report_specs(["Bad-Label=a.json", "other=b.json"])
    with pytest.raises(ValueError, match="at least two"):
        _parse_report_specs(["reference=a.json"])


def test_custom_thresholds_can_disqualify_predicted_delay(tmp_path: Path) -> None:
    reference = _live_report(request_p95=0.100, request_p99=0.120)
    candidate = _live_report(request_p95=0.040, request_p99=0.080)
    candidate["measured"]["client_telemetry"]["distributions"]["predicted_delay_steps"] = _distribution(
        p50=0.0,
        p95=2.0,
        p99=3.0,
    )
    reports = {
        "reference": _write(tmp_path / "reference.json", reference),
        "candidate": _write(tmp_path / "candidate.json", candidate),
    }

    summary = build_summary(reports, thresholds=PerformanceThresholds())

    assert "predicted_delay_p99" in summary["comparisons"]["candidate"]["performance_gate"]["failures"]


def test_slower_candidate_is_not_selected_for_shadow(tmp_path: Path) -> None:
    reports = {
        "reference": _write(tmp_path / "reference.json", _live_report(request_p95=0.100, request_p99=0.120)),
        "slower": _write(tmp_path / "slower.json", _live_report(request_p95=0.110, request_p99=0.130)),
    }

    summary = build_summary(reports)

    assert summary["selection"]["best_observed_candidate"] == "slower"
    assert summary["selection"]["best_observed_improves_reference"] is False
    assert summary["selection"]["selected_readonly_shadow_candidate"] is None
