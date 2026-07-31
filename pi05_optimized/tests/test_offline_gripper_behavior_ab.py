from __future__ import annotations

import numpy as np
import pytest

from tk_infer.pi05_optimized.tools.offline_gripper_behavior_ab import (
    _independent_chunk_diagnostics,
    _parse_named_values,
    _parse_seeds,
    _validate_thresholds,
    analyze_right_gripper,
)


def _processed_actions(right_gripper: list[float]) -> np.ndarray:
    actions = np.zeros((len(right_gripper), 18), dtype=np.float32)
    actions[:, 15] = 80.0
    actions[:, 16] = right_gripper
    actions[:, 17] = 80.0
    return actions


def test_gripper_analysis_uses_hysteresis_and_counts_reclose() -> None:
    report = analyze_right_gripper(
        _processed_actions([0.0, 30.0, 60.0, 100.0, 52.0, 40.0, 0.0, 90.0]),
        open_threshold=45.0,
        closed_threshold=55.0,
    )

    assert report["state_sequence"] == ["open", "closed", "open", "closed"]
    assert report["open_to_closed_transitions"] == 2
    assert report["reopen_after_close_transitions"] == 1
    assert report["closure_entries"] == 2
    assert report["reclose_entries"] == 1
    assert report["force_slots_exact_80"] is True


def test_gripper_analysis_reports_unresolved_band_and_force_failure() -> None:
    actions = _processed_actions([49.0, 50.0, 51.0])
    actions[1, 17] = 79.0

    report = analyze_right_gripper(actions, open_threshold=45.0, closed_threshold=55.0)

    assert report["state_sequence"] == []
    assert report["initial_resolved_state"] == "unresolved"
    assert report["final_resolved_state"] == "unresolved"
    assert report["force_slots_exact_80"] is False


def test_gripper_analysis_preserves_and_reports_execution_range_violations() -> None:
    report = analyze_right_gripper(
        _processed_actions([-2.0, 102.5]),
        open_threshold=45.0,
        closed_threshold=55.0,
    )

    assert report["state_sequence"] == ["open", "closed"]
    assert report["execution_range_valid"] is False
    assert report["below_range_steps"] == [0]
    assert report["above_range_steps"] == [1]
    assert report["maximum_range_violation"] == pytest.approx(2.5)


def test_independent_chunk_diagnostics_counts_only_resolved_boundary_flips() -> None:
    cases = []
    for seed, values in ((1, [0.0, 100.0]), (2, [0.0, 0.0]), (3, [50.0, 50.0])):
        stats = analyze_right_gripper(
            _processed_actions(values),
            open_threshold=45.0,
            closed_threshold=55.0,
        )
        cases.append(
            {
                "checkpoint": "step",
                "prompt": "task",
                "seed": seed,
                "backends": {"reference": stats},
            }
        )

    report = _independent_chunk_diagnostics(cases)

    assert report[0]["ordered_seeds"] == [1, 2, 3]
    assert report[0]["boundary_flip_count"] == 1
    assert report[0]["boundary_flips"][0]["from"] == "closed"
    assert report[0]["boundary_flips"][0]["to"] == "open"


def test_named_values_seeds_and_threshold_validation() -> None:
    assert _parse_named_values(["plate=Pick up the plate."], "prompt") == (
        ("plate", "Pick up the plate."),
    )
    assert _parse_seeds("3, 1,3") == (3, 1)
    _validate_thresholds(45.0, 55.0)

    with pytest.raises(ValueError, match="LABEL=VALUE"):
        _parse_named_values(["missing-separator"], "prompt")
    with pytest.raises(ValueError, match="duplicate"):
        _parse_named_values(["a=one", "a=two"], "prompt")
    with pytest.raises(ValueError, match="comma-separated"):
        _parse_seeds("one")
    with pytest.raises(ValueError, match="0 <= open"):
        _validate_thresholds(55.0, 45.0)
