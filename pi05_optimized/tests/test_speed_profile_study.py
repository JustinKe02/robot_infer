from __future__ import annotations

from argparse import Namespace

import pytest

from tk_infer.pi05_optimized.runtime.speed_profile_study import (
    FIXED_SPEED_PROFILES,
    LabeledSpeedTrial,
    evaluate_labeled_speed_trials,
)
from tk_infer.pi05_optimized.tools.offline_phase9_speed_profiles import run_study


def _trial(profile: float, index: int) -> LabeledSpeedTrial:
    return LabeledSpeedTrial(
        profile=profile,
        trial_id=f"{profile}-{index}",
        task_success=index % 3 != 0,
        cycle_time_s=10.0 / profile + index * 0.01,
        label_source="approved_fixed_profile_protocol",
        checkpoint_fingerprint="checkpoint",
        task_id="task",
    )


def test_labeled_trial_protocol_requires_all_fixed_profiles() -> None:
    blocked = evaluate_labeled_speed_trials([], min_trials_per_profile=2)
    assert blocked.status == "BLOCKED"
    assert len(blocked.blocking_reasons) == 3

    trials = [_trial(profile, index) for profile in FIXED_SPEED_PROFILES for index in range(3)]
    result = evaluate_labeled_speed_trials(trials, min_trials_per_profile=3)
    assert result.status == "PASS"
    assert [curve.profile for curve in result.curves] == list(FIXED_SPEED_PROFILES)
    assert all(curve.trial_count == 3 for curve in result.curves)
    assert all(0.0 <= curve.success_wilson95_low <= curve.task_success_rate for curve in result.curves)
    assert result.learned_adaptation_enabled is False


def test_trial_protocol_rejects_non_fixed_profile() -> None:
    with pytest.raises(ValueError, match="profile must be one of"):
        _trial(2.0, 0)


def test_phase9_offline_scheduler_profiles_do_not_claim_task_success() -> None:
    report = run_study(
        Namespace(
            source="synthetic",
            control_hz=20.0,
            max_joint_step_rad=0.02,
            solver_timeout_s=0.05,
            outcomes_json=None,
            min_trials_per_profile=2,
        )
    )

    assert report["status"] == "PASS"
    assert report["fixed_profiles"] == [1.0, 1.25, 1.5]
    assert report["learned_speed_adaptation_enabled"] is False
    assert report["hardware_access"] is False
    assert report["network_access"] is False
    assert report["task_curve_status"] == "BLOCKED"
    assert all(profile["task_success_rate"] is None for profile in report["scheduler_profiles"])
    assert report["robot_throttle_rollout"]["collected"] is False
    assert report["performance_context"]["may_be_described_as_20hz_realtime"] is False
