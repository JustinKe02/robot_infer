from __future__ import annotations

from argparse import Namespace

import pytest

from tk_infer.pi05_optimized.tools.offline_phase6_temporal_replay import run_replay


def _args(**overrides: object) -> Namespace:
    values = {
        "source": "synthetic",
        "speed_factor": 1.0,
        "max_joint_step_rad": 0.02,
        "solver_timeout_s": 0.05,
    }
    values.update(overrides)
    return Namespace(**values)


def test_synthetic_phase6_replay_passes_without_hardware_or_payload() -> None:
    report = run_replay(_args())

    assert report["status"] == "PASS"
    assert report["hardware_access"] is False
    assert report["network_access"] is False
    assert report["action_transport_created"] is False
    assert report["payload_retained_in_report"] is False
    assert report["single_interpolation_map_applied_to_pair"] is True
    assert report["force_slots_exact_80"] is True
    assert report["finite"] is True
    assert report["gate_failures"] == []
    assert report["temporal_report"]["limited_output_steps"] > 0
    assert report["temporal_report"]["output_max_joint_step_rad"] <= 0.0200001
    assert report["processor_config"]["acceleration_objective_enabled"] is False
    assert report["initial_state_guard_evaluated"] is False


def test_phase6_replay_rejects_invalid_config_before_processing() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        run_replay(_args(max_joint_step_rad=0.0))
