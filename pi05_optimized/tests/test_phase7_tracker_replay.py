from argparse import Namespace

from tk_infer.pi05_optimized.tools.offline_phase7_tracker_replay import run_replay


def test_phase7_tracker_replay_is_deterministic_and_mpc_stays_blocked() -> None:
    report = run_replay(
        Namespace(duration_s=1.0, rate_hz=20.0, max_joint_step_rad=0.02)
    )

    assert report["status"] == "PASS"
    assert report["hardware_access"] is False
    assert report["network_access"] is False
    assert report["deterministic"] is True
    assert report["tracker_replay_passed"] is True
    assert report["first_run"]["max_output_joint_step_rad"] <= 0.0200001
    assert report["first_run"]["contact_used_as_safety"] is False
    assert report["mpc"]["evaluated"] is False
    assert report["mpc"]["status"] == "BLOCKED"
    assert report["mpc"]["silent_fallback_allowed"] is False
