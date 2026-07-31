from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PI05_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = PI05_ROOT / "profiles/rtc_conditioned_010600"
TRAINING_TASK = "Put the bottle on the right into the basket on the left."


def _environment(**updates: str) -> dict[str, str]:
    excluded = {
        "CAMERA_PROFILE",
        "I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED",
        "I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT",
        "JZ_PI05_DISABLE_JOINT_DELTA_CHECKS",
        "JZ_PI05_RTC_CONDITIONED_CONTINUOUS_ARMED_CONFIRMED",
        "JZ_PI05_RTC_CONDITIONED_CONTINUOUS_DRY_RUN_PASSED",
        "JZ_PI05_RTC_CONDITIONED_JOINT_DELTA_BYPASS_CONFIRMED",
        "JZ_PI05_RTC_CONDITIONED_SINGLE_STEP_ARMED_PASSED",
        "JZ_POLICY_INFERENCE_ARMED",
        "JZ_ROBOT_PIN_ARMED",
        "MAX_CAMERA_STATE_RECEIVE_SKEW_MS",
        "ORIN_IP",
        "PROFILE_MAX_SENT_ACTIONS",
        "PROFILE_RTC_RUN_MODE",
        "PROFILE_RUN_TIME_S",
        "SERVER_URL",
        "STATE_BIND_IP",
        "STATE_PORT",
        "TASK",
        "TK_PI05_RTC_CONDITIONED_010600_CONFIRMED",
        "COMMAND_PORT",
    }
    env = {key: value for key, value in os.environ.items() if key not in excluded}
    env.update(
        {
            "CONDA_PYTHON": sys.executable,
            "PRINT_COMMAND_ONLY": "true",
            "TK_PI05_RTC_CONDITIONED_010600_CONFIRMED": "1",
            "RUN_STAMP": "pytest_rtc_conditioned_010600",
        }
    )
    env.update(updates)
    return env


def _run(script: str, **updates: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(PROFILE_ROOT / script)],
        cwd=PI05_ROOT.parents[1],
        env=_environment(**updates),
        check=False,
        capture_output=True,
        text=True,
    )


def test_profile_locks_path_unproven_checkpoint_and_live_smoke() -> None:
    result = _run("run_inference_smoke.sh")
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "--server-url=http://127.0.0.1:18089" in output
    assert "--mode=single_step" in output
    assert "--execution=dry_run" in output
    assert "--camera-profile=three_camera" in output
    assert "--task=Put\\ the\\ bottle\\ on\\ the\\ right\\ into\\ the\\ basket\\ on\\ the\\ left." in output
    assert "--inference-smoke=true" in output
    common = (PROFILE_ROOT / "common.sh").read_text(encoding="utf-8")
    assert "JZ_PI05_EXPECTED_CHECKPOINT_STEP=null" in common
    assert "JZ_PI05_EXPECTED_COMPLETE_STEP=null" in common
    assert "039ef411871f75e8504b7b72ccb299c29c4cdf3a99e7bfbc241a3daae7bfaa57" in common


def test_profile_accepts_only_the_training_task() -> None:
    accepted = _run("run_inference_smoke.sh", TASK=TRAINING_TASK)
    rejected = _run("run_inference_smoke.sh", TASK="jz robot pin timed vr teleoperation")

    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert rejected.returncode == 2
    assert "forbids TASK='jz robot pin timed vr teleoperation'" in rejected.stderr


def test_single_step_armed_is_one_action_and_requires_global_gates() -> None:
    rejected = _run("run_single_step_armed.sh")
    assert rejected.returncode == 2
    assert "JZ_ROBOT_PIN_ARMED=1" in rejected.stderr

    result = _run(
        "run_single_step_armed.sh",
        JZ_ROBOT_PIN_ARMED="1",
        I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT="1",
        JZ_POLICY_INFERENCE_ARMED="1",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "--mode=single_step" in output
    assert "--execution=armed" in output
    assert "--run-time-s=1" in output
    assert "--max-sent-actions=1" in output
    assert "joint_delta_checks=enabled" in output


def test_rtc_armed_requires_prior_single_step_pass() -> None:
    armed = {
        "JZ_ROBOT_PIN_ARMED": "1",
        "I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT": "1",
        "JZ_POLICY_INFERENCE_ARMED": "1",
    }
    rejected = _run("run_rtc_armed.sh", **armed)
    assert rejected.returncode == 2
    assert "JZ_PI05_RTC_CONDITIONED_SINGLE_STEP_ARMED_PASSED=1" in rejected.stderr

    result = _run(
        "run_rtc_armed.sh",
        **armed,
        JZ_PI05_RTC_CONDITIONED_SINGLE_STEP_ARMED_PASSED="1",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "--mode=rtc" in output
    assert "--execution=armed" in output
    assert "--run-time-s=1" in output
    assert "--max-sent-actions=10" in output


def test_profile_forbids_joint_delta_bypass() -> None:
    disabled = _run(
        "run_single_step_dry_run.sh",
        JZ_PI05_DISABLE_JOINT_DELTA_CHECKS="1",
    )
    acknowledged = _run(
        "run_single_step_dry_run.sh",
        I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED="1",
    )

    assert disabled.returncode == 2
    assert "forbids JZ_PI05_DISABLE_JOINT_DELTA_CHECKS=1 outside the bypass launcher" in disabled.stderr
    assert acknowledged.returncode == 2
    assert "forbids I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED=1 outside the bypass launcher" in (
        acknowledged.stderr
    )


def test_explicit_joint_delta_bypass_is_single_step_and_one_action_only() -> None:
    missing_ack = _run("run_single_step_dry_run_joint_delta_bypass.sh")
    assert missing_ack.returncode == 2
    assert "JZ_PI05_RTC_CONDITIONED_JOINT_DELTA_BYPASS_CONFIRMED=1" in missing_ack.stderr

    dry_run = _run(
        "run_single_step_dry_run_joint_delta_bypass.sh",
        JZ_PI05_RTC_CONDITIONED_JOINT_DELTA_BYPASS_CONFIRMED="1",
    )
    dry_output = dry_run.stdout + dry_run.stderr
    assert dry_run.returncode == 0, dry_output
    assert "--mode=single_step" in dry_output
    assert "--execution=dry_run" in dry_output
    assert "--run-time-s=1" in dry_output
    assert "--max-sent-actions=1" in dry_output
    assert "joint_delta_checks=bypass" in dry_output

    armed = _run(
        "run_single_step_armed_joint_delta_bypass.sh",
        JZ_PI05_RTC_CONDITIONED_JOINT_DELTA_BYPASS_CONFIRMED="1",
        JZ_ROBOT_PIN_ARMED="1",
        I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT="1",
        JZ_POLICY_INFERENCE_ARMED="1",
    )
    armed_output = armed.stdout + armed.stderr
    assert armed.returncode == 0, armed_output
    assert "--mode=single_step" in armed_output
    assert "--execution=armed" in armed_output
    assert "--run-time-s=1" in armed_output
    assert "--max-sent-actions=1" in armed_output
    assert "joint_delta_checks=bypass" in armed_output

    rtc_without_launcher = _run(
        "run_rtc_armed.sh",
        JZ_PI05_RTC_CONDITIONED_JOINT_DELTA_BYPASS_CONFIRMED="1",
        JZ_PI05_RTC_CONDITIONED_SINGLE_STEP_ARMED_PASSED="1",
        JZ_ROBOT_PIN_ARMED="1",
        I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT="1",
        JZ_POLICY_INFERENCE_ARMED="1",
    )
    assert rtc_without_launcher.returncode == 2
    assert "outside the bypass launcher" in rtc_without_launcher.stderr


def test_explicit_rtc_joint_delta_bypass_is_bounded_and_requires_prior_single_step() -> None:
    bypass = {
        "JZ_PI05_RTC_CONDITIONED_JOINT_DELTA_BYPASS_CONFIRMED": "1",
    }
    dry_run = _run(
        "run_rtc_dry_run_joint_delta_bypass.sh",
        **bypass,
    )
    dry_output = dry_run.stdout + dry_run.stderr
    assert dry_run.returncode == 0, dry_output
    assert "--mode=rtc" in dry_output
    assert "--execution=dry_run" in dry_output
    assert "--sensor-fps=20" in dry_output
    assert "--control-fps=20" in dry_output
    assert "--run-time-s=1" in dry_output
    assert "--max-sent-actions=10" in dry_output
    assert "joint_delta_checks=bypass" in dry_output

    armed = {
        **bypass,
        "JZ_ROBOT_PIN_ARMED": "1",
        "I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT": "1",
        "JZ_POLICY_INFERENCE_ARMED": "1",
    }
    missing_single_step = _run(
        "run_rtc_armed_joint_delta_bypass.sh",
        **armed,
    )
    assert missing_single_step.returncode == 2
    assert "JZ_PI05_RTC_CONDITIONED_SINGLE_STEP_ARMED_PASSED=1" in missing_single_step.stderr

    armed_run = _run(
        "run_rtc_armed_joint_delta_bypass.sh",
        **armed,
        JZ_PI05_RTC_CONDITIONED_SINGLE_STEP_ARMED_PASSED="1",
    )
    armed_output = armed_run.stdout + armed_run.stderr
    assert armed_run.returncode == 0, armed_output
    assert "--mode=rtc" in armed_output
    assert "--execution=armed" in armed_output
    assert "--sensor-fps=20" in armed_output
    assert "--control-fps=20" in armed_output
    assert "--run-time-s=1" in armed_output
    assert "--max-sent-actions=10" in armed_output
    assert "joint_delta_checks=bypass" in armed_output


def test_continuous_rtc_requires_qualification_and_two_extra_confirmations() -> None:
    bypass = {
        "JZ_PI05_RTC_CONDITIONED_JOINT_DELTA_BYPASS_CONFIRMED": "1",
    }
    qualification = _run(
        "run_rtc_dry_run_joint_delta_bypass_qualification.sh",
        **bypass,
    )
    qualification_output = qualification.stdout + qualification.stderr
    assert qualification.returncode == 0, qualification_output
    assert "--mode=rtc" in qualification_output
    assert "--execution=dry_run" in qualification_output
    assert "--sensor-fps=20" in qualification_output
    assert "--control-fps=20" in qualification_output
    assert "--run-time-s=5" in qualification_output
    assert "--max-sent-actions=0" in qualification_output
    assert "joint_delta_checks=bypass rtc_run_mode=qualification" in qualification_output

    armed = {
        **bypass,
        "JZ_PI05_RTC_CONDITIONED_SINGLE_STEP_ARMED_PASSED": "1",
        "JZ_ROBOT_PIN_ARMED": "1",
        "I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT": "1",
        "JZ_POLICY_INFERENCE_ARMED": "1",
    }
    missing_qualification = _run(
        "run_rtc_armed_joint_delta_bypass_continuous.sh",
        **armed,
    )
    assert missing_qualification.returncode == 2
    assert "JZ_PI05_RTC_CONDITIONED_CONTINUOUS_DRY_RUN_PASSED=1" in (missing_qualification.stderr)

    missing_continuous_ack = _run(
        "run_rtc_armed_joint_delta_bypass_continuous.sh",
        **armed,
        JZ_PI05_RTC_CONDITIONED_CONTINUOUS_DRY_RUN_PASSED="1",
    )
    assert missing_continuous_ack.returncode == 2
    assert "JZ_PI05_RTC_CONDITIONED_CONTINUOUS_ARMED_CONFIRMED=1" in (missing_continuous_ack.stderr)

    continuous = _run(
        "run_rtc_armed_joint_delta_bypass_continuous.sh",
        **armed,
        JZ_PI05_RTC_CONDITIONED_CONTINUOUS_DRY_RUN_PASSED="1",
        JZ_PI05_RTC_CONDITIONED_CONTINUOUS_ARMED_CONFIRMED="1",
    )
    continuous_output = continuous.stdout + continuous.stderr
    assert continuous.returncode == 0, continuous_output
    assert "--mode=rtc" in continuous_output
    assert "--execution=armed" in continuous_output
    assert "--sensor-fps=20" in continuous_output
    assert "--control-fps=20" in continuous_output
    assert "--run-time-s=0" in continuous_output
    assert "--max-sent-actions=0" in continuous_output
    assert "joint_delta_checks=bypass rtc_run_mode=continuous" in continuous_output


def test_continuous_rtc_rejects_bounded_overrides() -> None:
    confirmations = {
        "JZ_PI05_RTC_CONDITIONED_JOINT_DELTA_BYPASS_CONFIRMED": "1",
        "JZ_PI05_RTC_CONDITIONED_SINGLE_STEP_ARMED_PASSED": "1",
        "JZ_PI05_RTC_CONDITIONED_CONTINUOUS_DRY_RUN_PASSED": "1",
        "JZ_PI05_RTC_CONDITIONED_CONTINUOUS_ARMED_CONFIRMED": "1",
        "JZ_ROBOT_PIN_ARMED": "1",
        "I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT": "1",
        "JZ_POLICY_INFERENCE_ARMED": "1",
    }
    bounded_actions = _run(
        "run_rtc_armed_joint_delta_bypass_continuous.sh",
        **confirmations,
        PROFILE_MAX_SENT_ACTIONS="10",
    )
    bounded_time = _run(
        "run_rtc_armed_joint_delta_bypass_continuous.sh",
        **confirmations,
        PROFILE_RUN_TIME_S="10",
    )

    assert bounded_actions.returncode == 2
    assert "PROFILE_MAX_SENT_ACTIONS unset or 0" in bounded_actions.stderr
    assert bounded_time.returncode == 2
    assert "PROFILE_RUN_TIME_S unset or 0" in bounded_time.stderr


def test_profile_locks_robot_network_endpoints() -> None:
    for name, value in (
        ("ORIN_IP", "192.168.1.99"),
        ("STATE_BIND_IP", "127.0.0.1"),
        ("STATE_PORT", "39011"),
        ("COMMAND_PORT", "39021"),
        ("MAX_CAMERA_STATE_RECEIVE_SKEW_MS", "500"),
    ):
        result = _run("run_single_step_dry_run.sh", **{name: value})
        assert result.returncode == 2
        assert f"forbids {name}={value}" in result.stderr
