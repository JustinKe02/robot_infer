from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PI05_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = PI05_ROOT / "profiles" / "step_010600"
CONFIRM_ENV = "TK_PI05_010600_INTERMEDIATE_CONFIRMED"


def _env(**overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "POLICY_PATH",
        "JZ_PI05_SINGLE_STEP_ARMED_PASSED",
        "JZ_PI05_DISABLE_JOINT_DELTA_CHECKS",
    ):
        env.pop(name, None)
    env.update(
        {
            "CONDA_PYTHON": sys.executable,
            "PRINT_COMMAND_ONLY": "true",
            CONFIRM_ENV: "1",
            "RUN_STAMP": "pytest_step_010600",
        }
    )
    env.update(overrides)
    return env


def _run(script: str, *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(PROFILE_ROOT / script)],
        cwd=PI05_ROOT.parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_profile_server_locks_intermediate_checkpoint_without_action_transform() -> None:
    completed = _run("run_policy_server.sh", env=_env())
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output
    assert "checkpoints/010600/pretrained_model" in output
    assert "--require-complete-step=false" in output
    assert "--action-profile" not in output


def test_profile_requires_explicit_intermediate_confirmation() -> None:
    env = _env()
    env.pop(CONFIRM_ENV)
    completed = _run("run_policy_server.sh", env=env)

    assert completed.returncode == 2
    assert CONFIRM_ENV in completed.stderr


def test_profile_forbids_policy_path_override() -> None:
    completed = _run("run_policy_server.sh", env=_env(POLICY_PATH="/tmp/wrong/pretrained_model"))

    assert completed.returncode == 2
    assert "forbids POLICY_PATH override" in completed.stderr


def test_profile_single_step_dry_run_uses_only_head_and_right_cameras() -> None:
    completed = _run("run_single_step_dry_run.sh", env=_env())
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output
    assert "--mode=single_step" in output
    assert "--execution=dry_run" in output
    assert "--camera-profile=head_right" in output
    assert "--max-camera-state-receive-skew-ms=250" in output
    assert "--action-profile" not in output
    assert "action_boundary=full_raw18" in output
    assert "--sensor-fps=5" in output
    assert "--control-fps=5" in output


def test_profile_async_dry_run_has_independent_launcher() -> None:
    completed = _run("run_async_single_step_dry_run.sh", env=_env())
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output
    assert "--mode=async_single_step" in output
    assert "--execution=dry_run" in output
    assert "--sensor-fps=20" in output


def test_profile_async_armed_requires_single_step_passed() -> None:
    armed = {
        "JZ_ROBOT_PIN_ARMED": "1",
        "I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT": "1",
        "JZ_POLICY_INFERENCE_ARMED": "1",
    }
    completed = _run("run_async_single_step_armed.sh", env=_env(**armed))

    assert completed.returncode == 2
    assert "JZ_PI05_SINGLE_STEP_ARMED_PASSED=1" in completed.stderr

    passed = _run(
        "run_async_single_step_armed.sh",
        env=_env(**armed, JZ_PI05_SINGLE_STEP_ARMED_PASSED="1"),
    )
    output = passed.stdout + passed.stderr
    assert passed.returncode == 0, output
    assert "--mode=async_single_step" in output
    assert "--execution=armed" in output


def test_profile_single_step_armed_is_bounded_to_one_action() -> None:
    completed = _run(
        "run_single_step_armed.sh",
        env=_env(
            JZ_ROBOT_PIN_ARMED="1",
            I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT="1",
            JZ_POLICY_INFERENCE_ARMED="1",
        ),
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output
    assert "--max-sent-actions=1" in output
    assert "--run-time-s=1" in output


def test_profile_enforces_onboard_fps_range() -> None:
    completed = _run("run_single_step_dry_run.sh", env=_env(PROFILE_SENSOR_FPS="21"))

    assert completed.returncode == 2
    assert "integer in 1..20" in completed.stderr
