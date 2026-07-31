from __future__ import annotations

import os
import subprocess
from pathlib import Path

OPTIMIZED_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = OPTIMIZED_ROOT / "profiles/rtc_conditioned_010600"
LAUNCHER = PROFILE_ROOT / "run_policy_server.sh"
TRAINING_TASK = "Put the bottle on the right into the basket on the left."


def _environment(**updates: str) -> dict[str, str]:
    excluded = {
        "CONFIG_ONLY",
        "PRINT_COMMAND_ONLY",
        "PI05_OPT_BACKEND",
        "PI05_OPT_POLICY_PATH",
        "PI05_OPT_REQUIRE_COMPLETE_STEP",
        "PI05_OPT_RTC_CONDITIONED_TASK",
        "JZ_PI05_OPT_SERVER_HOST",
        "JZ_PI05_OPT_SERVER_PORT",
    }
    env = {key: value for key, value in os.environ.items() if key not in excluded}
    env.update(updates)
    return env


def test_profile_prints_locked_independent_backend_without_loading_model() -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=OPTIMIZED_ROOT.parents[1],
        env=_environment(PRINT_COMMAND_ONLY="true"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "rtc_conditioned_step_010600_three_camera" in result.stdout
    assert "expert_b_rtc_e10_seed1000_010600/pretrained_model" in result.stdout
    assert "backend=torch_rtc_conditioned" in result.stdout
    assert "--backend=torch_rtc_conditioned" in result.stdout
    assert f"task='{TRAINING_TASK}'" in result.stdout
    assert "--rtc-conditioned-task=Put\\ the\\ bottle\\ on\\ the\\ right" in result.stdout
    assert "--port=18089" in result.stdout
    assert "PRINT_COMMAND_ONLY=true; nothing was started" in result.stdout


def test_profile_rejects_backend_and_policy_overrides() -> None:
    backend = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=OPTIMIZED_ROOT.parents[1],
        env=_environment(PRINT_COMMAND_ONLY="true", PI05_OPT_BACKEND="torch"),
        check=False,
        capture_output=True,
        text=True,
    )
    policy = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=OPTIMIZED_ROOT.parents[1],
        env=_environment(PRINT_COMMAND_ONLY="true", PI05_OPT_POLICY_PATH="/tmp/wrong"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert backend.returncode == 2
    assert "forbids PI05_OPT_BACKEND" in backend.stderr
    assert policy.returncode == 2
    assert "forbids PI05_OPT_POLICY_PATH" in policy.stderr


def test_profile_verifier_pins_rtc_contract_and_all_file_hashes() -> None:
    source = (PROFILE_ROOT / "verify_checkpoint.py").read_text(encoding="utf-8")

    assert '"configured_steps": 10600' in source
    assert '"camera_profile": "three_camera"' in source
    assert "039ef411871f75e8504b7b72ccb299c29c4cdf3a99e7bfbc241a3daae7bfaa57" in source
    assert "a532c9cfbb56a6feb1b9da8ec5d40bcb17d5aec0a9f14e29923ff0bf2aa7021f" in source
    assert "ef6773c135b77b834de1d13c75a4c98ab7a3684ffd602d1831e1f1bf5467c563" in source
    assert "EXPECTED_TOKENIZER_HASHES" in source
    assert "inspect_rtc_conditioned_checkpoint" in source
    assert "require_complete_step=False" in source
