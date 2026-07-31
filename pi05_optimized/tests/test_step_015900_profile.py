from __future__ import annotations

import os
import subprocess
from pathlib import Path

OPTIMIZED_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = OPTIMIZED_ROOT / "profiles/step_015900"
LAUNCHER = PROFILE_ROOT / "run_policy_server.sh"


def _environment(**updates: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "CONFIG_ONLY",
            "PRINT_COMMAND_ONLY",
            "PI05_OPT_POLICY_PATH",
            "JZ_PI05_OPT_SERVER_HOST",
            "JZ_PI05_OPT_SERVER_PORT",
        }
    }
    env.update(updates)
    return env


def test_latest_profile_prints_locked_complete_checkpoint_without_loading_model() -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=OPTIMIZED_ROOT.parents[1],
        env=_environment(PRINT_COMMAND_ONLY="true"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "step_015900_epoch15_head_right_complete" in result.stdout
    assert "checkpoints/015900/pretrained_model" in result.stdout
    assert "step=15900/15900 complete=true" in result.stdout
    assert "--policy-path=" in result.stdout
    assert "PRINT_COMMAND_ONLY=true; nothing was started" in result.stdout


def test_latest_profile_rejects_policy_path_override() -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=OPTIMIZED_ROOT.parents[1],
        env=_environment(
            PRINT_COMMAND_ONLY="true",
            PI05_OPT_POLICY_PATH="/tmp/not-the-audited-step-015900",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "forbids PI05_OPT_POLICY_PATH override" in result.stderr


def test_latest_profile_verifier_pins_complete_checkpoint_metadata_and_weight_hash() -> None:
    source = (PROFILE_ROOT / "verify_checkpoint.py").read_text(encoding="utf-8")

    assert '"checkpoint_step": 15900' in source
    assert '"configured_steps": 15900' in source
    assert '"complete_step": True' in source
    assert "9d6d37f6111a034209c9bdc2899423a3258cc35070cb8294194c9c594197b58a" in source
    assert "00d75a7857fdc3eb45a4dbe6f40de90a1d789d9b1d54e876786dc0e9f908b0b9" in source
    assert "require_complete_step=True" in source
