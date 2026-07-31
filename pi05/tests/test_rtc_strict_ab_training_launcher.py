from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = (
    REPO_ROOT
    / "tk_infer/pi05/rtc_training_code/my_devs/jz_robot_pin_timed/pi05"
    / "train_rtc_strict_ab_full_head_right_15_epochs.sh"
)
CONTRACT = LAUNCHER.with_name("rtc_strict_ab_full_head_right_15e_d5_seed1000.json")


def _run(**updates: str) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "CUDA_VISIBLE_DEVICES",
            "DRY_RUN",
            "PRINT_CONTRACT_ONLY",
            "RUN_STAMP",
            "SOURCE_SMOKE_ONLY",
            "STEPS_OVERRIDE",
        }
    }
    env.update({"PRINT_CONTRACT_ONLY": "true", "RUN_STAMP": "pytest", **updates})
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_strict_ab_contract_prints_without_accessing_training_resources() -> None:
    result = _run()
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "source-training-host-only local_training=forbidden" in output
    assert f"contract={CONTRACT}" in output
    assert "view=jz_robot_pin_timed_merged_100eps_20260728_pi05_head_right" in output
    assert "task=Put the bottle on the right into the basket on the left." in output
    assert "finetune=full training=rtc epochs=15 delay=0..5 seed=1000" in output
    assert "per_device_batch=32 num_processes=1 effective_batch=32" in output
    assert "scheduler_config=warmup1000_decay30000 effective=warmup530_decay15900" in output
    assert "source_smoke_only=false save_checkpoint=true" in output
    assert "FINETUNE_MODE=full" in output
    assert "TRAINING_MODE=rtc" in output
    assert "RTC_MAX_DELAY=5" in output
    assert "CAMERA_MODE=head_right" in output
    assert "PRINT_CONTRACT_ONLY PASS; no data, model, GPU, or trainer access occurred" in output


def test_strict_ab_manifest_matches_the_working_full_baseline() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert config["dataset"]["episodes"] == 100
    assert config["dataset"]["frames"] == 33898
    assert config["dataset"]["camera_keys"] == [
        "observation.images.camera_head",
        "observation.images.camera_right",
    ]
    assert config["dataset"]["task"] == "Put the bottle on the right into the basket on the left."
    assert config["policy"]["finetune_mode"] == "full"
    assert config["policy"]["train_expert_only"] is False
    assert config["policy"]["rtc_config"] is None
    assert config["policy"]["rtc_training"] == {
        "enabled": True,
        "max_delay": 5,
        "min_postfix_steps": 1,
        "observed_delay_histogram": [],
        "observed_histogram_weight": 0.9,
        "loss_scope": "postfix_only",
    }
    expected_training = {
        "epochs": 15,
        "batch_size": 32,
        "num_processes": 1,
        "effective_batch_size": 32,
        "steps_per_epoch": 1060,
        "steps": 15900,
        "save_freq_steps": 5300,
        "seed": 1000,
    }
    assert {key: config["training"][key] for key in expected_training} == expected_training
    assert config["scheduler"]["effective_warmup_steps"] == 530
    assert config["scheduler"]["effective_decay_steps"] == 15900


def test_strict_ab_source_smoke_is_locked_to_two_steps_without_checkpoints() -> None:
    result = _run(SOURCE_SMOKE_ONLY="true")
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "source_smoke_only=true save_checkpoint=false" in output
    assert "STEPS_OVERRIDE=2" in output
    assert "SAVE_CHECKPOINT=false" in output
    assert "strict_ab_2step_smoke_pytest" in output


def test_strict_ab_contract_rejects_step_and_gpu_overrides() -> None:
    steps = _run(STEPS_OVERRIDE="1")
    gpu_count = _run(CUDA_VISIBLE_DEVICES="0,1")
    invalid_gpu = _run(CUDA_VISIBLE_DEVICES="gpu0")

    assert steps.returncode == 2
    assert "STEPS_OVERRIDE is forbidden" in steps.stderr
    assert gpu_count.returncode == 2
    assert "exactly one GPU" in gpu_count.stderr
    assert invalid_gpu.returncode == 2
    assert "invalid GPU id" in invalid_gpu.stderr
