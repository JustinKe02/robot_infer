#!/usr/bin/env python

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def resolve_repo_root(path: Path) -> Path:
    for candidate in path.resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from lerobot.configs.types import RTCAttentionSchedule  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from tk_infer.pi05.profiles.step_010600.verify_checkpoint import (  # noqa: E402
    POLICY_PATH,
    TOKENIZER_PATH,
    verify_checkpoint,
)
from tk_infer.pi05.runtime.policy_service import PolicyService, PolicyServiceConfig  # noqa: E402
from tk_infer.pi05.runtime.protocol import MAX_ACTION_CHUNK_STEPS, InferenceRequest  # noqa: E402

DATASET_ROOT = REPO_ROOT / "data/jz_robot_pin_timed_merged_100eps_20260728"
DATASET_REPO_ID = "local/jz_robot_pin_timed_merged_100eps_20260728_pi05_head_right"
TASK = "jz robot pin timed vr teleoperation"


def _state_to_wire(value: torch.Tensor) -> np.ndarray:
    state = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if state.shape != (18,) or not torch.isfinite(state).all():
        raise RuntimeError(f"dataset state must be one finite raw18 vector, got {tuple(state.shape)}")
    return np.ascontiguousarray(state.numpy())


def _image_to_wire(value: torch.Tensor, *, expected_chw: tuple[int, int, int]) -> np.ndarray:
    image = torch.as_tensor(value).detach().cpu()
    if image.shape != expected_chw:
        raise RuntimeError(f"dataset image must be CHW {expected_chw}, got {tuple(image.shape)}")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise RuntimeError("dataset image must contain finite floating-point values")
    if torch.any(image < 0) or torch.any(image > 1):
        raise RuntimeError("dataset image values must be normalized to [0,1]")
    image_hwc = image.permute(1, 2, 0).mul(255).round().to(torch.uint8)
    return np.ascontiguousarray(image_hwc.numpy())


def main() -> int:
    verify_checkpoint()
    service = PolicyService.from_config(
        PolicyServiceConfig(
            policy_path=POLICY_PATH,
            tokenizer_path=TOKENIZER_PATH,
            device="cuda",
            require_complete_step=False,
        ),
        rtc_config=RTCConfig(
            enabled=True,
            prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
            max_guidance_weight=10.0,
            execution_horizon=10,
        ),
    )
    dataset = LeRobotDataset(DATASET_REPO_ID, root=DATASET_ROOT, video_backend="pyav")
    sample = dataset[0]
    observation_frame = {
        "observation.state": _state_to_wire(sample["observation.state"]),
        "observation.images.camera_head": _image_to_wire(
            sample["observation.images.camera_head"], expected_chw=(3, 720, 1280)
        ),
        "observation.images.camera_right": _image_to_wire(
            sample["observation.images.camera_right"], expected_chw=(3, 480, 640)
        ),
    }
    expected_observation_keys = {
        "observation.state",
        "observation.images.camera_head",
        "observation.images.camera_right",
    }
    if set(observation_frame) != expected_observation_keys:
        raise RuntimeError(f"offline smoke observation keys differ: {sorted(observation_frame)}")
    started_s = time.perf_counter()
    response = service.infer(
        InferenceRequest(
            request_id=1,
            mode="single_step",
            observation_frame=observation_frame,
            task=TASK,
            robot_type="jz_robot_pin_timed",
            obs_sequence_id=1,
            execution_horizon=10,
        )
    )
    elapsed_s = time.perf_counter() - started_s

    raw = torch.as_tensor(response.raw_actions, dtype=torch.float32)
    processed = torch.as_tensor(response.processed_actions, dtype=torch.float32)
    if raw.ndim != 2 or raw.shape[1] != 16:
        raise RuntimeError(f"offline profile model action must be (T,16), got {tuple(raw.shape)}")
    if processed.ndim != 2 or processed.shape != (raw.shape[0], 18):
        raise RuntimeError(
            "offline profile processed action must match the model chunk as (T,18), "
            f"got model={tuple(raw.shape)} processed={tuple(processed.shape)}"
        )
    if not 1 <= raw.shape[0] <= MAX_ACTION_CHUNK_STEPS:
        raise RuntimeError(f"offline profile temporal length is invalid: {raw.shape[0]}")
    if not torch.isfinite(raw).all() or not torch.isfinite(processed).all():
        raise RuntimeError("offline profile inference produced non-finite actions")
    expected_force = torch.full((raw.shape[0],), 80.0, dtype=torch.float32)
    torch.testing.assert_close(processed[:, 15], expected_force, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(processed[:, 17], expected_force, rtol=0.0, atol=1e-6)

    health = service.health()
    expected_camera_keys = [
        "observation.images.camera_head",
        "observation.images.camera_right",
    ]
    if health.get("camera_profile") != "head_right" or health.get("camera_keys") != expected_camera_keys:
        raise RuntimeError(
            "loaded checkpoint does not expose the expected head+right camera contract: "
            f"profile={health.get('camera_profile')} keys={health.get('camera_keys')}"
        )

    report = {
        "status": "PASS",
        "hardware_access": False,
        "checkpoint_step": 10600,
        "configured_steps": 15900,
        "camera_profile": health["camera_profile"],
        "camera_keys": health["camera_keys"],
        "model_action_shape": list(response.raw_actions.shape),
        "processed_action_shape": list(response.processed_actions.shape),
        "processed_action_min": float(np.min(response.processed_actions)),
        "processed_action_max": float(np.max(response.processed_actions)),
        "force_slots": [15, 17],
        "force_value": 80.0,
        "server_inference_count": service.inference_count,
        "elapsed_s": elapsed_s,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
