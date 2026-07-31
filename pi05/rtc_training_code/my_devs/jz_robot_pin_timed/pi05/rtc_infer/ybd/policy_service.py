#!/usr/bin/env python

"""YBD-only adapter around the existing JZ PI0.5 inference runtime."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

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

from lerobot.robots.jz_robot_pin_timed.training_schema import JZPinTrainingSchema  # noqa: E402
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime import checkpoint  # noqa: E402
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.policy_service import (  # noqa: E402
    PolicyService,
)

HEAD_RIGHT_CAMERA_KEYS = (
    "observation.images.camera_head",
    "observation.images.camera_right",
)
HEAD_RIGHT_CAMERA_SHAPES = {
    "observation.images.camera_head": (3, 720, 1280),
    "observation.images.camera_right": (3, 480, 640),
}
RAW_STATE_DIM = 18
WIRE_ACTION_DIM = 18


def install_head_right_camera_contract() -> None:
    """Limit this process to the YBD checkpoint's exact two-camera contract."""

    checkpoint.EXPECTED_CAMERA_KEYS = HEAD_RIGHT_CAMERA_KEYS
    checkpoint.EXPECTED_CAMERA_SHAPES = dict(HEAD_RIGHT_CAMERA_SHAPES)


def hold_raw18_left_side(
    actions: Any,
    *,
    raw_observation_state: Any,
    schema: JZPinTrainingSchema,
) -> np.ndarray:
    """Replace executable raw18 left-side commands after checkpoint postprocessing."""

    processed_actions = torch.as_tensor(actions, dtype=torch.float32, device="cpu")
    if processed_actions.ndim != 2 or processed_actions.shape[-1] != WIRE_ACTION_DIM:
        raise ValueError(f"Expected processed actions (T,18), got {tuple(processed_actions.shape)}")
    if not torch.isfinite(processed_actions).all():
        raise ValueError("YBD processed raw18 actions must be finite")
    if raw_observation_state is None:
        raise ValueError("YBD inference requires raw18 observation.state")
    raw_state = torch.as_tensor(raw_observation_state, dtype=torch.float32, device="cpu")
    if raw_state.shape != (RAW_STATE_DIM,) or not torch.isfinite(raw_state).all():
        raise ValueError(
            "YBD inference requires one finite raw18 observation.state, "
            f"got {tuple(raw_state.shape)}"
        )
    model_state = torch.as_tensor(schema.project_observation(raw_state), dtype=torch.float32)
    model_hold = torch.zeros(16, dtype=torch.float32)
    model_hold[0:7] = model_state[0:7]
    model_hold[14] = model_state[14]
    raw_hold = torch.as_tensor(schema.expand_action(model_hold), dtype=torch.float32)

    held = processed_actions.clone()
    held[..., 0:7] = raw_hold[0:7]
    held[..., 14:16] = raw_hold[14:16]
    held[..., 17] = raw_hold[17]
    return held.numpy()


class YBDPolicyService(PolicyService):
    """Existing service with a request-local two-camera/right-arm adapter."""

    def infer(self, request: Any) -> Any:
        frame = dict(request.observation_frame)
        raw_state = frame.get("observation.state")
        frame.pop("observation.images.camera_left", None)
        adapted_request = copy.copy(request)
        adapted_request.observation_frame = frame
        response = super().infer(adapted_request)
        response.processed_actions = hold_raw18_left_side(
            response.processed_actions,
            raw_observation_state=raw_state,
            schema=self.bundle.schema,
        )
        response.processed_action_shape = tuple(response.processed_actions.shape)
        response.validate()
        return response


def server_main(argv: list[str]) -> int:
    install_head_right_camera_contract()
    from my_devs.jz_robot_pin_timed.pi05.rtc_infer import run_policy_server

    run_policy_server.PolicyService = YBDPolicyService
    return run_policy_server.main(argv)


def client_main(argv: list[str]) -> int:
    from my_devs.jz_robot_pin_timed.pi05.rtc_infer import run_robot_client

    run_robot_client.EXPECTED_CAMERA_KEYS = HEAD_RIGHT_CAMERA_KEYS
    run_robot_client.EXPECTED_CAMERA_SHAPES = dict(HEAD_RIGHT_CAMERA_SHAPES)
    return run_robot_client.main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("server", "client"))
    args, remainder = parser.parse_known_args(argv)
    return server_main(remainder) if args.command == "server" else client_main(remainder)


if __name__ == "__main__":
    raise SystemExit(main())
