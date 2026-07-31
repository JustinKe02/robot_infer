from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.robots.jz_robot_pin_timed.training_schema import RAW_DIM, RAW_FEATURE_NAMES

DRY_RUN_EXECUTION = "dry_run"
ARMED_EXECUTION = "armed"
LOCAL_TRANSPORT = "local"
UDP_TRANSPORT = "udp"
WIRE_FORCE = 80.0

ARMED_CONFIRMATION_ENV_VARS = (
    "JZ_ROBOT_PIN_ARMED",
    "I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT",
    "JZ_POLICY_INFERENCE_ARMED",
)


def transport_for_execution(execution: str) -> str:
    if execution == DRY_RUN_EXECUTION:
        return LOCAL_TRANSPORT
    if execution == ARMED_EXECUTION:
        return UDP_TRANSPORT
    raise ValueError(f"execution must be dry_run or armed, got {execution!r}")


def require_execution_pair(*, execution: str, transport: str) -> None:
    expected_transport = transport_for_execution(execution)
    if transport != expected_transport:
        raise ValueError(
            f"execution={execution!r} requires transport={expected_transport!r}, got {transport!r}"
        )


def require_armed_confirmation(
    *,
    execution: str,
    transport: str,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Apply the client-side gate in addition to the Robot's armed env gate."""

    require_execution_pair(execution=execution, transport=transport)
    if execution != ARMED_EXECUTION:
        return
    values = os.environ if environ is None else environ
    missing = [name for name in ARMED_CONFIRMATION_ENV_VARS if values.get(name) != "1"]
    if missing:
        raise RuntimeError(
            "Refusing armed PI0.5 inference; after the on-site safety check, "
            "set all confirmations to 1: "
            + ", ".join(missing)
        )


@dataclass(slots=True)
class ActionSafety:
    """Validate the raw18 contract before the Robot applies its own safety.

    Joint initial/step limits and gripper clamps deliberately remain owned by
    :class:`JZRobotPin`. This layer checks only the cross-process contract:
    exact raw18 shape/key set, finite values, and fixed wire force slots.
    """

    wire_force: float = WIRE_FORCE
    force_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.wire_force)):
            raise ValueError("wire_force must be finite")
        if self.force_tolerance < 0 or not math.isfinite(float(self.force_tolerance)):
            raise ValueError("force_tolerance must be finite and non-negative")

    def check_tensor(self, action: Tensor | Any) -> Tensor:
        tensor = torch.as_tensor(action).detach().to(device="cpu", dtype=torch.float32)
        if tensor.ndim != 1 or tensor.shape[0] != RAW_DIM:
            raise ValueError(f"Executable JZ action must be raw18, got shape={tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError("Executable raw18 action contains non-finite values")
        self._check_force(float(tensor[15]), "left_gripper.force")
        self._check_force(float(tensor[17]), "right_gripper.force")
        return tensor

    def check_robot_action(self, action: Mapping[str, Any]) -> None:
        actual_names = set(action)
        expected_names = set(RAW_FEATURE_NAMES)
        if actual_names != expected_names:
            raise ValueError(
                "Robot action keys do not match jz_pin_raw18_v1; "
                f"missing={sorted(expected_names - actual_names)} "
                f"extra={sorted(actual_names - expected_names)}"
            )
        values = []
        for name in RAW_FEATURE_NAMES:
            value = action[name]
            if isinstance(value, bool):
                raise ValueError(f"Robot action {name} must be numeric, not bool")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Robot action {name} must be numeric") from exc
            if not math.isfinite(numeric):
                raise ValueError(f"Robot action {name} must be finite")
            values.append(numeric)
        self._check_force(values[15], "left_gripper.force")
        self._check_force(values[17], "right_gripper.force")

    def _check_force(self, value: float, name: str) -> None:
        if abs(value - float(self.wire_force)) > float(self.force_tolerance):
            raise ValueError(
                f"{name} must come from the serialized schema boundary and equal "
                f"{self.wire_force}, got {value}"
            )


__all__ = [
    "ARMED_CONFIRMATION_ENV_VARS",
    "ARMED_EXECUTION",
    "ActionSafety",
    "DRY_RUN_EXECUTION",
    "LOCAL_TRANSPORT",
    "UDP_TRANSPORT",
    "WIRE_FORCE",
    "require_armed_confirmation",
    "require_execution_pair",
    "transport_for_execution",
]
