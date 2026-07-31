#!/usr/bin/env python

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.jz_robot_pin.config_jz_robot_pin import JZRobotPinConfig


@RobotConfig.register_subclass("jz_robot_pin_timed")
@dataclass
class JZRobotPinTimedConfig(JZRobotPinConfig):
    """Pin robot variant with timestamp-preserving camera receivers."""

    zmq_cameras: dict[str, ZMQCameraConfig] = field(default_factory=dict)
    camera_buffer_size: int = 8
    camera_reconnect_delay_ms: int = 250
    camera_stale_frame_timeout_ms: int = 1000
    max_camera_state_receive_skew_ms: float = 100.0
    enforce_camera_state_receive_skew: bool = True
    reject_reused_camera_frames: bool = False
    timing_log_every_n: int = 30
    timing_sidecar: bool = True
    require_state_source_timing: bool = False
    require_state_advance_per_observation: bool = True
    state_advance_timeout_s: float = 0.1
    left_gripper_observation_source: str = "unavailable"
    right_gripper_observation_source: str = "unavailable"
    left_gripper_observation_raw_closed: float = 0.0
    left_gripper_observation_raw_open: float = 100.0
    right_gripper_observation_raw_closed: float = 100.0
    right_gripper_observation_raw_open: float = 0.0
    left_gripper_action_raw_closed: float = 100.0
    left_gripper_action_raw_open: float = 0.0
    right_gripper_action_raw_closed: float = 100.0
    right_gripper_action_raw_open: float = 0.0
    left_gripper_training_command_force: float = 80.0
    right_gripper_training_command_force: float = 80.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for field_name in (
            "enforce_camera_state_receive_skew",
            "reject_reused_camera_frames",
            "timing_sidecar",
            "require_state_source_timing",
            "require_state_advance_per_observation",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean")
        valid_sources = {"measured_opening", "commanded_opening", "unavailable"}
        for field_name in (
            "left_gripper_observation_source",
            "right_gripper_observation_source",
        ):
            value = getattr(self, field_name)
            if value not in valid_sources:
                raise ValueError(f"{field_name} must be one of {sorted(valid_sources)}, got {value!r}")
        for side in ("left", "right"):
            for modality in ("observation", "action"):
                closed_field = f"{side}_gripper_{modality}_raw_closed"
                open_field = f"{side}_gripper_{modality}_raw_open"
                closed = getattr(self, closed_field)
                opened = getattr(self, open_field)
                for field_name, value in ((closed_field, closed), (open_field, opened)):
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, Real)
                        or not math.isfinite(float(value))
                    ):
                        raise ValueError(f"{field_name} must be a finite number")
                if float(closed) == float(opened):
                    raise ValueError(f"{closed_field} and {open_field} must differ")
            force_field = f"{side}_gripper_training_command_force"
            force = getattr(self, force_field)
            if isinstance(force, bool) or not isinstance(force, Real) or not math.isfinite(float(force)):
                raise ValueError(f"{force_field} must be a finite number")
        if isinstance(self.camera_buffer_size, bool) or not isinstance(self.camera_buffer_size, int):
            raise ValueError("camera_buffer_size must be an integer")
        if self.camera_buffer_size <= 0:
            raise ValueError("camera_buffer_size must be positive")
        if self.rtsp_cameras and self.zmq_cameras:
            raise ValueError("rtsp_cameras and zmq_cameras are mutually exclusive")
        for key, camera in self.zmq_cameras.items():
            if key != camera.camera_name:
                raise ValueError(
                    f"zmq_cameras key {key!r} must equal configured camera_name {camera.camera_name!r}"
                )
        if (
            isinstance(self.camera_stale_frame_timeout_ms, bool)
            or not isinstance(self.camera_stale_frame_timeout_ms, int)
            or self.camera_stale_frame_timeout_ms <= 0
        ):
            raise ValueError("camera_stale_frame_timeout_ms must be a positive integer")
        if isinstance(self.camera_reconnect_delay_ms, bool) or not isinstance(
            self.camera_reconnect_delay_ms, int
        ):
            raise ValueError("camera_reconnect_delay_ms must be an integer")
        if self.camera_reconnect_delay_ms < 0:
            raise ValueError("camera_reconnect_delay_ms must be non-negative")
        if isinstance(self.timing_log_every_n, bool) or not isinstance(self.timing_log_every_n, int):
            raise ValueError("timing_log_every_n must be an integer")
        if self.timing_log_every_n < 0:
            raise ValueError("timing_log_every_n must be non-negative")
        if (
            isinstance(self.max_camera_state_receive_skew_ms, bool)
            or not isinstance(self.max_camera_state_receive_skew_ms, Real)
            or not math.isfinite(float(self.max_camera_state_receive_skew_ms))
            or self.max_camera_state_receive_skew_ms <= 0
        ):
            raise ValueError("max_camera_state_receive_skew_ms must be a positive finite number")
        if (
            isinstance(self.state_advance_timeout_s, bool)
            or not isinstance(self.state_advance_timeout_s, Real)
            or not math.isfinite(float(self.state_advance_timeout_s))
            or self.state_advance_timeout_s <= 0
        ):
            raise ValueError("state_advance_timeout_s must be a positive finite number")


__all__ = ["JZRobotPinTimedConfig"]
