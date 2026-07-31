from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.robots.jz_robot_pin_timed import JZRobotPinTimedConfig
from lerobot.robots.jz_robot_pin_timed.training_schema import RAW_FEATURE_NAMES

from .safety import ARMED_EXECUTION, DRY_RUN_EXECUTION, transport_for_execution

DEFAULT_ORIN_IP = "192.168.1.81"
DEFAULT_STATE_BIND_IP = "0.0.0.0"
DEFAULT_STATE_PORT = 39010
DEFAULT_COMMAND_PORT = 39020
DEFAULT_CONTROL_FPS = 20
DEFAULT_CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "runtime" / "calibration"

CAMERA_SPECS: dict[str, tuple[int, int, int]] = {
    "camera_head": (5555, 1280, 720),
    "camera_left": (5556, 640, 480),
    "camera_right": (5557, 640, 480),
}


def build_zmq_camera_configs(
    *,
    orin_ip: str = DEFAULT_ORIN_IP,
    fps: int = 30,
    timeout_ms: int = 5000,
) -> dict[str, ZMQCameraConfig]:
    if not orin_ip:
        raise ValueError("orin_ip must be non-empty")
    if fps <= 0:
        raise ValueError("camera fps must be positive")
    cameras: dict[str, ZMQCameraConfig] = {}
    for camera_name, (port, width, height) in CAMERA_SPECS.items():
        cameras[camera_name] = ZMQCameraConfig(
            server_address=orin_ip,
            port=port,
            camera_name=camera_name,
            fps=fps,
            width=width,
            height=height,
            color_mode=ColorMode.RGB,
            timeout_ms=timeout_ms,
        )
    return cameras


def build_robot_config(
    *,
    robot_id: str = "jz_robot_pin_timed_pi05_rtc_infer",
    execution: str = DRY_RUN_EXECUTION,
    orin_ip: str = DEFAULT_ORIN_IP,
    state_bind_ip: str = DEFAULT_STATE_BIND_IP,
    state_port: int = DEFAULT_STATE_PORT,
    command_port: int = DEFAULT_COMMAND_PORT,
    connect_timeout_s: float = 300.0,
    state_timeout_s: float = 1.0,
    camera_fps: int = 30,
    camera_timeout_ms: int = 5000,
    camera_buffer_size: int = 8,
    camera_stale_frame_timeout_ms: int = 1000,
    max_camera_state_receive_skew_ms: float = 100.0,
    state_advance_timeout_s: float = 0.1,
    calibration_dir: str | Path = DEFAULT_CALIBRATION_DIR,
) -> JZRobotPinTimedConfig:
    """Build the audited JZ PI0.5 live boundary without connecting it."""

    if execution not in {DRY_RUN_EXECUTION, ARMED_EXECUTION}:
        raise ValueError("execution must be dry_run or armed")
    _validate_port("state_port", state_port)
    _validate_port("command_port", command_port)
    return JZRobotPinTimedConfig(
        id=robot_id,
        calibration_dir=Path(calibration_dir).expanduser(),
        bind_ip=state_bind_ip,
        state_port=state_port,
        allowed_state_sender_ip=orin_ip,
        connect_timeout_s=connect_timeout_s,
        state_timeout_s=state_timeout_s,
        command_target_ip=orin_ip,
        command_target_port=command_port,
        command_robot="robot1",
        send_action_transport=transport_for_execution(execution),
        send_action_execution=execution,
        require_armed_env=True,
        armed_env_var="JZ_ROBOT_PIN_ARMED",
        max_initial_joint_delta_rad=0.02,
        max_joint_step_rad=0.02,
        gripper_width_min=0.0,
        gripper_width_max=100.0,
        gripper_force_min=0.0,
        gripper_force_max=100.0,
        use_gripper=True,
        zmq_cameras=build_zmq_camera_configs(
            orin_ip=orin_ip,
            fps=camera_fps,
            timeout_ms=camera_timeout_ms,
        ),
        rtsp_cameras={},
        camera_buffer_size=camera_buffer_size,
        camera_reconnect_delay_ms=250,
        camera_stale_frame_timeout_ms=camera_stale_frame_timeout_ms,
        max_camera_state_receive_skew_ms=max_camera_state_receive_skew_ms,
        enforce_camera_state_receive_skew=True,
        reject_reused_camera_frames=False,
        timing_log_every_n=DEFAULT_CONTROL_FPS,
        timing_sidecar=False,
        require_state_source_timing=True,
        require_state_advance_per_observation=True,
        state_advance_timeout_s=state_advance_timeout_s,
        left_gripper_observation_source="measured_opening",
        right_gripper_observation_source="commanded_opening",
        left_gripper_observation_raw_closed=0.0,
        left_gripper_observation_raw_open=100.0,
        right_gripper_observation_raw_closed=100.0,
        right_gripper_observation_raw_open=0.0,
        left_gripper_action_raw_closed=100.0,
        left_gripper_action_raw_open=0.0,
        right_gripper_action_raw_closed=100.0,
        right_gripper_action_raw_open=0.0,
        left_gripper_training_command_force=80.0,
        right_gripper_training_command_force=80.0,
    )


def build_dataset_artifacts(robot: Any) -> tuple[dict[str, dict[str, Any]], Any, Any]:
    """Build the raw18 live feature map used to serialize server requests/actions."""

    _, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )
    validate_live_dataset_features(dataset_features)
    return dataset_features, robot_action_processor, robot_observation_processor


def validate_live_dataset_features(dataset_features: Mapping[str, Mapping[str, Any]]) -> None:
    for feature_key in ("action", "observation.state"):
        feature = dataset_features.get(feature_key)
        if feature is None:
            raise ValueError(f"Live feature map lacks {feature_key}")
        if tuple(feature.get("shape", ())) != (18,):
            raise ValueError(f"Live {feature_key} must be raw18, got {feature.get('shape')}")
        if tuple(feature.get("names", ())) != RAW_FEATURE_NAMES:
            raise ValueError(f"Live {feature_key} names/order do not match jz_pin_raw18_v1")

    expected_camera_features = {
        f"observation.images.{name}": (height, width, 3)
        for name, (_port, width, height) in CAMERA_SPECS.items()
    }
    actual_camera_features = {
        key: tuple(value.get("shape", ()))
        for key, value in dataset_features.items()
        if key.startswith("observation.images.")
    }
    if actual_camera_features != expected_camera_features:
        raise ValueError(
            "Live camera features differ from the trained three-camera contract; "
            f"expected={expected_camera_features} got={actual_camera_features}"
        )


def _validate_port(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError(f"{name} must be an integer in 1..65535")


__all__ = [
    "CAMERA_SPECS",
    "DEFAULT_CALIBRATION_DIR",
    "DEFAULT_COMMAND_PORT",
    "DEFAULT_CONTROL_FPS",
    "DEFAULT_ORIN_IP",
    "DEFAULT_STATE_BIND_IP",
    "DEFAULT_STATE_PORT",
    "build_dataset_artifacts",
    "build_robot_config",
    "build_zmq_camera_configs",
    "validate_live_dataset_features",
]
