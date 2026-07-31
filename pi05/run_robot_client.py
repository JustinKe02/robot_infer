#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np


def resolve_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = resolve_repo_root(Path(__file__))
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from tk_infer.pi05.runtime.camera_profiles import (  # noqa: E402
    CAMERA_SPECS,
    DEFAULT_CAMERA_PROFILE,
    SUPPORTED_CAMERA_PROFILES,
    camera_feature_keys,
    camera_feature_shapes,
    camera_names,
)
from tk_infer.pi05.runtime.client_runtime import (  # noqa: E402
    ClientRuntimeConfig,
    run_client_runtime,
    run_inference_smoke,
)
from tk_infer.pi05.runtime.protocol import (  # noqa: E402
    PROTOCOL_VERSION,
)
from tk_infer.pi05.runtime.remote_client import (  # noqa: E402
    RemotePolicyClient,
)
from tk_infer.pi05.runtime.robot_builder import (  # noqa: E402
    DEFAULT_COMMAND_PORT,
    DEFAULT_CONTROL_FPS,
    DEFAULT_MAX_CAMERA_STATE_RECEIVE_SKEW_MS,
    DEFAULT_ORIN_IP,
    DEFAULT_STATE_BIND_IP,
    DEFAULT_STATE_PORT,
    build_dataset_artifacts,
    build_robot_config,
)
from tk_infer.pi05.runtime.robot_io import (  # noqa: E402
    SerializedRobotIO,
)
from tk_infer.pi05.runtime.safety import (  # noqa: E402
    ARMED_EXECUTION,
    DRY_RUN_EXECUTION,
    ActionSafety,
    require_armed_confirmation,
    transport_for_execution,
)

LOG_PREFIX = "[tk_infer/pi05/client]"
DEFAULT_TASK = "jz robot pin timed vr teleoperation"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "client"
EXPECTED_CAMERA_KEYS = camera_feature_keys(DEFAULT_CAMERA_PROFILE)
EXPECTED_CAMERA_SHAPES = camera_feature_shapes(DEFAULT_CAMERA_PROFILE)
DISABLE_JOINT_DELTA_CHECKS_ENV = "JZ_PI05_DISABLE_JOINT_DELTA_CHECKS"
JOINT_DELTA_BYPASS_ACK_ENV = "I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED"
EXPECTED_CHECKPOINT_ENV = {
    "checkpoint_step": "JZ_PI05_EXPECTED_CHECKPOINT_STEP",
    "configured_steps": "JZ_PI05_EXPECTED_CONFIGURED_STEPS",
    "checkpoint_fingerprint": "JZ_PI05_EXPECTED_CHECKPOINT_FINGERPRINT",
    "checkpoint_path": "JZ_PI05_EXPECTED_CHECKPOINT_PATH",
    "complete_step": "JZ_PI05_EXPECTED_COMPLETE_STEP",
}


@dataclass(frozen=True, slots=True)
class ExpectedCheckpoint:
    checkpoint_step: int | None
    configured_steps: int
    checkpoint_fingerprint: str
    checkpoint_path: str
    complete_step: bool | None


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else parse_bool(value)


def expected_checkpoint_from_env() -> ExpectedCheckpoint | None:
    values = {key: os.getenv(name) for key, name in EXPECTED_CHECKPOINT_ENV.items()}
    present = {key for key, value in values.items() if value not in {None, ""}}
    if not present:
        return None
    if present != set(values):
        missing = sorted(EXPECTED_CHECKPOINT_ENV[key] for key in set(values) - present)
        raise ValueError(f"custom checkpoint contract is incomplete; missing {missing}")

    checkpoint_step_raw = str(values["checkpoint_step"]).strip().lower()
    checkpoint_step = (
        None
        if checkpoint_step_raw == "null"
        else _positive_int("JZ_PI05_EXPECTED_CHECKPOINT_STEP", checkpoint_step_raw)
    )
    configured_steps = _positive_int(
        "JZ_PI05_EXPECTED_CONFIGURED_STEPS", str(values["configured_steps"]).strip()
    )
    fingerprint = str(values["checkpoint_fingerprint"])
    if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
        raise ValueError("custom checkpoint fingerprint must be a lowercase SHA-256 hex digest")
    complete_raw = str(values["complete_step"]).strip().lower()
    if complete_raw not in {"true", "false", "null"}:
        raise ValueError("JZ_PI05_EXPECTED_COMPLETE_STEP must be exactly true, false, or null")
    complete_step = None if complete_raw == "null" else complete_raw == "true"
    if (checkpoint_step is None) != (complete_step is None):
        raise ValueError(
            "path-unproven checkpoint contracts require both checkpoint step and complete step to be null"
        )
    checkpoint_path = Path(str(values["checkpoint_path"])).expanduser()
    if not checkpoint_path.is_absolute():
        raise ValueError("JZ_PI05_EXPECTED_CHECKPOINT_PATH must be absolute")
    return ExpectedCheckpoint(
        checkpoint_step=checkpoint_step,
        configured_steps=configured_steps,
        checkpoint_fingerprint=fingerprint,
        checkpoint_path=checkpoint_path.resolve(strict=False).as_posix(),
        complete_step=complete_step,
    )


def _positive_int(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def joint_delta_checks_disabled_from_env() -> bool:
    disabled = os.getenv(DISABLE_JOINT_DELTA_CHECKS_ENV)
    acknowledged = os.getenv(JOINT_DELTA_BYPASS_ACK_ENV)
    if disabled not in {None, "", "0", "1"}:
        raise ValueError(f"{DISABLE_JOINT_DELTA_CHECKS_ENV} must be exactly 0 or 1")
    if acknowledged not in {None, "", "0", "1"}:
        raise ValueError(f"{JOINT_DELTA_BYPASS_ACK_ENV} must be exactly 0 or 1")
    if disabled == "1":
        if acknowledged != "1":
            raise RuntimeError(f"{DISABLE_JOINT_DELTA_CHECKS_ENV}=1 requires {JOINT_DELTA_BYPASS_ACK_ENV}=1")
        return True
    if acknowledged == "1":
        raise RuntimeError(f"{JOINT_DELTA_BYPASS_ACK_ENV}=1 requires {DISABLE_JOINT_DELTA_CHECKS_ENV}=1")
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JZ Robot Pin Timed client for pure-PyTorch PI0.5 single-step/RTC inference."
    )
    parser.add_argument("--server-url", default=os.getenv("SERVER_URL", "http://127.0.0.1:8088"))
    parser.add_argument(
        "--auth-token",
        default=os.getenv("JZ_PI05_SERVER_AUTH_TOKEN"),
        help="Bearer token; prefer JZ_PI05_SERVER_AUTH_TOKEN so it is absent from process arguments.",
    )
    parser.add_argument(
        "--mode",
        choices=("single_step", "async_single_step", "rtc"),
        default=os.getenv("MODE", "rtc"),
    )
    parser.add_argument(
        "--execution",
        choices=(DRY_RUN_EXECUTION, ARMED_EXECUTION),
        default=os.getenv("EXECUTION", DRY_RUN_EXECUTION),
    )
    parser.add_argument(
        "--robot-id",
        default=os.getenv("ROBOT_ID", "jz_robot_pin_timed_pi05_rtc"),
    )
    parser.add_argument(
        "--camera-profile",
        choices=SUPPORTED_CAMERA_PROFILES,
        default=os.getenv("CAMERA_PROFILE", DEFAULT_CAMERA_PROFILE),
    )
    parser.add_argument("--orin-ip", default=os.getenv("ORIN_IP", DEFAULT_ORIN_IP))
    parser.add_argument("--state-bind-ip", default=os.getenv("STATE_BIND_IP", DEFAULT_STATE_BIND_IP))
    parser.add_argument(
        "--state-port",
        type=int,
        default=int(os.getenv("STATE_PORT", str(DEFAULT_STATE_PORT))),
    )
    parser.add_argument(
        "--state-timeout-s",
        type=float,
        default=float(os.getenv("STATE_TIMEOUT_S", "1.0")),
    )
    parser.add_argument(
        "--max-camera-state-receive-skew-ms",
        type=float,
        default=float(
            os.getenv(
                "MAX_CAMERA_STATE_RECEIVE_SKEW_MS",
                str(DEFAULT_MAX_CAMERA_STATE_RECEIVE_SKEW_MS),
            )
        ),
    )
    parser.add_argument(
        "--connect-timeout-s",
        type=float,
        default=float(os.getenv("CONNECT_TIMEOUT_S", "300.0")),
    )
    parser.add_argument(
        "--command-port",
        type=int,
        default=int(os.getenv("COMMAND_PORT", str(DEFAULT_COMMAND_PORT))),
    )
    parser.add_argument("--task", default=os.getenv("TASK", DEFAULT_TASK))
    parser.add_argument(
        "--sensor-fps",
        type=int,
        default=int(os.getenv("SENSOR_FPS", str(DEFAULT_CONTROL_FPS))),
    )
    parser.add_argument(
        "--control-fps",
        type=int,
        default=int(os.getenv("CONTROL_FPS", str(DEFAULT_CONTROL_FPS))),
    )
    parser.add_argument("--run-time-s", type=float, default=float(os.getenv("RUN_TIME_S", "10")))
    parser.add_argument(
        "--queue-low-watermark",
        type=int,
        default=int(os.getenv("QUEUE_LOW_WATERMARK", "30")),
    )
    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=int(os.getenv("MAX_QUEUE_SIZE", "50")),
    )
    parser.add_argument(
        "--first-chunk-timeout-s",
        type=float,
        default=float(os.getenv("FIRST_CHUNK_TIMEOUT_S", "120")),
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=int(os.getenv("RTC_EXECUTION_HORIZON", "10")),
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=float(os.getenv("REQUEST_TIMEOUT_S", "120")),
    )
    parser.add_argument(
        "--empty-queue-strategy",
        choices=("stop", "skip_send", "hold_last_action"),
        default=os.getenv("EMPTY_QUEUE_STRATEGY", "stop"),
    )
    parser.add_argument(
        "--fully-stale-chunk-limit",
        type=int,
        default=int(os.getenv("FULLY_STALE_CHUNK_LIMIT", "3")),
    )
    parser.add_argument(
        "--metrics-log-interval-s",
        type=float,
        default=float(os.getenv("METRICS_LOG_INTERVAL_S", "2")),
    )
    parser.add_argument(
        "--max-sent-actions",
        type=int,
        default=int(os.getenv("MAX_SENT_ACTIONS", "0")),
    )
    parser.add_argument(
        "--config-only",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("CONFIG_ONLY"),
    )
    parser.add_argument(
        "--health-only",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("HEALTH_ONLY"),
    )
    parser.add_argument(
        "--connect-smoke",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("CONNECT_SMOKE"),
    )
    parser.add_argument(
        "--inference-smoke",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("INFERENCE_SMOKE"),
        help="Run one live observation and policy inference without calling robot.send_action().",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_ROOT.as_posix())),
    )
    return parser


def build_runtime_config(args: argparse.Namespace) -> ClientRuntimeConfig:
    return ClientRuntimeConfig(
        task=args.task,
        server_url=args.server_url,
        mode=args.mode,
        sensor_fps=args.sensor_fps,
        control_fps=args.control_fps,
        run_time_s=args.run_time_s,
        queue_low_watermark=args.queue_low_watermark,
        max_queue_size=args.max_queue_size,
        first_chunk_timeout_s=args.first_chunk_timeout_s,
        rtc_execution_horizon=args.rtc_execution_horizon,
        empty_queue_strategy=args.empty_queue_strategy,
        metrics_log_interval_s=args.metrics_log_interval_s,
        fully_stale_chunk_limit=args.fully_stale_chunk_limit,
        max_sent_actions=args.max_sent_actions,
    )


def validate_server_url_security(server_url: str, auth_token: str | None) -> None:
    parsed = urlparse(server_url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.port is None:
        raise ValueError("server_url must be an explicit http://host:port URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("server_url must not contain credentials; use the Bearer auth token")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("server_url must be a bare http://host:port base URL")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} and not auth_token:
        raise ValueError("Non-loopback policy server requires JZ_PI05_SERVER_AUTH_TOKEN")


def validate_server_health(
    health: dict[str, Any],
    *,
    mode: str,
    camera_profile: str = DEFAULT_CAMERA_PROFILE,
    expected_checkpoint: ExpectedCheckpoint | None = None,
) -> None:
    expected_camera_keys = camera_feature_keys(camera_profile)
    expected_camera_shapes = camera_feature_shapes(camera_profile)
    expected = {
        "ok": True,
        "protocol_version": PROTOCOL_VERSION,
        "policy_type": "pi05",
        "model_state_dim": 16,
        "model_action_dim": 16,
        "wire_action_dim": 18,
        "schema_id": "jz_pin_opening16_v1",
        "schema_version": 1,
        "camera_profile": camera_profile,
    }
    mismatches = {
        key: {"expected": value, "actual": health.get(key)}
        for key, value in expected.items()
        if health.get(key) != value
    }
    if tuple(health.get("camera_keys", ())) != expected_camera_keys:
        mismatches["camera_keys"] = {
            "expected": list(expected_camera_keys),
            "actual": health.get("camera_keys"),
        }
    actual_camera_shapes = health.get("camera_shapes")
    normalized_camera_shapes = (
        {
            key: tuple(value)
            for key, value in actual_camera_shapes.items()
            if isinstance(key, str) and isinstance(value, (list, tuple))
        }
        if isinstance(actual_camera_shapes, dict)
        else None
    )
    if normalized_camera_shapes != expected_camera_shapes:
        mismatches["camera_shapes"] = {
            "expected": {key: list(value) for key, value in expected_camera_shapes.items()},
            "actual": actual_camera_shapes,
        }
    wire_mode = "single_step" if mode == "async_single_step" else mode
    if wire_mode not in health.get("supported_modes", ()):
        mismatches["supported_modes"] = {
            "expected_contains": wire_mode,
            "actual": health.get("supported_modes"),
        }
    if expected_checkpoint is not None:
        checkpoint_expected = asdict(expected_checkpoint)
        mismatches.update(
            {
                key: {"expected": value, "actual": health.get(key)}
                for key, value in checkpoint_expected.items()
                if health.get(key) != value
            }
        )
    else:
        checkpoint_step = health.get("checkpoint_step")
        configured_steps = health.get("configured_steps")
        if health.get("complete_step") is not True:
            mismatches["complete_step"] = {
                "expected": True,
                "actual": health.get("complete_step"),
            }
        if (
            isinstance(checkpoint_step, bool)
            or not isinstance(checkpoint_step, int)
            or checkpoint_step <= 0
            or isinstance(configured_steps, bool)
            or not isinstance(configured_steps, int)
            or configured_steps <= 0
            or checkpoint_step != configured_steps
        ):
            mismatches["checkpoint_step"] = {
                "expected": "same positive integer as configured_steps",
                "actual": checkpoint_step,
                "configured_steps": configured_steps,
            }
    if mismatches:
        raise RuntimeError(f"Policy server health is incompatible with JZ PI0.5 client: {mismatches}")


def print_resolved_config(
    args: argparse.Namespace,
    runtime_config: ClientRuntimeConfig,
    robot_config: Any,
) -> None:
    camera_endpoints = ",".join(
        f"{name}:{CAMERA_SPECS[name].port}" for name in camera_names(args.camera_profile)
    )
    print(f"{LOG_PREFIX} repo={REPO_ROOT}")
    print(
        f"{LOG_PREFIX} server={runtime_config.server_url} "
        f"auth={'enabled' if args.auth_token else 'disabled'} mode={runtime_config.mode}"
    )
    print(
        f"{LOG_PREFIX} execution={args.execution} "
        f"transport={robot_config.send_action_transport} robot_id={robot_config.id}"
    )
    print(
        f"{LOG_PREFIX} state=udp://{args.state_bind_ip}:{args.state_port} "
        f"allowed_source={args.orin_ip} command=udp://{args.orin_ip}:{args.command_port}"
    )
    print(
        f"{LOG_PREFIX} sensor_fps={runtime_config.sensor_fps} "
        f"control_fps={runtime_config.control_fps} task={runtime_config.task!r}"
    )
    print(
        f"{LOG_PREFIX} camera_profile={args.camera_profile} "
        f"cameras=zmq://{args.orin_ip}/[{camera_endpoints}] schema=raw18->model16->raw18 force=80"
    )
    print(f"{LOG_PREFIX} camera_state_receive_skew_limit_ms={robot_config.max_camera_state_receive_skew_ms}")
    print(f"{LOG_PREFIX} action_boundary=full_raw18")
    print(
        f"{LOG_PREFIX} joint_delta_checks="
        f"{'disabled' if robot_config.allow_armed_joint_delta_bypass else 'enabled'} "
        f"initial={robot_config.max_initial_joint_delta_rad} "
        f"step={robot_config.max_joint_step_rad}"
    )


def write_summary(
    *,
    output_dir: Path,
    execution: str,
    runtime_config: ClientRuntimeConfig,
    health: dict[str, Any],
    result: Any,
) -> Path:
    output_dir = resolve_local_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "client_summary.json"
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "execution": execution,
        "transport": transport_for_execution(execution),
        "runtime": asdict(runtime_config),
        "server": health,
        "result": result.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def write_inference_smoke_artifacts(
    *,
    output_dir: Path,
    runtime_config: ClientRuntimeConfig,
    health: dict[str, Any],
    result: Any,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = output_dir / "inference_smoke.json"
    observation_path = output_dir / "observation_frame.npz"
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "execution": DRY_RUN_EXECUTION,
        "send_action_called": False,
        "runtime": asdict(runtime_config),
        "server": health,
        "result": result.to_dict(),
        "selected_robot_action": result.selected_robot_action,
        "raw_actions_model16": result.raw_actions.tolist(),
        "processed_actions_raw18": result.processed_actions.tolist(),
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(observation_path, **result.observation_frame)
    return metadata_path, observation_path


def resolve_local_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    if not resolved.is_relative_to(SCRIPT_DIR):
        raise ValueError(f"output-dir must stay inside {SCRIPT_DIR}, got {resolved}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    disable_joint_delta_checks = joint_delta_checks_disabled_from_env()
    expected_checkpoint = expected_checkpoint_from_env()
    args.auth_token = None if args.auth_token is None else args.auth_token.strip() or None
    args.output_dir = resolve_local_output_dir(args.output_dir)
    inspection_modes = sum(
        bool(value)
        for value in (args.config_only, args.health_only, args.connect_smoke, args.inference_smoke)
    )
    if inspection_modes > 1:
        raise ValueError(
            "config-only, health-only, connect-smoke, and inference-smoke are mutually exclusive"
        )
    if args.connect_smoke and args.execution != DRY_RUN_EXECUTION:
        raise ValueError("connect-smoke is read-only and requires execution=dry_run")
    if args.inference_smoke and args.execution != DRY_RUN_EXECUTION:
        raise ValueError("inference-smoke is read-only and requires execution=dry_run")
    if args.inference_smoke and args.mode != "single_step":
        raise ValueError("inference-smoke requires mode=single_step")
    runtime_config = build_runtime_config(args)
    robot_config = build_robot_config(
        robot_id=args.robot_id,
        execution=args.execution,
        camera_profile=args.camera_profile,
        orin_ip=args.orin_ip,
        state_bind_ip=args.state_bind_ip,
        state_port=args.state_port,
        command_port=args.command_port,
        connect_timeout_s=args.connect_timeout_s,
        state_timeout_s=args.state_timeout_s,
        max_camera_state_receive_skew_ms=args.max_camera_state_receive_skew_ms,
        disable_joint_delta_checks=disable_joint_delta_checks,
    )
    require_armed_confirmation(
        execution=args.execution,
        transport=robot_config.send_action_transport,
    )
    if not args.connect_smoke:
        validate_server_url_security(runtime_config.server_url, args.auth_token)
    print_resolved_config(args, runtime_config, robot_config)
    if args.config_only:
        print(f"{LOG_PREFIX} CONFIG_ONLY PASS; no server or robot connection was made")
        return 0

    if args.connect_smoke:
        from lerobot.robots import make_robot_from_config

        robot = make_robot_from_config(robot_config)
        try:
            robot.connect()
            observation = robot.get_observation()
            print(f"{LOG_PREFIX} CONNECT_SMOKE PASS keys={sorted(observation)}")
            return 0
        finally:
            if getattr(robot, "is_connected", False):
                robot.disconnect()
                print(f"{LOG_PREFIX} robot disconnected")

    remote_policy = RemotePolicyClient(
        runtime_config.server_url,
        timeout_s=args.request_timeout_s,
        auth_token=args.auth_token,
    )
    health = remote_policy.health()
    validate_server_health(
        health,
        mode=runtime_config.mode,
        camera_profile=args.camera_profile,
        expected_checkpoint=expected_checkpoint,
    )
    print(f"{LOG_PREFIX} server health PASS checkpoint={health.get('checkpoint_path')}")
    if args.health_only:
        print(f"{LOG_PREFIX} HEALTH_ONLY PASS; no robot connection was made")
        return 0

    if args.inference_smoke:
        from lerobot.robots import make_robot_from_config

        robot = make_robot_from_config(robot_config)
        try:
            robot.connect()
            print(f"{LOG_PREFIX} robot connected type={robot.robot_type}")
            dataset_features, robot_action_processor, robot_observation_processor = build_dataset_artifacts(
                robot,
                camera_profile=args.camera_profile,
            )
            result = run_inference_smoke(
                config=runtime_config,
                remote_policy=remote_policy,
                robot_io=SerializedRobotIO(robot),
                dataset_features=dataset_features,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                safety=ActionSafety(),
            )
            print(
                f"{LOG_PREFIX} INFERENCE_SMOKE PASS send_action=not-called "
                + json.dumps(result.to_dict(), sort_keys=True)
            )
            metadata_path, observation_path = write_inference_smoke_artifacts(
                output_dir=args.output_dir,
                runtime_config=runtime_config,
                health=health,
                result=result,
            )
            print(
                f"{LOG_PREFIX} INFERENCE_SMOKE artifacts metadata={metadata_path} "
                f"observation={observation_path}"
            )
            return 0
        finally:
            if getattr(robot, "is_connected", False):
                robot.disconnect()
                print(f"{LOG_PREFIX} robot disconnected")

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output-dir: {args.output_dir}")
    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(robot_config)
    robot_io = SerializedRobotIO(robot)
    try:
        robot.connect()
        print(f"{LOG_PREFIX} robot connected type={robot.robot_type}")
        dataset_features, robot_action_processor, robot_observation_processor = build_dataset_artifacts(
            robot,
            camera_profile=args.camera_profile,
        )
        result = run_client_runtime(
            config=runtime_config,
            remote_policy=remote_policy,
            robot_io=robot_io,
            dataset_features=dataset_features,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            safety=ActionSafety(),
        )
        summary_path = write_summary(
            output_dir=args.output_dir,
            execution=args.execution,
            runtime_config=runtime_config,
            health=health,
            result=result,
        )
        print(
            f"{LOG_PREFIX} PASS sent={result.sent_actions} requests={result.inference_requests} "
            f"summary={summary_path}"
        )
        return 0
    finally:
        if getattr(robot, "is_connected", False):
            robot.disconnect()
            print(f"{LOG_PREFIX} robot disconnected")


if __name__ == "__main__":
    raise SystemExit(main())
