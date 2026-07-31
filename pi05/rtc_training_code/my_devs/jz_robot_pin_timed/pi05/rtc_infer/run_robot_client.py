#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


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

from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.client_runtime import (  # noqa: E402
    ClientRuntimeConfig,
    run_client_runtime,
)
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.protocol import (  # noqa: E402
    PROTOCOL_VERSION,
)
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.remote_client import (  # noqa: E402
    RemotePolicyClient,
)
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.robot_builder import (  # noqa: E402
    DEFAULT_COMMAND_PORT,
    DEFAULT_CONTROL_FPS,
    DEFAULT_ORIN_IP,
    DEFAULT_STATE_BIND_IP,
    DEFAULT_STATE_PORT,
    build_dataset_artifacts,
    build_robot_config,
)
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.robot_io import (  # noqa: E402
    SerializedRobotIO,
)
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.safety import (  # noqa: E402
    ARMED_EXECUTION,
    DRY_RUN_EXECUTION,
    ActionSafety,
    require_armed_confirmation,
    transport_for_execution,
)

LOG_PREFIX = "[jz/pi05/rtc-client]"
DEFAULT_TASK = "jz robot pin timed vr teleoperation"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "client"
EXPECTED_CAMERA_KEYS = (
    "observation.images.camera_head",
    "observation.images.camera_left",
    "observation.images.camera_right",
)
EXPECTED_CAMERA_SHAPES = {
    "observation.images.camera_head": (3, 720, 1280),
    "observation.images.camera_left": (3, 480, 640),
    "observation.images.camera_right": (3, 480, 640),
}


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
    parser.add_argument("--mode", choices=("single_step", "rtc"), default=os.getenv("MODE", "rtc"))
    parser.add_argument(
        "--execution",
        choices=(DRY_RUN_EXECUTION, ARMED_EXECUTION),
        default=os.getenv("EXECUTION", DRY_RUN_EXECUTION),
    )
    parser.add_argument(
        "--robot-id",
        default=os.getenv("ROBOT_ID", "jz_robot_pin_timed_pi05_rtc"),
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


def validate_server_health(health: dict[str, Any], *, mode: str) -> None:
    expected = {
        "ok": True,
        "protocol_version": PROTOCOL_VERSION,
        "policy_type": "pi05",
        "model_state_dim": 16,
        "model_action_dim": 16,
        "wire_action_dim": 18,
        "schema_id": "jz_pin_opening16_v1",
        "schema_version": 1,
        "complete_step": True,
    }
    mismatches = {
        key: {"expected": value, "actual": health.get(key)}
        for key, value in expected.items()
        if health.get(key) != value
    }
    if tuple(health.get("camera_keys", ())) != EXPECTED_CAMERA_KEYS:
        mismatches["camera_keys"] = {
            "expected": list(EXPECTED_CAMERA_KEYS),
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
    if normalized_camera_shapes != EXPECTED_CAMERA_SHAPES:
        mismatches["camera_shapes"] = {
            "expected": {key: list(value) for key, value in EXPECTED_CAMERA_SHAPES.items()},
            "actual": actual_camera_shapes,
        }
    if mode not in health.get("supported_modes", ()):
        mismatches["supported_modes"] = {
            "expected_contains": mode,
            "actual": health.get("supported_modes"),
        }
    if mismatches:
        raise RuntimeError(f"Policy server health is incompatible with JZ PI0.5 client: {mismatches}")


def print_resolved_config(
    args: argparse.Namespace,
    runtime_config: ClientRuntimeConfig,
    robot_config: Any,
) -> None:
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
        f"{LOG_PREFIX} cameras=zmq://{args.orin_ip}:5555,5556,5557 "
        "schema=raw18->model16->raw18 force=80"
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


def resolve_local_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    if not resolved.is_relative_to(SCRIPT_DIR):
        raise ValueError(f"output-dir must stay inside {SCRIPT_DIR}, got {resolved}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.auth_token = None if args.auth_token is None else args.auth_token.strip() or None
    args.output_dir = resolve_local_output_dir(args.output_dir)
    inspection_modes = sum(bool(value) for value in (args.config_only, args.health_only, args.connect_smoke))
    if inspection_modes > 1:
        raise ValueError("config-only, health-only, and connect-smoke are mutually exclusive")
    if args.connect_smoke and args.execution != DRY_RUN_EXECUTION:
        raise ValueError("connect-smoke is read-only and requires execution=dry_run")

    runtime_config = build_runtime_config(args)
    robot_config = build_robot_config(
        robot_id=args.robot_id,
        execution=args.execution,
        orin_ip=args.orin_ip,
        state_bind_ip=args.state_bind_ip,
        state_port=args.state_port,
        command_port=args.command_port,
        connect_timeout_s=args.connect_timeout_s,
        state_timeout_s=args.state_timeout_s,
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
    validate_server_health(health, mode=runtime_config.mode)
    print(f"{LOG_PREFIX} server health PASS checkpoint={health.get('checkpoint_path')}")
    if args.health_only:
        print(f"{LOG_PREFIX} HEALTH_ONLY PASS; no robot connection was made")
        return 0

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output-dir: {args.output_dir}")
    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(robot_config)
    try:
        robot.connect()
        print(f"{LOG_PREFIX} robot connected type={robot.robot_type}")
        dataset_features, robot_action_processor, robot_observation_processor = build_dataset_artifacts(robot)
        result = run_client_runtime(
            config=runtime_config,
            remote_policy=remote_policy,
            robot_io=SerializedRobotIO(robot),
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
