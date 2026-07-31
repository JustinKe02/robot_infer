#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


def resolve_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from tk_infer.pi05.runtime.protocol import PROTOCOL_VERSION, InferenceRequest, InferenceResponse  # noqa: E402
from tk_infer.pi05.runtime.remote_client import RemotePolicyClient  # noqa: E402
from tk_infer.pi05_optimized.runtime.local_tracker import (  # noqa: E402
    LocalActionTracker,
    LocalTrackerConfig,
)
from tk_infer.pi05_optimized.runtime.optimized_client import (  # noqa: E402
    OptimizedClient,
    OptimizedClientConfig,
)
from tk_infer.pi05_optimized.runtime.timed_observation import TimedObservation  # noqa: E402

DEFAULT_SERVER_URL = "http://127.0.0.1:18088"
DEFAULT_TASK = "jz robot pin timed vr teleoperation"
LOG_PREFIX = "[tk_infer/pi05_optimized/client]"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="No-hardware client entry point for optimized PI0.5 inference."
    )
    parser.add_argument("--mode", choices=["single_step", "rtc"], default="rtc")
    parser.add_argument("--task", default=os.getenv("PI05_OPT_TASK", DEFAULT_TASK))
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument(
        "--server-url",
        default=os.getenv("JZ_PI05_OPT_SERVER_URL", DEFAULT_SERVER_URL),
    )
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--local-tracker", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--config-only", action="store_true")
    action.add_argument("--health-only", action="store_true")
    action.add_argument("--offline-smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = OptimizedClientConfig(
        task=args.task,
        mode=args.mode,
        control_hz=args.control_hz,
        execution_horizon=args.execution_horizon,
        local_tracker_enabled=args.local_tracker,
    )
    payload = {
        "config_only": not args.health_only and not args.offline_smoke,
        "mode": config.mode,
        "task": config.task,
        "control_hz": config.control_hz,
        "execution_horizon": config.execution_horizon,
        "server_url": args.server_url,
        "hardware_adapter": None,
        "action_transport": None,
        "armed_capability": False,
        "local_tracker_enabled": config.local_tracker_enabled,
        "mpc_enabled": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.health_only:
        client = RemotePolicyClient(
            args.server_url,
            timeout_s=args.timeout_s,
            auth_token=os.getenv("JZ_PI05_OPT_SERVER_AUTH_TOKEN"),
        )
        health = client.health()
        if health.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(
                f"optimized server protocol mismatch: {health.get('protocol_version')} != {PROTOCOL_VERSION}"
            )
        if health.get("optimized_runtime") is not True:
            raise RuntimeError("server health does not identify the optimized runtime")
        print(json.dumps(health, indent=2, sort_keys=True))
        print(f"{LOG_PREFIX} HEALTH_ONLY passed; no observation source or action sink was created.")
        return 0

    if args.offline_smoke:
        report = _run_offline_smoke(config)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"{LOG_PREFIX} OFFLINE_SMOKE passed; in-memory source/policy/sink only.")
        return 0

    print(f"{LOG_PREFIX} CONFIG_ONLY passed; no socket, observation source, or action sink was created.")
    return 0


@dataclass
class _OfflineObservationSource:
    observation: TimedObservation
    reads: int = 0

    def read(self) -> TimedObservation:
        if self.reads:
            raise RuntimeError("offline observation source supports one cycle")
        self.reads += 1
        return self.observation


@dataclass
class _OfflinePolicyClient:
    requests: list[InferenceRequest] = field(default_factory=list)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        model_actions = np.arange(3 * 16, dtype=np.float32).reshape(3, 16)
        robot_actions = np.zeros((3, 18), dtype=np.float32)
        robot_actions[:, :14] = model_actions[:, :14]
        robot_actions[:, 14] = model_actions[:, 14]
        robot_actions[:, 15] = 80.0
        robot_actions[:, 16] = model_actions[:, 15]
        robot_actions[:, 17] = 80.0
        return InferenceResponse(
            request_id=request.request_id,
            mode=request.mode,
            raw_actions=model_actions,
            processed_actions=robot_actions,
            server_latency_s=0.005,
            model_latency_s=0.004,
            raw_action_shape=model_actions.shape,
            processed_action_shape=robot_actions.shape,
        )


@dataclass
class _OfflineActionSink:
    actions: list[torch.Tensor] = field(default_factory=list)

    def write(self, action: torch.Tensor) -> None:
        self.actions.append(action.detach().clone())


def _run_offline_smoke(config: OptimizedClientConfig) -> dict[str, object]:
    source = _OfflineObservationSource(
        TimedObservation(
            observation_frame={"observation.state": np.zeros(18, dtype=np.float32)},
            sequence_id=1,
            receive_monotonic_s=1.0,
            build_started_monotonic_s=1.0,
            build_ready_monotonic_s=1.0,
        )
    )
    policy = _OfflinePolicyClient()
    sink = _OfflineActionSink()
    clock = iter((1.0, 1.01, 1.02))
    client = OptimizedClient(
        config=config,
        observation_source=source,
        policy_client=policy,
        action_sink=sink,
        local_tracker=(
            LocalActionTracker(
                LocalTrackerConfig(control_period_s=config.control_period_s)
            )
            if config.local_tracker_enabled
            else None
        ),
        clock=lambda: next(clock),
    )
    result = client.run_cycle()
    return {
        "status": "PASS",
        "hardware_access": False,
        "network_access": False,
        "request_count": len(policy.requests),
        "action_count": len(sink.actions),
        "action_shape": list(sink.actions[0].shape),
        "force_slots": [float(sink.actions[0][15]), float(sink.actions[0][17])],
        "cycle": {
            "request_id": result.request_id,
            "dropped_steps": result.dropped_steps,
            "queue_depth": result.queue_depth,
            "action_written": result.action_written,
        },
        "telemetry": client.telemetry.snapshot().to_dict(),
        "local_tracker": None if client.local_tracker is None else client.local_tracker.health(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
