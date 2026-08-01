#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path


def resolve_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from lerobot.configs.types import RTCAttentionSchedule  # noqa: E402
from tk_infer.pi05.runtime.checkpoint import inspect_checkpoint  # noqa: E402
from tk_infer.pi05.runtime.http_server import make_server, validate_bind_security  # noqa: E402
from tk_infer.pi05_optimized.backends.realtime_vla_v2_backend import (  # noqa: E402
    RealtimeVLAV2PolicyBackend,
)
from tk_infer.pi05_optimized.backends.torch_backend import TorchPolicyBackend  # noqa: E402
from tk_infer.pi05_optimized.backends.torch_optimized_backend import (  # noqa: E402
    TorchOptimizedBackend,
)
from tk_infer.pi05_optimized.backends.torch_rtc_conditioned_backend import (  # noqa: E402
    TorchRTCConditionedBackend,
)
from tk_infer.pi05_optimized.backends.triton_backend import TritonPolicyBackend  # noqa: E402
from tk_infer.pi05_optimized.config import (  # noqa: E402
    DEFAULT_MAX_REQUEST_BYTES,
    DEFAULT_METRICS_WINDOW_SIZE,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    OptimizedRuntimeConfig,
)
from tk_infer.pi05_optimized.runtime.metrics import InferenceMetrics  # noqa: E402
from tk_infer.pi05_optimized.runtime.policy_service import OptimizedPolicyService  # noqa: E402
from tk_infer.pi05_optimized.runtime.trace import JsonlTraceWriter  # noqa: E402
from tk_infer.pi05_optimized.runtime.trajectory_processor import (  # noqa: E402
    build_trajectory_processor,
)

DEFAULT_TOKENIZER_PATH = REPO_ROOT / "assets/modelscope/google/paligemma-3b-pt-224"
DEFAULT_TRACE_PATH = REPO_ROOT / "tk_infer/pi05_optimized/logs/server/inference_trace.jsonl"
LOG_PREFIX = "[tk_infer/pi05_optimized/server]"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baseline-preserving optimized PI0.5 policy server for JZ Robot Pin Timed."
    )
    parser.add_argument("--host", default=os.getenv("JZ_PI05_OPT_SERVER_HOST", DEFAULT_SERVER_HOST))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("JZ_PI05_OPT_SERVER_PORT", str(DEFAULT_SERVER_PORT))),
    )
    parser.add_argument(
        "--backend",
        choices=[
            "torch",
            "torch_optimized",
            "torch_rtc_conditioned",
            "triton",
            "realtime_vla_v2",
        ],
        default=os.getenv("PI05_OPT_BACKEND", "torch"),
    )
    parser.add_argument(
        "--trajectory-processor",
        choices=["pass_through", "paired_temporal"],
        default=os.getenv("PI05_OPT_TRAJECTORY_PROCESSOR", "pass_through"),
    )
    parser.add_argument(
        "--policy-path",
        default=os.getenv("PI05_OPT_POLICY_PATH"),
        help="Completed pretrained_model directory; required unless --config-only is used.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=os.getenv("PI05_OPT_TOKENIZER_PATH", DEFAULT_TOKENIZER_PATH.as_posix()),
    )
    parser.add_argument(
        "--triton-artifact-path",
        type=Path,
        default=os.getenv("PI05_OPT_TRITON_ARTIFACT_PATH"),
    )
    parser.add_argument(
        "--realtime-vla-v2-artifact-path",
        type=Path,
        default=os.getenv("PI05_OPT_REALTIME_VLA_V2_ARTIFACT_PATH"),
    )
    parser.add_argument("--device", default=os.getenv("PI05_OPT_DEVICE", "cuda"))
    parser.add_argument("--auth-token", default=os.getenv("JZ_PI05_OPT_SERVER_AUTH_TOKEN"))
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=int(os.getenv("JZ_PI05_OPT_MAX_REQUEST_BYTES", str(DEFAULT_MAX_REQUEST_BYTES))),
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=int(os.getenv("PI05_OPT_RTC_EXECUTION_HORIZON", "10")),
    )
    parser.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=float(os.getenv("PI05_OPT_RTC_MAX_GUIDANCE_WEIGHT", "10.0")),
    )
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=[schedule.value for schedule in RTCAttentionSchedule],
        default=os.getenv("PI05_OPT_RTC_PREFIX_ATTENTION_SCHEDULE", RTCAttentionSchedule.LINEAR.value),
    )
    parser.add_argument(
        "--rtc-debug",
        type=parse_bool,
        nargs="?",
        const=True,
        default=parse_bool(os.getenv("PI05_OPT_RTC_DEBUG", "false")),
    )
    parser.add_argument(
        "--rtc-conditioned-task",
        default=os.getenv("PI05_OPT_RTC_CONDITIONED_TASK"),
        help="Exact training task accepted by a training-time RTC-conditioned backend.",
    )
    parser.add_argument(
        "--require-complete-step",
        type=parse_bool,
        nargs="?",
        const=True,
        default=parse_bool(os.getenv("PI05_OPT_REQUIRE_COMPLETE_STEP", "true")),
    )
    parser.add_argument(
        "--metrics-window-size",
        type=int,
        default=int(os.getenv("PI05_OPT_METRICS_WINDOW_SIZE", str(DEFAULT_METRICS_WINDOW_SIZE))),
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=Path(os.getenv("PI05_OPT_TRACE_PATH", str(DEFAULT_TRACE_PATH))),
    )
    parser.add_argument(
        "--trace-strict",
        type=parse_bool,
        nargs="?",
        const=True,
        default=parse_bool(os.getenv("PI05_OPT_TRACE_STRICT", "false")),
    )
    parser.add_argument(
        "--trace-max-bytes",
        type=int,
        default=int(os.getenv("PI05_OPT_TRACE_MAX_BYTES", str(16 * 1024 * 1024))),
    )
    parser.add_argument(
        "--trace-backup-count",
        type=int,
        default=int(os.getenv("PI05_OPT_TRACE_BACKUP_COUNT", "2")),
    )
    parser.add_argument(
        "--torch-inference-mode",
        type=parse_bool,
        nargs="?",
        const=True,
        default=parse_bool(os.getenv("PI05_OPT_TORCH_INFERENCE_MODE", "false")),
    )
    parser.add_argument(
        "--torch-bf16-autocast",
        type=parse_bool,
        nargs="?",
        const=True,
        default=parse_bool(os.getenv("PI05_OPT_TORCH_BF16_AUTOCAST", "false")),
    )
    parser.add_argument(
        "--torch-pinned-memory",
        type=parse_bool,
        nargs="?",
        const=True,
        default=parse_bool(os.getenv("PI05_OPT_TORCH_PINNED_MEMORY", "false")),
    )
    parser.add_argument(
        "--torch-non-blocking-copies",
        type=parse_bool,
        nargs="?",
        const=True,
        default=parse_bool(os.getenv("PI05_OPT_TORCH_NON_BLOCKING_COPIES", "false")),
    )
    parser.add_argument(
        "--torch-static-buffers",
        type=parse_bool,
        nargs="?",
        const=True,
        default=parse_bool(os.getenv("PI05_OPT_TORCH_STATIC_BUFFERS", "false")),
    )
    parser.add_argument(
        "--torch-cuda-graph",
        type=parse_bool,
        nargs="?",
        const=True,
        default=parse_bool(os.getenv("PI05_OPT_TORCH_CUDA_GRAPH", "false")),
    )
    parser.add_argument(
        "--torch-warmup-iterations",
        type=int,
        default=int(os.getenv("PI05_OPT_TORCH_WARMUP_ITERATIONS", "0")),
    )
    parser.add_argument(
        "--torch-warmup-seed",
        type=int,
        default=int(os.getenv("PI05_OPT_TORCH_WARMUP_SEED", "12345")),
    )
    parser.add_argument(
        "--temporal-speed-factor",
        type=float,
        default=float(os.getenv("PI05_OPT_TEMPORAL_SPEED_FACTOR", "1.0")),
    )
    parser.add_argument(
        "--temporal-max-joint-step-rad",
        type=float,
        default=float(os.getenv("PI05_OPT_TEMPORAL_MAX_JOINT_STEP_RAD", "0.02")),
    )
    parser.add_argument(
        "--temporal-solver-timeout-s",
        type=float,
        default=float(os.getenv("PI05_OPT_TEMPORAL_SOLVER_TIMEOUT_S", "0.05")),
    )
    parser.add_argument("--disable-trace", action="store_true")
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Validate configuration without reading checkpoint/tokenizer files or opening a socket.",
    )
    parser.add_argument(
        "--check-policy-load",
        action="store_true",
        help="Load the reference backend and print health, without inference or socket listen.",
    )
    return parser


def build_runtime_config(args: argparse.Namespace) -> OptimizedRuntimeConfig:
    return OptimizedRuntimeConfig(
        backend=args.backend,
        trajectory_processor=args.trajectory_processor,
        server_host=args.host,
        server_port=args.port,
        max_request_bytes=args.max_request_bytes,
        device=args.device,
        policy_path=args.policy_path,
        tokenizer_path=args.tokenizer_path,
        triton_artifact_path=args.triton_artifact_path,
        realtime_vla_v2_artifact_path=args.realtime_vla_v2_artifact_path,
        require_complete_step=args.require_complete_step,
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
        rtc_debug=args.rtc_debug,
        rtc_conditioned_task=args.rtc_conditioned_task,
        metrics_window_size=args.metrics_window_size,
        trace_path=None if args.disable_trace else args.trace_path,
        trace_strict=args.trace_strict,
        trace_max_bytes=args.trace_max_bytes,
        trace_backup_count=args.trace_backup_count,
        torch_inference_mode=args.torch_inference_mode,
        torch_bf16_autocast=args.torch_bf16_autocast,
        torch_pinned_memory=args.torch_pinned_memory,
        torch_non_blocking_copies=args.torch_non_blocking_copies,
        torch_static_buffers=args.torch_static_buffers,
        torch_cuda_graph=args.torch_cuda_graph,
        torch_warmup_iterations=args.torch_warmup_iterations,
        torch_warmup_seed=args.torch_warmup_seed,
        temporal_speed_factor=args.temporal_speed_factor,
        temporal_max_joint_step_rad=args.temporal_max_joint_step_rad,
        temporal_solver_timeout_s=args.temporal_solver_timeout_s,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config_only and args.check_policy_load:
        raise ValueError("config-only and check-policy-load are mutually exclusive")
    config = build_runtime_config(args)
    if args.check_policy_load and config.torch_warmup_iterations:
        raise ValueError(
            "check-policy-load requires torch_warmup_iterations=0 because it promises no inference"
        )
    auth_token = validate_bind_security(config.server_host, args.auth_token)
    print(f"{LOG_PREFIX} Repo root: {REPO_ROOT}")
    print(
        f"{LOG_PREFIX} Bind: http://{config.server_host}:{config.server_port} "
        f"authentication={'enabled' if auth_token else 'loopback-only'}"
    )

    if args.config_only:
        single_step_only = config.backend == "triton" or (
            config.backend == "torch_optimized" and config.torch_inference_mode
        )
        supported_modes = ["single_step"] if single_step_only else ["single_step", "rtc"]
        payload = {
            "config_only": True,
            **config.public_dict(),
            "authentication_configured": auth_token is not None,
            "supported_modes": supported_modes,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        print(
            f"{LOG_PREFIX} CONFIG_ONLY passed; no checkpoint/tokenizer file was read, "
            "no model was loaded, and no socket was opened."
        )
        return 0

    policy_path, tokenizer_path = config.require_model_paths()
    metadata, _ = inspect_checkpoint(
        policy_path,
        tokenizer_path=tokenizer_path,
        require_complete_step=config.require_complete_step,
    )
    config = replace(config, policy_path=metadata.policy_path)
    print(f"{LOG_PREFIX} Policy: {metadata.policy_path}")
    print(f"{LOG_PREFIX} Tokenizer: {tokenizer_path}")
    print(
        f"{LOG_PREFIX} Checkpoint: fingerprint={metadata.checkpoint_fingerprint} "
        f"step={metadata.checkpoint_step}/{metadata.configured_steps}"
    )
    print(
        f"{LOG_PREFIX} Loading backend={config.backend} on {config.device}; "
        f"warmup_iterations={config.torch_warmup_iterations}."
    )
    if config.backend == "torch":
        backend = TorchPolicyBackend.from_runtime_config(config)
    elif config.backend == "torch_optimized":
        backend = TorchOptimizedBackend.from_runtime_config(config)
    elif config.backend == "torch_rtc_conditioned":
        backend = TorchRTCConditionedBackend.from_runtime_config(config)
    elif config.backend == "triton":
        backend = TritonPolicyBackend.from_runtime_config(config)
    else:
        backend = RealtimeVLAV2PolicyBackend.from_runtime_config(config)
    service = OptimizedPolicyService(
        backend=backend,
        trajectory_processor=build_trajectory_processor(config),
        metrics=InferenceMetrics(window_size=config.metrics_window_size),
        trace_recorder=(
            None
            if config.trace_path is None
            else JsonlTraceWriter(
                config.trace_path,
                strict=config.trace_strict,
                max_bytes=config.trace_max_bytes,
                backup_count=config.trace_backup_count,
            )
        ),
    )
    print(json.dumps(service.health(), indent=2, ensure_ascii=False, sort_keys=True))
    if args.check_policy_load:
        print(f"{LOG_PREFIX} CHECK_POLICY_LOAD passed; no inference request or socket was created.")
        return 0

    server = make_server(
        host=config.server_host,
        port=config.server_port,
        service=service,
        auth_token=auth_token,
        max_request_bytes=config.max_request_bytes,
    )
    print(
        f"{LOG_PREFIX} Listening on http://{config.server_host}:{config.server_port}; "
        f"backend={config.backend} trajectory_processor={config.trajectory_processor}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"{LOG_PREFIX} KeyboardInterrupt received, shutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
