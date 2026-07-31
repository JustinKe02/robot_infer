#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
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
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from tk_infer.pi05.runtime.checkpoint import (  # noqa: E402
    inspect_checkpoint,
)
from tk_infer.pi05.runtime.http_server import (  # noqa: E402
    DEFAULT_MAX_REQUEST_BYTES,
    make_server,
    validate_bind_security,
)
from tk_infer.pi05.runtime.policy_service import (  # noqa: E402
    PolicyService,
    PolicyServiceConfig,
)

DEFAULT_TOKENIZER_PATH = REPO_ROOT / "assets/modelscope/google/paligemma-3b-pt-224"
LOG_PREFIX = "[tk_infer/pi05/server]"


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
    policy_default = os.getenv("POLICY_PATH")
    parser = argparse.ArgumentParser(
        description="Pure-PyTorch PI0.5 single-step and RTC policy server for JZ Robot Pin Timed."
    )
    parser.add_argument("--host", default=os.getenv("JZ_PI05_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("JZ_PI05_SERVER_PORT", "8088")))
    parser.add_argument(
        "--policy-path",
        default=policy_default,
        help="Completed pretrained_model directory; required unless --config-only is used.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=os.getenv("PALIGEMMA_TOKENIZER_PATH", DEFAULT_TOKENIZER_PATH.as_posix()),
    )
    parser.add_argument("--device", default=os.getenv("DEVICE", "cuda"))
    parser.add_argument("--auth-token", default=os.getenv("JZ_PI05_SERVER_AUTH_TOKEN"))
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=int(os.getenv("JZ_PI05_MAX_REQUEST_BYTES", str(DEFAULT_MAX_REQUEST_BYTES))),
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=int(os.getenv("RTC_EXECUTION_HORIZON", "10")),
    )
    parser.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=float(os.getenv("RTC_MAX_GUIDANCE_WEIGHT", "10.0")),
    )
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=[schedule.value for schedule in RTCAttentionSchedule],
        default=os.getenv("RTC_PREFIX_ATTENTION_SCHEDULE", RTCAttentionSchedule.LINEAR.value),
    )
    parser.add_argument(
        "--rtc-debug",
        type=parse_bool,
        nargs="?",
        const=True,
        default=parse_bool(os.getenv("RTC_DEBUG", "false")),
    )
    parser.add_argument(
        "--require-complete-step",
        type=parse_bool,
        nargs="?",
        const=True,
        default=parse_bool(os.getenv("REQUIRE_COMPLETE_STEP", "true")),
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Validate arguments/bind security without reading checkpoint/tokenizer files or listening.",
    )
    parser.add_argument(
        "--check-policy-load",
        action="store_true",
        help="Load the model and processors, print health metadata, then exit without listening.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config_only and args.check_policy_load:
        raise ValueError("config-only and check-policy-load are mutually exclusive")
    if not 0 <= args.port <= 65535:
        raise ValueError("port must be in 0..65535")
    if args.max_request_bytes <= 0:
        raise ValueError("max-request-bytes must be positive")
    if args.rtc_execution_horizon <= 0:
        raise ValueError("rtc-execution-horizon must be positive")
    if not args.device.strip():
        raise ValueError("device must be non-empty")
    auth_token = validate_bind_security(args.host, args.auth_token)
    print(f"{LOG_PREFIX} Repo root: {REPO_ROOT}")
    print(
        f"{LOG_PREFIX} Bind: http://{args.host}:{args.port} "
        f"authentication={'enabled' if auth_token else 'loopback-only'}"
    )
    if args.config_only:
        rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=RTCAttentionSchedule(args.rtc_prefix_attention_schedule),
            max_guidance_weight=args.rtc_max_guidance_weight,
            execution_horizon=args.rtc_execution_horizon,
            debug=args.rtc_debug,
        )
        payload = {
            "config_only": True,
            "host": args.host,
            "port": args.port,
            "authentication_configured": auth_token is not None,
            "max_request_bytes": args.max_request_bytes,
            "device": args.device,
            "policy_path": args.policy_path,
            "tokenizer_path": args.tokenizer_path,
            "require_complete_step": args.require_complete_step,
            "supported_modes": ["single_step", "rtc"],
            "rtc_execution_horizon": rtc_config.execution_horizon,
            "rtc_max_guidance_weight": rtc_config.max_guidance_weight,
            "rtc_prefix_attention_schedule": rtc_config.prefix_attention_schedule.value,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        print(
            f"{LOG_PREFIX} CONFIG_ONLY passed; no checkpoint/tokenizer file was read, "
            "no model was loaded, and no socket was opened."
        )
        return 0

    if not args.policy_path:
        raise ValueError("policy-path is required unless --config-only is used")
    metadata, _ = inspect_checkpoint(
        args.policy_path,
        tokenizer_path=args.tokenizer_path,
        require_complete_step=args.require_complete_step,
    )
    print(f"{LOG_PREFIX} Policy: {metadata.policy_path}")
    print(f"{LOG_PREFIX} Tokenizer: {Path(args.tokenizer_path).expanduser()}")
    print(
        f"{LOG_PREFIX} Checkpoint: fingerprint={metadata.checkpoint_fingerprint} "
        f"step={metadata.checkpoint_step}/{metadata.configured_steps}"
    )

    service_config = PolicyServiceConfig(
        policy_path=metadata.policy_path,
        tokenizer_path=Path(args.tokenizer_path).expanduser(),
        device=args.device,
        require_complete_step=args.require_complete_step,
    )
    rtc_config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule(args.rtc_prefix_attention_schedule),
        max_guidance_weight=args.rtc_max_guidance_weight,
        execution_horizon=args.rtc_execution_horizon,
        debug=args.rtc_debug,
    )
    print(f"{LOG_PREFIX} Loading PI0.5 policy on {args.device}; no inference is run during startup.")
    service = PolicyService.from_config(service_config, rtc_config=rtc_config)
    print(json.dumps(service.health(), indent=2, ensure_ascii=False, sort_keys=True))
    if args.check_policy_load:
        print(f"{LOG_PREFIX} CHECK_POLICY_LOAD passed; no inference request was executed.")
        return 0

    http_server = make_server(
        host=args.host,
        port=args.port,
        service=service,
        auth_token=auth_token,
        max_request_bytes=args.max_request_bytes,
    )
    print(f"{LOG_PREFIX} Listening on http://{args.host}:{args.port}; modes=single_step,rtc")
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        print(f"{LOG_PREFIX} KeyboardInterrupt received, shutting down.")
    finally:
        http_server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
