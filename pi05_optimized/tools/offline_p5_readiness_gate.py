#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def resolve_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
OPTIMIZED_ROOT = REPO_ROOT / "tk_infer/pi05_optimized"
DEFAULT_BACKEND_AB = OPTIMIZED_ROOT / "outputs/live/live_backend_ab_step015900_20260730.json"
DEFAULT_TRACKER_REPLAY = OPTIMIZED_ROOT / "outputs/phase7_tracker_replay.json"
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/p5_software_readiness.json"

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from tk_infer.pi05_optimized.runtime.p5_readiness import evaluate_p5_readiness  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline P5 single-action readiness gate; never creates hardware or action transport."
    )
    parser.add_argument("--backend-ab-json", type=Path, default=DEFAULT_BACKEND_AB)
    parser.add_argument("--tracker-replay-json", type=Path, default=DEFAULT_TRACKER_REPLAY)
    parser.add_argument(
        "--authorization-json",
        type=Path,
        help="Optional evidence only; this tool cannot authorize or execute motion.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    backend_path = args.backend_ab_json.expanduser().resolve(strict=True)
    tracker_path = args.tracker_replay_json.expanduser().resolve(strict=True)
    authorization_path = (
        None if args.authorization_json is None else args.authorization_json.expanduser().resolve(strict=True)
    )
    backend_ab = _read_json_object(backend_path)
    tracker_replay = _read_json_object(tracker_path)
    authorization = None if authorization_path is None else _read_json_object(authorization_path)
    decision = evaluate_p5_readiness(
        backend_ab=backend_ab,
        tracker_replay=tracker_replay,
        authorization=authorization,
    )
    return {
        "schema_version": 1,
        "phase": "P5_software_readiness",
        "status": decision.status,
        "hardware_access": False,
        "network_access": False,
        "robot_created": False,
        "action_transport_created": False,
        "action_sent": False,
        "armed_launcher_created": False,
        "source_evidence": {
            "backend_ab_json": str(backend_path),
            "tracker_replay_json": str(tracker_path),
            "authorization_json": None if authorization_path is None else str(authorization_path),
        },
        "decision": decision.to_dict(),
        "authorization_note": (
            "READY only means the evidence is eligible for a separately controlled P5 trial; this tool never "
            "creates a Robot, socket, command transport, or action."
        ),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def _write_report(path: Path, report: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(OPTIMIZED_ROOT.resolve()):
        raise ValueError(f"output-json must stay inside {OPTIMIZED_ROOT}, got {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gate(args)
    output_path = _write_report(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
