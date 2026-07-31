#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/p4_readonly_state_probe.json"

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from lerobot.robots.jz_robot_udp.state_cache import StateCache  # noqa: E402
from lerobot.robots.jz_robot_udp.udp_client import UDPStateReceiver  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P4 read-only JZ raw state probe; never creates a command sender."
    )
    parser.add_argument("--bind-ip", default="0.0.0.0")
    parser.add_argument("--state-port", type=int, default=39010)
    parser.add_argument("--expected-sender-ip", default="192.168.1.81")
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--min-packets", type=int, default=2)
    parser.add_argument("--min-rate-hz", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def run_probe(
    *,
    bind_ip: str,
    state_port: int,
    expected_sender_ip: str,
    duration_s: float,
    connect_timeout_s: float,
    min_packets: int,
    min_rate_hz: float,
) -> dict[str, Any]:
    _validate_inputs(
        bind_ip=bind_ip,
        state_port=state_port,
        expected_sender_ip=expected_sender_ip,
        duration_s=duration_s,
        connect_timeout_s=connect_timeout_s,
        min_packets=min_packets,
        min_rate_hz=min_rate_hz,
    )
    cache = StateCache()
    receiver = UDPStateReceiver(bind_ip, state_port, cache, label="p4_readonly_state")
    samples = []
    errors = []
    started_s = time.monotonic()
    try:
        receiver.start()
        first = cache.wait_after_revision(timeout_s=connect_timeout_s, after_revision=0)
        if first is None:
            errors.append(
                f"no valid state packet received on udp://{bind_ip}:{state_port} "
                f"within {connect_timeout_s:.3f}s"
            )
        else:
            samples.append(first)
            deadline_s = started_s + duration_s
            revision = first.revision
            while time.monotonic() < deadline_s:
                remaining_s = deadline_s - time.monotonic()
                item = cache.wait_after_revision(
                    timeout_s=max(0.0, min(0.25, remaining_s)),
                    after_revision=revision,
                )
                if item is None:
                    continue
                revision = item.revision
                samples.append(item)
    finally:
        receiver.stop()
    stopped_s = time.monotonic()

    senders = sorted({sample.sender[0] for sample in samples})
    unexpected_senders = [sender for sender in senders if sender != expected_sender_ip]
    if unexpected_senders:
        errors.append(f"unexpected state sender IPs: {unexpected_senders}")
    sequences = [int(sample.packet["seq"]) for sample in samples]
    source_stamp_ns = [int(sample.packet["stamp_ns"]) for sample in samples]
    sequence_regressions = sum(
        current <= previous for previous, current in zip(sequences, sequences[1:], strict=False)
    )
    stamp_regressions = sum(
        current < previous for previous, current in zip(source_stamp_ns, source_stamp_ns[1:], strict=False)
    )
    if sequence_regressions:
        errors.append(f"state sequence failed to advance {sequence_regressions} times")
    if stamp_regressions:
        errors.append(f"state source timestamp regressed {stamp_regressions} times")
    elapsed_s = stopped_s - started_s
    receive_intervals_ms = [
        (current.received_monotonic_s - previous.received_monotonic_s) * 1000.0
        for previous, current in zip(samples, samples[1:], strict=False)
    ]
    packet_rate_hz = len(samples) / elapsed_s if elapsed_s > 0 else 0.0
    if len(samples) < min_packets:
        errors.append(f"received {len(samples)} packets, below minimum {min_packets}")
    if packet_rate_hz < min_rate_hz:
        errors.append(f"packet rate {packet_rate_hz:.3f} Hz below minimum {min_rate_hz:.3f} Hz")

    raw_dimensions = []
    source_skew_ms = []
    source_timing_samples = 0
    robots = set()
    for sample in samples:
        packet = sample.packet
        robots.add(str(packet["robot"]))
        raw_dimensions.append(
            len(packet["joints"]["left"])
            + len(packet["joints"]["right"])
            + len(packet["grippers"]["left"])
            + len(packet["grippers"]["right"])
        )
        timing = packet.get("source_timing")
        if timing is not None:
            source_timing_samples += 1
            source_skew_ms.append(float(timing["source_skew_ms"]))
    wrong_dimensions = sorted({dimension for dimension in raw_dimensions if dimension != 18})
    if wrong_dimensions:
        errors.append(f"state packets do not map to raw18 dimensions: {wrong_dimensions}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "phase": "P4_read_only",
        "bind": f"udp://{bind_ip}:{state_port}",
        "expected_sender_ip": expected_sender_ip,
        "duration_s": elapsed_s,
        "packet_count": len(samples),
        "packet_rate_hz": packet_rate_hz,
        "sender_ips": senders,
        "robots": sorted(robots),
        "first_sequence": None if not sequences else sequences[0],
        "last_sequence": None if not sequences else sequences[-1],
        "sequence_regressions": sequence_regressions,
        "source_stamp_regressions": stamp_regressions,
        "receive_interval_ms": _distribution(receive_intervals_ms),
        "raw18_dimensions_exact": bool(samples) and not wrong_dimensions,
        "source_timing_samples": source_timing_samples,
        "source_skew_ms": _distribution(source_skew_ms),
        "receiver_stopped": not receiver.is_running,
        "command_socket_created": False,
        "command_port_opened": False,
        "action_sink_created": False,
        "action_sent": False,
        "errors": errors,
    }


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _validate_inputs(**values: object) -> None:
    bind_ip = values["bind_ip"]
    expected_sender_ip = values["expected_sender_ip"]
    if not isinstance(bind_ip, str) or not bind_ip.strip():
        raise ValueError("bind_ip must be a non-empty string")
    if not isinstance(expected_sender_ip, str) or not expected_sender_ip.strip():
        raise ValueError("expected_sender_ip must be a non-empty string")
    state_port = values["state_port"]
    if isinstance(state_port, bool) or not isinstance(state_port, int) or not 1 <= state_port <= 65535:
        raise ValueError("state_port must be an integer in 1..65535")
    min_packets = values["min_packets"]
    if isinstance(min_packets, bool) or not isinstance(min_packets, int) or min_packets <= 0:
        raise ValueError("min_packets must be a positive integer")
    for name in ("duration_s", "connect_timeout_s", "min_rate_hz"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} must be a real number")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")


def _write_report(path: Path, report: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(OPTIMIZED_ROOT.resolve()):
        raise ValueError(f"output-json must stay inside {OPTIMIZED_ROOT}, got {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_probe(
        bind_ip=args.bind_ip,
        state_port=args.state_port,
        expected_sender_ip=args.expected_sender_ip,
        duration_s=args.duration_s,
        connect_timeout_s=args.connect_timeout_s,
        min_packets=args.min_packets,
        min_rate_hz=args.min_rate_hz,
    )
    output_path = _write_report(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
