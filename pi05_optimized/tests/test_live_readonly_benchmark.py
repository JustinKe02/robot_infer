from __future__ import annotations

from pathlib import Path

import pytest

from tk_infer.pi05_optimized.tools import live_readonly_benchmark

OPTIMIZED_ROOT = Path(__file__).resolve().parents[1]


def test_parser_defaults_to_bounded_head_right_live_measurement() -> None:
    args = live_readonly_benchmark.build_parser().parse_args([])

    assert args.server_url == "http://127.0.0.1:18088"
    assert args.state_port == 39010
    assert args.warmup_requests == 3
    assert args.measure_requests == 30
    assert args.control_hz == 5.0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--server-url=http://192.168.1.1:18088",), "loopback"),
        (("--warmup-requests=-1",), "warmup_requests"),
        (("--measure-requests=0",), "measure_requests"),
        (("--control-hz=31",), "control_hz"),
    ],
)
def test_argument_validation_fails_closed(arguments: tuple[str, ...], message: str) -> None:
    args = live_readonly_benchmark.build_parser().parse_args(arguments)

    with pytest.raises(ValueError, match=message):
        live_readonly_benchmark._validate_args(args)


def test_health_requires_latest_complete_optimized_checkpoint() -> None:
    good = {
        "ok": True,
        "optimized_runtime": True,
        "camera_profile": "head_right",
        "checkpoint_step": 15900,
        "configured_steps": 15900,
        "complete_step": True,
        "model_state_dim": 16,
        "model_action_dim": 16,
        "wire_action_dim": 18,
        "supported_modes": ["single_step"],
    }
    live_readonly_benchmark._validate_health(good)

    bad = dict(good, checkpoint_step=10600, complete_step=False)
    with pytest.raises(ValueError, match="health contract mismatch"):
        live_readonly_benchmark._validate_health(bad)


def test_server_count_prefers_optimized_runtime_counter() -> None:
    assert (
        live_readonly_benchmark._server_inference_count(
            {"inference_count": 0, "optimized_inference_count": 33}
        )
        == 33
    )
    assert live_readonly_benchmark._server_inference_count({"inference_count": 7}) == 7

    with pytest.raises(ValueError, match="invalid inference count"):
        live_readonly_benchmark._server_inference_count({"optimized_inference_count": True})


def test_live_benchmark_sources_have_no_external_action_path() -> None:
    paths = [
        OPTIMIZED_ROOT / "runtime/live_readonly.py",
        OPTIMIZED_ROOT / "tools/live_readonly_benchmark.py",
        OPTIMIZED_ROOT / "run_live_readonly_benchmark.sh",
    ]
    forbidden = ("UDPCommandSender", "command_port", "command_target", "send" + "_action")

    for path in paths:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{path} contains forbidden token {token!r}"
