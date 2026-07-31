from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tk_infer.pi05_optimized import run_policy_server

OPTIMIZED_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = OPTIMIZED_ROOT.parent / "pi05"


def test_server_parser_uses_isolated_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "JZ_PI05_OPT_SERVER_HOST",
        "JZ_PI05_OPT_SERVER_PORT",
        "PI05_OPT_BACKEND",
        "PI05_OPT_TRAJECTORY_PROCESSOR",
        "PI05_OPT_POLICY_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    args = run_policy_server.build_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 18088
    assert args.backend == "torch"
    assert args.trajectory_processor == "pass_through"
    assert args.policy_path is None


def test_config_only_does_not_load_backend_or_open_server(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("config-only attempted a model or socket operation")

    monkeypatch.setattr(run_policy_server.TorchPolicyBackend, "from_runtime_config", forbidden)
    monkeypatch.setattr(run_policy_server, "make_server", forbidden)

    assert run_policy_server.main(["--config-only"]) == 0

    output = capsys.readouterr().out
    assert '"config_only": true' in output
    assert '"server_port": 18088' in output
    assert "no model was loaded, and no socket was opened" in output


def test_config_only_reports_inference_mode_as_single_step_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("config-only attempted a model or socket operation")

    monkeypatch.setattr(run_policy_server.TorchOptimizedBackend, "from_runtime_config", forbidden)
    monkeypatch.setattr(run_policy_server, "make_server", forbidden)

    assert (
        run_policy_server.main(
            ["--config-only", "--backend=torch_optimized", "--torch-inference-mode=true"]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert '"supported_modes": [\n    "single_step"\n  ]' in output


def test_config_only_reports_rtc_conditioned_as_independent_rtc_backend(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("config-only attempted a model or socket operation")

    monkeypatch.setattr(run_policy_server.TorchRTCConditionedBackend, "from_runtime_config", forbidden)
    monkeypatch.setattr(run_policy_server, "make_server", forbidden)

    assert (
        run_policy_server.main(
            [
                "--config-only",
                "--backend=torch_rtc_conditioned",
                "--rtc-conditioned-task=jz robot pin timed vr teleoperation",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert '"backend": "torch_rtc_conditioned"' in output
    assert '"supported_modes": [\n    "single_step",\n    "rtc"\n  ]' in output


def test_optimized_python_and_shell_sources_have_no_robot_execution_path() -> None:
    forbidden = (
        "run_robot_client",
        "robot_builder",
        "robot.connect(",
        "send_action(",
        "run_single_step_armed",
        "run_rtc_armed",
    )
    for path in OPTIMIZED_ROOT.rglob("*"):
        if "tests" in path.parts or not path.is_file() or path.suffix not in {".py", ".sh"}:
            continue
        source = path.read_text(encoding="utf-8")
        for value in forbidden:
            assert value not in source, f"{path} contains forbidden Phase 0 execution reference {value!r}"


def test_baseline_sources_do_not_import_optimized_runtime() -> None:
    for path in BASELINE_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".sh"}:
            continue
        assert "tk_infer.pi05_optimized" not in path.read_text(encoding="utf-8"), path


def test_shell_launchers_use_optimized_local_write_root() -> None:
    common = (OPTIMIZED_ROOT / "common.sh").read_text(encoding="utf-8")
    server = (OPTIMIZED_ROOT / "run_server.sh").read_text(encoding="utf-8")

    assert 'PI05_OPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in common
    assert '"${PI05_OPT_DIR}"|"${PI05_OPT_DIR}"/*' in common
    assert "JZ_PI05_OPT_SERVER_AUTH_TOKEN" in server
    assert "JZ_PI05_SERVER_AUTH_TOKEN" not in server


def test_shell_launcher_rejects_cli_path_override() -> None:
    result = subprocess.run(
        ["bash", str(OPTIMIZED_ROOT / "run_server.sh"), "--policy-path=/tmp/unvalidated"],
        cwd=OPTIMIZED_ROOT.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "environment variables only" in result.stderr
