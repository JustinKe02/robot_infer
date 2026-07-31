from __future__ import annotations

from pathlib import Path

from tk_infer.pi05 import run_policy_server
from tk_infer.pi05.runtime.robot_builder import DEFAULT_CALIBRATION_DIR

PI05_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PARTS = ("my_devs", "jz_robot_pin_timed", "pi05", "rtc_infer")
LEGACY_IMPORT = ".".join(LEGACY_PARTS)
LEGACY_PATH = "/".join(LEGACY_PARTS)


def test_pi05_sources_do_not_depend_on_legacy_private_runtime() -> None:
    for path in PI05_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".sh"}:
            continue
        source = path.read_text(encoding="utf-8")
        assert LEGACY_IMPORT not in source, path
        assert LEGACY_PATH not in source, path


def test_runtime_state_is_separate_from_runtime_source() -> None:
    assert DEFAULT_CALIBRATION_DIR == PI05_ROOT / "run_state" / "calibration"
    assert (PI05_ROOT / "runtime" / "protocol.py").is_file()


def test_policy_server_has_no_implicit_checkpoint(monkeypatch) -> None:
    monkeypatch.delenv("POLICY_PATH", raising=False)
    args = run_policy_server.build_parser().parse_args([])

    assert args.policy_path is None
