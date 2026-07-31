from __future__ import annotations

import pytest
import torch

from lerobot.robots.jz_robot_pin_timed.training_schema import RAW_FEATURE_NAMES
from tk_infer.pi05.run_robot_client import (
    DISABLE_JOINT_DELTA_CHECKS_ENV,
    JOINT_DELTA_BYPASS_ACK_ENV,
    SCRIPT_DIR,
    joint_delta_checks_disabled_from_env,
    resolve_local_output_dir,
)
from tk_infer.pi05.runtime.robot_builder import (
    CAMERA_SPECS,
    build_robot_config,
)
from tk_infer.pi05.runtime.safety import (
    ARMED_CONFIRMATION_ENV_VARS,
    ARMED_EXECUTION,
    DRY_RUN_EXECUTION,
    LOCAL_TRANSPORT,
    UDP_TRANSPORT,
    ActionSafety,
    require_armed_confirmation,
)


def test_build_robot_config_is_side_effect_free_and_matches_live_contract() -> None:
    config = build_robot_config(execution=DRY_RUN_EXECUTION)

    assert config.bind_ip == "0.0.0.0"
    assert config.state_port == 39010
    assert config.allowed_state_sender_ip == "192.168.1.81"
    assert config.command_target_ip == "192.168.1.81"
    assert config.command_target_port == 39020
    assert config.send_action_transport == LOCAL_TRANSPORT
    assert config.send_action_execution == DRY_RUN_EXECUTION
    assert config.max_initial_joint_delta_rad == pytest.approx(0.02)
    assert config.max_joint_step_rad == pytest.approx(0.02)
    assert config.allow_armed_joint_delta_bypass is False
    assert config.max_camera_state_receive_skew_ms == pytest.approx(250.0)
    assert config.enforce_camera_state_receive_skew is True
    assert config.left_gripper_training_command_force == pytest.approx(80.0)
    assert config.right_gripper_training_command_force == pytest.approx(80.0)

    assert set(config.zmq_cameras) == set(CAMERA_SPECS)
    for name, spec in CAMERA_SPECS.items():
        camera = config.zmq_cameras[name]
        assert camera.server_address == "192.168.1.81"
        assert camera.port == spec.port
        assert camera.width == spec.width
        assert camera.height == spec.height


def test_build_robot_config_supports_head_right_camera_profile() -> None:
    config = build_robot_config(
        execution=DRY_RUN_EXECUTION,
        camera_profile="head_right",
    )

    assert set(config.zmq_cameras) == {"camera_head", "camera_right"}
    assert config.zmq_cameras["camera_head"].port == 5555
    assert config.zmq_cameras["camera_right"].port == 5557


def test_build_robot_config_can_reject_reused_frames_for_fixed_profile() -> None:
    config = build_robot_config(
        execution=DRY_RUN_EXECUTION,
        camera_profile="head_right",
        reject_reused_camera_frames=True,
    )

    assert config.reject_reused_camera_frames is True


def test_build_robot_config_accepts_explicit_camera_state_receive_skew_limit() -> None:
    config = build_robot_config(max_camera_state_receive_skew_ms=175.0)

    assert config.max_camera_state_receive_skew_ms == pytest.approx(175.0)


def test_build_robot_config_can_explicitly_disable_only_joint_delta_checks() -> None:
    config = build_robot_config(
        execution=ARMED_EXECUTION,
        disable_joint_delta_checks=True,
    )

    assert config.max_initial_joint_delta_rad == 0.0
    assert config.max_joint_step_rad == 0.0
    assert config.allow_armed_joint_delta_bypass is True
    assert config.require_armed_env is True
    assert config.state_timeout_s == pytest.approx(1.0)
    assert config.gripper_width_min == pytest.approx(0.0)
    assert config.gripper_width_max == pytest.approx(100.0)
    assert config.gripper_force_min == pytest.approx(0.0)
    assert config.gripper_force_max == pytest.approx(100.0)


def test_local_transport_can_audit_with_joint_delta_checks_disabled() -> None:
    config = build_robot_config(
        execution=DRY_RUN_EXECUTION,
        disable_joint_delta_checks=True,
    )

    assert config.send_action_transport == LOCAL_TRANSPORT
    assert config.send_action_execution == DRY_RUN_EXECUTION
    assert config.max_initial_joint_delta_rad == 0.0
    assert config.max_joint_step_rad == 0.0
    assert config.allow_armed_joint_delta_bypass is True


def test_client_output_directory_cannot_escape_rtc_infer(tmp_path) -> None:
    assert resolve_local_output_dir(SCRIPT_DIR / "outputs" / "test") == (SCRIPT_DIR / "outputs" / "test")
    with pytest.raises(ValueError, match="must stay inside"):
        resolve_local_output_dir(tmp_path / "outside")


def test_dry_run_gate_requires_local_transport_but_no_armed_environment() -> None:
    require_armed_confirmation(
        execution=DRY_RUN_EXECUTION,
        transport=LOCAL_TRANSPORT,
        environ={},
    )

    with pytest.raises(ValueError, match="requires transport"):
        require_armed_confirmation(
            execution=DRY_RUN_EXECUTION,
            transport=UDP_TRANSPORT,
            environ={},
        )


@pytest.mark.parametrize("missing_name", ARMED_CONFIRMATION_ENV_VARS)
def test_armed_gate_requires_every_confirmation(missing_name: str) -> None:
    environ = dict.fromkeys(ARMED_CONFIRMATION_ENV_VARS, "1")
    del environ[missing_name]

    with pytest.raises(RuntimeError, match=missing_name):
        require_armed_confirmation(
            execution=ARMED_EXECUTION,
            transport=UDP_TRANSPORT,
            environ=environ,
        )


def test_armed_gate_accepts_only_explicit_three_way_confirmation() -> None:
    require_armed_confirmation(
        execution=ARMED_EXECUTION,
        transport=UDP_TRANSPORT,
        environ=dict.fromkeys(ARMED_CONFIRMATION_ENV_VARS, "1"),
    )


def test_joint_delta_bypass_python_gate_requires_both_exact_confirmations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DISABLE_JOINT_DELTA_CHECKS_ENV, "1")
    monkeypatch.delenv(JOINT_DELTA_BYPASS_ACK_ENV, raising=False)
    with pytest.raises(RuntimeError, match=JOINT_DELTA_BYPASS_ACK_ENV):
        joint_delta_checks_disabled_from_env()

    monkeypatch.setenv(JOINT_DELTA_BYPASS_ACK_ENV, "1")
    assert joint_delta_checks_disabled_from_env() is True

    monkeypatch.setenv(DISABLE_JOINT_DELTA_CHECKS_ENV, "0")
    with pytest.raises(RuntimeError, match=DISABLE_JOINT_DELTA_CHECKS_ENV):
        joint_delta_checks_disabled_from_env()


def _valid_raw18() -> torch.Tensor:
    action = torch.zeros(18, dtype=torch.float32)
    action[15] = 80.0
    action[17] = 80.0
    return action


def test_action_safety_accepts_only_finite_raw18_with_fixed_force() -> None:
    safety = ActionSafety()

    checked = safety.check_tensor(_valid_raw18())

    assert checked.shape == (18,)
    assert checked[15].item() == pytest.approx(80.0)
    assert checked[17].item() == pytest.approx(80.0)

    with pytest.raises(ValueError, match="raw18"):
        safety.check_tensor(torch.zeros(16))

    non_finite = _valid_raw18()
    non_finite[3] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        safety.check_tensor(non_finite)

    wrong_force = _valid_raw18()
    wrong_force[15] = 0.0
    with pytest.raises(ValueError, match="left_gripper.force"):
        safety.check_tensor(wrong_force)


def test_action_safety_validates_robot_action_keys_and_values() -> None:
    safety = ActionSafety()
    action = {
        name: float(value)
        for name, value in zip(
            RAW_FEATURE_NAMES,
            _valid_raw18(),
            strict=True,
        )
    }
    safety.check_robot_action(action)

    missing = dict(action)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing"):
        safety.check_robot_action(missing)

    boolean = dict(action)
    boolean[next(iter(boolean))] = True
    with pytest.raises(ValueError, match="not bool"):
        safety.check_robot_action(boolean)

    non_finite = dict(action)
    non_finite[next(iter(non_finite))] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        safety.check_robot_action(non_finite)
