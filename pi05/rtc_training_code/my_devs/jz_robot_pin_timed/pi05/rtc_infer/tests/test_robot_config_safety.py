from __future__ import annotations

import pytest

from my_devs.jz_robot_pin_timed.pi05.rtc_infer.run_robot_client import (
    SCRIPT_DIR,
    resolve_local_output_dir,
)
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.robot_builder import (
    CAMERA_SPECS,
    build_robot_config,
)
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.safety import (
    ARMED_CONFIRMATION_ENV_VARS,
    ARMED_EXECUTION,
    DRY_RUN_EXECUTION,
    LOCAL_TRANSPORT,
    UDP_TRANSPORT,
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
    assert config.left_gripper_training_command_force == pytest.approx(80.0)
    assert config.right_gripper_training_command_force == pytest.approx(80.0)

    assert set(config.zmq_cameras) == set(CAMERA_SPECS)
    for name, (port, width, height) in CAMERA_SPECS.items():
        camera = config.zmq_cameras[name]
        assert camera.server_address == "192.168.1.81"
        assert camera.port == port
        assert camera.width == width
        assert camera.height == height


def test_client_output_directory_cannot_escape_rtc_infer(tmp_path) -> None:
    assert resolve_local_output_dir(SCRIPT_DIR / "outputs" / "test") == (
        SCRIPT_DIR / "outputs" / "test"
    )
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
    environ = {name: "1" for name in ARMED_CONFIRMATION_ENV_VARS}
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
        environ={name: "1" for name in ARMED_CONFIRMATION_ENV_VARS},
    )
