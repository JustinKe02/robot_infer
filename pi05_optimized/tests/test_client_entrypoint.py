from __future__ import annotations

import pytest

from tk_infer.pi05_optimized import run_client


def test_client_defaults_to_config_only_and_has_no_execution_or_armed_option(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("default client path attempted a network operation")

    monkeypatch.setattr(run_client, "RemotePolicyClient", forbidden)
    assert run_client.main([]) == 0

    output = capsys.readouterr().out
    assert '"config_only": true' in output
    assert '"hardware_adapter": null' in output
    assert '"armed_capability": false' in output
    option_strings = {
        option for action in run_client.build_parser()._actions for option in action.option_strings
    }
    assert "--execution" not in option_strings
    assert "--armed" not in option_strings


@pytest.mark.parametrize("mode", ["single_step", "rtc"])
def test_client_offline_smoke_uses_only_in_memory_adapters(
    mode: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_client.main(["--offline-smoke", f"--mode={mode}"]) == 0

    output = capsys.readouterr().out
    assert '"hardware_access": false' in output
    assert '"network_access": false' in output
    assert '"action_count": 1' in output
    assert '"force_slots"' in output


def test_health_only_validates_optimized_server_without_building_adapters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeRemotePolicyClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def health(self) -> dict[str, object]:
            return {"protocol_version": 3, "optimized_runtime": True, "ok": True}

    monkeypatch.setattr(run_client, "RemotePolicyClient", FakeRemotePolicyClient)

    assert run_client.main(["--health-only"]) == 0
    assert "HEALTH_ONLY passed" in capsys.readouterr().out


def test_offline_smoke_can_explicitly_enable_local_tracker(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_client.main(["--offline-smoke", "--mode=single_step", "--local-tracker"]) == 0

    output = capsys.readouterr().out
    assert '"local_tracker_enabled": true' in output
    assert '"contact_innovation_role": "slowdown_only_not_safety"' in output
    assert '"mpc_enabled": false' in output
