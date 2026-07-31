from __future__ import annotations

import socket
import threading
import time

from lerobot.robots.jz_robot_udp.protocol import encode_state_packet
from tk_infer.pi05_optimized.tools.p4_readonly_state_probe import run_probe


def _packet(sequence: int) -> dict[str, object]:
    return {
        "version": 1,
        "type": "state",
        "robot": "offline_test",
        "seq": sequence,
        "stamp_ns": sequence * 1_000_000,
        "joints": {
            "left": {f"joint_{index}": float(index) for index in range(7)},
            "right": {f"joint_{index}": float(index) for index in range(7)},
        },
        "grippers": {
            "left": {"width": 1.0, "force": 2.0},
            "right": {"width": 3.0, "force": 4.0},
        },
    }


def test_p4_probe_receives_state_and_closes_receiver_without_command_socket() -> None:
    reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    reservation.bind(("127.0.0.1", 0))
    port = reservation.getsockname()[1]
    reservation.close()

    def send_packets() -> None:
        time.sleep(0.05)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            for sequence in range(1, 8):
                sender.sendto(encode_state_packet(_packet(sequence)), ("127.0.0.1", port))
                time.sleep(0.02)

    thread = threading.Thread(target=send_packets)
    thread.start()
    report = run_probe(
        bind_ip="127.0.0.1",
        state_port=port,
        expected_sender_ip="127.0.0.1",
        duration_s=0.25,
        connect_timeout_s=0.5,
        min_packets=2,
        min_rate_hz=1.0,
    )
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert report["status"] == "PASS"
    assert report["packet_count"] >= 2
    assert report["raw18_dimensions_exact"] is True
    assert report["receiver_stopped"] is True
    assert report["command_socket_created"] is False
    assert report["action_sink_created"] is False
    assert report["action_sent"] is False
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as rebound:
        rebound.bind(("127.0.0.1", port))
