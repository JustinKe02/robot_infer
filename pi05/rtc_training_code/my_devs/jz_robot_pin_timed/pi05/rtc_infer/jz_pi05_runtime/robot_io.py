from __future__ import annotations

import threading
from typing import Any


class SerializedRobotIO:
    """Serialize sensor and actor access to one JZ robot instance.

    Construction has no side effects: connecting and disconnecting remain
    explicit responsibilities of the entrypoint.
    """

    def __init__(self, robot: Any) -> None:
        self.robot = robot
        self._lock = threading.RLock()

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            return self.robot.get_observation()

    def send_action(self, action: Any) -> Any:
        with self._lock:
            return self.robot.send_action(action)

    @property
    def robot_type(self) -> str:
        with self._lock:
            return str(self.robot.robot_type)

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return bool(self.robot.is_connected)


__all__ = ["SerializedRobotIO"]
