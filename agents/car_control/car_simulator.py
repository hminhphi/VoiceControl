import os
from dataclasses import dataclass, field


@dataclass
class CarSimulatorState:
    door_locked: bool = True
    door_open: bool = False
    window_open: bool = False
    trunk_open: bool = False
    light_on: bool = False
    ac_on: bool = False
    mirror_folded: bool = False


def _read_bool_env(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, str(default)).strip().lower()
    return raw in ("1", "true", "yes")


class CarSimulator:
    def __init__(self) -> None:
        self._state = CarSimulatorState()
        self._connected = _read_bool_env("CAR_SIMULATOR_CONNECTED", True)

    @property
    def connected(self) -> bool:
        return self._connected

    def set_connected(self, value: bool) -> None:
        self._connected = value

    def get_state(self) -> CarSimulatorState:
        return CarSimulatorState(
            door_locked=self._state.door_locked,
            door_open=self._state.door_open,
            window_open=self._state.window_open,
            trunk_open=self._state.trunk_open,
            light_on=self._state.light_on,
            ac_on=self._state.ac_on,
            mirror_folded=self._state.mirror_folded,
        )

    def set_door(self, action: str) -> tuple[bool, str]:
        if not self._connected:
            return False, "no_connection"
        action = (action or "").strip().lower()
        if action in ("open", "unlock"):
            self._state.door_locked = False
            self._state.door_open = True
            return True, "open"
        if action in ("close", "lock"):
            self._state.door_open = False
            self._state.door_locked = True
            return True, "closed"
        return False, "unknown"

    def set_window(self, action: str) -> tuple[bool, str]:
        if not self._connected:
            return False, "no_connection"
        action = (action or "").strip().lower()
        if action == "open":
            self._state.window_open = True
            return True, "open"
        if action == "close":
            self._state.window_open = False
            return True, "closed"
        return False, "unknown"

    def set_trunk(self, action: str) -> tuple[bool, str]:
        if not self._connected:
            return False, "no_connection"
        action = (action or "").strip().lower()
        if action == "open":
            self._state.trunk_open = True
            return True, "open"
        if action == "close":
            self._state.trunk_open = False
            return True, "closed"
        return False, "unknown"

    def set_light(self, action: str) -> tuple[bool, str]:
        if not self._connected:
            return False, "no_connection"
        action = (action or "").strip().lower()
        if action == "on":
            self._state.light_on = True
            return True, "on"
        if action == "off":
            self._state.light_on = False
            return True, "off"
        return False, "unknown"

    def set_ac(self, action: str) -> tuple[bool, str]:
        if not self._connected:
            return False, "no_connection"
        action = (action or "").strip().lower()
        if action == "on":
            self._state.ac_on = True
            return True, "on"
        if action == "off":
            self._state.ac_on = False
            return True, "off"
        return False, "unknown"

    def set_mirror(self, action: str) -> tuple[bool, str]:
        if not self._connected:
            return False, "no_connection"
        action = (action or "").strip().lower()
        if action == "fold":
            self._state.mirror_folded = True
            return True, "folded"
        if action == "unfold":
            self._state.mirror_folded = False
            return True, "unfolded"
        return False, "unknown"
