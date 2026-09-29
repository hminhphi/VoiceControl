import os
import subprocess
import time
import numpy as np
import sounddevice as sd


def find_device_id(name="default"):
    devices = sd.query_devices()
    for d in devices:
        if name in d["name"]:
            return d["index"]
    return None


def get_hostapi_name(device_info):
    try:
        return sd.query_hostapis(device_info.get("hostapi", -1)).get("name", "?")
    except Exception:
        return "?"


def get_valid_input_device_id(preferred_name=None):
    """
    Return (device_id, device_info) for an input with max_input_channels >= 1.
    Prefer device name containing preferred_name; else try pulse, default, alsa; else first valid.
    """
    devices = sd.query_devices()
    if preferred_name:
        key = preferred_name.lower()
        for index, dev in enumerate(devices):
            if key in dev.get("name", "").lower() and dev.get("max_input_channels", 0) >= 1:
                return index, dev
        if key.startswith(("bluez_source.", "bluez_sink.")):
            dev_id = find_device_id("pulse")
            if dev_id is not None:
                dev = sd.query_devices(dev_id)
                if dev.get("max_input_channels", 0) >= 1:
                    return dev_id, dev
    for name in ("pulse", "default", "alsa"):
        dev_id = find_device_id(name)
        if dev_id is not None:
            dev = sd.query_devices(dev_id)
            if dev.get("max_input_channels", 0) >= 1:
                return dev_id, dev
    for index, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) >= 1:
            return index, dev
    return None, None


def get_pulse_default_source():
    """Query Pulse server for the system default source name."""
    try:
        out = subprocess.run(
            ["pactl", "info"],
            capture_output=True,
            text=True,
            timeout=2,
            env={**os.environ},
        )
        if out.returncode != 0:
            return None
        for line in out.stdout.splitlines():
            if line.strip().startswith("Default Source:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def get_pulse_default_sink():
    """Query Pulse server for the system default sink name."""
    try:
        out = subprocess.run(
            ["pactl", "info"],
            capture_output=True,
            text=True,
            timeout=2,
            env={**os.environ},
        )
        if out.returncode != 0:
            return None
        for line in out.stdout.splitlines():
            if line.strip().startswith("Default Sink:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _run_pactl(args, timeout=2):
    try:
        out = subprocess.run(
            ["pactl", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ},
        )
        if out.returncode == 0:
            return out.stdout
    except Exception:
        pass
    return ""


def _pulse_source_names_by_index():
    sources = {}
    for line in _run_pactl(["list", "sources", "short"]).splitlines():
        parts = line.split()
        if len(parts) >= 2:
            sources[parts[0]] = parts[1]
    return sources


def _pulse_sink_names_by_index():
    sinks = {}
    for line in _run_pactl(["list", "sinks", "short"]).splitlines():
        parts = line.split()
        if len(parts) >= 2:
            sinks[parts[0]] = parts[1]
    return sinks


def _pulse_source_output_blocks():
    blocks = []
    current = None
    for line in _run_pactl(["list", "source-outputs"]).splitlines():
        stripped = line.strip()
        if stripped.startswith("Source Output #"):
            if current:
                blocks.append(current)
            current = {"source_output": stripped.rsplit("#", 1)[-1]}
            continue
        if current is None:
            continue
        if stripped.startswith("Source:"):
            current["source_index"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("application.name ="):
            current["application_name"] = stripped.split("=", 1)[1].strip().strip('"')
        elif stripped.startswith("application.process.id ="):
            current["process_id"] = stripped.split("=", 1)[1].strip().strip('"')
    if current:
        blocks.append(current)
    return blocks


def _pulse_sink_input_blocks():
    blocks = []
    current = None
    for line in _run_pactl(["list", "sink-inputs"]).splitlines():
        stripped = line.strip()
        if stripped.startswith("Sink Input #"):
            if current:
                blocks.append(current)
            current = {"sink_input": stripped.rsplit("#", 1)[-1]}
            continue
        if current is None:
            continue
        if stripped.startswith("Sink:"):
            current["sink_index"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("application.name ="):
            current["application_name"] = stripped.split("=", 1)[1].strip().strip('"')
        elif stripped.startswith("application.process.id ="):
            current["process_id"] = stripped.split("=", 1)[1].strip().strip('"')
    if current:
        blocks.append(current)
    return blocks


def get_current_pulse_record_source(process_id=None, attempts=5, delay=0.1):
    """Return PulseAudio source-output routing for this process after capture starts."""
    process_id = str(process_id or os.getpid())
    for _ in range(attempts):
        sources = _pulse_source_names_by_index()
        blocks = _pulse_source_output_blocks()

        selected = None
        for block in blocks:
            if block.get("process_id") == process_id:
                selected = block
                break
        if selected is None and len(blocks) == 1:
            selected = blocks[0]

        if selected and selected.get("source_index"):
            source_index = selected["source_index"]
            return {
                **selected,
                "source_name": sources.get(source_index),
            }

        time.sleep(delay)
    return None


def get_current_pulse_playback_sink(process_id=None, attempts=5, delay=0.1):
    """Return PulseAudio sink-input routing for this process after playback starts."""
    process_id = str(process_id or os.getpid())
    for _ in range(attempts):
        sinks = _pulse_sink_names_by_index()
        blocks = _pulse_sink_input_blocks()

        selected = None
        for block in blocks:
            if block.get("process_id") == process_id:
                selected = block
                break
        if selected is None and len(blocks) == 1:
            selected = blocks[0]

        if selected and selected.get("sink_index"):
            sink_index = selected["sink_index"]
            return {
                **selected,
                "sink_name": sinks.get(sink_index),
            }

        time.sleep(delay)
    return None


class CircularBuffer:
    def __init__(self, maxsize):
        self.buffer = np.zeros(maxsize, dtype=np.int16)
        self.maxsize = maxsize
        self.size = 0
        self.head = 0
        self.tail = 0

    def put(self, data):
        data_len = len(data)
        if data_len >= self.maxsize:
            self.buffer = data[-self.maxsize:]
            self.head = 0
            self.tail = 0
            self.size = self.maxsize
        else:
            space_right = self.maxsize - self.tail
            if data_len <= space_right:
                self.buffer[self.tail : self.tail + data_len] = data
                self.tail = (self.tail + data_len) % self.maxsize
            else:
                self.buffer[self.tail :] = data[:space_right]
                self.buffer[: data_len - space_right] = data[space_right:]
                self.tail = data_len - space_right
            self.size = min(self.size + data_len, self.maxsize)
            if self.size == self.maxsize:
                self.head = self.tail

    def get(self, size):
        if size > self.size:
            return None
        if size <= self.maxsize - self.head:
            data = self.buffer[self.head : self.head + size].copy()
            self.head = (self.head + size) % self.maxsize
        else:
            data = np.concatenate(
                (
                    self.buffer[self.head :],
                    self.buffer[: size - (self.maxsize - self.head)],
                )
            )
            self.head = size - (self.maxsize - self.head)
        self.size -= size
        return data
