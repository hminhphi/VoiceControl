"""Compact pipeline state snapshot for the visualization UI.

The audio loop updates this in-process object; a background thread writes it
atomically (10 Hz, only when dirty) to `cache/pipeline_state.json`, which
`car_control_ui` reads through its `/api/pipeline` route. Updates are
best-effort and never raise into the audio path.
"""
import json
import os
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("/.dockerenv"):
    DEFAULT_PATH = os.path.join(_HERE, "cache", "pipeline_state.json")
else:
    DEFAULT_PATH = os.path.join(
        _HERE, "..", "cache", "voice_processing", "pipeline_state.json"
    )
MAX_EVENTS = 120


def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class PipelineState:
    def __init__(self, path=None, flush_hz=10.0):
        self.path = path or os.environ.get("PIPELINE_STATE_FILE") or DEFAULT_PATH
        self.flush_interval = 1.0 / max(flush_hz, 1.0)
        self._lock = threading.Lock()
        self._dirty = False
        self._stop = threading.Event()
        self._thread = None
        self._data = {
            "ts": 0.0,
            "updated": None,
            "stages": {
                "aec": {"enabled": None, "active": None, "erle_db": None},
                "vad": {"prob": 0.0, "speech": False, "threshold": None, "window": None},
                "smart_turn": {"prob": None, "threshold": None, "decision": None, "extends": 0},
                "wake": {"detected": False, "listening": False},
                "stt": {"state": "idle", "text": "", "lang": None, "latency_ms": None, "drop_reason": None},
                "diar": {"enabled": None, "n_speakers": 0, "overlap_ratio": 0.0, "latency_ms": None},
                "identity": {"label": None, "score": None, "known": 0},
                "llm": {"state": "idle", "chars": 0},
                "tts": {"speaking": False, "text": ""},
            },
            "turn": {"gen": 0, "text": "", "lang": None, "speaker": None, "turns": 0},
            "events": [],
        }

    # ── api ──────────────────────────────────────────────────────────────
    def update(self, stage, **fields):
        with self._lock:
            bucket = self._data["stages"].get(stage)
            if bucket is None:
                bucket = self._data["stages"][stage] = {}
            bucket.update(fields)
            self._dirty = True

    def set_turn(self, **fields):
        with self._lock:
            self._data["turn"].update(fields)
            self._dirty = True

    def bump_turn(self):
        with self._lock:
            self._data["turn"]["turns"] += 1
            self._dirty = True

    def event(self, kind, **data):
        with self._lock:
            entry = {"t": round(time.time(), 3), "kind": kind}
            entry.update(data)
            events = self._data["events"]
            events.append(entry)
            if len(events) > MAX_EVENTS:
                del events[: len(events) - MAX_EVENTS]
            self._dirty = True

    def snapshot(self):
        with self._lock:
            data = json.loads(json.dumps(self._data))
        data["ts"] = round(time.time(), 3)
        data["updated"] = _now_iso()
        return data

    def flush(self):
        with self._lock:
            if not self._dirty:
                return
            self._dirty = False
            self._data["ts"] = round(time.time(), 3)
            self._data["updated"] = _now_iso()
            payload = json.dumps(self._data, ensure_ascii=False)
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, self.path)
        except Exception:
            pass

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="pipeline-state", daemon=True
        )
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            self.flush()
            time.sleep(self.flush_interval)

    def stop(self):
        self._stop.set()
        self.flush()


_SINGLETON = None
_LOCK = threading.Lock()


def get_state():
    global _SINGLETON
    with _LOCK:
        if _SINGLETON is None:
            _SINGLETON = PipelineState()
            _SINGLETON.start()
        return _SINGLETON
