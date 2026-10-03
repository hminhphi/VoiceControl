"""Standalone unit tests for car_control multi-action matching.

Run: python scripts/test_car_control_multi.py
No GPU / no Docker needed: torch & transformers imports are stubbed.
"""
import json
import sys
import types
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agents" / "car_control"))

# --- stub heavy deps so agent_executor imports without torch/a2a ---
def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m

_stub("torch", Tensor=object, no_grad=lambda: types.SimpleNamespace(
    __enter__=lambda s: None, __exit__=lambda s, *a: None))
sys.modules["torch"].nn = types.SimpleNamespace(
    functional=types.SimpleNamespace(normalize=lambda *a, **k: None))
_stub("torch.nn", functional=sys.modules["torch"].nn.functional)
_stub("transformers", AutoModel=object, AutoTokenizer=object)
_stub("a2a")
_stub("a2a.server")
_stub("a2a.server.agent_execution", AgentExecutor=object, RequestContext=object)
_stub("a2a.server.events", EventQueue=object)
_stub("a2a.utils", new_agent_text_message=lambda *a, **k: None)
_stub("typing_extensions", override=lambda f: f)
_stub("graph_ql", RemoteGraphQLClient=object)

import agent_executor as ce  # noqa: E402

FAILED = []


def check(name, actual, expected):
    ok = actual == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"       expected: {expected}")
        print(f"       actual:   {actual}")
        FAILED.append(name)


# ---------- _parse_batch_json ----------
pairs, meta = ce._parse_batch_json(json.dumps({
    "actions": [
        {"action": "open", "component": "left_door"},
        {"action": "open", "component": "right_door"},
    ]
}))
check("batch_json 2 actions", pairs, [("left_door", "open"), ("right_door", "open")])
check("batch_json meta source", meta["match_source"], "batch_json")

pairs, _ = ce._parse_batch_json(json.dumps({
    "actions": [{"action": "turn_on", "component": "light"},
                {"action": "turn_off", "component": "ac"}]
}))
check("batch_json turn_on/off norm", pairs, [("light", "open"), ("ac", "close")])

check("batch_json invalid action dropped",
      ce._parse_batch_json(json.dumps({"actions": [{"action": "fly", "component": "trunk"}]}))[0], [])
check("batch_json not json -> None", ce._parse_batch_json("open left door"), None)
check("batch_json no actions key -> None", ce._parse_batch_json('{"foo": 1}'), None)

# ---------- _expand_elliptical ----------
check("expand left and right door",
      ce._expand_elliptical("open left and right door"),
      "open left door open right door")
check("expand the left and the right doors",
      ce._expand_elliptical("close the left and the right doors"),
      "close left door close right door")
check("expand both doors", ce._expand_elliptical("open both doors"),
      "open left door open right door")
check("expand all doors", ce._expand_elliptical("close all doors"),
      "close left door close right door")
check("expand light and ac", ce._expand_elliptical("turn on light and ac"),
      "turn on light turn on ac")
check("expand trunk and left door",
      ce._expand_elliptical("open the trunk and left door"),
      "open trunk open left door")
check("expand vi c?a tr?i va ph?i",
      ce._expand_elliptical("m\u1edf c\u1eeda tr\u00e1i v\u00e0 ph\u1ea3i"),
      "m\u1edf c\u1eeda tr\u00e1i m\u1edf c\u1eeda ph\u1ea3i")

# ---------- _match_actions_with_meta (no matcher) ----------
def match(text):
    return ce._match_actions_with_meta(text, None)

pairs, meta = match("open left and right door")
check("match: open left and right door", pairs,
      [("left_door", "open"), ("right_door", "open")])
check("match: source elliptical", meta["match_source"], "elliptical")

pairs, meta = match("open the trunk, and open left door")
check("match: trunk + left door", pairs, [("trunk", "open"), ("left_door", "open")])
check("match: source phrase", meta["match_source"], "phrase")

pairs, meta = match("open both doors")
check("match: both doors", pairs, [("left_door", "open"), ("right_door", "open")])

pairs, meta = match("turn on the light and the ac")
check("match: light and ac", pairs, [("light", "open"), ("ac", "open")])

pairs, meta = match("open trunk")
check("match: single trunk", pairs, [("trunk", "open")])

pairs, meta = match("turn off light")
check("match: single light off", pairs, [("light", "close")])

pairs, meta = match("hello how are you")
check("match: chit-chat empty", pairs, [])

# legacy single-command regression
pairs, meta = match("open the left door")
check("regression: single left door", pairs, [("left_door", "open")])

print()
if FAILED:
    print(f"FAILED: {len(FAILED)} -> {FAILED}")
    sys.exit(1)
print("ALL TESTS PASSED")
