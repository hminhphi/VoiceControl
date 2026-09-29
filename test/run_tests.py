import json
import os
import sys
from pathlib import Path

import httpx

TEST_DIR = Path(__file__).resolve().parent
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000").rstrip("/")


def post_message(client: httpx.Client, message: str, session_id: str = "test") -> tuple[int, str, dict]:
    r = client.post(
        f"{ORCHESTRATOR_URL}/message",
        json={"message": message, "session_id": session_id},
        timeout=30.0,
    )
    try:
        raw = r.json()
        body = raw if isinstance(raw, dict) else {}
    except Exception:
        body = {}
    resp_text = body.get("message", body.get("response", "")) or ""
    return r.status_code, resp_text, body


def run_single(cases: list[dict], client: httpx.Client) -> tuple[int, int]:
    passed = 0
    failed = 0
    for i, item in enumerate(cases):
        msg = item.get("message", "")
        expected_contains = item.get("expected_contains", [])
        if isinstance(expected_contains, str):
            expected_contains = [expected_contains]
        expected_not_contains = item.get("expected_not_contains", [])
        if isinstance(expected_not_contains, str):
            expected_not_contains = [expected_not_contains]
        expected_contains_any = item.get("expected_contains_any", [])
        if isinstance(expected_contains_any, str):
            expected_contains_any = [expected_contains_any]
        status, resp, full = post_message(client, msg)
        print(f"  [{i+1}] message: {msg!r}")
        print(f"       status={status} orchestrator_response={json.dumps(full, ensure_ascii=False)[:250]}{'...' if len(json.dumps(full)) > 250 else ''}")
        ok = status == 200
        if ok and expected_contains:
            for s in expected_contains:
                if s not in resp:
                    ok = False
                    print(f"       FAIL: expected to contain {s!r}")
        if ok and expected_not_contains:
            for s in expected_not_contains:
                if s in resp:
                    ok = False
                    print(f"       FAIL: expected not to contain {s!r}")
        if ok and expected_contains_any:
            if not any(s in resp for s in expected_contains_any):
                ok = False
                print(f"       FAIL: expected one of {expected_contains_any!r}")
        if ok:
            passed += 1
            print(f"       PASS")
        else:
            failed += 1
            if status != 200:
                print(f"       FAIL: status {status}")
    return passed, failed


def main() -> int:
    single_path = TEST_DIR / "test_routing_single.json"
    complex_path = TEST_DIR / "test_flow_complex.json"
    if not single_path.is_file():
        print(f"Missing {single_path}")
        return 1
    if not complex_path.is_file():
        print(f"Missing {complex_path}")
        return 1
    with open(single_path, encoding="utf-8") as f:
        single_cases = json.load(f)
    with open(complex_path, encoding="utf-8") as f:
        complex_cases = json.load(f)
    print(f"ORCHESTRATOR_URL={ORCHESTRATOR_URL}\n")
    total_passed = 0
    total_failed = 0
    with httpx.Client() as client:
        print("=== test_routing_single ===")
        p, f = run_single(single_cases, client)
        total_passed += p
        total_failed += f
        print()
        print("=== test_flow_complex ===")
        p, f = run_single(complex_cases, client)
        total_passed += p
        total_failed += f
    print()
    print(f"Total: {total_passed} passed, {total_failed} failed")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
