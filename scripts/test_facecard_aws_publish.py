#!/usr/bin/env python3
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MUTATE_SDV = """
mutation MutateSDV($id: ID!, $name: String!, $value: String!) {
  mutateSDV(id: $id, name: $name, value: $value) {
    id
    name
    value
    success
  }
}
"""

MUTATE_VEHICLE = """
mutation MutateVehicle($id: ID!, $name: String!, $value: String!) {
  mutateVehicle(id: $id, name: $name, value: $value) {
    id
    name
    value
    success
  }
}
"""

BASELINE_SIGNALS = {
    "trunk-open": ("Body.Trunk.Rear.IsOpen", "true"),
    "trunk-close": ("Body.Trunk.Rear.IsOpen", "false"),
    "driver-door-lock": ("Cabin.Door.Row1.DriverSide.IsLocked", "true"),
    "driver-door-unlock": ("Cabin.Door.Row1.DriverSide.IsLocked", "false"),
}


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def post_graphql(url: str, api_key: str, query: str, variables: dict[str, str]) -> tuple[int, dict[str, Any]]:
    payload = {
        "query": query,
        "variables": variables,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        status = exc.code

    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, {"raw": body}


def print_result(label: str, variables: dict[str, str], status: int, result: dict[str, Any]) -> None:
    print(f"[{label}]")
    print("sent:")
    print(json.dumps(variables, indent=2, ensure_ascii=False))
    print(f"http_status={status}")
    print("response:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()


def send_baseline(url: str, api_key: str, vehicle_id: str, baseline: str, label: str) -> None:
    name, value = BASELINE_SIGNALS[baseline]
    variables = {
        "id": vehicle_id,
        "name": name,
        "value": value,
    }
    status, result = post_graphql(url, api_key, MUTATE_VEHICLE, variables)
    print_result(label, variables, status, result)


def send_facecard(url: str, api_key: str, vehicle_id: str, value: str, label: str) -> None:
    variables = {
        "id": vehicle_id,
        "name": "FaceCard",
        "value": value,
    }
    status, result = post_graphql(url, api_key, MUTATE_SDV, variables)
    print_result(label, variables, status, result)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Send a unique FaceCard mutateSDV command, optionally wrapped by a known "
            "mutateVehicle door/trunk baseline for AWS IoT publish comparison."
        )
    )
    parser.add_argument("--env-file", default=".env", help="Path to env file with GRAPHQL_* values.")
    parser.add_argument("--host", default=None, help="GraphQL endpoint. Defaults to GRAPHQL_HOST.")
    parser.add_argument("--api-key", default=None, help="GraphQL API key. Defaults to GRAPHQL_API_KEY.")
    parser.add_argument("--vehicle-id", default=None, help="Vehicle id. Defaults to GRAPHQL_VEHICLE_ID.")
    parser.add_argument("--value", default=None, help="FaceCard value. Defaults to a unique timestamp marker.")
    parser.add_argument(
        "--baseline",
        choices=sorted(BASELINE_SIGNALS),
        default="trunk-open",
        help="Known mutateVehicle signal to send before and after FaceCard.",
    )
    parser.add_argument("--no-baseline", action="store_true", help="Only send FaceCard.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of FaceCard sends.")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between sends.")
    args = parser.parse_args()

    load_dotenv(Path(args.env_file))

    url = args.host or os.environ["GRAPHQL_HOST"]
    api_key = args.api_key or os.environ["GRAPHQL_API_KEY"]
    vehicle_id = args.vehicle_id or os.environ["GRAPHQL_VEHICLE_ID"]
    marker = args.value or f"FaceCard AWS publish test {datetime.now(timezone.utc).isoformat(timespec='seconds')}"

    print("Ask AWS side to subscribe before running:")
    print("  iot/sdv/#")
    print()
    print(f"vehicle_id={vehicle_id}")
    print(f"facecard_marker={marker!r}")
    print()

    if not args.no_baseline:
        send_baseline(url, api_key, vehicle_id, args.baseline, f"baseline-before:{args.baseline}")
        time.sleep(args.interval)

    for index in range(1, args.repeat + 1):
        value = marker if args.repeat == 1 else f"{marker} #{index}"
        send_facecard(url, api_key, vehicle_id, value, f"facecard:{index}")
        if index != args.repeat:
            time.sleep(args.interval)

    if not args.no_baseline:
        time.sleep(args.interval)
        send_baseline(url, api_key, vehicle_id, args.baseline, f"baseline-after:{args.baseline}")

    print("Expected interpretation:")
    print("- If baseline appears on iot/sdv/# but FaceCard does not, AWS FaceCard projection/publish is missing or broken.")
    print("- If neither appears, the subscriber is in the wrong AWS account/region/topic or publish bridge is down.")
    print("- If both appear, orchestrator/AWS publish path is good; the issue is in the live STT/orchestrator trigger path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
