#!/usr/bin/env python3
import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


DATA_DIR = Path("agents/car_manual/data")
FACE_CARD_NAME = "Body.FaceCard"

MUTATION = """
mutation MutateVehicle($id: ID!, $name: String!, $value: String!) {
  mutateVehicle(id: $id, name: $name, value: $value) {
    id
    name
    value
    success
  }
}
"""


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_entries(data_dir: Path, brand: str | None) -> list[dict]:
    paths = [data_dir / f"{brand}.json"] if brand else sorted(data_dir.glob("*.json"))
    entries = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            raw_items = json.load(f)
        for index, item in enumerate(raw_items, 1):
            if not isinstance(item, dict):
                continue
            q = str(item.get("q") or "").strip()
            a = str(item.get("a") or "").strip()
            if not q or not a:
                continue
            entries.append({
                "file": path.name,
                "index": index,
                "q": q,
                "a": a,
            })
    return entries


def send_facecard(url: str, api_key: str, vehicle_id: str, answer: str) -> tuple[int, dict]:
    payload = {
        "query": MUTATION,
        "variables": {
            "id": vehicle_id,
            "name": FACE_CARD_NAME,
            "value": answer,
        },
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


def success_from_result(result: dict) -> bool | None:
    mutation_result = (result.get("data") or {}).get("mutateVehicle")
    if not isinstance(mutation_result, dict):
        return None
    success = mutation_result.get("success")
    return success if isinstance(success, bool) else None


def shorten(text: str, max_len: int = 100) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test Body.FaceCard mutateVehicle with answers from agents/car_manual/data."
    )
    parser.add_argument("--brand", choices=["toyota", "mercedes", "mmc"], help="Only test one data file.")
    parser.add_argument("--send", action="store_true", help="Actually call AWS. Without this, only prints entries.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of entries for a quick test.")
    parser.add_argument("--env-file", default=".env", help="Path to env file with GRAPHQL_* values.")
    args = parser.parse_args()

    entries = load_entries(DATA_DIR, args.brand)
    if args.limit > 0:
        entries = entries[: args.limit]

    print(f"entries={len(entries)} send={args.send}")
    print()

    if args.send:
        load_dotenv(Path(args.env_file))
        url = os.environ["GRAPHQL_HOST"]
        api_key = os.environ["GRAPHQL_API_KEY"]
        vehicle_id = os.environ["GRAPHQL_VEHICLE_ID"]

    counts = {True: 0, False: 0, None: 0}
    for entry in entries:
        label = f"{entry['file']}#{entry['index']}"
        print(f"[{label}]")
        print(f"  q     : {shorten(entry['q'])}")
        print(f"  name  : {FACE_CARD_NAME}")
        print(f"  value : {shorten(entry['a'])}")

        if args.send:
            status, result = send_facecard(url, api_key, vehicle_id, entry["a"])
            success = success_from_result(result)
            counts[success] += 1
            print(f"  http  : {status}")
            print(f"  ok    : {success}")
            if result.get("errors"):
                print(f"  errors: {json.dumps(result['errors'], ensure_ascii=False)}")
        print()

    if args.send:
        print("Summary:")
        print(f"  success true : {counts[True]}")
        print(f"  success false: {counts[False]}")
        print(f"  unknown/error: {counts[None]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
