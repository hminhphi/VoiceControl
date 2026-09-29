#!/usr/bin/env python3
import json
import os
import urllib.request
from pathlib import Path


# Change only these two lines when testing.
name = "Body.FaceCard"
value = "George Leondis"


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, env_value = line.split("=", 1)
        os.environ.setdefault(key.strip(), env_value.strip().strip('"').strip("'"))


mutation = """
mutation MutateVehicle($id: ID!, $name: String!, $value: String!) {
  mutateVehicle(id: $id, name: $name, value: $value) {
    id
    name
    value
    success
  }
}
"""


load_dotenv(Path(".env"))

payload = {
    "query": mutation,
    "variables": {
        "id": os.environ["GRAPHQL_VEHICLE_ID"],
        "name": name,
        "value": value,
    },
}

print("Sending:")
print(json.dumps(payload["variables"], indent=2, ensure_ascii=False))
print()

req = urllib.request.Request(
    os.environ["GRAPHQL_HOST"],
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "x-api-key": os.environ["GRAPHQL_API_KEY"],
    },
    method="POST",
)

with urllib.request.urlopen(req, timeout=10) as response:
    body = response.read().decode("utf-8")
    print(response.status)
    print(body)
