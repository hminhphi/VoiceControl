import os
import logging

import requests

logger = logging.getLogger("car_control.graph_ql")


class RemoteGraphQLClient:
    def __init__(self, url=None, api_key=None, vehicle_id=None):
        self.url = url or os.environ.get("GRAPHQL_HOST")
        self.api_key = api_key or os.environ.get("GRAPHQL_API_KEY")
        self.vehicle_id = vehicle_id or os.environ.get("GRAPHQL_VEHICLE_ID", "03174526-5baa-11f0-947d-00155d08e663")

        if not self.url:
            raise ValueError("GRAPHQL_HOST is missing")
        if not self.api_key:
            raise ValueError("GRAPHQL_API_KEY is missing")

        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

        self.mutation = """
        mutation MutateVehicle($id: ID!, $name: String!, $value: String!) {
          mutateVehicle(id: $id, name: $name, value: $value) {
            id
            name
            value
            success
          }
        }
        """

    ACTION_MAP = {
        "can_door_left": {
            "open": ("Cabin.Door.Row1.DriverSide.IsOpen", "true"),
            "close": ("Cabin.Door.Row1.DriverSide.IsOpen", "false"),
            "lock": ("Cabin.Door.Row1.DriverSide.IsLocked", "true"),
            "unlock": ("Cabin.Door.Row1.DriverSide.IsLocked", "false"),
        },
        "can_door_right": {
            "open": ("Cabin.Door.Row2.DriverSide.IsOpen", "true"),
            "close": ("Cabin.Door.Row2.DriverSide.IsOpen", "false"),
            "lock": ("Cabin.Door.Row2.DriverSide.IsLocked", "true"),
            "unlock": ("Cabin.Door.Row2.DriverSide.IsLocked", "false"),
        },
        "can_trunk": {
            "open": ("Body.Trunk.Rear.IsOpen", "true"),
            "close": ("Body.Trunk.Rear.IsOpen", "false"),
        },
    }

    def send_request(self, function_name: str, params: dict) -> bool:
        if function_name not in self.ACTION_MAP:
            raise ValueError(f"Unknown function: {function_name}")

        action = params.get("action")
        if action not in self.ACTION_MAP[function_name]:
            raise ValueError(f"Invalid action '{action}' for function '{function_name}'")

        name, value = self.ACTION_MAP[function_name][action]
        payload = {
            "query": self.mutation,
            "variables": {
                "id": self.vehicle_id,
                "name": name,
                "value": value,
            },
        }
        response = requests.post(self.url, json=payload, headers=self.headers, timeout=10)
        result = response.json()
        logger.info("GraphQL response: %s", result)
        return True
