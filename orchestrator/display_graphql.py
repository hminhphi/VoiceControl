import logging
import os
import re
import uuid
from typing import Any

logger = logging.getLogger("orchestrator.display_graphql")

FACE_CARD_SIGNAL_NAME = "LCD_INTENT"

_MUTATION = """
mutation SendVehicleCommand($input: VehicleCommandInput!) {
  sendVehicleCommand(input: $input) {
    requestId
    vehicleId
    status
    reasonCode
    success
    timestamp
  }
}
"""

_missing_config_logged = False

_FACE_QUERY_RE = re.compile(
    r"\b(?:"
    r"who\s+is|"
    r"ceo|cfo|cio|president|tony|cherry|fpt\s+automotive|fpt\s+logo"
    r")\b",
    re.IGNORECASE,
)


_INTENT_PATTERNS: tuple[tuple[int, str], ...] = (
    (0, "fpt logo"),
    (1, "nguyen dinh thanh"),
    (2, "duong nguyen"),
    (3, "nguyen duc kinh"),
    (4, "pham minh tuan"),
    (5, "george leondis"),
    (6, "shibata hideki"),
    (7, "president kato"),
)


def _looks_like_face_query(*texts: str | None) -> bool:
    return any(_FACE_QUERY_RE.search(text or "") for text in texts)


def _resolve_lcd_intent_code(*texts: str | None) -> int | None:
    haystack = " ".join(text or "" for text in texts).lower()
    for intent_code, pattern in _INTENT_PATTERNS:
        if pattern in haystack:
            return intent_code
    return None


def build_face_card_payload(agent_response: dict[str, Any]) -> dict[str, Any] | None:
    """Build the AWS screen payload from a confident person/executive car_manual response."""
    if agent_response.get("agent_id") != "car_manual":
        return None
    if agent_response.get("success") is not True:
        return None

    meta = agent_response.get("meta") or {}
    if meta.get("passes_threshold") is not True:
        return None

    documents = (agent_response.get("data") or {}).get("documents") or []
    if not documents:
        return None

    first_doc = documents[0] or {}
    matched_q = meta.get("matched_q") or first_doc.get("q") or ""
    original_query = meta.get("original_query") or ""
    answer = first_doc.get("a") or agent_response.get("message") or ""
    if not answer:
        return None

    if not _looks_like_face_query(original_query, matched_q, answer):
        return None

    intent_code = _resolve_lcd_intent_code(original_query, matched_q, answer)
    if intent_code is None:
        return None

    return {
        "type": "famous_face",
        "original_query": original_query,
        "matched_q": matched_q,
        "answer": answer,
        "display_name": answer,
        "intent_code": intent_code,
        "agent_id": "car_manual",
        "source": "car_manual",
        "score": meta.get("score"),
    }


class DisplayGraphQLClient:
    def __init__(self, url: str | None = None, api_key: str | None = None, vehicle_id: str | None = None) -> None:
        self.url = url or os.environ.get("GRAPHQL_HOST")
        self.api_key = api_key or os.environ.get("GRAPHQL_API_KEY")
        self.vehicle_id = vehicle_id or os.environ.get(
            "GRAPHQL_VEHICLE_ID",
            "03174526-5baa-11f0-947d-00155d08e663",
        )

        if not self.url:
            raise ValueError("GRAPHQL_HOST is missing")
        if not self.api_key:
            raise ValueError("GRAPHQL_API_KEY is missing")

        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

    def build_mutation_payload(self, face_card: dict[str, Any]) -> dict[str, Any]:
        intent_code = face_card.get("intent_code")
        if not isinstance(intent_code, int):
            raise ValueError(f"Face card payload is missing intent_code: {face_card}")

        return {
            "query": _MUTATION,
            "variables": {
                "input": {
                    "requestId": f"cmd-lcd-{uuid.uuid4()}",
                    "vehicleId": self.vehicle_id,
                    "commandType": FACE_CARD_SIGNAL_NAME,
                    "intentCode": intent_code,
                    "schemaVersion": "1.0",
                },
            },
        }

    def send_face_card(self, face_card: dict[str, Any]) -> dict[str, Any]:
        import httpx

        payload = self.build_mutation_payload(face_card)
        command_input = payload["variables"]["input"]
        logger.info(
            "Sending face card GraphQL command: request_id=%s vehicle_id=%s command_type=%s intent_code=%s value=%r",
            command_input.get("requestId"),
            command_input.get("vehicleId"),
            command_input.get("commandType"),
            command_input.get("intentCode"),
            face_card.get("display_name") or face_card.get("answer"),
        )
        response = httpx.post(self.url, json=payload, headers=self.headers, timeout=10.0)
        logger.info("Face card GraphQL HTTP response: status_code=%s", response.status_code)
        response.raise_for_status()
        result = response.json()
        if result.get("errors"):
            logger.warning("Face card GraphQL errors: %s", result["errors"])
            raise RuntimeError(f"GraphQL errors: {result['errors']}")
        mutation_result = (result.get("data") or {}).get("sendVehicleCommand") or {}
        mutation_success = mutation_result.get("success")
        logger.info(
            "Face card GraphQL command result: success=%s request_id=%s vehicle_id=%s status=%s reason_code=%s",
            mutation_success,
            mutation_result.get("requestId"),
            mutation_result.get("vehicleId"),
            mutation_result.get("status"),
            mutation_result.get("reasonCode"),
        )
        if mutation_success is not True:
            raise RuntimeError(f"sendVehicleCommand returned success={mutation_success}: {mutation_result}")
        return result


def send_face_card_with_details(face_card: dict[str, Any]) -> dict[str, Any]:
    global _missing_config_logged

    display_value = str(face_card.get("display_name") or face_card.get("answer") or "")
    details: dict[str, Any] = {
        "attempted": False,
        "success": False,
        "name": FACE_CARD_SIGNAL_NAME,
        "value": display_value,
        "intent_code": face_card.get("intent_code"),
        "matched_q": face_card.get("matched_q"),
        "original_query": face_card.get("original_query"),
    }

    if not os.environ.get("GRAPHQL_HOST") or not os.environ.get("GRAPHQL_API_KEY"):
        details["error"] = "GRAPHQL_HOST or GRAPHQL_API_KEY is missing"
        if not _missing_config_logged:
            logger.info("Skipping face card upload because %s", details["error"])
            _missing_config_logged = True
        return details

    try:
        client = DisplayGraphQLClient()
        details["attempted"] = True
        details["vehicle_id"] = client.vehicle_id
        result = client.send_face_card(face_card)
        mutation_result = (result.get("data") or {}).get("sendVehicleCommand") or {}
        details.update(
            {
                "success": mutation_result.get("success") is True,
                "mutation_success": mutation_result.get("success"),
                "request_id": mutation_result.get("requestId"),
                "returned_id": mutation_result.get("vehicleId"),
                "status": mutation_result.get("status"),
                "reason_code": mutation_result.get("reasonCode"),
            }
        )
        return details
    except Exception as exc:
        details["error"] = str(exc)
        logger.warning("Face card GraphQL upload failed: %s", exc)
        return details


def send_face_card_best_effort(face_card: dict[str, Any]) -> bool:
    return bool(send_face_card_with_details(face_card).get("success"))


def send_car_manual_face_card_best_effort(agent_response: dict[str, Any]) -> bool:
    face_card = build_face_card_payload(agent_response)
    if face_card is None:
        return False
    return send_face_card_best_effort(face_card)


def send_car_manual_face_card_with_details(agent_response: dict[str, Any]) -> dict[str, Any]:
    face_card = build_face_card_payload(agent_response)
    if face_card is None:
        return {
            "attempted": False,
            "success": False,
            "reason": "agent response did not produce a face card payload",
        }
    return send_face_card_with_details(face_card)
