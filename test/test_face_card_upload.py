import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "orchestrator"))

from display_graphql import (  # noqa: E402
    FACE_CARD_SIGNAL_NAME,
    DisplayGraphQLClient,
    build_face_card_payload,
)


def _manual_response(question: str, answer: str, *, success=True, passes_threshold=True):
    return {
        "agent_id": "car_manual",
        "success": success,
        "message": answer if success else "No answer found.",
        "data": {
            "documents": [{"q": question, "a": answer}] if success else [],
        },
        "meta": {
            "original_query": question,
            "matched_q": question,
            "score": 1.0,
            "passes_threshold": passes_threshold,
        },
    }


class FaceCardUploadTest(unittest.TestCase):
    def test_cfo_answer_builds_payload(self):
        payload = build_face_card_payload(_manual_response("Who is CFO of Nissan?", "George Leondis"))

        self.assertIsNotNone(payload)
        self.assertEqual(payload["answer"], "George Leondis")
        self.assertEqual(payload["display_name"], "George Leondis")
        self.assertEqual(payload["intent_code"], 5)
        self.assertEqual(payload["matched_q"], "Who is CFO of Nissan?")
        self.assertEqual(payload["source"], "car_manual")

    def test_tony_answer_builds_payload(self):
        payload = build_face_card_payload(_manual_response("Who is Tony?", "Nguyen Dinh Thanh - Tony"))

        self.assertIsNotNone(payload)
        self.assertEqual(payload["answer"], "Nguyen Dinh Thanh - Tony")
        self.assertEqual(payload["intent_code"], 1)

    def test_fpt_automotive_ceo_maps_to_nguyen_intent(self):
        payload = build_face_card_payload(
            _manual_response(
                "Who is CEO of FPT Automotive?",
                "Nguyen Duc Kinh - FPT Software Executive Vice President - "
                "FPT Automotive Chief Executive Officer. Leads the team to secure "
                "prestigious projects in software-defined and electric vehicles across "
                "the entire engineering and manufacturing spectrum.",
            )
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["intent_code"], 3)

    def test_fpt_software_ceo_and_vp_maps_to_pham_intent(self):
        payload = build_face_card_payload(
            _manual_response(
                "Who is Vice President of FPT?",
                "Pham Minh Tuan - FPT Software Vice President - CEO of FPT Software. "
                "Responsible for promoting the corporation and its member companies' "
                "global business activities.",
            )
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["intent_code"], 4)

    def test_fpt_logo_answer_builds_zero_intent_payload(self):
        payload = build_face_card_payload(
            _manual_response("FPT logo", "Ok, I will show you the FPT logo")
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["answer"], "Ok, I will show you the FPT logo")
        self.assertEqual(payload["display_name"], "Ok, I will show you the FPT logo")
        self.assertEqual(payload["intent_code"], 0)
        self.assertEqual(payload["matched_q"], "FPT logo")

    def test_fpt_logo_answer_builds_payload_from_misheard_question(self):
        payload = build_face_card_payload(
            _manual_response("FBT logo", "Ok, I will show you the FPT logo")
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["intent_code"], 0)
        self.assertEqual(payload["matched_q"], "FBT logo")

    def test_generic_manual_answer_does_not_upload(self):
        payload = build_face_card_payload(
            _manual_response(
                "How do I reset the tire pressure warning?",
                "Use the vehicle settings menu.",
            )
        )

        self.assertIsNone(payload)

    def test_low_confidence_answer_does_not_upload(self):
        payload = build_face_card_payload(
            _manual_response("Who is CFO of Nissan?", "George Leondis", passes_threshold=False)
        )

        self.assertIsNone(payload)

    def test_graphql_mutation_payload_shape(self):
        client = DisplayGraphQLClient(
            url="https://example.test/graphql",
            api_key="test-key",
            vehicle_id="vehicle-1",
        )
        mutation_payload = client.build_mutation_payload({
            "type": "famous_face",
            "answer": "George Leondis",
            "intent_code": 5,
        })

        command_input = mutation_payload["variables"]["input"]
        self.assertTrue(command_input["requestId"].startswith("cmd-lcd-"))
        self.assertEqual(command_input["vehicleId"], "vehicle-1")
        self.assertEqual(command_input["commandType"], FACE_CARD_SIGNAL_NAME)
        self.assertEqual(command_input["intentCode"], 5)
        self.assertEqual(command_input["schemaVersion"], "1.0")
        self.assertEqual(client.headers["x-api-key"], "test-key")

    def test_graphql_mutation_payload_preserves_zero_intent_code(self):
        client = DisplayGraphQLClient(
            url="https://example.test/graphql",
            api_key="test-key",
            vehicle_id="vehicle-1",
        )
        mutation_payload = client.build_mutation_payload({
            "type": "famous_face",
            "answer": "Ok, I will show you the FPT logo",
            "intent_code": 0,
        })

        command_input = mutation_payload["variables"]["input"]
        self.assertEqual(command_input["commandType"], FACE_CARD_SIGNAL_NAME)
        self.assertEqual(command_input["intentCode"], 0)


if __name__ == "__main__":
    unittest.main()
