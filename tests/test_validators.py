from __future__ import annotations

import unittest

from toolgen.generator.models import ConversationRecord, Message, ToolCall
from toolgen.generator.validators import (
    validate_chain_grounding,
    validate_conversation,
    validate_conversation_structure,
    validate_required_params_present,
    validate_tool_calls_exist_in_registry,
)


def make_registry() -> list[dict]:
    return [
        {
            "endpoint_id": "hotels.search",
            "required_params": ["city"],
        },
        {
            "endpoint_id": "hotels.book",
            "required_params": ["hotel_id"],
        },
    ]


class TestValidators(unittest.TestCase):
    def test_validate_conversation_structure_passes_for_valid_record(self) -> None:
        record = ConversationRecord(
            conversation_id="conv_001",
            messages=[
                Message(role="user", content="Find me a hotel in Paris"),
                Message(role="assistant", content="I can search for hotels."),
                Message(
                    role="assistant",
                    tool_calls=[ToolCall(endpoint_id="hotels.search", arguments={"city": "Paris"})],
                ),
                Message(role="tool", content={"results": [{"id": "hotel_001", "name": "Hotel 1"}], "count": 1}),
                Message(role="assistant", content="I found one option."),
            ],
        )

        result = validate_conversation_structure(record)
        self.assertTrue(result["passed"])
        self.assertEqual(result["issues"], [])

    def test_validate_conversation_structure_fails_for_empty_messages(self) -> None:
        record = ConversationRecord(conversation_id="conv_001", messages=[])
        result = validate_conversation_structure(record)
        self.assertFalse(result["passed"])
        self.assertIn("messages must be a non-empty list", result["issues"])

    def test_validate_tool_calls_exist_in_registry_flags_unknown_endpoint(self) -> None:
        record = ConversationRecord(
            conversation_id="conv_001",
            messages=[
                Message(
                    role="assistant",
                    tool_calls=[ToolCall(endpoint_id="unknown.endpoint", arguments={})],
                )
            ],
        )

        result = validate_tool_calls_exist_in_registry(record, make_registry())
        self.assertFalse(result["passed"])
        self.assertIn("unknown endpoint_id", result["issues"][0])

    def test_validate_required_params_present_flags_missing_required_arg(self) -> None:
        record = ConversationRecord(
            conversation_id="conv_001",
            messages=[
                Message(
                    role="assistant",
                    tool_calls=[ToolCall(endpoint_id="hotels.book", arguments={})],
                )
            ],
        )

        result = validate_required_params_present(record, make_registry())
        self.assertFalse(result["passed"])
        self.assertIn("missing required param 'hotel_id'", result["issues"][0])

    def test_validate_chain_grounding_passes_when_later_call_reuses_prior_id(self) -> None:
        record = ConversationRecord(
            conversation_id="conv_001",
            messages=[
                Message(role="assistant", tool_calls=[ToolCall(endpoint_id="hotels.search", arguments={"city": "Paris"})]),
                Message(role="tool", content={"results": [{"id": "hotel_001", "name": "Hotel 1"}], "count": 1}),
                Message(
                    role="assistant",
                    tool_calls=[ToolCall(endpoint_id="hotels.book", arguments={"hotel_id": "hotel_001"})],
                ),
            ],
        )

        result = validate_chain_grounding(record)
        self.assertTrue(result["passed"])
        self.assertEqual(result["issues"], [])

    def test_validate_chain_grounding_fails_for_hallucinated_id(self) -> None:
        record = ConversationRecord(
            conversation_id="conv_001",
            messages=[
                Message(role="assistant", tool_calls=[ToolCall(endpoint_id="hotels.search", arguments={"city": "Paris"})]),
                Message(role="tool", content={"results": [{"id": "hotel_001", "name": "Hotel 1"}], "count": 1}),
                Message(
                    role="assistant",
                    tool_calls=[ToolCall(endpoint_id="hotels.book", arguments={"hotel_id": "hotel_999"})],
                ),
            ],
        )

        result = validate_chain_grounding(record)
        self.assertFalse(result["passed"])
        self.assertIn("uses unknown id 'hotel_999'", result["issues"][0])

    def test_validate_conversation_combines_failures(self) -> None:
        record = ConversationRecord(
            conversation_id="conv_001",
            messages=[
                Message(
                    role="assistant",
                    tool_calls=[ToolCall(endpoint_id="missing.endpoint", arguments={"hotel_id": "hotel_999"})],
                ),
                Message(role="tool", content=""),
            ],
        )

        result = validate_conversation(record, make_registry())
        self.assertFalse(result["passed"])
        self.assertGreaterEqual(len(result["issues"]), 2)


if __name__ == "__main__":
    unittest.main()
