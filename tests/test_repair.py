from __future__ import annotations

import unittest
from unittest.mock import patch

from toolgen.generator.models import ConversationRecord, Message, ToolCall
from toolgen.generator.orchestrator import generate_conversation
from toolgen.generator.repair import repair_conversation
from toolgen.graph.builder import build_graph
from toolgen.graph.models import ToolGraph
from toolgen.registry.models import Endpoint, Parameter


def make_registry() -> list[dict]:
    return [
        {
            "endpoint_id": "hotels.search",
            "required_params": ["city"],
            "input_params": [{"name": "city", "type": "string", "required": True}],
        },
        {
            "endpoint_id": "hotels.book",
            "required_params": ["hotel_id"],
            "input_params": [{"name": "hotel_id", "type": "string", "required": True}],
        },
    ]


class TestRepair(unittest.TestCase):
    def test_repair_fills_missing_required_param(self) -> None:
        record = ConversationRecord(
            conversation_id="conv_001",
            messages=[
                Message(
                    role="assistant",
                    content="I'll search.",
                    tool_calls=[ToolCall(endpoint_id="hotels.search", arguments={})],
                ),
                Message(role="tool", content={"results": [{"id": "hotel_001"}], "count": 1}),
            ],
            metadata={"initial_arguments": {"city": "Paris"}},
        )

        result = repair_conversation(record, make_registry())
        repaired = result["record"]
        tool_call = repaired["messages"][0]["tool_calls"][0]
        self.assertEqual(tool_call["arguments"]["city"], "Paris")
        self.assertTrue(result["final_validation"]["passed"] or len(result["final_validation"]["issues"]) < 2)

    def test_repair_replaces_hallucinated_id_with_prior_known_id(self) -> None:
        record = {
            "conversation_id": "conv_001",
            "messages": [
                {"role": "assistant", "content": "Searching", "tool_calls": [{"endpoint_id": "hotels.search", "arguments": {"city": "Paris"}}]},
                {"role": "tool", "content": {"results": [{"id": "hotel_001"}], "count": 1}, "tool_calls": []},
                {"role": "assistant", "content": "Booking", "tool_calls": [{"endpoint_id": "hotels.book", "arguments": {"hotel_id": "hotel_999"}}]},
            ],
            "metadata": {},
        }

        result = repair_conversation(record, make_registry())
        repaired_id = result["record"]["messages"][2]["tool_calls"][0]["arguments"]["hotel_id"]
        self.assertEqual(repaired_id, "hotel_001")

    def test_repair_adds_final_assistant_message_if_missing(self) -> None:
        record = {
            "conversation_id": "conv_001",
            "messages": [
                {"role": "assistant", "content": "Searching", "tool_calls": [{"endpoint_id": "hotels.search", "arguments": {"city": "Paris"}}]},
                {"role": "tool", "content": {"results": [{"id": "hotel_001"}], "count": 1}, "tool_calls": []},
            ],
            "metadata": {},
        }

        result = repair_conversation(record, make_registry())
        final_message = result["record"]["messages"][-1]
        self.assertEqual(final_message["role"], "assistant")
        self.assertTrue(final_message["content"])

    def test_generator_uses_repair_when_validation_fails(self) -> None:
        endpoints = [
            Endpoint(
                endpoint_id="hotels.search",
                tool_name="hotels",
                api_name="search_hotels",
                category="travel",
                description="Search hotels",
                required_params=["city"],
                input_params=[Parameter(name="city", required=True)],
            )
        ]
        graph = build_graph(endpoints)

        with patch("toolgen.generator.orchestrator.validate_conversation", return_value={"passed": False, "issues": ["synthetic failure"]}):
            with patch(
                "toolgen.generator.orchestrator.repair_conversation",
                return_value={
                    "record": {
                        "conversation_id": "conv_0001",
                        "messages": [
                            {"role": "user", "content": "Find a hotel", "tool_calls": []},
                            {"role": "assistant", "content": "I completed the requested steps.", "tool_calls": []},
                        ],
                        "metadata": {"used_repair": True},
                        "validation": {"passed": True, "issues": []},
                    },
                    "repair_applied": True,
                    "repair_attempts": 1,
                    "final_validation": {"passed": True, "issues": []},
                },
            ):
                record = generate_conversation(endpoints, graph, chain_length=1, seed=42)

        self.assertTrue(record.metadata["used_repair"])
        self.assertGreaterEqual(record.metadata["repair_attempts"], 1)

    def test_repair_stops_when_record_already_valid(self) -> None:
        record = {
            "conversation_id": "conv_001",
            "messages": [
                {"role": "user", "content": "Find a hotel", "tool_calls": []},
                {"role": "assistant", "content": "Done.", "tool_calls": []},
            ],
            "validation": {"passed": True, "issues": []},
            "metadata": {},
        }

        result = repair_conversation(record, [])
        self.assertFalse(result["repair_applied"])
        self.assertEqual(result["repair_attempts"], 0)
        self.assertTrue(result["final_validation"]["passed"])


if __name__ == "__main__":
    unittest.main()
