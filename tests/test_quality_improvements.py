from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from toolgen.generator.orchestrator import load_graph_from_dict
from toolgen.generator.repair import repair_conversation
from toolgen.generator.validators import validate_chain_grounding
from toolgen.graph.sampler import sample_chain
from toolgen.utils.io import read_json, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestQualityImprovements(unittest.TestCase):
    def _build_rich_artifacts(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        subprocess.run(
            [sys.executable, "-m", "toolgen.cli", "build", "--input", "data/raw/sample_tools_rich.json"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

    def test_entity_type_grounding_accepts_matching_family(self) -> None:
        record = {
            "conversation_id": "conv_001",
            "messages": [
                {"role": "tool", "content": {"results": [{"id": "hotel_001"}]}, "tool_calls": []},
                {
                    "role": "assistant",
                    "content": "Book it",
                    "tool_calls": [{"endpoint_id": "travel_hotels_book", "arguments": {"hotel_id": "hotel_001"}}],
                },
            ],
        }
        result = validate_chain_grounding(record)
        self.assertTrue(result["passed"])

    def test_entity_type_grounding_rejects_incompatible_family(self) -> None:
        record = {
            "conversation_id": "conv_001",
            "messages": [
                {"role": "tool", "content": {"results": [{"id": "flight_001"}]}, "tool_calls": []},
                {
                    "role": "assistant",
                    "content": "Book it",
                    "tool_calls": [{"endpoint_id": "travel_hotels_book", "arguments": {"hotel_id": "flight_001"}}],
                },
            ],
        }
        result = validate_chain_grounding(record)
        self.assertFalse(result["passed"])
        self.assertTrue(any("incompatible id" in issue for issue in result["issues"]))

    def test_restaurant_id_rejects_hotel_id(self) -> None:
        record = {
            "conversation_id": "conv_001",
            "messages": [
                {"role": "tool", "content": {"results": [{"id": "hotel_001"}]}, "tool_calls": []},
                {
                    "role": "assistant",
                    "content": "Reserve it",
                    "tool_calls": [{"endpoint_id": "food_restaurants_book", "arguments": {"restaurant_id": "hotel_001"}}],
                },
            ],
        }
        result = validate_chain_grounding(record)
        self.assertFalse(result["passed"])

    def test_product_id_rejects_booking_id(self) -> None:
        record = {
            "conversation_id": "conv_001",
            "messages": [
                {"role": "tool", "content": {"booking_id": "booking_001"}, "tool_calls": []},
                {
                    "role": "assistant",
                    "content": "Create order",
                    "tool_calls": [{"endpoint_id": "shopping_orders_create", "arguments": {"product_id": "booking_001"}}],
                },
            ],
        }
        result = validate_chain_grounding(record)
        self.assertFalse(result["passed"])

    def test_sampler_prefers_search_first_when_possible(self) -> None:
        self._build_rich_artifacts()
        graph = load_graph_from_dict(read_json(PROJECT_ROOT / "data" / "processed" / "sample_tools_rich_graph.json"))
        chain = sample_chain(graph, chain_length=3, seed=42)
        self.assertTrue(chain)
        self.assertIn("search", str(chain[0]["api_name"]).lower())

    def test_rich_graph_avoids_unrelated_cross_domain_edges(self) -> None:
        self._build_rich_artifacts()
        graph = read_json(PROJECT_ROOT / "data" / "processed" / "sample_tools_rich_graph.json")
        adjacency = graph["adjacency"]
        hotel_targets = {edge["target"]: edge["score"] for edge in adjacency["travel_hotels_search"]}
        self.assertIn("travel_hotels_details", hotel_targets)
        self.assertIn("travel_hotels_book", hotel_targets)
        self.assertNotIn("food_restaurants_book", hotel_targets)
        self.assertNotIn("shopping_orders_create", hotel_targets)

        product_targets = {edge["target"]: edge["score"] for edge in adjacency["shopping_catalog_search"]}
        self.assertIn("shopping_catalog_details", product_targets)
        self.assertIn("shopping_orders_create", product_targets)
        self.assertNotIn("travel_flights_book", product_targets)

    def test_repair_does_not_use_incompatible_ids(self) -> None:
        self._build_rich_artifacts()
        registry = read_json(PROJECT_ROOT / "data" / "processed" / "sample_tools_rich_registry.json")["endpoints"]
        record = {
            "conversation_id": "conv_001",
            "messages": [
                {"role": "tool", "content": {"results": [{"id": "flight_001"}]}, "tool_calls": []},
                {
                    "role": "assistant",
                    "content": "Book hotel",
                    "tool_calls": [{"endpoint_id": "travel_hotels_book", "arguments": {"hotel_id": "restaurant_001"}}],
                },
            ],
            "metadata": {},
        }
        repaired = repair_conversation(record, registry)
        hotel_id = repaired["record"]["messages"][1]["tool_calls"][0]["arguments"]["hotel_id"]
        self.assertEqual(hotel_id, "restaurant_001")
        self.assertFalse(repaired["final_validation"]["passed"])

    def test_build_creates_input_specific_artifacts_and_generate_can_use_them(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

        subprocess.run(
            [sys.executable, "-m", "toolgen.cli", "build", "--input", "data/raw/sample_tools_rich.json"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        rich_registry = PROJECT_ROOT / "data" / "processed" / "sample_tools_rich_registry.json"
        rich_graph = PROJECT_ROOT / "data" / "processed" / "sample_tools_rich_graph.json"
        rich_graph_before = read_json(rich_graph)

        subprocess.run(
            [sys.executable, "-m", "toolgen.cli", "build", "--input", "data/raw/sample_tools.json"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertTrue(rich_registry.exists())
        self.assertTrue(rich_graph.exists())
        self.assertEqual(read_json(rich_graph)["summary"], rich_graph_before["summary"])

        output_path = PROJECT_ROOT / "data" / "outputs" / "quality_fix_check.jsonl"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "toolgen.cli",
                "generate",
                "--num-samples",
                "5",
                "--seed",
                "42",
                "--output",
                str(output_path),
                "--registry-path",
                str(rich_registry),
                "--graph-path",
                str(rich_graph),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        records = read_jsonl(output_path)
        self.assertEqual(len(records), 5)
        self.assertTrue(any(record["metadata"].get("chain_pattern") for record in records))

    def test_rich_generation_produces_search_led_three_step_chain(self) -> None:
        self._build_rich_artifacts()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        output_path = PROJECT_ROOT / "data" / "outputs" / "quality_locality_check.jsonl"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "toolgen.cli",
                "generate",
                "--num-samples",
                "20",
                "--seed",
                "42",
                "--output",
                str(output_path),
                "--registry-path",
                str(PROJECT_ROOT / "data" / "processed" / "sample_tools_rich_registry.json"),
                "--graph-path",
                str(PROJECT_ROOT / "data" / "processed" / "sample_tools_rich_graph.json"),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        records = read_jsonl(output_path)
        three_step_records = [record for record in records if record.get("metadata", {}).get("chain_length") == 3]
        self.assertTrue(three_step_records)
        self.assertTrue(any(record.get("metadata", {}).get("chain_pattern", "").startswith("search->") for record in three_step_records))
        for record in records:
            pattern = record.get("metadata", {}).get("chain_pattern", "")
            self.assertNotIn("other", pattern)
            self.assertNotIn("generic->book", pattern)
            for message in record.get("messages", []):
                if not isinstance(message, dict):
                    continue
                for tool_call in message.get("tool_calls", []):
                    if not isinstance(tool_call, dict):
                        continue
                    arguments = tool_call.get("arguments", {})
                    if not isinstance(arguments, dict):
                        continue
                    if "hotel_id" in arguments:
                        self.assertTrue(str(arguments["hotel_id"]).startswith("hotel_"))
                    if "product_id" in arguments:
                        self.assertTrue(str(arguments["product_id"]).startswith("product_"))


if __name__ == "__main__":
    unittest.main()
