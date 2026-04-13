from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from toolgen.generator.models import ConversationRecord
from toolgen.generator.orchestrator import generate_conversation
from toolgen.graph.builder import build_graph
from toolgen.graph.models import ToolGraph
from toolgen.registry.models import Endpoint, Parameter
from toolgen.utils.io import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = PROJECT_ROOT / "tests" / "_tmp"


def make_temp_dir(name: str) -> Path:
    path = TMP_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_endpoint(
    endpoint_id: str,
    tool_name: str,
    api_name: str,
    category: str,
    description: str,
    required_params: list[str] | None = None,
    input_params: list[Parameter] | None = None,
    output_schema: dict | None = None,
) -> Endpoint:
    return Endpoint(
        endpoint_id=endpoint_id,
        tool_name=tool_name,
        api_name=api_name,
        category=category,
        description=description,
        required_params=required_params or [],
        input_params=input_params or [],
        output_schema=output_schema or {},
    )


class TestGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoints = [
            make_endpoint(
                endpoint_id="hotels.search",
                tool_name="hotels",
                api_name="search_hotels",
                category="travel",
                description="Search hotels by city",
                required_params=["city"],
                input_params=[Parameter(name="city", required=True)],
                output_schema={"properties": {"hotel_id": {"type": "string"}}},
            ),
            make_endpoint(
                endpoint_id="hotels.book",
                tool_name="booking",
                api_name="book_hotel",
                category="travel",
                description="Book hotel details",
                required_params=["hotel_id"],
                input_params=[Parameter(name="hotel_id", required=True)],
            ),
        ]
        self.graph = build_graph(self.endpoints)

    def test_generate_conversation_returns_record_with_messages(self) -> None:
        record = generate_conversation(self.endpoints, self.graph, chain_length=2, seed=42)
        self.assertIsInstance(record, ConversationRecord)
        self.assertTrue(record.messages)

    def test_generate_conversation_includes_tool_and_assistant_messages(self) -> None:
        record = generate_conversation(self.endpoints, self.graph, chain_length=2, seed=42)
        self.assertTrue(any(message.role == "tool" for message in record.messages))
        self.assertTrue(any(message.tool_calls for message in record.messages if message.role == "assistant"))

    def test_generate_conversation_attaches_validation_result(self) -> None:
        record = generate_conversation(self.endpoints, self.graph, chain_length=2, seed=42)
        self.assertIn("passed", record.validation)
        self.assertIn("issues", record.validation)

    def test_generate_command_writes_jsonl(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

        subprocess.run(
            [sys.executable, "-m", "toolgen.cli", "build", "--input", "data/raw/sample_tools.json"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "toolgen.cli", "generate", "--num-samples", "2", "--seed", "42"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        records = read_jsonl(PROJECT_ROOT / "data" / "outputs" / "conversations.jsonl")
        self.assertEqual(len(records), 2)
        self.assertIn("conversation_id", records[0])
        self.assertTrue(records[0]["messages"])

    def test_generate_falls_back_to_single_step_when_graph_has_no_edges(self) -> None:
        endpoint = make_endpoint(
            endpoint_id="search.only",
            tool_name="search",
            api_name="search_items",
            category="general",
            description="Search for items",
            required_params=["query"],
            input_params=[Parameter(name="query", required=True)],
        )
        graph = ToolGraph(
            nodes=[endpoint.to_dict()],
            edges=[],
            adjacency={endpoint.endpoint_id: []},
            summary={"num_nodes": 1, "num_edges": 0},
        )

        record = generate_conversation([endpoint], graph, chain_length=3, seed=42)
        tool_messages = [message for message in record.messages if message.role == "tool"]
        self.assertEqual(len(tool_messages), 1)


if __name__ == "__main__":
    unittest.main()
