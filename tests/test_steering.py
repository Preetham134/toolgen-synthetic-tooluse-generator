from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from toolgen.evaluation.metrics import compute_dataset_metrics
from toolgen.generator.orchestrator import generate_conversation
from toolgen.generator.steering import GenerationCorpusState
from toolgen.graph.builder import build_graph
from toolgen.registry.models import Endpoint, Parameter
from toolgen.utils.io import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


class TestSteering(unittest.TestCase):
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

    def test_corpus_state_records_usage_counts(self) -> None:
        state = GenerationCorpusState()
        record = {
            "messages": [
                {"role": "assistant", "tool_calls": [{"endpoint_id": "hotels.search", "arguments": {"city": "Paris"}}]},
                {"role": "assistant", "tool_calls": [{"endpoint_id": "hotels.book", "arguments": {"hotel_id": "hotel_001"}}]},
            ],
            "metadata": {"categories": ["travel", "travel"], "chain_length": 2},
        }
        state.record_conversation(record)
        self.assertEqual(state.tool_usage["hotels.search"], 1)
        self.assertEqual(state.tool_usage["hotels.book"], 1)
        self.assertEqual(state.category_usage["travel"], 2)
        self.assertEqual(state.tool_pair_usage["hotels.search->hotels.book"], 1)
        self.assertEqual(state.chain_length_usage[2], 1)

    def test_steering_weights_downweight_repeated_tools(self) -> None:
        state = GenerationCorpusState(tool_usage={"hotels.search": 3})
        self.assertLess(state.get_tool_weight("hotels.search"), state.get_tool_weight("hotels.book"))

    def test_dataset_metrics_include_diversity_fields(self) -> None:
        records = [
            {
                "messages": [
                    {"role": "assistant", "tool_calls": [{"endpoint_id": "hotels.search", "arguments": {}}]},
                    {"role": "assistant", "tool_calls": [{"endpoint_id": "hotels.book", "arguments": {}}]},
                ],
                "validation": {"passed": True},
                "judge_scores": {"naturalness": 4.0, "tool_correctness": 4.0, "task_completion": 4.0},
                "metadata": {"categories": ["travel", "travel"]},
            }
        ]
        metrics = compute_dataset_metrics(records)
        self.assertIn("tool_usage_entropy", metrics)
        self.assertIn("distinct_tool_pair_ratio", metrics)
        self.assertIn("category_coverage", metrics)

    def test_generate_with_and_without_steering_both_work(self) -> None:
        off_record = generate_conversation(
            self.endpoints,
            self.graph,
            chain_length=2,
            seed=42,
            cross_conversation_steering=False,
            corpus_state=None,
        )
        on_state = GenerationCorpusState()
        on_record = generate_conversation(
            self.endpoints,
            self.graph,
            chain_length=2,
            seed=42,
            cross_conversation_steering=True,
            corpus_state=on_state,
        )
        self.assertTrue(off_record.messages)
        self.assertTrue(on_record.messages)

    def test_generate_records_include_steering_metadata(self) -> None:
        state = GenerationCorpusState()
        record = generate_conversation(
            self.endpoints,
            self.graph,
            chain_length=2,
            seed=42,
            cross_conversation_steering=True,
            corpus_state=state,
        )
        self.assertIn("steering_enabled", record.metadata)
        self.assertTrue(record.metadata["steering_enabled"])

    def test_cli_run_a_and_run_b_outputs_work(self) -> None:
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
            [
                sys.executable,
                "-m",
                "toolgen.cli",
                "generate",
                "--num-samples",
                "2",
                "--seed",
                "42",
                "--output",
                "data/outputs/run_a_test.jsonl",
                "--no-cross-conversation-steering",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "toolgen.cli",
                "generate",
                "--num-samples",
                "2",
                "--seed",
                "42",
                "--output",
                "data/outputs/run_b_test.jsonl",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        run_a = read_jsonl(PROJECT_ROOT / "data" / "outputs" / "run_a_test.jsonl")
        run_b = read_jsonl(PROJECT_ROOT / "data" / "outputs" / "run_b_test.jsonl")
        self.assertEqual(len(run_a), 2)
        self.assertEqual(len(run_b), 2)


if __name__ == "__main__":
    unittest.main()
