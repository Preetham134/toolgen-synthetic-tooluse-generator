from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from toolgen.evaluation.judge import judge_conversation
from toolgen.evaluation.metrics import compute_dataset_metrics
from toolgen.utils.io import read_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_good_record() -> dict:
    return {
        "conversation_id": "conv_good",
        "messages": [
            {"role": "user", "content": "Find me a hotel", "tool_calls": []},
            {
                "role": "assistant",
                "content": "I can help with that.",
                "tool_calls": [{"endpoint_id": "hotels.search", "arguments": {"city": "Paris"}}],
            },
            {"role": "tool", "content": {"results": [{"id": "hotel_001"}], "count": 1}, "tool_calls": []},
            {"role": "assistant", "content": "I found an option for you.", "tool_calls": []},
        ],
        "validation": {"passed": True, "issues": []},
    }


def make_bad_record() -> dict:
    return {
        "conversation_id": "conv_bad",
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": []},
        ],
        "validation": {"passed": False, "issues": ["missing structure"]},
    }


class TestEvaluation(unittest.TestCase):
    def test_heuristic_judge_returns_expected_keys(self) -> None:
        result = judge_conversation(make_good_record())
        self.assertIn("naturalness", result)
        self.assertIn("tool_correctness", result)
        self.assertIn("task_completion", result)
        self.assertIn("summary", result)
        self.assertIn("judge_mode", result)

    def test_heuristic_judge_scores_valid_record_higher_than_invalid_record(self) -> None:
        good = judge_conversation(make_good_record())
        bad = judge_conversation(make_bad_record())
        good_overall = (good["naturalness"] + good["tool_correctness"] + good["task_completion"]) / 3.0
        bad_overall = (bad["naturalness"] + bad["tool_correctness"] + bad["task_completion"]) / 3.0
        self.assertGreater(good_overall, bad_overall)

    def test_compute_dataset_metrics_returns_expected_fields(self) -> None:
        good_record = make_good_record()
        bad_record = make_bad_record()
        good_record["judge_scores"] = judge_conversation(good_record)
        bad_record["judge_scores"] = judge_conversation(bad_record)

        metrics = compute_dataset_metrics([good_record, bad_record])
        for key in (
            "num_records",
            "mean_naturalness",
            "mean_tool_correctness",
            "mean_task_completion",
            "mean_overall_score",
            "validation_pass_rate",
            "avg_num_messages",
            "avg_num_tool_calls",
        ):
            self.assertIn(key, metrics)
        self.assertIsInstance(metrics["mean_overall_score"], float)

    def test_evaluate_command_writes_report(self) -> None:
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
            [sys.executable, "-m", "toolgen.cli", "generate", "--num-samples", "3", "--seed", "42"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "toolgen.cli", "evaluate"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        report = read_json(PROJECT_ROOT / "data" / "outputs" / "evaluation_report.json")
        self.assertGreater(report["records_scored"], 0)
        self.assertIn("summary", report)
        self.assertIn("mean_overall_score", report["summary"])


if __name__ == "__main__":
    unittest.main()
