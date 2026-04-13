from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from toolgen.utils.io import read_json, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = PROJECT_ROOT / "tests" / "_tmp"


def make_temp_dir(name: str) -> Path:
    path = TMP_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class TestE2EPipeline(unittest.TestCase):
    def test_full_pipeline_generates_and_evaluates_100_samples(self) -> None:
        temp_dir = make_temp_dir("e2e_pipeline")
        project_root = temp_dir / "project"
        raw_dir = project_root / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        sample_input = raw_dir / "sample_tools_rich.json"
        sample_input.write_text((PROJECT_ROOT / "data" / "raw" / "sample_tools_rich.json").read_text(encoding="utf-8"), encoding="utf-8")

        output_path = project_root / "data" / "outputs" / "final_run.jsonl"
        report_path = project_root / "data" / "outputs" / "final_report.json"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "toolgen.cli",
                "--project-root",
                str(project_root),
                "build",
                "--input",
                str(sample_input),
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
                "--project-root",
                str(project_root),
                "generate",
                "--num-samples",
                "100",
                "--seed",
                "42",
                "--output",
                str(output_path),
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
                "--project-root",
                str(project_root),
                "evaluate",
                "--input",
                str(output_path),
                "--report-out",
                str(report_path),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        graph = read_json(project_root / "data" / "processed" / "graph.json")
        records = read_jsonl(output_path)
        report = read_json(report_path)

        self.assertTrue(output_path.exists())
        self.assertTrue(report_path.exists())
        self.assertGreater(graph["summary"]["num_nodes"], 0)
        self.assertGreater(graph["summary"]["num_edges"], 0)
        self.assertGreaterEqual(len(records), 100)
        self.assertGreaterEqual(report["records_scored"], 100)
        self.assertIn("mean_overall_score", report["summary"])
        self.assertGreaterEqual(report["summary"]["mean_overall_score"], 4.0)
        self.assertTrue(any(message.get("tool_calls") for record in records for message in record.get("messages", [])))
        self.assertTrue(
            any(message.get("role") == "tool" for record in records for message in record.get("messages", []))
        )


if __name__ == "__main__":
    unittest.main()
