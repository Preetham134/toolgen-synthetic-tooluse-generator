from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from toolgen.registry.loader import load_registry
from toolgen.registry.normalize import normalize_endpoint
from toolgen.utils.io import read_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = PROJECT_ROOT / "tests" / "_tmp"


def make_temp_dir(name: str) -> Path:
    path = TMP_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class TestRegistryNormalization(unittest.TestCase):
    def test_loader_supports_top_level_list(self) -> None:
        temp_dir = make_temp_dir("list_case")
        input_path = temp_dir / "tools.json"
        input_path.write_text(
            json.dumps([{"tool_name": "search", "api_name": "query"}]),
            encoding="utf-8",
        )

        endpoints, summary = load_registry(input_path)

        self.assertEqual(summary, {"loaded": 1, "skipped": 0})
        self.assertEqual(endpoints[0].tool_name, "search")
        shutil.rmtree(temp_dir)

    def test_loader_supports_dict_with_tools_key(self) -> None:
        temp_dir = make_temp_dir("dict_tools_case")
        input_path = temp_dir / "tools.json"
        input_path.write_text(
            json.dumps({"tools": [{"tool_name": "calendar", "api_name": "create"}]}),
            encoding="utf-8",
        )

        endpoints, summary = load_registry(input_path)

        self.assertEqual(summary, {"loaded": 1, "skipped": 0})
        self.assertEqual(endpoints[0].api_name, "create")
        shutil.rmtree(temp_dir)

    def test_loader_supports_single_record_dict(self) -> None:
        temp_dir = make_temp_dir("single_record_case")
        input_path = temp_dir / "tools.json"
        input_path.write_text(
            json.dumps({"tool_name": "maps", "api_name": "geocode", "parameters": []}),
            encoding="utf-8",
        )

        endpoints, summary = load_registry(input_path)

        self.assertEqual(summary, {"loaded": 1, "skipped": 0})
        self.assertEqual(endpoints[0].endpoint_id, "maps.geocode")
        shutil.rmtree(temp_dir)

    def test_normalization_handles_inconsistent_fields(self) -> None:
        endpoint = normalize_endpoint(
            {
                "tool_name": "weather",
                "endpoint": "forecast",
                "domain": "climate",
                "summary": "Get forecast",
                "id": "tool-1",
                "parameters": [
                    {"name": "city", "required": True, "description": "Target city"},
                    {"name": "units"},
                ],
                "required": ["units"],
                "response": {"type": "object"},
                "tags": ["forecast", "weather"],
            }
        )

        self.assertIsNotNone(endpoint)
        assert endpoint is not None
        self.assertEqual(endpoint.tool_name, "weather")
        self.assertEqual(endpoint.api_name, "forecast")
        self.assertEqual(endpoint.category, "climate")
        self.assertEqual(endpoint.description, "Get forecast")
        self.assertEqual(endpoint.source_tool_id, "tool-1")
        self.assertEqual(endpoint.required_params, ["city", "units"])
        self.assertEqual(endpoint.input_params[0].type, "string")
        self.assertTrue(endpoint.input_params[0].required)
        self.assertTrue(endpoint.input_params[1].required)
        self.assertEqual(endpoint.output_schema, {"type": "object"})

    def test_input_schema_properties_are_normalized(self) -> None:
        endpoint = normalize_endpoint(
            {
                "name": "calendar",
                "action": "create_event",
                "group": "productivity",
                "input_schema": {
                    "required": ["title"],
                    "properties": {
                        "title": {"description": "Event title"},
                        "timezone": {"type": "string"},
                        "ignored": {"type": "string", "name": ""},
                    },
                },
            }
        )

        self.assertIsNotNone(endpoint)
        assert endpoint is not None
        self.assertEqual(endpoint.category, "productivity")
        self.assertEqual([param.name for param in endpoint.input_params], ["title", "timezone"])
        self.assertEqual(endpoint.input_params[0].type, "string")
        self.assertEqual(endpoint.required_params, ["title"])

    def test_loader_skips_malformed_records_safely(self) -> None:
        temp_dir = make_temp_dir("loader_case")
        input_path = temp_dir / "tools.json"
        input_path.write_text(
            json.dumps(
                {
                    "tools": [
                        {"tool_name": "search", "api_name": "query"},
                        {"description": "missing names"},
                        "not-a-record",
                    ]
                }
            ),
            encoding="utf-8",
        )

        endpoints, summary = load_registry(input_path)

        self.assertEqual(summary, {"loaded": 1, "skipped": 2})
        self.assertEqual(len(endpoints), 1)
        self.assertEqual(endpoints[0].endpoint_id, "search.query")
        shutil.rmtree(temp_dir)


class TestBuildCommand(unittest.TestCase):
    def test_build_creates_normalized_registry(self) -> None:
        project_root = make_temp_dir("build_case") / "project"
        raw_dir = project_root / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        input_path = raw_dir / "tools.json"
        input_path.write_text(
            json.dumps(
                [
                    {
                        "tool": "maps",
                        "name": "geocode",
                        "category": "location",
                        "desc": "Find coordinates",
                        "params": [{"name": "address", "required": True}],
                        "output_schema": {"type": "object"},
                    },
                    {"summary": "bad record"},
                ]
            ),
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "toolgen.cli",
                "--project-root",
                str(project_root),
                "build",
                "--input",
                str(input_path),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        registry = read_json(project_root / "data" / "processed" / "registry.json")
        graph = read_json(project_root / "data" / "processed" / "graph.json")

        self.assertIn("Loaded registry endpoint count: 1", result.stdout)
        self.assertIn("Skipped: 1", result.stdout)
        self.assertIn("Graph node count: 1", result.stdout)
        self.assertIn("Graph edge count: 0", result.stdout)
        self.assertEqual(registry["summary"], {"loaded": 1, "skipped": 1})
        self.assertEqual(len(registry["endpoints"]), 1)
        self.assertEqual(registry["endpoints"][0]["tool_name"], "maps")
        self.assertEqual(registry["endpoints"][0]["api_name"], "geocode")
        self.assertEqual(registry["endpoints"][0]["required_params"], ["address"])
        self.assertEqual(graph["summary"], {"num_nodes": 1, "num_edges": 0})
        self.assertEqual(len(graph["nodes"]), 1)
        self.assertEqual(graph["edges"], [])
        self.assertIn("maps.geocode", graph["adjacency"])
        shutil.rmtree(project_root.parent)


if __name__ == "__main__":
    unittest.main()
