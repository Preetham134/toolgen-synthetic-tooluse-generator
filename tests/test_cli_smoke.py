from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "toolgen.cli", *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


class TestCliSmoke(unittest.TestCase):
    def test_cli_smoke(self) -> None:
        run_cli("build")
        run_cli("generate")
        run_cli("evaluate")

        self.assertTrue((PROJECT_ROOT / "data" / "processed" / "registry.json").exists())
        self.assertTrue((PROJECT_ROOT / "data" / "processed" / "graph.json").exists())
        self.assertTrue((PROJECT_ROOT / "data" / "outputs" / "conversations.jsonl").exists())
        self.assertTrue((PROJECT_ROOT / "data" / "outputs" / "evaluation_report.json").exists())


if __name__ == "__main__":
    unittest.main()
