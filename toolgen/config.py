from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    outputs_dir: Path
    registry_path: Path
    graph_path: Path
    conversations_path: Path
    evaluation_report_path: Path


def get_paths(project_root: Path | None = None) -> Paths:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    outputs_dir = data_dir / "outputs"
    return Paths(
        project_root=root,
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        processed_dir=processed_dir,
        outputs_dir=outputs_dir,
        registry_path=processed_dir / "registry.json",
        graph_path=processed_dir / "graph.json",
        conversations_path=outputs_dir / "conversations.jsonl",
        evaluation_report_path=outputs_dir / "evaluation_report.json",
    )


def ensure_directories(paths: Paths) -> None:
    for directory in (paths.raw_dir, paths.processed_dir, paths.outputs_dir):
        directory.mkdir(parents=True, exist_ok=True)
