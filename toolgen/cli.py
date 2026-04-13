from __future__ import annotations

import argparse
from pathlib import Path

from toolgen.config import ensure_directories, get_paths
from toolgen.evaluation.judge import judge_conversation
from toolgen.evaluation.metrics import compute_dataset_metrics
from toolgen.generator.orchestrator import generate_conversation, load_graph_from_dict
from toolgen.generator.steering import GenerationCorpusState
from toolgen.graph.builder import build_graph
from toolgen.registry.loader import load_registry
from toolgen.utils.io import append_jsonl, read_json, read_jsonl, write_json


def _artifact_paths_for_input(paths, raw_input_path: Path) -> tuple[Path, Path]:
    stem = raw_input_path.stem
    return (
        paths.processed_dir / f"{stem}_registry.json",
        paths.processed_dir / f"{stem}_graph.json",
    )


def cmd_build(project_root: Path | None = None, input_path: Path | None = None) -> int:
    paths = get_paths(project_root)
    ensure_directories(paths)
    raw_input_path = input_path if input_path is not None else paths.raw_dir / "tools.json"
    if raw_input_path.exists():
        endpoints, summary = load_registry(raw_input_path)
    else:
        endpoints, summary = [], {"loaded": 0, "skipped": 0}

    write_json(
        paths.registry_path,
        {
            "summary": summary,
            "endpoints": [endpoint.to_dict() for endpoint in endpoints],
        },
    )
    graph = build_graph(endpoints)
    write_json(paths.graph_path, graph.to_dict())
    named_registry_path, named_graph_path = _artifact_paths_for_input(paths, raw_input_path)
    write_json(
        named_registry_path,
        {
            "summary": summary,
            "endpoints": [endpoint.to_dict() for endpoint in endpoints],
        },
    )
    write_json(named_graph_path, graph.to_dict())
    print(f"Loaded registry endpoint count: {summary['loaded']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Graph node count: {graph.summary['num_nodes']}")
    print(f"Graph edge count: {graph.summary['num_edges']}")
    print(f"Default registry output: {paths.registry_path}")
    print(f"Default graph output: {paths.graph_path}")
    print(f"Input-specific registry output: {named_registry_path}")
    print(f"Input-specific graph output: {named_graph_path}")
    return 0


def cmd_generate(
    project_root: Path | None = None,
    num_samples: int = 5,
    seed: int = 42,
    output_path: Path | None = None,
    cross_conversation_steering: bool = True,
    registry_path: Path | None = None,
    graph_path: Path | None = None,
) -> int:
    paths = get_paths(project_root)
    ensure_directories(paths)
    resolved_registry_path = registry_path if registry_path is not None else paths.registry_path
    resolved_graph_path = graph_path if graph_path is not None else paths.graph_path
    registry_data = read_json(resolved_registry_path) if resolved_registry_path.exists() else {"endpoints": []}
    graph_data = read_json(resolved_graph_path) if resolved_graph_path.exists() else {"nodes": [], "edges": [], "adjacency": {}, "summary": {}}
    registry_endpoints = registry_data.get("endpoints", [])
    graph = load_graph_from_dict(graph_data)
    target_output_path = output_path if output_path is not None else paths.conversations_path
    corpus_state = GenerationCorpusState() if cross_conversation_steering else None

    target_output_path.parent.mkdir(parents=True, exist_ok=True)
    target_output_path.write_text("", encoding="utf-8")
    chain_lengths = [1, 2, 3, 4]
    for index in range(num_samples):
        requested_chain_length = chain_lengths[index % len(chain_lengths)]
        if graph.summary.get("num_edges", 0) == 0 and graph.summary.get("num_nodes", 0) > 0:
            requested_chain_length = 1
        record = generate_conversation(
            registry_endpoints=registry_endpoints,
            graph=graph,
            chain_length=requested_chain_length,
            seed=seed + index,
            conversation_id=f"conv_{index + 1:04d}",
            cross_conversation_steering=cross_conversation_steering,
            corpus_state=corpus_state,
        )
        append_jsonl(target_output_path, record.to_dict())
    print(f"Generated {num_samples} conversation records")
    return 0


def cmd_evaluate(project_root: Path | None = None, input_path: Path | None = None, report_out: Path | None = None) -> int:
    paths = get_paths(project_root)
    ensure_directories(paths)
    source_path = input_path if input_path is not None else paths.conversations_path
    report_path = report_out if report_out is not None else paths.evaluation_report_path
    conversations = read_jsonl(source_path) if source_path.exists() else []
    scored_records = []
    judge_mode = "heuristic"
    for record in conversations:
        if not isinstance(record, dict):
            continue
        scores = judge_conversation(record, use_llm=False)
        record["judge_scores"] = scores
        judge_mode = scores.get("judge_mode", "heuristic")
        scored_records.append(record)

    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("", encoding="utf-8")
    for record in scored_records:
        append_jsonl(source_path, record)

    summary = compute_dataset_metrics(scored_records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        report_path,
        {
            "summary": summary,
            "records_scored": len(scored_records),
            "judge_mode": judge_mode,
            "input_path": str(source_path),
            "output_report_path": str(report_path),
        },
    )
    print(f"Records scored: {len(scored_records)}")
    print(f"Judge mode used: {judge_mode}")
    print(f"Mean overall score: {summary['mean_overall_score']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toolgen", description="Minimal ToolGen CLI.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root containing data/, src/, and tests/.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a normalized registry from raw tool JSON.")
    build_parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to raw tool JSON. Defaults to data/raw/tools.json under the project root.",
    )
    generate_parser = subparsers.add_parser("generate", help="Create conversation outputs from processed artifacts.")
    generate_parser.add_argument("--num-samples", type=int, default=5, help="Number of conversation records to generate.")
    generate_parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic generation.")
    generate_parser.add_argument("--output", type=Path, default=None, help="Output JSONL path. Defaults to data/outputs/conversations.jsonl.")
    generate_parser.add_argument("--registry-path", type=Path, default=None, help="Explicit registry artifact path to use for generation.")
    generate_parser.add_argument("--graph-path", type=Path, default=None, help="Explicit graph artifact path to use for generation.")
    generate_parser.add_argument(
        "--no-cross-conversation-steering",
        action="store_true",
        help="Disable lightweight frequency-based steering. By default, steering is enabled.",
    )
    evaluate_parser = subparsers.add_parser("evaluate", help="Score generated conversations and write an evaluation report.")
    evaluate_parser.add_argument("--input", type=Path, default=None, help="Input JSONL path. Defaults to data/outputs/conversations.jsonl.")
    evaluate_parser.add_argument("--report-out", type=Path, default=None, help="Evaluation report path. Defaults to data/outputs/evaluation_report.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        return cmd_build(args.project_root, args.input)
    if args.command == "generate":
        return cmd_generate(
            args.project_root,
            args.num_samples,
            args.seed,
            args.output,
            not args.no_cross_conversation_steering,
            args.registry_path,
            args.graph_path,
        )
    if args.command == "evaluate":
        return cmd_evaluate(args.project_root, args.input, args.report_out)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
