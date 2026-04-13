# ToolGen

ToolGen is an offline synthetic multi-step tool-use conversation generator. It takes a raw tool/API catalog, normalizes it into a registry, builds a lightweight tool graph, samples tool chains, generates synthetic conversations with a mock executor, validates and repairs them, and scores the results with a heuristic judge.

The repository is designed for reproducible submission runs. It does not depend on live APIs for generation, and it supports paired steering-off / steering-on runs for simple diversity experiments.

## Main CLI Commands

- `build`: create normalized registry and graph artifacts
- `generate`: create synthetic conversations
- `evaluate`: score generated conversations and write a report

## Repository Structure

- `src/toolgen`
  Core implementation: registry loading, graph building, sampling, generation, execution, validation, repair, evaluation, and CLI entrypoints.
- `tests`
  Unit, integration, and end-to-end tests covering the pipeline and quality constraints.
- `data/raw`
  Raw sample tool catalogs. `sample_tools_rich.json` is the main submission-ready example.
- `data/processed`
  Built artifacts such as normalized registries and graph files. Both default and input-specific outputs are written here.
- `data/outputs`
  Generated conversation datasets and evaluation reports.

## Installation And Setup

From the project root:

```powershell
python -m pip install -e .
```

For local execution without installing into the environment, the repository also works with:

```powershell
$env:PYTHONPATH="src"
```

## End-To-End Usage

Use the rich sample input for the main run:

```powershell
python -m toolgen.cli build --input data/raw/sample_tools_rich.json
python -m toolgen.cli generate --num-samples 100 --seed 42 --output data/outputs/final_run.jsonl --registry-path data/processed/sample_tools_rich_registry.json --graph-path data/processed/sample_tools_rich_graph.json
python -m toolgen.cli evaluate --input data/outputs/final_run.jsonl --report-out data/outputs/final_report.json
```

What each stage does:

- `build`
  Loads raw tool JSON, normalizes the registry, builds the graph, and writes both default and input-specific processed artifacts.
- `generate`
  Loads a registry and graph, samples chains, generates conversations, validates them, applies deterministic repair when needed, and writes JSONL output.
- `evaluate`
  Scores generated conversations with the heuristic judge and writes aggregate dataset metrics.

## Reproducibility

The pipeline is seed-driven. `--seed` controls deterministic choices in planning, sampling, and generation. If the same artifact inputs and seed are used, the output should be stable.

### Input-Specific Processed Artifacts

To avoid stale artifact mismatch, `build` writes:

- Default artifacts:
  - `data/processed/registry.json`
  - `data/processed/graph.json`
- Input-specific artifacts:
  - `data/processed/sample_tools_rich_registry.json`
  - `data/processed/sample_tools_rich_graph.json`

For submission runs, it is recommended to use the input-specific artifact paths explicitly during generation.

### Run A vs Run B

Run A disables cross-conversation steering:

```powershell
python -m toolgen.cli generate --num-samples 100 --seed 42 --output data/outputs/run_a.jsonl --registry-path data/processed/sample_tools_rich_registry.json --graph-path data/processed/sample_tools_rich_graph.json --no-cross-conversation-steering
python -m toolgen.cli evaluate --input data/outputs/run_a.jsonl --report-out data/outputs/run_a_report.json
```

Run B enables cross-conversation steering:

```powershell
python -m toolgen.cli generate --num-samples 100 --seed 42 --output data/outputs/run_b.jsonl --registry-path data/processed/sample_tools_rich_registry.json --graph-path data/processed/sample_tools_rich_graph.json
python -m toolgen.cli evaluate --input data/outputs/run_b.jsonl --report-out data/outputs/run_b_report.json
```

## Output Files

### Processed Artifacts

- `registry.json`
  Normalized endpoint registry with a summary and endpoint list.
- `graph.json`
  Tool graph with nodes, scored edges, adjacency, and summary counts.

### Generated Dataset

- `*.jsonl`
  One conversation record per line. Each record includes:
  - `conversation_id`
  - `messages`
  - `validation`
  - `metadata`
  - optionally `judge_scores` after evaluation

Generated JSONL records may contain empty or placeholder `judge_scores` before evaluation. The `evaluate` command computes the final scoring summary, writes the report JSON, and in the current pipeline rewrites the input JSONL with attached scoring fields.

Metadata includes items such as:

- `chain_length`
- `chain_pattern`
- `tools_used`
- `categories`
- `steering_enabled`
- `used_repair`
- `repair_attempts`
- `semantic_quality_flags`

### Evaluation Report

- `*.json`
  Aggregate dataset metrics including:
  - mean judge scores
  - validation pass rate
  - average message/tool-call counts
  - diversity metrics such as tool usage entropy and distinct tool-pair ratio

## Testing

Run the full test suite:

```powershell
python -m pytest -q
```

Test coverage includes:

- unit tests for registry normalization, graph building, sampling, executor behavior, validators, repair, steering, and evaluation
- integration tests for CLI flows
- an end-to-end test that builds rich artifacts, generates 100 samples, evaluates them, and checks the aggregate score threshold

## Current Observed Results

Based on saved repository artifacts:

- Rich sample graph:
  - `10` nodes
  - `28` edges after the final graph-tightening pass
- `data/outputs/final_report.json`:
  - `100` records scored
  - `mean_overall_score = 4.67`
  - `validation_pass_rate = 0.96`
  - `tool_usage_entropy = 3.298`
  - `distinct_tool_pair_ratio = 0.3`
- `data/outputs/quality_fix_check_report.json`:
  - `20` records scored
  - `mean_overall_score = 4.667`
  - `validation_pass_rate = 0.9`

## Limitations

- The graph and sampler are much tighter than earlier versions, but semantic chain quality is still not perfect.
- Some generated chains can still be less natural than a hand-authored conversation.
- The current judge is heuristic-first; the LLM judge path is only a stub wrapper for future extension.
- The executor is intentionally offline and synthetic, so outputs are structurally useful but not realistic API responses.

## Submission Checklist

- `build` works on `data/raw/sample_tools_rich.json`
- `generate` works with explicit rich artifact paths
- `evaluate` works on generated JSONL output
- Run A / Run B commands are documented
- rich sample input is included
- input-specific artifact paths are documented
- tests pass
- README and DESIGN documentation are included

## Final Submission Run

```powershell
python -m pip install -e .
python -m toolgen.cli build --input data/raw/sample_tools_rich.json
python -m toolgen.cli generate --num-samples 100 --seed 42 --output data/outputs/final_run.jsonl --registry-path data/processed/sample_tools_rich_registry.json --graph-path data/processed/sample_tools_rich_graph.json
python -m toolgen.cli evaluate --input data/outputs/final_run.jsonl --report-out data/outputs/final_report.json
python -m pytest -q
```
