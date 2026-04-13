from __future__ import annotations

import math
from typing import Any


def _round(value: float) -> float:
    return round(value, 3)


def _extract_endpoint_ids(record: dict[str, Any]) -> list[str]:
    endpoint_ids: list[str] = []
    for message in record.get("messages", []):
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if isinstance(tool_call, dict) and tool_call.get("endpoint_id"):
                endpoint_ids.append(str(tool_call["endpoint_id"]))
    return endpoint_ids


def compute_tool_usage_entropy(records: list[dict]) -> float:
    counts: dict[str, int] = {}
    total = 0
    for record in records:
        for endpoint_id in _extract_endpoint_ids(record):
            counts[endpoint_id] = counts.get(endpoint_id, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return _round(entropy)


def compute_distinct_tool_pair_ratio(records: list[dict]) -> float:
    total_pairs = 0
    distinct_pairs: set[str] = set()
    for record in records:
        endpoint_ids = _extract_endpoint_ids(record)
        for source, target in zip(endpoint_ids, endpoint_ids[1:]):
            total_pairs += 1
            distinct_pairs.add(f"{source}->{target}")
    if total_pairs == 0:
        return 0.0
    return _round(len(distinct_pairs) / total_pairs)


def compute_category_coverage(records: list[dict]) -> float:
    observed_categories: list[str] = []
    distinct_categories: set[str] = set()
    for record in records:
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        categories = metadata.get("categories", [])
        if not isinstance(categories, list):
            continue
        for category in categories:
            category_name = str(category)
            observed_categories.append(category_name)
            distinct_categories.add(category_name)
    if not observed_categories:
        return 0.0
    return _round(len(distinct_categories) / len(observed_categories))


def compute_dataset_metrics(records: list[dict]) -> dict[str, Any]:
    num_records = len(records)
    if num_records == 0:
        return {
            "num_records": 0,
            "mean_naturalness": 0.0,
            "mean_tool_correctness": 0.0,
            "mean_task_completion": 0.0,
            "mean_overall_score": 0.0,
            "validation_pass_rate": 0.0,
            "avg_num_messages": 0.0,
            "avg_num_tool_calls": 0.0,
            "tool_usage_entropy": 0.0,
            "distinct_tool_pair_ratio": 0.0,
            "category_coverage": 0.0,
        }

    total_naturalness = 0.0
    total_tool_correctness = 0.0
    total_task_completion = 0.0
    total_overall = 0.0
    total_validation_passed = 0
    total_messages = 0
    total_tool_calls = 0

    for record in records:
        judge_scores = record.get("judge_scores", {})
        naturalness = float(judge_scores.get("naturalness", 0.0))
        tool_correctness = float(judge_scores.get("tool_correctness", 0.0))
        task_completion = float(judge_scores.get("task_completion", 0.0))
        overall = (naturalness + tool_correctness + task_completion) / 3.0

        total_naturalness += naturalness
        total_tool_correctness += tool_correctness
        total_task_completion += task_completion
        total_overall += overall

        validation = record.get("validation", {})
        if isinstance(validation, dict) and validation.get("passed") is True:
            total_validation_passed += 1

        messages = record.get("messages", [])
        total_messages += len(messages) if isinstance(messages, list) else 0
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict):
                    tool_calls = message.get("tool_calls", [])
                    if isinstance(tool_calls, list):
                        total_tool_calls += len(tool_calls)

    return {
        "num_records": num_records,
        "mean_naturalness": _round(total_naturalness / num_records),
        "mean_tool_correctness": _round(total_tool_correctness / num_records),
        "mean_task_completion": _round(total_task_completion / num_records),
        "mean_overall_score": _round(total_overall / num_records),
        "validation_pass_rate": _round(total_validation_passed / num_records),
        "avg_num_messages": _round(total_messages / num_records),
        "avg_num_tool_calls": _round(total_tool_calls / num_records),
        "tool_usage_entropy": compute_tool_usage_entropy(records),
        "distinct_tool_pair_ratio": compute_distinct_tool_pair_ratio(records),
        "category_coverage": compute_category_coverage(records),
    }
