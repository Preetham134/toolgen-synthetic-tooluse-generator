from __future__ import annotations

from pathlib import Path
from typing import Any

from toolgen.registry.models import Endpoint
from toolgen.registry.normalize import normalize_endpoint
from toolgen.utils.io import read_json


KNOWN_LIST_KEYS = ("tools", "apis", "endpoints", "data", "records", "items")
RECORD_HINT_KEYS = (
    "tool_name",
    "name",
    "tool",
    "api_name",
    "endpoint",
    "action",
    "parameters",
    "params",
    "input_params",
)


def describe_top_level(data: Any) -> str:
    if isinstance(data, list):
        return "list"
    if isinstance(data, dict):
        return "dict"
    return type(data).__name__


def _looks_like_record(data: dict[str, Any]) -> bool:
    return any(key in data for key in RECORD_HINT_KEYS)


def _extract_records(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in KNOWN_LIST_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return value
        if _looks_like_record(data):
            return [data]

    return []


def load_registry(path: str | Path) -> tuple[list[Endpoint], dict[str, int]]:
    data = read_json(path)
    records = _extract_records(data)
    endpoints: list[Endpoint] = []
    skipped = 0

    for record in records:
        endpoint = normalize_endpoint(record) if isinstance(record, dict) else None
        if endpoint is None:
            skipped += 1
            continue
        endpoints.append(endpoint)

    print(f"Input path: {Path(path)}")
    print(f"Detected top-level type: {describe_top_level(data)}")
    print(f"Extracted raw record count: {len(records)}")
    return endpoints, {"loaded": len(endpoints), "skipped": skipped}
