from __future__ import annotations

from typing import Any

from toolgen.registry.models import Endpoint, Parameter


def _first_string(record: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _normalize_required_names(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _get_raw_params(record: dict[str, Any]) -> Any:
    for key in ("input_params", "parameters", "params"):
        if key in record:
            return record.get(key)

    input_schema = record.get("input_schema")
    if isinstance(input_schema, dict):
        return input_schema.get("properties")
    return None


def _normalize_param(name: str, raw_param: Any, required_names: set[str]) -> Parameter | None:
    if not name:
        return None

    if isinstance(raw_param, dict):
        param_name = raw_param.get("name", name)
        if not isinstance(param_name, str) or not param_name.strip():
            return None

        enum_values = raw_param.get("enum") or raw_param.get("enum_values") or []
        if not isinstance(enum_values, list):
            enum_values = []

        required = bool(raw_param.get("required", False)) or param_name in required_names
        description = raw_param.get("description") or raw_param.get("desc") or ""
        param_type = raw_param.get("type") or "string"
        return Parameter(
            name=param_name.strip(),
            type=str(param_type).strip() or "string",
            required=required,
            description=str(description).strip(),
            enum_values=[str(value) for value in enum_values if str(value).strip()],
        )

    if isinstance(raw_param, str):
        param_name = raw_param.strip()
        if not param_name:
            return None
        return Parameter(name=param_name, required=param_name in required_names)

    return None


def _normalize_params(record: dict[str, Any]) -> tuple[list[Parameter], list[str]]:
    required_names = _normalize_required_names(record.get("required"))
    required_names.update(_normalize_required_names(record.get("required_params")))

    input_schema = record.get("input_schema")
    if isinstance(input_schema, dict):
        required_names.update(_normalize_required_names(input_schema.get("required")))

    raw_params = _get_raw_params(record)
    params: list[Parameter] = []

    if isinstance(raw_params, dict):
        for name, raw_param in raw_params.items():
            param = _normalize_param(str(name).strip(), raw_param, required_names)
            if param is not None:
                params.append(param)
    elif isinstance(raw_params, list):
        for raw_param in raw_params:
            name = raw_param.get("name", "") if isinstance(raw_param, dict) else str(raw_param).strip()
            param = _normalize_param(name, raw_param, required_names)
            if param is not None:
                params.append(param)

    for param in params:
        if param.required:
            required_names.add(param.name)

    ordered_required = [param.name for param in params if param.name in required_names]
    return params, ordered_required


def normalize_endpoint(record: dict[str, Any]) -> Endpoint | None:
    if not isinstance(record, dict):
        return None

    tool_name = _first_string(record, ["tool_name", "tool", "name", "api_name"])
    api_name = _first_string(record, ["api_name", "endpoint", "name", "action"])

    if not tool_name or not api_name:
        return None

    source_tool_id = _first_string(record, ["tool_id", "id", "source_tool_id"])
    endpoint_id = source_tool_id or f"{tool_name}.{api_name}"
    category = _first_string(record, ["category", "domain", "group"], default="general")
    description = _first_string(record, ["description", "desc", "summary"], default="")

    output_schema = {}
    for key in ("output_schema", "response_schema", "output", "response"):
        value = record.get(key)
        if isinstance(value, dict):
            output_schema = value
            break

    tags = record.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    input_params, required_params = _normalize_params(record)

    return Endpoint(
        endpoint_id=endpoint_id,
        tool_name=tool_name,
        api_name=api_name,
        category=category,
        description=description,
        input_params=input_params,
        required_params=required_params,
        output_schema=output_schema,
        tags=[str(tag) for tag in tags if str(tag).strip()],
        source_tool_id=source_tool_id,
    )
