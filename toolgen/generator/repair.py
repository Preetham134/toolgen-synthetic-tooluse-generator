from __future__ import annotations

from copy import deepcopy
from typing import Any

from toolgen.generator.models import ConversationRecord
from toolgen.generator.validators import extract_seen_ids, infer_expected_id_family, infer_id_family, validate_conversation


DEFAULT_STRING_FIELDS = {
    "city": "Paris",
    "location": "Paris",
    "category": "general",
    "date": "2026-04-11",
}


def _as_record_dict(record: ConversationRecord | dict) -> dict[str, Any]:
    if isinstance(record, ConversationRecord):
        return record.to_dict()
    return deepcopy(record)


def find_registry_endpoint(endpoint_id: str, registry_endpoints: list[Any]) -> dict[str, Any] | None:
    for endpoint in registry_endpoints:
        data = endpoint.to_dict() if hasattr(endpoint, "to_dict") else endpoint
        if isinstance(data, dict) and data.get("endpoint_id") == endpoint_id:
            return data
    return None


def extract_known_ids(messages: list[dict[str, Any]]) -> list[str]:
    return list(extract_seen_ids(messages).keys())


def _compatible_known_id(messages: list[dict[str, Any]], param_name: str) -> str | None:
    expected_family = infer_expected_id_family(param_name)
    if expected_family is None:
        known_ids = extract_known_ids(messages)
        return known_ids[-1] if known_ids else None
    seen_ids = extract_seen_ids(messages)
    compatible = [identifier for identifier, families in seen_ids.items() if expected_family in families]
    return compatible[-1] if compatible else None


def _get_param_type(endpoint: dict[str, Any], param_name: str) -> str:
    for param in endpoint.get("input_params", []):
        if isinstance(param, dict) and param.get("name") == param_name:
            return str(param.get("type", "string"))
    return "string"


def _default_value(param_name: str, endpoint: dict[str, Any], known_ids: list[str]) -> object:
    if param_name == "id" or param_name.endswith("_id"):
        explicit_family = infer_expected_id_family(param_name)
        if explicit_family is not None:
            return f"{explicit_family}_001"
        if known_ids:
            return known_ids[-1]
        return "item_001"
    if param_name in DEFAULT_STRING_FIELDS:
        return DEFAULT_STRING_FIELDS[param_name]

    param_type = _get_param_type(endpoint, param_name).lower()
    if param_type in {"int", "integer", "number", "float"}:
        return 1
    if param_type in {"bool", "boolean"}:
        return True
    return "sample_value"


def _get_initial_arguments(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    initial_arguments = metadata.get("initial_arguments", {})
    if isinstance(initial_arguments, dict):
        return initial_arguments
    return {}


def fill_missing_required_params(record: dict[str, Any], registry_endpoints: list[Any]) -> bool:
    changed = False
    initial_arguments = _get_initial_arguments(record)
    messages = record.get("messages", [])
    known_ids = extract_known_ids(messages)

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls", []):
            if not isinstance(tool_call, dict):
                continue
            endpoint = find_registry_endpoint(str(tool_call.get("endpoint_id", "")), registry_endpoints)
            if endpoint is None:
                continue
            arguments = tool_call.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
                tool_call["arguments"] = arguments
            for required_param in endpoint.get("required_params", []):
                if required_param in arguments:
                    continue
                if required_param in initial_arguments:
                    arguments[required_param] = initial_arguments[required_param]
                    changed = True
                    continue
                if required_param == "id" or str(required_param).endswith("_id"):
                    compatible_id = _compatible_known_id(messages[:message_index], str(required_param))
                    if compatible_id is not None:
                        arguments[required_param] = compatible_id
                        changed = True
                        continue
                    endpoint_name = str(endpoint.get("api_name", "")).lower()
                    if message_index == 0 and any(word in endpoint_name for word in ("detail", "details", "get")):
                        arguments[required_param] = _default_value(str(required_param), endpoint, known_ids)
                        changed = True
                    continue
                arguments[required_param] = _default_value(str(required_param), endpoint, known_ids)
                changed = True
    return changed


def _replace_hallucinated_ids(record: dict[str, Any]) -> bool:
    changed = False
    seen_ids: dict[str, set[str]] = {}

    for message in record.get("messages", []):
        if not isinstance(message, dict):
            continue

        if message.get("role") == "tool":
            content = message.get("content")
            if isinstance(content, dict):
                for identifier, families in extract_seen_ids([message]).items():
                    seen_ids.setdefault(identifier, set()).update(families)
            continue

        for tool_call in message.get("tool_calls", []):
            if not isinstance(tool_call, dict):
                continue
            arguments = tool_call.get("arguments", {})
            if not isinstance(arguments, dict):
                continue
            for key, value in list(arguments.items()):
                if not (key == "id" or key.endswith("_id")):
                    continue
                if not isinstance(value, (str, int)):
                    continue
                expected_family = infer_expected_id_family(key)
                compatible_ids = []
                if expected_family is None:
                    compatible_ids = list(seen_ids.keys())
                else:
                    compatible_ids = [
                        identifier for identifier, families in seen_ids.items() if expected_family in families
                    ]
                if not compatible_ids:
                    continue
                if str(value) not in compatible_ids:
                    arguments[key] = compatible_ids[-1]
                    changed = True
    return changed


def _ensure_final_assistant_message(record: dict[str, Any]) -> bool:
    messages = record.get("messages", [])
    if not isinstance(messages, list):
        return False
    if messages:
        last_message = messages[-1]
        if (
            isinstance(last_message, dict)
            and last_message.get("role") == "assistant"
            and last_message.get("content")
        ):
            return False
    messages.append(
        {
            "role": "assistant",
            "content": "I completed the requested steps using the available tool results.",
            "tool_calls": [],
        }
    )
    return True


def _has_final_assistant_message(record: dict[str, Any]) -> bool:
    messages = record.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return False
    last_message = messages[-1]
    return (
        isinstance(last_message, dict)
        and last_message.get("role") == "assistant"
        and bool(last_message.get("content"))
    )


def repair_conversation(
    record,
    registry_endpoints,
    max_repair_attempts: int = 2,
) -> dict:
    record_dict = _as_record_dict(record)
    initial_validation = validate_conversation(record_dict, registry_endpoints)
    if initial_validation["passed"] and _has_final_assistant_message(record_dict):
        return {
            "record": record_dict,
            "repair_applied": False,
            "repair_attempts": 0,
            "final_validation": initial_validation,
        }

    repair_applied = False
    attempts_used = 0
    final_validation = initial_validation

    for attempt in range(1, max_repair_attempts + 1):
        changed = False
        changed = fill_missing_required_params(record_dict, registry_endpoints) or changed
        changed = _replace_hallucinated_ids(record_dict) or changed
        changed = _ensure_final_assistant_message(record_dict) or changed

        final_validation = validate_conversation(record_dict, registry_endpoints)
        attempts_used = attempt
        repair_applied = repair_applied or changed
        if final_validation["passed"]:
            break
        if not changed:
            break

    return {
        "record": record_dict,
        "repair_applied": repair_applied,
        "repair_attempts": attempts_used,
        "final_validation": final_validation,
    }
