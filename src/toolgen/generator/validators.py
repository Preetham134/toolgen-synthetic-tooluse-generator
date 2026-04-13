from __future__ import annotations

from typing import Any

from toolgen.generator.models import ConversationRecord


VALID_ROLES = {"user", "assistant", "tool"}


def _as_record_dict(record: ConversationRecord | dict) -> dict[str, Any]:
    if isinstance(record, ConversationRecord):
        return record.to_dict()
    return record


def _registry_index(registry_endpoints: list[dict] | list[Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for endpoint in registry_endpoints:
        if hasattr(endpoint, "to_dict"):
            data = endpoint.to_dict()
        else:
            data = endpoint
        if isinstance(data, dict) and data.get("endpoint_id"):
            index[str(data["endpoint_id"])] = data
    return index


def infer_expected_id_family(argument_key: str) -> str | None:
    key = str(argument_key)
    if key == "id":
        return None
    if not key.endswith("_id"):
        return None
    return key[: -len("_id")] or None


def infer_id_family(value: str | int) -> str | None:
    text = str(value).strip().lower()
    if not text:
        return None
    if "_" in text:
        prefix = text.split("_", 1)[0]
        if prefix:
            return prefix
    return None


def _record_seen_id(seen_ids: dict[str, set[str]], identifier: str | int, family: str | None) -> None:
    identifier_text = str(identifier)
    families = seen_ids.setdefault(identifier_text, set())
    inferred_family = family or infer_id_family(identifier_text)
    if inferred_family is not None:
        families.add(inferred_family)


def _extract_ids_from_output(content: dict[str, Any]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}

    results = content.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            if item.get("id"):
                _record_seen_id(found, item["id"], None)
            for key, value in item.items():
                if key.endswith("_id") and isinstance(value, (str, int)):
                    _record_seen_id(found, value, infer_expected_id_family(key))

    if content.get("id"):
        _record_seen_id(found, content["id"], None)

    for key, value in content.items():
        if key.endswith("_id") and isinstance(value, (str, int)):
            _record_seen_id(found, value, infer_expected_id_family(key))

    return found


def _extract_argument_ids(arguments: dict[str, Any]) -> list[tuple[str, str]]:
    ids: list[tuple[str, str]] = []
    for key, value in arguments.items():
        if (key == "id" or key.endswith("_id")) and isinstance(value, (str, int)):
            ids.append((key, str(value)))
    return ids


def extract_seen_ids(messages: list[dict[str, Any]]) -> dict[str, set[str]]:
    seen_ids: dict[str, set[str]] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        output_ids = _extract_ids_from_output(content)
        for identifier, families in output_ids.items():
            seen_ids.setdefault(identifier, set()).update(families)
    return seen_ids


def validate_conversation_structure(record: ConversationRecord | dict) -> dict:
    data = _as_record_dict(record)
    issues: list[str] = []

    conversation_id = data.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        issues.append("conversation_id is missing or empty")

    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        issues.append("messages must be a non-empty list")
        return {"passed": not issues, "issues": issues}

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            issues.append(f"message {index} is not a dict")
            continue

        role = message.get("role")
        if role not in VALID_ROLES:
            issues.append(f"message {index} has invalid role")

        content = message.get("content")
        tool_calls = message.get("tool_calls", [])
        has_content = content is not None and content != "" and content != {}
        has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
        if not has_content and not has_tool_calls:
            issues.append(f"message {index} must have content or tool_calls")

        if role == "tool" and not (isinstance(content, dict) or has_content):
            issues.append(f"tool message {index} must have dict or non-empty content")

    return {"passed": not issues, "issues": issues}


def validate_tool_calls_exist_in_registry(record: ConversationRecord | dict, registry_endpoints: list[dict] | list[Any]) -> dict:
    data = _as_record_dict(record)
    issues: list[str] = []
    registry = _registry_index(registry_endpoints)

    for index, message in enumerate(data.get("messages", [])):
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls", []):
            if not isinstance(tool_call, dict):
                continue
            endpoint_id = tool_call.get("endpoint_id")
            if endpoint_id not in registry:
                issues.append(f"message {index} references unknown endpoint_id '{endpoint_id}'")

    return {"passed": not issues, "issues": issues}


def validate_required_params_present(record: ConversationRecord | dict, registry_endpoints: list[dict] | list[Any]) -> dict:
    data = _as_record_dict(record)
    issues: list[str] = []
    registry = _registry_index(registry_endpoints)

    for index, message in enumerate(data.get("messages", [])):
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls", []):
            if not isinstance(tool_call, dict):
                continue
            endpoint_id = tool_call.get("endpoint_id")
            endpoint = registry.get(str(endpoint_id))
            if endpoint is None:
                continue
            arguments = tool_call.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            for required_param in endpoint.get("required_params", []):
                if required_param not in arguments:
                    issues.append(
                        f"message {index} tool call '{endpoint_id}' is missing required param '{required_param}'"
                    )

    return {"passed": not issues, "issues": issues}


def validate_chain_grounding(record: ConversationRecord | dict) -> dict:
    data = _as_record_dict(record)
    issues: list[str] = []
    seen_ids: dict[str, set[str]] = {}
    saw_tool_output = False

    for index, message in enumerate(data.get("messages", [])):
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")
        if role == "tool" and isinstance(content, dict):
            output_ids = _extract_ids_from_output(content)
            for identifier, families in output_ids.items():
                seen_ids.setdefault(identifier, set()).update(families)
            saw_tool_output = True
            continue

        for tool_call in message.get("tool_calls", []):
            if not isinstance(tool_call, dict):
                continue
            arguments = tool_call.get("arguments", {})
            if not isinstance(arguments, dict):
                continue
            for key, value in _extract_argument_ids(arguments):
                if not saw_tool_output:
                    continue
                expected_family = infer_expected_id_family(key)
                seen_families = seen_ids.get(value)
                if seen_families is None:
                    issues.append(
                        f"message {index} tool call '{tool_call.get('endpoint_id')}' uses unknown id '{value}' in '{key}'"
                    )
                    continue
                if expected_family is not None and seen_families and expected_family not in seen_families:
                    issues.append(
                        f"message {index} tool call '{tool_call.get('endpoint_id')}' uses incompatible id '{value}' in '{key}'"
                    )

    return {"passed": not issues, "issues": issues}


def validate_conversation(record: ConversationRecord | dict, registry_endpoints: list[dict] | list[Any]) -> dict:
    checks = [
        validate_conversation_structure(record),
        validate_tool_calls_exist_in_registry(record, registry_endpoints),
        validate_required_params_present(record, registry_endpoints),
        validate_chain_grounding(record),
    ]
    issues: list[str] = []
    for check in checks:
        issues.extend(check["issues"])
    return {"passed": not issues, "issues": issues}
