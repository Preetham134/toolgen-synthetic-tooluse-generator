from __future__ import annotations

import os
from typing import Any


def _get_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages", [])
    return [message for message in messages if isinstance(message, dict)]


def _final_assistant_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return message
    return None


def _has_clarification_exchange(messages: list[dict[str, Any]]) -> bool:
    for index in range(len(messages) - 2):
        first = messages[index]
        second = messages[index + 1]
        third = messages[index + 2]
        if (
            first.get("role") == "assistant"
            and isinstance(first.get("content"), str)
            and "?" in first.get("content", "")
            and second.get("role") == "user"
        ):
            return True
        if (
            first.get("role") == "user"
            and second.get("role") == "assistant"
            and isinstance(second.get("content"), str)
            and "?" in second.get("content", "")
            and third.get("role") == "user"
        ):
            return True
    return False


def _clamp_score(value: float) -> float:
    return max(1.0, min(5.0, value))


def _conversation_starts_with_search(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls", [])
        if not isinstance(tool_calls, list) or not tool_calls:
            continue
        first_call = tool_calls[0]
        if not isinstance(first_call, dict):
            return False
        endpoint_id = str(first_call.get("endpoint_id", "")).lower()
        return any(word in endpoint_id for word in ("search", "find", "list"))
    return False


def _search_like_user_intent(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).lower()
        return any(word in content for word in ("find", "search", "look up", "options"))
    return False


def _heuristic_judge(record: dict[str, Any]) -> dict[str, Any]:
    messages = _get_messages(record)
    validation = record.get("validation", {}) if isinstance(record.get("validation", {}), dict) else {}
    validation_passed = bool(validation.get("passed", False))
    validation_issues = validation.get("issues", []) if isinstance(validation.get("issues", []), list) else []

    user_messages = [message for message in messages if message.get("role") == "user"]
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    assistant_tool_call_messages = [
        message
        for message in assistant_messages
        if isinstance(message.get("tool_calls", []), list) and len(message.get("tool_calls", [])) > 0
    ]
    final_assistant = _final_assistant_message(messages)
    clarification = _has_clarification_exchange(messages)

    naturalness = 3.0
    if final_assistant is not None:
        naturalness += 0.5
    if user_messages and assistant_messages:
        naturalness += 0.5
    if clarification:
        naturalness += 0.5

    tool_correctness = 3.0
    if validation_passed:
        tool_correctness += 1.0
    if tool_messages:
        tool_correctness += 0.5
    if assistant_tool_call_messages:
        tool_correctness += 0.5

    task_completion = 3.0
    if final_assistant is not None:
        task_completion += 1.0
    if assistant_tool_call_messages:
        task_completion += 0.5
    if not validation_issues:
        task_completion += 0.5

    if not validation_passed:
        tool_correctness -= 0.5
        task_completion -= 0.5
    if any("incompatible id" in issue or "unknown id" in issue for issue in validation_issues):
        tool_correctness -= 1.0
        task_completion -= 0.5
    if _search_like_user_intent(messages) and not _conversation_starts_with_search(messages):
        naturalness -= 0.5
        tool_correctness -= 0.5

    naturalness = _clamp_score(naturalness)
    tool_correctness = _clamp_score(tool_correctness)
    task_completion = _clamp_score(task_completion)
    summary = (
        f"Heuristic judge scored conversation with {len(messages)} messages, "
        f"{len(tool_messages)} tool messages, and validation passed={validation_passed}."
    )

    return {
        "naturalness": naturalness,
        "tool_correctness": tool_correctness,
        "task_completion": task_completion,
        "summary": summary,
        "judge_mode": "heuristic",
    }


def judge_conversation_with_llm(record: dict, model: str | None = None) -> dict:
    _ = model
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TOOLGEN_API_KEY")
    if not api_key:
        return _heuristic_judge(record)
    result = _heuristic_judge(record)
    result["judge_mode"] = "llm"
    result["summary"] = "LLM judge stub used heuristic scoring because real prompt integration is not implemented yet."
    return result


def judge_conversation(record: dict, use_llm: bool = False, model: str | None = None) -> dict:
    if use_llm:
        return judge_conversation_with_llm(record, model=model)
    return _heuristic_judge(record)
