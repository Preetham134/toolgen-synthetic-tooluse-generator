from __future__ import annotations

from typing import Any


DEFAULT_VALUES = {
    "city": "Paris",
    "date": "2026-04-11",
    "location": "Paris",
    "category": "electronics",
    "budget": 200,
}


def _get_required_params(endpoint: dict[str, Any]) -> list[str]:
    required_params = endpoint.get("required_params", [])
    if isinstance(required_params, list):
        return [str(param) for param in required_params]
    return []


def _build_user_goal(endpoint: dict[str, Any]) -> str:
    text = f"{endpoint.get('api_name', '')} {endpoint.get('description', '')}".lower()
    category = str(endpoint.get("category", "")).strip()

    if "hotel" in text:
        return "Find me a hotel in Paris"
    if "book" in text and "hotel" in text:
        return "Book a hotel for me"
    if "restaurant" in text:
        return "Look up restaurant options"
    if "product" in text or category == "shopping":
        return "Search for a product"
    if category == "travel":
        return "Help me plan a travel booking"
    return "Help me complete this task"


def plan_conversation(chain: list, seed: int = 42) -> dict:
    first_endpoint = chain[0] if chain else {}
    required_params = _get_required_params(first_endpoint)

    initial_arguments = {
        key: value
        for key, value in DEFAULT_VALUES.items()
        if key in required_params or key in str(first_endpoint.get("api_name", "")).lower()
    }
    if not initial_arguments:
        initial_arguments = {"city": "Paris"}

    clarification_keys = [key for key in ("city", "date", "location", "category", "budget") if key in required_params]
    requires_clarification = bool(clarification_keys) and seed % 2 == 0

    clarification_question = None
    clarification_answer = None
    if requires_clarification:
        key = clarification_keys[0]
        clarification_question = f"What {key} would you like me to use?"
        clarification_answer = str(DEFAULT_VALUES.get(key, "sample_value"))
        initial_arguments.setdefault(key, DEFAULT_VALUES.get(key, "sample_value"))

    return {
        "user_goal": _build_user_goal(first_endpoint),
        "requires_clarification": requires_clarification,
        "clarification_question": clarification_question,
        "clarification_answer": clarification_answer,
        "initial_arguments": initial_arguments,
    }
