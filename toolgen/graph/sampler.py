from __future__ import annotations

import random
from typing import Any

from toolgen.graph.models import ToolGraph


def _find_node(graph: ToolGraph, endpoint_id: str) -> dict | None:
    for node in graph.nodes:
        if node.get("endpoint_id") == endpoint_id:
            return node
    return None


def _intent(node: dict[str, Any]) -> str:
    text = f"{node.get('api_name', '')} {node.get('description', '')}".lower()
    if any(word in text for word in ("search", "list", "find")):
        return "search"
    if any(word in text for word in ("detail", "details", "get")):
        return "details"
    if any(word in text for word in ("book", "reserve", "order", "buy")):
        return "book"
    return "other"


def _entity_family(node: dict[str, Any]) -> str | None:
    text = f"{node.get('endpoint_id', '')} {node.get('tool_name', '')} {node.get('api_name', '')}".lower()
    for family in ("hotel", "flight", "product", "restaurant", "booking", "order", "reservation"):
        if family in text:
            return family
    return None


def _compatible_family_pair(source_family: str | None, target_family: str | None) -> bool:
    if source_family is None or target_family is None:
        return False
    return {source_family, target_family} in ({"product", "order"}, {"hotel", "flight"})


def _eligible_start_nodes(graph: ToolGraph, required_category: str | None) -> list[dict[str, Any]]:
    starts: list[dict[str, Any]] = []
    for node in graph.nodes:
        if required_category and node.get("category") != required_category:
            continue
        starts.append(node)
    preferred = [node for node in starts if _intent(node) == "search"]
    return preferred or starts


def _ordering_bonus(current_node: dict[str, Any], target_node: dict[str, Any], step_index: int) -> float:
    current_intent = _intent(current_node)
    target_intent = _intent(target_node)
    current_family = _entity_family(current_node)
    target_family = _entity_family(target_node)
    current_category = str(current_node.get("category", ""))
    target_category = str(target_node.get("category", ""))

    bonus = 0.0
    if step_index == 0 and target_intent != "search":
        bonus -= 4.0
    if current_intent == "search" and target_intent == "details":
        bonus += 3.0
    if current_intent == "search" and target_intent == "book":
        bonus += 2.0
    if current_intent == "details" and target_intent == "book":
        bonus += 3.0
    if current_intent == "search" and "order" in str(target_node.get("api_name", "")).lower():
        bonus += 2.0

    if current_category and target_category and current_category != target_category:
        bonus -= 4.0

    if current_family and target_family and current_family != target_family:
        if current_category == target_category == "travel":
            bonus -= 1.5
        else:
            bonus -= 3.0
    elif current_family and target_family and current_family == target_family:
        bonus += 2.0

    return bonus


def _steered_node_score(node: dict[str, Any], corpus_state: Any) -> float:
    if corpus_state is None:
        return 1.0
    endpoint_id = str(node.get("endpoint_id", ""))
    category = str(node.get("category", "general"))
    return corpus_state.get_tool_weight(endpoint_id) * corpus_state.get_category_weight(category)


def _choose_start_node(start_nodes: list[dict[str, Any]], rng: random.Random, corpus_state: Any) -> str:
    if corpus_state is None:
        return str(rng.choice(start_nodes)["endpoint_id"])
    scored = sorted(
        ((-_steered_node_score(node, corpus_state), str(node.get("endpoint_id", ""))) for node in start_nodes),
        key=lambda item: (item[0], item[1]),
    )
    return scored[0][1]


def _pick_next_target(
    graph: ToolGraph,
    current_id: str,
    visited: set[str],
    required_category: str | None,
    rng: random.Random,
    corpus_state: Any = None,
    step_index: int = 0,
) -> str | None:
    neighbors = graph.adjacency.get(current_id, [])
    preferred: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    current_node = _find_node(graph, current_id)
    if current_node is None:
        return None

    for neighbor in neighbors:
        target_id = neighbor["target"]
        node = _find_node(graph, target_id)
        if node is None:
            continue
        if required_category and node.get("category") != required_category:
            continue
        if target_id in visited:
            fallback.append(neighbor)
        else:
            preferred.append(neighbor)

    choices = preferred or fallback
    if not choices:
        return None

    family_filtered: list[dict[str, Any]] = []
    compatible_filtered: list[dict[str, Any]] = []
    current_family = _entity_family(current_node)
    current_category = str(current_node.get("category", ""))
    for choice in choices:
        node = _find_node(graph, str(choice["target"]))
        if node is None:
            continue
        target_family = _entity_family(node)
        target_category = str(node.get("category", ""))
        if current_family is not None and target_family == current_family:
            family_filtered.append(choice)
        elif _compatible_family_pair(current_family, target_family):
            compatible_filtered.append(choice)
        elif current_category and target_category and current_category == target_category:
            compatible_filtered.append(choice)
    if family_filtered:
        choices = family_filtered
    elif compatible_filtered:
        choices = compatible_filtered

    scored_choices = []
    for choice in choices:
        target_id = str(choice["target"])
        node = _find_node(graph, target_id)
        if node is None:
            continue
        pair_key = f"{current_id}->{target_id}"
        score = float(choice.get("score", 1)) + _ordering_bonus(current_node, node, step_index)
        if corpus_state is not None:
            score = score * _steered_node_score(node, corpus_state) * corpus_state.get_pair_weight(pair_key)
        scored_choices.append((score, target_id))
    if not scored_choices:
        return None
    max_score = max(score for score, _ in scored_choices)
    best_targets = sorted(target_id for score, target_id in scored_choices if score == max_score)
    return rng.choice(best_targets)


def sample_chain(
    graph: ToolGraph,
    chain_length: int = 3,
    required_category: str | None = None,
    min_distinct_tools: int = 1,
    seed: int = 42,
    cross_conversation_steering: bool = False,
    corpus_state: Any = None,
) -> list[dict]:
    if chain_length <= 0:
        raise ValueError("chain_length must be positive")
    if min_distinct_tools <= 0:
        raise ValueError("min_distinct_tools must be positive")
    if not graph.nodes:
        return []

    rng = random.Random(seed)
    start_nodes = _eligible_start_nodes(graph, required_category)
    if not start_nodes:
        return []
    steering_state = corpus_state if cross_conversation_steering else None

    max_attempts = max(10, len(start_nodes) * 5)
    for _ in range(max_attempts):
        current_id = _choose_start_node(start_nodes, rng, steering_state)
        chain_ids = [current_id]
        visited = {current_id}

        while len(chain_ids) < chain_length:
            next_id = _pick_next_target(
                graph,
                current_id,
                visited,
                required_category,
                rng,
                steering_state,
                step_index=len(chain_ids) - 1,
            )
            if next_id is None:
                break
            chain_ids.append(next_id)
            visited.add(next_id)
            current_id = next_id

        if len(chain_ids) != chain_length:
            continue

        chain = [_find_node(graph, endpoint_id) for endpoint_id in chain_ids]
        if any(node is None for node in chain):
            continue

        distinct_tools = {node["tool_name"] for node in chain if "tool_name" in node}
        if len(distinct_tools) < min_distinct_tools:
            continue

        return [node for node in chain if node is not None]

    return []
