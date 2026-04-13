from __future__ import annotations

from toolgen.graph.models import GraphEdge, ToolGraph
from toolgen.registry.models import Endpoint


SEARCH_WORDS = ("search", "list", "find")
ACTION_WORDS = ("get", "book", "order", "reserve", "buy", "track", "details")
GENERIC_KEYS = {"city", "date", "category", "budget", "location", "status", "name", "count"}


def _endpoint_family(endpoint: Endpoint) -> str:
    text = f"{endpoint.endpoint_id} {endpoint.tool_name} {endpoint.api_name}".lower()
    for family in ("hotel", "flight", "product", "order", "restaurant"):
        if family in text:
            return family
    return "generic"


def _compatible_family_pair(source_family: str, target_family: str) -> bool:
    return {source_family, target_family} in ({"product", "order"}, {"hotel", "flight"})


def _intent(endpoint: Endpoint) -> str:
    text = f"{endpoint.api_name} {endpoint.description}".lower()
    if any(word in text for word in SEARCH_WORDS):
        return "search"
    if any(word in text for word in ("detail", "details", "get")):
        return "details"
    if "order" in text and "search" not in text:
        return "order"
    if any(word in text for word in ("book", "reserve", "buy")):
        return "book"
    if "create" in text:
        return "create"
    return "generic"


def _extract_schema_keys(schema: dict) -> set[str]:
    keys = {str(key).strip() for key in schema.keys() if str(key).strip()}
    properties = schema.get("properties")
    if isinstance(properties, dict):
        keys.update(str(key).strip() for key in properties.keys() if str(key).strip())
    return keys


def _relation_score(source: Endpoint, target: Endpoint) -> tuple[int, list[str]]:
    score = 0
    relations: list[str] = []
    source_family = _endpoint_family(source)
    target_family = _endpoint_family(target)
    same_family = source_family == target_family and source_family != "generic"
    compatible_family = _compatible_family_pair(source_family, target_family)

    if same_family:
        score += 6
        relations.append("same_family")
    elif compatible_family:
        score += 4
        relations.append("compatible_family")

    if source.tool_name == target.tool_name:
        score += 3
        relations.append("same_tool")

    if source.category and source.category == target.category:
        score += 1
        relations.append("same_category")

    source_output_keys = _extract_schema_keys(source.output_schema)
    target_input_keys = set(target.required_params)
    target_input_keys.update(param.name for param in target.input_params)
    overlap = source_output_keys.intersection(target_input_keys)
    if overlap:
        entity_overlap = {
            key for key in overlap if key.endswith("_id") and key not in GENERIC_KEYS and _endpoint_family(source) in key
        }
        generic_overlap = {key for key in overlap if key in GENERIC_KEYS}
        if entity_overlap and (same_family or compatible_family):
            score += 5
            relations.append("entity_id_overlap")
        elif entity_overlap:
            score += 1
            relations.append("weak_entity_overlap")
        elif generic_overlap and (same_family or compatible_family or source.tool_name == target.tool_name):
            score += 1
            relations.append("generic_overlap")

    source_text = f"{source.api_name} {source.description}".lower()
    target_text = f"{target.api_name} {target.description}".lower()
    if (
        any(word in source_text for word in SEARCH_WORDS)
        and any(word in target_text for word in ACTION_WORDS)
        and (same_family or compatible_family)
    ):
        score += 2
        relations.append("search_to_action")

    source_intent = _intent(source)
    target_intent = _intent(target)
    if source_intent == "search" and target_intent == "details" and (same_family or compatible_family):
        score += 3
        relations.append("search_to_details")
    if source_intent == "details" and target_intent in {"book", "order", "create"} and (same_family or compatible_family):
        score += 3
        relations.append("details_to_action")
    if source_intent == "search" and target_intent in {"book", "order", "create"} and (same_family or compatible_family):
        score += 2
        relations.append("search_to_specific_action")

    if not (same_family or compatible_family or source.tool_name == target.tool_name):
        score -= 1

    return score, relations


def build_graph(endpoints: list[Endpoint]) -> ToolGraph:
    nodes = [endpoint.to_dict() for endpoint in endpoints]
    edges: list[GraphEdge] = []
    adjacency: dict[str, list[dict]] = {endpoint.endpoint_id: [] for endpoint in endpoints}

    for source in endpoints:
        for target in endpoints:
            if source.endpoint_id == target.endpoint_id:
                continue

            score, relations = _relation_score(source, target)
            if score <= 0:
                continue

            relation_type = "|".join(relations)
            edge = GraphEdge(
                source_endpoint_id=source.endpoint_id,
                target_endpoint_id=target.endpoint_id,
                relation_type=relation_type,
                score=score,
            )
            edges.append(edge)
            adjacency[source.endpoint_id].append(
                {
                    "target": target.endpoint_id,
                    "score": score,
                    "relation_type": relation_type,
                }
            )

    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: (-item["score"], item["target"]))

    return ToolGraph(
        nodes=nodes,
        edges=edges,
        adjacency=adjacency,
        summary={"num_nodes": len(nodes), "num_edges": len(edges)},
    )
