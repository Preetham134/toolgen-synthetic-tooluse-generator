from __future__ import annotations

from typing import Any

from toolgen.executor.mock_executor import MockToolExecutor
from toolgen.executor.state import ConversationState
from toolgen.generator.models import ConversationRecord, Message, ToolCall
from toolgen.generator.planner import plan_conversation
from toolgen.generator.repair import repair_conversation
from toolgen.generator.steering import GenerationCorpusState
from toolgen.generator.validators import extract_seen_ids, infer_expected_id_family, validate_conversation
from toolgen.graph.models import GraphEdge, ToolGraph
from toolgen.graph.sampler import sample_chain
from toolgen.registry.models import Endpoint, Parameter


FINAL_RESPONSES = [
    "I found some options and completed the requested steps.",
    "The task has been completed successfully.",
    "I searched and used the available tool results to complete your request.",
]


def _endpoint_intent(endpoint: Endpoint) -> str:
    text = f"{endpoint.api_name} {endpoint.description}".lower()
    if any(word in text for word in ("search", "list", "find")):
        return "search"
    if any(word in text for word in ("detail", "details", "get")):
        return "details"
    if "order" in text and "search" not in text:
        return "order"
    if "create" in text and "order" not in text:
        return "create"
    if any(word in text for word in ("book", "reserve", "buy")):
        return "book"
    return "generic"


def _entity_family_from_endpoint(endpoint: Endpoint) -> str | None:
    text = f"{endpoint.endpoint_id} {endpoint.tool_name} {endpoint.api_name}".lower()
    for family in ("hotel", "flight", "product", "restaurant", "booking", "order", "reservation"):
        if family in text:
            return family
    return None


def _endpoint_from_data(data: Endpoint | dict[str, Any]) -> Endpoint:
    if isinstance(data, Endpoint):
        return data
    input_params = []
    for param in data.get("input_params", []):
        if isinstance(param, Parameter):
            input_params.append(param)
        elif isinstance(param, dict):
            input_params.append(
                Parameter(
                    name=str(param.get("name", "")),
                    type=str(param.get("type", "string")),
                    required=bool(param.get("required", False)),
                    description=str(param.get("description", "")),
                    enum_values=list(param.get("enum_values", [])),
                )
            )
    return Endpoint(
        endpoint_id=str(data.get("endpoint_id", "")),
        tool_name=str(data.get("tool_name", "")),
        api_name=str(data.get("api_name", "")),
        category=str(data.get("category", "general")),
        description=str(data.get("description", "")),
        input_params=input_params,
        required_params=[str(value) for value in data.get("required_params", [])],
        output_schema=data.get("output_schema", {}) if isinstance(data.get("output_schema", {}), dict) else {},
        tags=[str(value) for value in data.get("tags", [])],
        source_tool_id=str(data.get("source_tool_id", "")),
    )


def _record_from_dict(data: dict[str, Any]) -> ConversationRecord:
    messages = []
    for message in data.get("messages", []):
        if not isinstance(message, dict):
            continue
        tool_calls = []
        for tool_call in message.get("tool_calls", []):
            if isinstance(tool_call, dict):
                tool_calls.append(
                    ToolCall(
                        endpoint_id=str(tool_call.get("endpoint_id", "")),
                        arguments=tool_call.get("arguments", {}) if isinstance(tool_call.get("arguments", {}), dict) else {},
                    )
                )
        messages.append(
            Message(
                role=str(message.get("role", "")),
                content=message.get("content"),
                tool_calls=tool_calls,
            )
        )
    return ConversationRecord(
        conversation_id=str(data.get("conversation_id", "")),
        messages=messages,
        judge_scores=data.get("judge_scores", {}) if isinstance(data.get("judge_scores", {}), dict) else {},
        validation=data.get("validation", {}) if isinstance(data.get("validation", {}), dict) else {},
        metadata=data.get("metadata", {}) if isinstance(data.get("metadata", {}), dict) else {},
    )


def load_graph_from_dict(data: dict[str, Any]) -> ToolGraph:
    edges = []
    for edge in data.get("edges", []):
        if isinstance(edge, dict):
            edges.append(
                GraphEdge(
                    source_endpoint_id=str(edge.get("source_endpoint_id", "")),
                    target_endpoint_id=str(edge.get("target_endpoint_id", "")),
                    relation_type=str(edge.get("relation_type", "")),
                    score=int(edge.get("score", 0)),
                )
            )
    return ToolGraph(
        nodes=[node for node in data.get("nodes", []) if isinstance(node, dict)],
        edges=edges,
        adjacency=data.get("adjacency", {}) if isinstance(data.get("adjacency", {}), dict) else {},
        summary=data.get("summary", {}) if isinstance(data.get("summary", {}), dict) else {},
    )


def _extract_recent_id(state: ConversationState) -> str | None:
    last_output = state.get_last_tool_output()
    if not isinstance(last_output, dict):
        return None
    results = last_output.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict) and first.get("id"):
            return str(first["id"])
    if last_output.get("id"):
        return str(last_output["id"])
    for key, value in last_output.items():
        if key.endswith("_id") and isinstance(value, (str, int)):
            return str(value)
    return None


def _extract_compatible_recent_id(state: ConversationState, param_name: str) -> str | None:
    expected_family = infer_expected_id_family(param_name)
    history_messages = []
    for item in reversed(state.history):
        output = item.get("output")
        if isinstance(output, dict):
            history_messages.append({"role": "tool", "content": output})
    seen_ids = extract_seen_ids(history_messages)
    if expected_family is None:
        return next(reversed(seen_ids), None) if seen_ids else None
    compatible = [identifier for identifier, families in seen_ids.items() if expected_family in families]
    return compatible[-1] if compatible else None


def _default_value_for_param(endpoint: Endpoint, param_name: str) -> object:
    param_type = "string"
    for param in endpoint.input_params:
        if param.name == param_name:
            param_type = param.type
            break
    lowered = param_type.lower()
    if lowered in {"int", "integer", "number", "float"}:
        return 1
    if lowered in {"bool", "boolean"}:
        return True
    return "sample_value"


def _build_arguments(endpoint: Endpoint, state: ConversationState, planner_args: dict[str, Any]) -> dict[str, Any]:
    arguments = dict(planner_args)
    endpoint_intent = _endpoint_intent(endpoint)
    endpoint_family = _entity_family_from_endpoint(endpoint)

    for required_param in endpoint.required_params:
        if required_param in arguments:
            continue
        if required_param == "id" or required_param.endswith("_id"):
            compatible_id = _extract_compatible_recent_id(state, required_param)
            if compatible_id is not None:
                arguments[required_param] = compatible_id
                continue
            if not state.history and endpoint_intent == "details" and endpoint_family is not None:
                arguments[required_param] = f"{endpoint_family}_001"
        elif state.get_slot(required_param) is not None:
            arguments[required_param] = state.get_slot(required_param)
        else:
            arguments[required_param] = _default_value_for_param(endpoint, required_param)

    return arguments


def _sample_endpoint_chain(
    graph: ToolGraph,
    chain_length: int,
    required_category: str | None,
    min_distinct_tools: int,
    seed: int,
    cross_conversation_steering: bool = False,
    corpus_state: GenerationCorpusState | None = None,
) -> list[dict[str, Any]]:
    chain = sample_chain(
        graph,
        chain_length=chain_length,
        required_category=required_category,
        min_distinct_tools=min_distinct_tools,
        seed=seed,
        cross_conversation_steering=cross_conversation_steering,
        corpus_state=corpus_state,
    )
    if chain:
        return chain
    if graph.nodes:
        fallback = sample_chain(
            graph,
            chain_length=1,
            required_category=required_category,
            min_distinct_tools=1,
            seed=seed,
            cross_conversation_steering=cross_conversation_steering,
            corpus_state=corpus_state,
        )
        if fallback:
            return fallback
        return [graph.nodes[0]]
    return []


def generate_conversation(
    registry_endpoints: list,
    graph: ToolGraph,
    chain_length: int = 3,
    seed: int = 42,
    required_category: str | None = None,
    min_distinct_tools: int = 1,
    conversation_id: str = "conv_0001",
    cross_conversation_steering: bool = False,
    corpus_state: GenerationCorpusState | None = None,
) -> ConversationRecord:
    chain_data = _sample_endpoint_chain(
        graph,
        chain_length,
        required_category,
        min_distinct_tools,
        seed,
        cross_conversation_steering=cross_conversation_steering,
        corpus_state=corpus_state,
    )
    chain = [_endpoint_from_data(item) for item in chain_data]

    plan = plan_conversation([endpoint.to_dict() for endpoint in chain], seed=seed)
    state = ConversationState()
    executor = MockToolExecutor()
    messages = [Message(role="user", content=plan["user_goal"])]

    if plan["requires_clarification"]:
        messages.append(Message(role="assistant", content=plan["clarification_question"]))
        messages.append(Message(role="user", content=plan["clarification_answer"]))

    planner_args = dict(plan["initial_arguments"])
    for endpoint in chain:
        arguments = _build_arguments(endpoint, state, planner_args)
        messages.append(
            Message(
                role="assistant",
                content=f"I'll use {endpoint.api_name} to help with this request.",
                tool_calls=[ToolCall(endpoint_id=endpoint.endpoint_id, arguments=arguments)],
            )
        )
        output = executor.execute(endpoint, arguments, state)
        messages.append(Message(role="tool", content=output))

    final_response = FINAL_RESPONSES[seed % len(FINAL_RESPONSES)]
    messages.append(Message(role="assistant", content=final_response))

    record = ConversationRecord(
        conversation_id=conversation_id,
        messages=messages,
        metadata={
            "seed": seed,
            "chain_length": len(chain),
            "tools_used": [endpoint.tool_name for endpoint in chain],
            "categories": [endpoint.category for endpoint in chain],
            "chain_pattern": "->".join(_endpoint_intent(endpoint) for endpoint in chain),
            "had_clarification": plan["requires_clarification"],
            "initial_arguments": planner_args,
            "steering_enabled": cross_conversation_steering,
            "placeholder_judge": True,
        },
    )
    record.validation = validate_conversation(record, registry_endpoints)
    record.metadata["semantic_quality_flags"] = list(record.validation.get("issues", []))
    if record.validation["passed"]:
        record.metadata["used_repair"] = False
        record.metadata["repair_attempts"] = 0
        if corpus_state is not None:
            corpus_state.record_conversation(record.to_dict())
        return record

    repair_result = repair_conversation(record, registry_endpoints)
    repaired_record = _record_from_dict(repair_result["record"])
    repaired_record.validation = repair_result["final_validation"]
    repaired_record.metadata["used_repair"] = repair_result["repair_applied"]
    repaired_record.metadata["repair_attempts"] = repair_result["repair_attempts"]
    repaired_record.metadata.setdefault("steering_enabled", cross_conversation_steering)
    repaired_record.metadata["semantic_quality_flags"] = list(repaired_record.validation.get("issues", []))
    if corpus_state is not None:
        corpus_state.record_conversation(repaired_record.to_dict())
    return repaired_record
