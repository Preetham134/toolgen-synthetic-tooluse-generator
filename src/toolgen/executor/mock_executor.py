from __future__ import annotations

from toolgen.executor.state import ConversationState
from toolgen.registry.models import Endpoint


SEARCH_WORDS = ("search", "list", "find")
DETAIL_WORDS = ("get", "detail", "details")
BOOK_WORDS = ("book", "reserve", "create", "order", "buy")
UPDATE_WORDS = ("update", "modify")
CANCEL_WORDS = ("cancel", "delete")
ENTITY_TYPES = ("hotel", "flight", "restaurant", "product", "order", "event", "item")


def infer_entity_type(endpoint: Endpoint) -> str:
    text = f"{endpoint.api_name} {endpoint.description}".lower()
    for entity_type in ENTITY_TYPES:
        if entity_type in text:
            return entity_type
    return "item"


def extract_reference_id(arguments: dict) -> tuple[str, str] | None:
    for key, value in arguments.items():
        if not isinstance(value, (str, int)):
            continue
        if key == "id" or key.endswith("_id"):
            return key, str(value)
    return None


def _infer_intent(endpoint: Endpoint) -> str:
    text = f"{endpoint.api_name} {endpoint.description}".lower()
    if any(word in text for word in SEARCH_WORDS):
        return "search"
    if any(word in text for word in DETAIL_WORDS):
        return "detail"
    if any(word in text for word in BOOK_WORDS):
        return "book"
    if any(word in text for word in UPDATE_WORDS):
        return "update"
    if any(word in text for word in CANCEL_WORDS):
        return "cancel"
    return "fallback"


def _is_primitive(value: object) -> bool:
    return isinstance(value, (str, int, float, bool))


def _find_entity_by_id(state: ConversationState, entity_id: str) -> dict | None:
    for entities in state.entities.values():
        for entity in entities:
            if str(entity.get("id")) == entity_id:
                return entity
    return None


class MockToolExecutor:
    def execute(self, endpoint: Endpoint, arguments: dict, state: ConversationState) -> dict:
        for key, value in arguments.items():
            if _is_primitive(value):
                state.set_slot(key, value)

        intent = _infer_intent(endpoint)
        if intent == "search":
            output = self._execute_search(endpoint, arguments, state)
        elif intent == "detail":
            output = self._execute_detail(arguments, state)
        elif intent == "book":
            output = self._execute_book(arguments, state)
        elif intent == "update":
            output = self._execute_update(arguments)
        elif intent == "cancel":
            output = self._execute_cancel(arguments)
        else:
            output = {"status": "ok", "echo": arguments}

        state.remember_tool_output(endpoint.endpoint_id, output)
        return output

    def _execute_search(self, endpoint: Endpoint, arguments: dict, state: ConversationState) -> dict:
        entity_type = infer_entity_type(endpoint)
        results = []
        for index in range(2):
            entity = {"id": state.next_id(entity_type), "name": f"{entity_type.title()} {index + 1}"}
            for key, value in arguments.items():
                if _is_primitive(value):
                    entity[key] = value
            state.add_entity(entity_type, entity)
            results.append(entity)
        return {"results": results, "count": len(results)}

    def _execute_detail(self, arguments: dict, state: ConversationState) -> dict:
        reference = extract_reference_id(arguments)
        if reference is not None:
            _, entity_id = reference
            entity = _find_entity_by_id(state, entity_id)
            if entity is not None:
                return entity
            return {"id": entity_id, "name": "Unknown Item", "status": "found"}
        return {"status": "found", "name": "Unknown Item"}

    def _execute_book(self, arguments: dict, state: ConversationState) -> dict:
        reference = extract_reference_id(arguments)
        record_type = "booking"
        if "order" in " ".join(arguments.keys()).lower():
            record_type = "order"
        elif "reserve" in " ".join(arguments.keys()).lower():
            record_type = "reservation"

        output = {"id": state.next_id(record_type), "status": "confirmed"}
        if reference is not None:
            key, value = reference
            output[key] = value
        for key, value in arguments.items():
            if key not in output and _is_primitive(value):
                output[key] = value
        state.add_entity(record_type, output)
        return output

    def _execute_update(self, arguments: dict) -> dict:
        output = {"status": "updated"}
        for key, value in arguments.items():
            if _is_primitive(value):
                output[key] = value
        return output

    def _execute_cancel(self, arguments: dict) -> dict:
        output = {"status": "cancelled"}
        for key, value in arguments.items():
            if _is_primitive(value):
                output[key] = value
        return output
