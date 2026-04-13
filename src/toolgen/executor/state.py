from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationState:
    entities: dict[str, list[dict]] = field(default_factory=dict)
    slots: dict[str, object] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)

    def next_id(self, prefix: str) -> str:
        value = self.counters.get(prefix, 0) + 1
        self.counters[prefix] = value
        return f"{prefix}_{value:03d}"

    def add_entity(self, entity_type: str, entity: dict) -> None:
        self.entities.setdefault(entity_type, []).append(entity)

    def get_entities(self, entity_type: str) -> list[dict]:
        return list(self.entities.get(entity_type, []))

    def remember_tool_output(self, endpoint_id: str, output: dict) -> None:
        self.history.append({"endpoint_id": endpoint_id, "output": output})

    def get_last_tool_output(self, endpoint_id: str | None = None) -> dict | None:
        for item in reversed(self.history):
            if endpoint_id is None or item.get("endpoint_id") == endpoint_id:
                return item.get("output")
        return None

    def set_slot(self, key: str, value: object) -> None:
        self.slots[key] = value

    def get_slot(self, key: str, default: Any = None) -> Any:
        return self.slots.get(key, default)
