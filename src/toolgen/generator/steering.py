from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationCorpusState:
    tool_usage: dict[str, int] = field(default_factory=dict)
    category_usage: dict[str, int] = field(default_factory=dict)
    tool_pair_usage: dict[str, int] = field(default_factory=dict)
    chain_length_usage: dict[int, int] = field(default_factory=dict)

    def record_conversation(self, record: dict) -> None:
        messages = record.get("messages", [])
        endpoint_ids: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            tool_calls = message.get("tool_calls", [])
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and tool_call.get("endpoint_id"):
                    endpoint_ids.append(str(tool_call["endpoint_id"]))
                    self.tool_usage[str(tool_call["endpoint_id"])] = self.tool_usage.get(str(tool_call["endpoint_id"]), 0) + 1

        metadata = record.get("metadata", {})
        if isinstance(metadata, dict):
            categories = metadata.get("categories", [])
            if isinstance(categories, list):
                for category in categories:
                    category_key = str(category)
                    self.category_usage[category_key] = self.category_usage.get(category_key, 0) + 1
            chain_length = metadata.get("chain_length")
            if isinstance(chain_length, int):
                self.chain_length_usage[chain_length] = self.chain_length_usage.get(chain_length, 0) + 1

        for source, target in zip(endpoint_ids, endpoint_ids[1:]):
            pair_key = f"{source}->{target}"
            self.tool_pair_usage[pair_key] = self.tool_pair_usage.get(pair_key, 0) + 1

    def get_tool_weight(self, endpoint_id: str) -> float:
        return 1.0 / (1 + self.tool_usage.get(endpoint_id, 0))

    def get_category_weight(self, category: str) -> float:
        return 1.0 / (1 + self.category_usage.get(category, 0))

    def get_pair_weight(self, pair_key: str) -> float:
        return 1.0 / (1 + self.tool_pair_usage.get(pair_key, 0))
