from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    endpoint_id: str
    arguments: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"endpoint_id": self.endpoint_id, "arguments": dict(self.arguments)}


@dataclass
class Message:
    role: str
    content: str | dict | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
        }


@dataclass
class ConversationRecord:
    conversation_id: str
    messages: list[Message] = field(default_factory=list)
    judge_scores: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "messages": [message.to_dict() for message in self.messages],
            "judge_scores": dict(self.judge_scores),
            "validation": dict(self.validation),
            "metadata": dict(self.metadata),
        }
