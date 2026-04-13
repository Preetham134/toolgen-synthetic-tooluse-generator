from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GraphEdge:
    source_endpoint_id: str
    target_endpoint_id: str
    relation_type: str
    score: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolGraph:
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    adjacency: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [edge.to_dict() for edge in self.edges],
            "adjacency": self.adjacency,
            "summary": self.summary,
        }
