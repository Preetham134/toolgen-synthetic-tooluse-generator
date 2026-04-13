from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Parameter:
    name: str
    type: str = "string"
    required: bool = False
    description: str = ""
    enum_values: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Endpoint:
    endpoint_id: str
    tool_name: str
    api_name: str
    category: str
    description: str
    input_params: list[Parameter] = field(default_factory=list)
    required_params: list[str] = field(default_factory=list)
    output_schema: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source_tool_id: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["input_params"] = [param.to_dict() for param in self.input_params]
        return data
