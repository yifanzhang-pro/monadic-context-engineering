from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    task: str
    history: list[str] = Field(default_factory=list)

    def with_history(self, entry: str) -> AgentState:
        return self.model_copy(update={"history": [*self.history, entry]})


class ToolCall(BaseModel):
    tool_id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    tool_id: str
    content: str
    is_error: bool = False

    @classmethod
    def success(cls, tool_id: str, content: str) -> ToolResult:
        return cls(tool_id=tool_id, content=content, is_error=False)

    @classmethod
    def failure(cls, tool_id: str, content: str) -> ToolResult:
        return cls(tool_id=tool_id, content=content, is_error=True)


ToolHandler = Callable[[AgentState, ToolCall], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        self._tools[name] = handler

    def run(self, state: AgentState, call: ToolCall) -> ToolResult:
        handler = self._tools.get(call.name)
        if handler is None:
            return ToolResult.failure(call.tool_id, f"Tool not found: {call.name}")
        try:
            content = handler(state, call)
            return ToolResult.success(call.tool_id, content)
        except Exception as exc:
            return ToolResult.failure(call.tool_id, f"Tool error: {exc}")


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def search_tool(state: AgentState, call: ToolCall) -> str:
        query = call.arguments.get("query", state.task)
        return f"Search results for '{query}': MCE formalizes agent steps using monads."

    registry.register("search", search_tool)
    return registry
